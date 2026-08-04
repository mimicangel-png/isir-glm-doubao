#!/usr/bin/env python3
"""
VCP (波动收缩形态) A股回测
=========================
检测 Minervini VCP 形态在 A股 445只股票池中的预测力。

流程:
  1. Stage 2 趋势模板过滤 (7项)
  2. Swing High/Low 识别 + 收缩形态检测
  3. 量能枯竭检测
  4. 突破确认 → 买入信号
  5. 统计前向收益 vs 基准

用法: python3 vcp_backtest.py
"""

import os, json, math, sys, time
from datetime import datetime
from collections import defaultdict
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stock_db import StockDB
import sector_map
import unified_scoring_engine as eng

# ================================================================
# VCP 检测参数
# ================================================================

SWING_WINDOW = 3        # swing high/low 两侧窗口
MIN_CONTRACTIONS = 2    # 最少收缩次数
MAX_CONTRACTIONS = 5    # 最多检测
CONTRACTION_RATIO = 0.85  # 每次收缩 ≤ 上次 × 0.85 (A股放宽)
T1_MIN_DEPTH = 0.03     # 首次回调最小 3% (A股小回调也算)
T1_MAX_DEPTH = 0.50     # 首次回调最大 50%
DRY_UP_THRESHOLD = 0.70 # 量能枯竭阈值
BREAKOUT_VOL_RATIO = 1.5  # 突破量能倍数
LOOKBACK_DAYS = 200     # VCP检测回看天数

# Stage 2 趋势模板参数 (适配A股)
STAGE2_PRICE_ABOVE_MA50 = True
STAGE2_PRICE_ABOVE_MA150 = True
STAGE2_MA150_ABOVE_MA200 = True
STAGE2_MA200_UPTREND_DAYS = 22
STAGE2_MIN_FROM_LOW = 0.15   # A股放宽到15% (原25%)
STAGE2_MAX_FROM_HIGH = 0.35  # A股放宽到35% (原25%)

# ================================================================
# Stage 2 趋势模板
# ================================================================

def check_stage2(closes, highs, lows):
    """检查 Stage 2 趋势模板 7 项"""
    n = len(closes)
    if n < 200:
        return 0, []

    price = closes[-1]
    ma50 = np.mean(closes[-50:])
    ma150 = np.mean(closes[-150:])
    ma200 = np.mean(closes[-200:])

    # MA200 是否上升 22 天
    ma200_22d_ago = np.mean(closes[-222:-22]) if n >= 222 else ma200
    ma200_uptrend = ma200 > ma200_22d_ago

    high_52w = max(highs[-250:]) if n >= 250 else max(highs)
    low_52w = min(lows[-250:]) if n >= 250 else min(lows)
    from_low = (price - low_52w) / low_52w if low_52w > 0 else 0
    from_high = (high_52w - price) / high_52w if high_52w > 0 else 1

    checks = []
    checks.append(("price>MA50", price > ma50))
    checks.append(("price>MA150", price > ma150))
    checks.append(("MA150>MA200", ma150 > ma200))
    checks.append(("MA200上升", ma200_uptrend))
    checks.append(("距低点>15%", from_low > STAGE2_MIN_FROM_LOW))
    checks.append(("距高点<35%", from_high < STAGE2_MAX_FROM_HIGH))
    # RS排名需要截面计算, 这里用绝对涨幅近似
    ret_250d = (price / closes[-250] - 1) if n >= 250 and closes[-250] > 0 else 0
    checks.append(("250d涨幅>0", ret_250d > 0))

    passed = sum(1 for _, ok in checks if ok)
    return passed, checks

# ================================================================
# Swing High/Low 检测
# ================================================================

def find_swings(highs, lows, window=3):
    """识别 Swing High 和 Swing Low"""
    n = len(highs)
    swings = []  # [(index, type, price)]

    for i in range(window, n - window):
        # Swing High: high[i] 是前后 window 范围内最高
        is_high = all(highs[i] >= highs[i + j] for j in range(-window, window + 1) if j != 0)
        if is_high:
            swings.append((i, "H", highs[i]))

        # Swing Low: low[i] 是前后 window 范围内最低
        is_low = all(lows[i] <= lows[i + j] for j in range(-window, window + 1) if j != 0)
        if is_low:
            swings.append((i, "L", lows[i]))

    # 按索引排序, 去重(同一根K线不可能同时是H和L, 除非窗口内有平价)
    swings.sort(key=lambda x: x[0])
    # 合并相邻同类型
    merged = []
    for s in swings:
        if merged and merged[-1][1] == s[1] and s[0] - merged[-1][0] <= window:
            if s[1] == "H":
                merged[-1] = (merged[-1][0], "H", max(merged[-1][2], s[2]))
            else:
                merged[-1] = (merged[-1][0], "L", min(merged[-1][2], s[2]))
        else:
            merged.append(list(s))

    return [(s[0], s[1], s[2]) for s in merged]

# ================================================================
# VCP 收缩形态检测
# ================================================================

def detect_vcp(swings, closes, volumes):
    """
    检测 VCP 收缩形态

    逻辑:
      1. 从 swings 中提取 H-L 对 (每次回调的 high→low)
      2. 从末尾往前找最长的"逐次收紧"子序列
      3. pivot = 最近的 swing high
      4. 量能枯竭 + 突破检测
    """
    result = {
        "is_vcp": False, "contractions": [], "pivot": 0,
        "dry_up_ratio": 1.0, "breakout": False, "breakout_vol": 0,
        "quality_score": 0,
    }

    if len(swings) < 5:
        return result

    # 提取 H-L 对: 每个 swing high 后面跟一个 swing low
    pairs = []
    i = 0
    while i < len(swings) - 1:
        if swings[i][1] == "H":
            # 找紧接的下一个 L
            j = i + 1
            while j < len(swings) and swings[j][1] != "L":
                j += 1
            if j < len(swings):
                h_price = swings[i][2]
                l_price = swings[j][2]
                if h_price > 0 and l_price < h_price:
                    depth = (h_price - l_price) / h_price
                    pairs.append({
                        "high_idx": swings[i][0], "high_price": h_price,
                        "low_idx": swings[j][0], "low_price": l_price,
                        "depth": depth,
                    })
            i = j + 1
        else:
            i += 1

    if len(pairs) < MIN_CONTRACTIONS:
        return result

    # pivot = 最近的 swing high
    pivot = None
    pivot_idx = None
    for s in reversed(swings):
        if s[1] == "H":
            pivot = s[2]
            pivot_idx = s[0]
            break
    if pivot is None:
        return result

    # 从末尾往前找最长的"逐次收紧"子序列
    # 要求: depth[i] <= depth[i-1] * CONTRACTION_RATIO
    best_seq = []
    # 从最后一对往前扫描
    current_seq = [pairs[-1]]
    for i in range(len(pairs) - 2, -1, -1):
        prev_depth = current_seq[0]["depth"]
        curr_depth = pairs[i]["depth"]
        # 当前收缩必须 >= 上次的 CONTRACTION_RATIO 倍 (即更深的在前)
        # 也就是: 后面的 depth <= 前面的 depth * RATIO
        if curr_depth > 0 and curr_depth >= prev_depth * 0.5:  # 不能太浅(噪音)
            if curr_depth <= prev_depth / CONTRACTION_RATIO + 0.01:  # 允许微小的误差
                current_seq.insert(0, pairs[i])
            else:
                break
        else:
            break

    if len(current_seq) < MIN_CONTRACTIONS:
        # 放宽: 只要最后2次在收紧就算
        if len(pairs) >= 2 and pairs[-1]["depth"] < pairs[-2]["depth"] * CONTRACTION_RATIO:
            current_seq = pairs[-2:]
        else:
            return result

    current_seq = current_seq[-MAX_CONTRACTIONS:]

    # 检查 T1 范围
    t1 = current_seq[0]["depth"]
    if t1 < T1_MIN_DEPTH or t1 > T1_MAX_DEPTH:
        return result

    # 量能枯竭检测
    n = len(volumes)
    vol_50d = np.mean(volumes[-50:]) if n >= 50 else np.mean(volumes)
    vol_10d = np.mean(volumes[-10:]) if n >= 10 else np.mean(volumes)
    dry_up = vol_10d / vol_50d if vol_50d > 0 else 1.0

    # 突破检测
    current_close = closes[-1]
    current_vol = volumes[-1]
    breakout = current_close > pivot
    breakout_vol = current_vol / vol_50d if vol_50d > 0 else 0
    breakout_confirmed = breakout and breakout_vol >= BREAKOUT_VOL_RATIO

    # 质量评分 (0-100)
    score = 0
    score += min(len(current_seq) * 10, 40)
    last_depth = current_seq[-1]["depth"]
    score += max(0, 20 - last_depth * 100)
    if dry_up < 0.30: score += 20
    elif dry_up < 0.50: score += 15
    elif dry_up < 0.70: score += 10
    if breakout_confirmed: score += 20
    elif breakout: score += 10

    result["is_vcp"] = True
    result["contractions"] = current_seq
    result["pivot"] = pivot
    result["dry_up_ratio"] = round(dry_up, 3)
    result["breakout"] = breakout
    result["breakout_vol"] = round(breakout_vol, 2)
    result["quality_score"] = round(score, 1)

    return result

# ================================================================
# 完整 VCP 检测流程
# ================================================================

def scan_vcp(klines_dict, verbose=True):
    """扫描所有股票的 VCP 形态"""
    results = []
    stage2_pass = 0
    vcp_found = 0
    breakout_found = 0

    for code, k in klines_dict.items():
        if len(k) < LOOKBACK_DAYS:
            continue

        closes = [b["close"] for b in k]
        highs = [b["high"] for b in k]
        lows = [b["low"] for b in k]
        volumes = [b["volume"] for b in k]

        # 1. Stage 2 过滤
        passed, checks = check_stage2(closes, highs, lows)
        if passed < 6:
            continue
        stage2_pass += 1

        # 2. Swing 检测
        swings = find_swings(highs, lows, SWING_WINDOW)

        # 3. VCP 检测
        vcp = detect_vcp(swings, closes, volumes)
        if not vcp["is_vcp"]:
            continue
        vcp_found += 1

        if vcp["breakout"]:
            breakout_found += 1

        results.append({
            "code": code,
            "close": closes[-1],
            "stage2_score": passed,
            "vcp": vcp,
            "n_contractions": len(vcp["contractions"]),
            "pivot": vcp["pivot"],
            "dry_up": vcp["dry_up_ratio"],
            "breakout": vcp["breakout"],
            "breakout_vol": vcp["breakout_vol"],
            "quality": vcp["quality_score"],
        })

    if verbose:
        print(f"  Stage2通过: {stage2_pass}只 | VCP形态: {vcp_found}只 | 突破确认: {breakout_found}只")

    return results

# ================================================================
# 回测: VCP信号 vs 基准
# ================================================================

def run_vcp_backtest(klines, extra_today, sectors, backtest_days=85):
    """回测 VCP 信号的预测力"""
    all_dates = set()
    for k in klines.values():
        for b in k:
            all_dates.add(b["date"])
    dates = sorted(all_dates)

    start_idx = max(LOOKBACK_DAYS + 10, len(dates) - backtest_days)
    test_dates = dates[start_idx:]
    # 需要留20日前向收益空间
    cutoff = len(dates) - 20
    test_dates = [d for d in test_dates if dates.index(d) < cutoff]

    print(f"  回测区间: {test_dates[0]} ~ {test_dates[-1]} ({len(test_dates)}日)")

    # 三组: VCP突破, VCP未突破(预突破), 全市场基准
    vcp_breakout_returns = {"5d": [], "10d": [], "20d": []}
    vcp_prebreakout_returns = {"5d": [], "10d": [], "20d": []}
    all_market_returns = {"5d": [], "10d": [], "20d": []}
    isir_top30_returns = {"5d": [], "10d": [], "20d": []}

    daily_vcp_counts = []

    for di, date in enumerate(test_dates):
        if di % 20 == 0:
            print(f"  进度: {di}/{len(test_dates)}")

        day_klines = {}
        for code, k in klines.items():
            day_bars = [b for b in k if b["date"] <= date]
            if len(day_bars) >= LOOKBACK_DAYS:
                day_klines[code] = day_bars

        if len(day_klines) < 100:
            continue

        # 计算 ISIR TOP30
        day_extra = {}
        for code in day_klines:
            kb = day_klines[code]
            if len(kb) >= 2:
                last = kb[-1]
                avg5 = sum(b["volume"] for b in kb[-6:-1]) / 5 if len(kb) >= 6 else last["volume"]
                day_extra[code] = {
                    "name": "", "price": last["close"],
                    "change_pct": 0, "pe_ttm": 0, "pb": 0, "mcap": 0, "turnover": 0,
                    "vol_ratio": last["volume"] / avg5 if avg5 > 0 else 1.0
                }

        factor_data, _ = eng.compute_all_factors(day_klines, day_extra, {}, {}, sectors)
        rankings = eng.compute_rankings(factor_data)
        isir_top30 = {r["code"] for r in rankings if r.get("isir_rank", 999) <= 30}

        # VCP 扫描
        vcp_results = scan_vcp(day_klines, verbose=False)
        daily_vcp_counts.append({
            "date": date,
            "stage2": sum(1 for v in vcp_results if v["stage2_score"] >= 6),
            "vcp": len(vcp_results),
            "breakout": sum(1 for v in vcp_results if v["breakout"]),
        })

        # 计算前向收益
        for code in day_klines:
            k = klines.get(code, [])
            fwd_bars = [b for b in k if b["date"] > date]
            entry = day_klines[code][-1]["close"]

            rets = {}
            for hold in [5, 10, 20]:
                if len(fwd_bars) >= hold:
                    exit_price = fwd_bars[hold - 1]["close"]
                    rets[f"{hold}d"] = round((exit_price / entry - 1) * 100, 2)

            if not rets:
                continue

            # 全市场基准
            for k_str, v in rets.items():
                all_market_returns[k_str].append(v)

            # ISIR TOP30 基准
            if code in isir_top30:
                for k_str, v in rets.items():
                    isir_top30_returns[k_str].append(v)

        # VCP 信号收益
        for vcp_stock in vcp_results:
            code = vcp_stock["code"]
            k = klines.get(code, [])
            fwd_bars = [b for b in k if b["date"] > date]
            entry = day_klines[code][-1]["close"]

            rets = {}
            for hold in [5, 10, 20]:
                if len(fwd_bars) >= hold:
                    exit_price = fwd_bars[hold - 1]["close"]
                    rets[f"{hold}d"] = round((exit_price / entry - 1) * 100, 2)

            if not rets:
                continue

            target = vcp_breakout_returns if vcp_stock["breakout"] else vcp_prebreakout_returns
            for k_str, v in rets.items():
                target[k_str].append(v)

    # 汇总统计
    def stats(returns_dict):
        result = {}
        for period, rets in returns_dict.items():
            if rets:
                wr = round(sum(1 for r in rets if r > 0) / len(rets) * 100, 1)
                ar = round(float(np.mean(rets)), 2)
                mr = round(float(np.median(rets)), 2)
                result[period] = {
                    "count": len(rets), "win_rate": wr,
                    "avg_return": ar, "median_return": mr,
                }
            else:
                result[period] = {"count": 0, "win_rate": 0, "avg_return": 0, "median_return": 0}
        return result

    return {
        "vcp_breakout": stats(vcp_breakout_returns),
        "vcp_prebreakout": stats(vcp_prebreakout_returns),
        "all_market": stats(all_market_returns),
        "isir_top30": stats(isir_top30_returns),
        "daily_counts": daily_vcp_counts,
    }

# ================================================================
# 报告生成
# ================================================================

def generate_report(results, output_path):
    def fmt(r):
        if r["count"] == 0:
            return '<td style="color:#999">无信号</td>'
        return f'<td>{r["win_rate"]}%</td><td class="{"pos" if r["avg_return"]>0 else "neg"}">{r["avg_return"]:+.1f}%</td><td>{r["count"]}</td>'

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>VCP 回测报告</title>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; background:#f5f5f5; color:#333; padding:20px; }}
.container {{ max-width:850px; margin:0 auto; }}
h1 {{ font-size:22px; font-weight:500; margin-bottom:8px; }}
.sub {{ font-size:13px; color:#888; margin-bottom:24px; }}
.card {{ background:#fff; border-radius:12px; padding:24px; margin-bottom:16px; box-shadow:0 1px 3px rgba(0,0,0,0.08); }}
h2 {{ font-size:16px; font-weight:500; margin-bottom:16px; }}
table {{ width:100%; border-collapse:collapse; font-size:13px; }}
th {{ text-align:left; padding:10px 8px; border-bottom:2px solid #e0e0e0; font-weight:500; color:#555; }}
td {{ padding:10px 8px; border-bottom:1px solid #f0f0f0; }}
.pos {{ color:#d32f2f; font-weight:500; }}
.neg {{ color:#2e7d32; font-weight:500; }}
.verdict {{ padding:16px; border-radius:8px; margin-top:16px; font-size:14px; }}
.verdict-good {{ background:#e8f5e9; border-left:4px solid #4caf50; }}
.verdict-neutral {{ background:#fff3e0; border-left:4px solid #ff9800; }}
.verdict-bad {{ background:#ffebee; border-left:4px solid #f44336; }}
</style>
</head>
<body>
<div class="container">
<h1>VCP 波动收缩形态 A股回测</h1>
<div class="sub">Minervini VCP · {results['daily_counts'][0]['date']} ~ {results['daily_counts'][-1]['date']} · 445只A股</div>

<div class="card">
<h2>前向收益对比 (胜率 / 均收益 / 样本数)</h2>
<table>
<tr><th>组别</th><th>5日胜率</th><th>5日均收益</th><th>5日样本</th><th>10日胜率</th><th>10日均收益</th><th>10日样本</th><th>20日胜率</th><th>20日均收益</th><th>20日样本</th></tr>
<tr><td><b>VCP突破</b></td>{fmt(results['vcp_breakout']['5d'])}{fmt(results['vcp_breakout']['10d'])}{fmt(results['vcp_breakout']['20d'])}</tr>
<tr><td>VCP预突破</td>{fmt(results['vcp_prebreakout']['5d'])}{fmt(results['vcp_prebreakout']['10d'])}{fmt(results['vcp_prebreakout']['20d'])}</tr>
<tr><td>ISIR TOP30</td>{fmt(results['isir_top30']['5d'])}{fmt(results['isir_top30']['10d'])}{fmt(results['isir_top30']['20d'])}</tr>
<tr><td>全市场基准</td>{fmt(results['all_market']['5d'])}{fmt(results['all_market']['10d'])}{fmt(results['all_market']['20d'])}</tr>
</table>
</div>

<div class="card">
<h2>每日VCP信号数量分布</h2>
<table>
<tr><th>日期</th><th>Stage2通过</th><th>VCP形态</th><th>突破确认</th></tr>
"""

    for d in results["daily_counts"][-20:]:
        html += f'<tr><td>{d["date"]}</td><td>{d["stage2"]}</td><td>{d["vcp"]}</td><td>{d["breakout"]}</td></tr>'

    html += '</table></div>'

    # 结论
    bo = results["vcp_breakout"]
    mkt = results["all_market"]
    isir = results["isir_top30"]

    bo_wr_10d = bo["10d"]["win_rate"]
    mkt_wr_10d = mkt["10d"]["win_rate"]
    isir_wr_10d = isir["10d"]["win_rate"]
    bo_ret_10d = bo["10d"]["avg_return"]
    mkt_ret_10d = mkt["10d"]["avg_return"]

    bo_count = bo["10d"]["count"]

    if bo_count < 10:
        cls = "verdict-neutral"
        txt = f"VCP突破信号样本量不足({bo_count}笔)，无法得出可靠结论。Stage2筛选+收缩检测在A股可能过于严格。"
    elif bo_wr_10d > mkt_wr_10d + 3 and bo_ret_10d > mkt_ret_10d + 0.5:
        cls = "verdict-good"
        txt = f"VCP有效! 10日胜率{bo_wr_10d}% vs 全市场{mkt_wr_10d}%(+{bo_wr_10d-mkt_wr_10d:.1f}pp), 均收益{bo_ret_10d:+.1f}% vs {mkt_ret_10d:+.1f}%。{bo_count}笔样本。"
    elif bo_wr_10d > mkt_wr_10d:
        cls = "verdict-neutral"
        txt = f"VCP有轻微优势: 10日胜率{bo_wr_10d}% vs 全市场{mkt_wr_10d}%(+{bo_wr_10d-mkt_wr_10d:.1f}pp), 但均收益差异不显著。{bo_count}笔样本。"
    else:
        cls = "verdict-bad"
        txt = f"VCP在A股无效: 10日胜率{bo_wr_10d}% vs 全市场{mkt_wr_10d}%({bo_wr_10d-mkt_wr_10d:+.1f}pp), 均收益{bo_ret_10d:+.1f}% vs {mkt_ret_10d:+.1f}%。{bo_count}笔样本。"

    html += f'<div class="card"><h2>结论</h2><div class="verdict {cls}">{txt}</div></div>'
    html += '</div></body></html>'

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    return output_path

# ================================================================
# 主流程
# ================================================================

def main():
    print("=" * 60)
    print("  VCP 波动收缩形态 A股回测")
    print("  Minervini Volatility Contraction Pattern")
    print("=" * 60)

    SELF_DIR = os.path.dirname(os.path.abspath(__file__))
    OUTPUT_DIR = os.path.join(SELF_DIR, "output")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n[1/3] 加载数据...")
    db = StockDB()
    with open(os.path.join(SELF_DIR, "stock_codes.txt")) as f:
        codes = [l.strip() for l in f if l.strip()]
    klines = db.get_klines(codes, days=300)
    extra_today = db.get_extra_info(codes)
    sectors = sector_map.STOCK_SECTOR
    print(f"  股票: {len(klines)}只")

    # 今日VCP扫描
    print("\n[2/3] 今日VCP扫描...")
    today_vcp = scan_vcp(klines)
    if today_vcp:
        breakout_stocks = [v for v in today_vcp if v["breakout"]]
        pre_stocks = [v for v in today_vcp if not v["breakout"]]
        print(f"\n  突破确认 ({len(breakout_stocks)}只):")
        for v in sorted(breakout_stocks, key=lambda x: x["quality"], reverse=True)[:10]:
            print(f"    {v['code']} | 收盘{v['close']:.2f} | pivot{v['pivot']:.2f} | "
                  f"收缩{v['n_contractions']}次 | dry_up{v['dry_up']} | 量比{v['breakout_vol']}x | 评分{v['quality']}")
        print(f"\n  预突破 ({len(pre_stocks)}只):")
        for v in sorted(pre_stocks, key=lambda x: x["quality"], reverse=True)[:10]:
            print(f"    {v['code']} | 收盘{v['close']:.2f} | pivot{v['pivot']:.2f} | "
                  f"收缩{v['n_contractions']}次 | dry_up{v['dry_up']} | 评分{v['quality']}")
    else:
        print("  今日无VCP形态")

    # 回测
    print(f"\n[3/3] 回测...")
    t0 = time.time()
    results = run_vcp_backtest(klines, extra_today, sectors, backtest_days=85)
    print(f"  回测耗时: {time.time()-t0:.0f}s")

    # 生成报告
    report_path = os.path.join(OUTPUT_DIR, "vcp_backtest.html")
    generate_report(results, report_path)

    # 终端汇总
    print(f"\n{'='*60}")
    print(f"  VCP 回测结果")
    print(f"{'='*60}")
    print(f"{'组别':<16} {'5日胜率':>8} {'5日收益':>8} {'10日胜率':>8} {'10日收益':>8} {'20日胜率':>8} {'20日收益':>8}")
    print("-" * 68)
    for name, key in [("VCP突破", "vcp_breakout"), ("VCP预突破", "vcp_prebreakout"), ("ISIR TOP30", "isir_top30"), ("全市场基准", "all_market")]:
        r = results[key]
        print(f"{name:<16} {r['5d']['win_rate']:>7.1f}% {r['5d']['avg_return']:>+7.1f}% {r['10d']['win_rate']:>7.1f}% {r['10d']['avg_return']:>+7.1f}% {r['20d']['win_rate']:>7.1f}% {r['20d']['avg_return']:>+7.1f}%")
    print(f"\n  样本数: VCP突破={results['vcp_breakout']['10d']['count']} | 预突破={results['vcp_prebreakout']['10d']['count']} | ISIR={results['isir_top30']['10d']['count']} | 全市场={results['all_market']['10d']['count']}")
    print(f"\n  报告: {report_path}")

if __name__ == "__main__":
    main()
