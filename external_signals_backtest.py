#!/usr/bin/env python3
"""
外部市场信号回测
================
测试外围市场信号(HK恒生指数+恒生科技)对A股因子模型的增量价值。

新增:
  1. sector_external 因子 — 板块级外部信号(截面因子, 有变异)
     科技板块 → 恒生科技权重高
     非科技板块 → 恒生指数权重高
  2. global_market_gate — 市场级门控(时序开关, 不改排名)
     恒生指数暴跌 > 2% → 减仓

用法: python3 external_signals_backtest.py
"""

import os, json, math, sys, time
from datetime import datetime
from collections import defaultdict
import numpy as np
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stock_db import StockDB
import sector_map
import unified_scoring_engine as eng
from factor_ab_test import run_backtest as run_strategy_backtest

# ================================================================
# 板块 → 外部信号映射
# ================================================================

# 恒生科技 → 科技板块强关联; 恒生指数 → 全市场弱关联
SECTOR_EXTERNAL_WEIGHTS = {
    "半导体/芯片":     {"hkHSTECH": 0.7, "hkHSI": 0.3},
    "AI/算力/通信":    {"hkHSTECH": 0.7, "hkHSI": 0.3},
    "电子/消费电子":   {"hkHSTECH": 0.6, "hkHSI": 0.4},
    "传媒/游戏":       {"hkHSTECH": 0.5, "hkHSI": 0.5},
    "新能源/电力":     {"hkHSTECH": 0.3, "hkHSI": 0.5},
    "金融/交通/基建":  {"hkHSTECH": 0.1, "hkHSI": 0.5},
    "智能制造":         {"hkHSTECH": 0.2, "hkHSI": 0.3},
    "医药生物":         {"hkHSTECH": 0.1, "hkHSI": 0.3},
    "消费":             {"hkHSTECH": 0.1, "hkHSI": 0.3},
    "化工/新材料":      {"hkHSTECH": 0.0, "hkHSI": 0.2},
}

# ================================================================
# 获取恒生指数K线
# ================================================================

def fetch_hk_klines(sym, days=300):
    url = f"https://web.ifzq.gtimg.cn/appstock/app/kline/kline?param={sym},day,,,{days},"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    resp = urllib.request.urlopen(req, timeout=15)
    data = json.loads(resp.read().decode("utf-8"))
    kls = data.get("data", {}).get(sym, {}).get("day", [])
    parsed = {}
    for k in kls:
        try:
            parsed[k[0]] = {"close": float(k[2]), "open": float(k[1]), "high": float(k[3]), "low": float(k[4])}
        except (IndexError, ValueError, TypeError):
            continue
    return parsed

# ================================================================
# 计算外部信号
# ================================================================

def compute_hk_returns(hk_hsi_data, hk_hstech_data):
    """计算恒生指数和恒生科技的日收益率"""
    hsi_returns = {}
    hstech_returns = {}

    hsi_dates = sorted(hk_hsi_data.keys())
    for i in range(1, len(hsi_dates)):
        d = hsi_dates[i]
        prev = hsi_dates[i - 1]
        prev_close = hk_hsi_data[prev]["close"]
        if prev_close > 0:
            hsi_returns[d] = (hk_hsi_data[d]["close"] / prev_close - 1) * 100

    hstech_dates = sorted(hk_hstech_data.keys())
    for i in range(1, len(hstech_dates)):
        d = hstech_dates[i]
        prev = hstech_dates[i - 1]
        prev_close = hk_hstech_data[prev]["close"]
        if prev_close > 0:
            hstech_returns[d] = (hk_hstech_data[d]["close"] / prev_close - 1) * 100

    return hsi_returns, hstech_returns

def get_sector_external(sector, date, hsi_returns, hstech_returns):
    """计算某板块某日的外部信号值"""
    weights = SECTOR_EXTERNAL_WEIGHTS.get(sector, {"hkHSI": 0.3})

    # 找最近有数据的日期 (港股和A股交易日历不完全重合)
    hsi_r = _find_nearest_return(date, hsi_returns)
    hstech_r = _find_nearest_return(date, hstech_returns)

    w_hstech = weights.get("hkHSTECH", 0)
    w_hsi = weights.get("hkHSI", 0)

    return hstech_r * w_hstech + hsi_r * w_hsi

def _find_nearest_return(date, returns_dict):
    """找最近有数据的日期的收益率"""
    if date in returns_dict:
        return returns_dict[date]
    # 往前找最近3天
    from datetime import datetime, timedelta
    d = datetime.strptime(date, "%Y-%m-%d")
    for i in range(1, 4):
        prev_d = (d - timedelta(days=i)).strftime("%Y-%m-%d")
        if prev_d in returns_dict:
            return returns_dict[prev_d]
    return 0.0

# ================================================================
# 增强版因子计算 (加 sector_external)
# ================================================================

NEW_FACTOR = "sector_external"

def compute_all_factors_with_external(klines, extra_info, fund_flows, events, sectors,
                                       hsi_returns, hstech_returns, date_str=None):
    """在原始因子基础上增加 sector_external"""
    factor_data, return_data = eng.compute_all_factors(klines, extra_info, fund_flows, events, sectors)

    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")

    for code, factors in factor_data.items():
        sector = sectors.get(code, "其他")
        ext_val = get_sector_external(sector, date_str, hsi_returns, hstech_returns)
        factors[NEW_FACTOR] = ext_val

    # 截面标准化
    values = [factor_data[c].get(NEW_FACTOR, 0) for c in factor_data]
    arr = np.array(values, dtype=float)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if np.std(arr) > 1e-10:
        z = (arr - np.mean(arr)) / np.std(arr)
        for ci, c in enumerate(factor_data.keys()):
            factor_data[c][f"{NEW_FACTOR}_z"] = float(z[ci])
    else:
        for c in factor_data:
            factor_data[c][f"{NEW_FACTOR}_z"] = 0.0

    return factor_data, return_data

def compute_rankings_with_external(factor_data, sector_external_weight=0.05):
    """用增强权重(31+sector_external)计算排名"""
    orig_v3 = dict(eng.ICIR_V3)
    orig_glm = dict(eng.ICIR_GLM)

    eng.ICIR_V3[NEW_FACTOR] = sector_external_weight
    eng.ICIR_GLM[NEW_FACTOR] = sector_external_weight

    try:
        results = eng.compute_rankings(factor_data)
    finally:
        eng.ICIR_V3.clear()
        eng.ICIR_V3.update(orig_v3)
        eng.ICIR_GLM.clear()
        eng.ICIR_GLM.update(orig_glm)

    return results

# ================================================================
# ICIR 计算
# ================================================================

def compute_sector_external_icir(klines, extra_today, sectors, hsi_returns, hstech_returns, backtest_days=85, fwd_days=5):
    """计算 sector_external 因子的 IC"""
    all_dates = set()
    for k in klines.values():
        for b in k:
            all_dates.add(b["date"])
    dates = sorted(all_dates)
    start_idx = max(60, len(dates) - backtest_days)
    test_dates = dates[start_idx:]
    cutoff = len(dates) - fwd_days
    test_dates = [d for d in test_dates if dates.index(d) < cutoff]

    daily_ics = []
    daily_values = []

    for di, date in enumerate(test_dates):
        if di % 20 == 0:
            print(f"  IC进度: {di}/{len(test_dates)}")

        day_klines = {}
        for code, k in klines.items():
            day_bars = [b for b in k if b["date"] <= date]
            if len(day_bars) >= 60:
                day_klines[code] = day_bars
        if len(day_klines) < 100:
            continue

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

        factor_data, _ = compute_all_factors_with_external(
            day_klines, day_extra, {}, {}, sectors, hsi_returns, hstech_returns, date)

        if len(factor_data) < 100:
            continue

        fwd_returns = {}
        for code in factor_data:
            k = klines.get(code, [])
            fwd_bars = [b for b in k if b["date"] > date]
            if len(fwd_bars) >= fwd_days:
                entry = day_klines[code][-1]["close"]
                exit_p = fwd_bars[fwd_days - 1]["close"]
                if entry > 0:
                    fwd_returns[code] = (exit_p / entry - 1) * 100

        if len(fwd_returns) < 50:
            continue

        codes = sorted(factor_data.keys())
        z_vals = np.array([factor_data[c].get(f"{NEW_FACTOR}_z", 0) for c in codes])
        fwd_arr = np.array([fwd_returns.get(c, 0) for c in codes])

        if np.std(z_vals) < 1e-10:
            continue

        z_ranks = _rank_data(z_vals)
        fwd_ranks = _rank_data(fwd_arr)
        ic = _spearman(z_ranks, fwd_ranks)
        daily_ics.append(ic)
        daily_values.append(np.mean([factor_data[c].get(NEW_FACTOR, 0) for c in codes]))

    if daily_ics:
        mean_ic = float(np.mean(daily_ics))
        std_ic = float(np.std(daily_ics))
        ir = mean_ic / std_ic if std_ic > 1e-10 else 0
        pos_pct = sum(1 for x in daily_ics if x > 0) / len(daily_ics) * 100
        return {
            "mean_ic": round(mean_ic, 4),
            "std_ic": round(std_ic, 4),
            "ir": round(ir, 4),
            "pos_pct": round(pos_pct, 1),
            "n_days": len(daily_ics),
        }
    return {"mean_ic": 0, "std_ic": 0, "ir": 0, "pos_pct": 0, "n_days": 0}

def _rank_data(arr):
    sorter = np.argsort(arr)
    ranks = np.empty_like(sorter, dtype=float)
    ranks[sorter] = np.arange(len(arr), dtype=float)
    unique_vals, counts = np.unique(arr, return_counts=True)
    for v, c in zip(unique_vals, counts):
        if c > 1:
            mask = arr == v
            ranks[mask] = np.mean(ranks[mask])
    return ranks

def _spearman(x, y):
    n = len(x)
    if n < 5:
        return 0
    mx, my = np.mean(x), np.mean(y)
    dx, dy = x - mx, y - my
    denom = np.sqrt(np.sum(dx**2) * np.sum(dy**2))
    if denom < 1e-10:
        return 0
    return float(np.sum(dx * dy) / denom)

# ================================================================
# A/B 回测
# ================================================================

def run_ab_backtest(klines, extra_today, sectors, hsi_returns, hstech_returns, backtest_days=85):
    """A/B 回测: 原始31因子 vs 31+sector_external"""
    all_dates = set()
    for k in klines.values():
        for b in k:
            all_dates.add(b["date"])
    dates = sorted(all_dates)
    start_idx = max(60, len(dates) - backtest_days)
    test_dates = dates[start_idx:]
    print(f"  回测区间: {test_dates[0]} ~ {test_dates[-1]} ({len(test_dates)}日)")

    daily_data = {}
    for date in test_dates:
        day_klines = {}
        for code, k in klines.items():
            day_bars = [b for b in k if b["date"] <= date]
            if len(day_bars) >= 60:
                day_klines[code] = day_bars
        if len(day_klines) < 100:
            continue
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
        next_opens = {}
        for code in day_klines:
            fwd = [b for b in klines.get(code, []) if b["date"] > date]
            if fwd:
                next_opens[code] = fwd[0]["open"]
        daily_data[date] = {"klines": day_klines, "extra": day_extra, "next_opens": next_opens}

    print(f"  有效交易日: {len(daily_data)}")

    strategies = ["isir", "glm"]
    max_positions = 30
    hold_days_max = 10
    stop_loss = -8.0
    take_profit = 15.0
    rank_collapse_pct = 0.5
    buy_count = 30

    results = {"baseline": {}, "enhanced": {}}

    for mode in ["baseline", "enhanced"]:
        print(f"\n  === {mode.upper()} ===")
        for strat in strategies:
            positions = []
            closed_trades = []

            for date in sorted(daily_data.keys()):
                dd = daily_data[date]
                day_klines = dd["klines"]
                day_extra = dd["extra"]
                next_opens = dd["next_opens"]

                if mode == "enhanced":
                    factor_data, _ = compute_all_factors_with_external(
                        day_klines, day_extra, {}, {}, sectors, hsi_returns, hstech_returns, date)
                    rankings = compute_rankings_with_external(factor_data)
                else:
                    factor_data, _ = eng.compute_all_factors(day_klines, day_extra, {}, {}, sectors)
                    rankings = eng.compute_rankings(factor_data)

                if len(factor_data) < 100 or not rankings:
                    continue

                rank_key = f"{strat}_rank"
                rank_map = {r["code"]: r for r in rankings}
                n_total = len(rankings)
                current_top = {r["code"] for r in rankings if r.get(rank_key, 999) <= buy_count}

                still_holding = []
                for pos in positions:
                    code = pos["code"]
                    price = day_extra.get(code, {}).get("price", 0)
                    if price <= 0:
                        still_holding.append(pos)
                        continue
                    ret = (price / pos["entry_price"] - 1) * 100
                    pos["hold_days"] += 1
                    current_rank = rank_map.get(code, {}).get(rank_key, n_total)
                    exit_reason = None
                    if pos["hold_days"] >= hold_days_max:
                        exit_reason = "time"
                    elif ret <= stop_loss:
                        exit_reason = "stop_loss"
                    elif ret >= take_profit:
                        exit_reason = "take_profit"
                    elif current_rank > n_total * rank_collapse_pct:
                        exit_reason = "rank_collapse"
                    if exit_reason:
                        exit_price = next_opens.get(code, price)
                        exit_ret = round((exit_price / pos["entry_price"] - 1) * 100, 2)
                        pos["exit_date"] = date
                        pos["return_pct"] = exit_ret
                        pos["is_win"] = exit_ret > 0
                        closed_trades.append(pos)
                    else:
                        still_holding.append(pos)
                positions = still_holding

                existing = {p["code"] for p in positions}
                available = max_positions - len(positions)
                if available > 0:
                    new_cands = []
                    for code in current_top - existing:
                        r = rank_map.get(code)
                        if r and code in next_opens and next_opens[code] > 0:
                            new_cands.append((code, r.get(rank_key, 999)))
                    new_cands.sort(key=lambda x: x[1])
                    for code, rank in new_cands[:available]:
                        entry_price = next_opens[code]
                        positions.append({
                            "code": code, "entry_date": date,
                            "entry_price": entry_price, "hold_days": 0,
                        })

            last_date = sorted(daily_data.keys())[-1]
            for pos in positions:
                price = daily_data[last_date]["extra"].get(pos["code"], {}).get("price", 0)
                ret = round((price / pos["entry_price"] - 1) * 100, 2) if price > 0 else 0
                pos["return_pct"] = ret
                closed_trades.append(pos)

            total = len(closed_trades)
            wins = sum(1 for t in closed_trades if t["return_pct"] > 0)
            wr = round(wins / total * 100, 1) if total > 0 else 0
            ar = round(float(np.mean([t["return_pct"] for t in closed_trades])), 2) if total > 0 else 0
            cum = round(sum(t["return_pct"] for t in closed_trades), 1)

            results[mode][strat] = {"total": total, "win_rate": wr, "avg_return": ar, "cumulative": cum}
            print(f"  {strat.upper()}: {total}笔 | 胜率{wr}% | 均收益{ar:+.1f}% | 累积{cum:+.1f}%")

    return results

# ================================================================
# 报告
# ================================================================

def generate_report(ic_stats, ab_results, today_signals, output_path):
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>外部市场信号回测</title>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family:-apple-system,sans-serif; background:#f5f5f5; color:#333; padding:20px; }}
.c {{ max-width:850px; margin:0 auto; }}
h1 {{ font-size:22px; font-weight:500; margin-bottom:8px; }}
.sub {{ font-size:13px; color:#888; margin-bottom:24px; }}
.card {{ background:#fff; border-radius:12px; padding:24px; margin-bottom:16px; box-shadow:0 1px 3px rgba(0,0,0,0.08); }}
h2 {{ font-size:16px; font-weight:500; margin-bottom:16px; }}
table {{ width:100%; border-collapse:collapse; font-size:13px; }}
th {{ text-align:left; padding:10px 8px; border-bottom:2px solid #e0e0e0; font-weight:500; }}
td {{ padding:10px 8px; border-bottom:1px solid #f0f0f0; }}
.pos {{ color:#d32f2f; font-weight:500; }}
.neg {{ color:#2e7d32; }}
.v {{ padding:16px; border-radius:8px; margin-top:16px; }}
.v-good {{ background:#e8f5e9; border-left:4px solid #4caf50; }}
.v-neutral {{ background:#fff3e0; border-left:4px solid #ff9800; }}
</style>
</head>
<body>
<div class="c">
<h1>外部市场信号回测</h1>
<div class="sub">恒生指数+恒生科技 → A股板块映射 · {datetime.now().strftime("%Y-%m-%d %H:%M")}</div>

<div class="card">
<h2>sector_external 因子 IC 统计</h2>
<table>
<tr><th>指标</th><th>值</th></tr>
<tr><td>IC 均值</td><td class="{"pos" if ic_stats["mean_ic"]>0 else "neg"}">{ic_stats["mean_ic"]:+.4f}</td></tr>
<tr><td>IC 标准差</td><td>{ic_stats["std_ic"]:.4f}</td></tr>
<tr><td>IR (IC/STD)</td><td>{ic_stats["ir"]:+.3f}</td></tr>
<tr><td>正 IC 天数占比</td><td>{ic_stats["pos_pct"]:.0f}%</td></tr>
<tr><td>有效天数</td><td>{ic_stats["n_days"]}</td></tr>
</table>
</div>

<div class="card">
<h2>A/B 回测: 原始31因子 vs 31+sector_external</h2>
<table>
<tr><th>策略</th><th>指标</th><th>Baseline</th><th>+外部信号</th><th>变化</th></tr>
"""

    for strat in ["isir", "glm"]:
        b = ab_results["baseline"].get(strat, {})
        e = ab_results["enhanced"].get(strat, {})
        for metric, label in [("win_rate", "胜率%"), ("avg_return", "均收益%"), ("cumulative", "累积%")]:
            bv = b.get(metric, 0)
            ev = e.get(metric, 0)
            d = ev - bv
            cls = "pos" if d > 0 else "neg"
            html += f'<tr><td>{strat.upper()}</td><td>{label}</td><td>{bv}</td><td>{ev}</td><td class="{cls}">{"+" if d>0 else ""}{d:.1f}</td></tr>'

    html += '</table></div>'

    # 今日外部信号
    html += '<div class="card"><h2>今日外部信号快照</h2><table>'
    html += '<tr><th>板块</th><th>恒生科技权重</th><th>恒生指数权重</th><th>今日信号值</th></tr>'
    for sector, weights in sorted(SECTOR_EXTERNAL_WEIGHTS.items()):
        sig = today_signals.get(sector, 0)
        cls = "pos" if sig > 0 else "neg"
        html += f'<tr><td>{sector}</td><td>{weights.get("hkHSTECH",0)*100:.0f}%</td><td>{weights.get("hkHSI",0)*100:.0f}%</td><td class="{cls}">{sig:+.2f}%</td></tr>'
    html += '</table></div>'

    # 结论
    isir_d = ab_results["enhanced"].get("isir",{}).get("win_rate",0) - ab_results["baseline"].get("isir",{}).get("win_rate",0)
    glm_d = ab_results["enhanced"].get("glm",{}).get("win_rate",0) - ab_results["baseline"].get("glm",{}).get("win_rate",0)
    avg_d = (isir_d + glm_d) / 2
    ic = ic_stats["mean_ic"]

    if avg_d > 1 and ic > 0.01:
        cls = "v-good"
        txt = f"外部信号有效! IC={ic:+.4f}, 平均胜率提升{avg_d:+.1f}pp。建议纳入引擎。"
    elif avg_d > 0 or ic > 0:
        cls = "v-neutral"
        txt = f"外部信号有轻微正向: IC={ic:+.4f}, 平均胜率变化{avg_d:+.1f}pp。效果有限但方向正确。"
    else:
        cls = "v-neutral"
        txt = f"外部信号无效: IC={ic:+.4f}, 平均胜率变化{avg_d:+.1f}pp。"

    html += f'<div class="card"><h2>结论</h2><div class="v {cls}">{txt}</div></div>'
    html += '</div></body></html>'

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    return output_path

# ================================================================
# 主流程
# ================================================================

def main():
    print("=" * 60)
    print("  外部市场信号回测")
    print("  恒生指数 + 恒生科技 → A股板块映射")
    print("=" * 60)

    SELF_DIR = os.path.dirname(os.path.abspath(__file__))
    OUTPUT_DIR = os.path.join(SELF_DIR, "output")

    print("\n[1/4] 加载数据...")
    db = StockDB()
    with open(os.path.join(SELF_DIR, "stock_codes.txt")) as f:
        codes = [l.strip() for l in f if l.strip()]
    klines = db.get_klines(codes, days=300)
    extra_today = db.get_extra_info(codes)
    sectors = sector_map.STOCK_SECTOR
    print(f"  股票: {len(klines)}只")

    print("\n[2/4] 获取恒生指数K线...")
    hk_hsi = fetch_hk_klines("hkHSI", 300)
    hk_hstech = fetch_hk_klines("hkHSTECH", 300)
    hsi_returns, hstech_returns = compute_hk_returns(hk_hsi, hk_hstech)
    print(f"  恒生指数: {len(hk_hsi)}条 | 恒生科技: {len(hk_hstech)}条")
    print(f"  收益率序列: HSI {len(hsi_returns)}日 | HSTECH {len(hstech_returns)}日")

    # 今日外部信号
    today = datetime.now().strftime("%Y-%m-%d")
    today_signals = {}
    for sector in SECTOR_EXTERNAL_WEIGHTS:
        today_signals[sector] = get_sector_external(sector, today, hsi_returns, hstech_returns)
    print(f"\n  今日外部信号:")
    for s, v in sorted(today_signals.items(), key=lambda x: x[1], reverse=True):
        print(f"    {s:<16} {v:+.2f}%")

    print(f"\n[3/4] 计算 sector_external IC...")
    ic_stats = compute_sector_external_icir(klines, extra_today, sectors, hsi_returns, hstech_returns, backtest_days=85)
    print(f"  IC={ic_stats['mean_ic']:+.4f} | IR={ic_stats['ir']:+.3f} | 正天数={ic_stats['pos_pct']:.0f}% | 天数={ic_stats['n_days']}")

    print(f"\n[4/4] A/B 回测...")
    ab_results = run_ab_backtest(klines, extra_today, sectors, hsi_returns, hstech_returns, backtest_days=85)

    # 生成报告
    report_path = os.path.join(OUTPUT_DIR, "external_signals.html")
    generate_report(ic_stats, ab_results, today_signals, report_path)

    print(f"\n{'='*60}")
    print(f"  报告: {report_path}")
    print(f"{'='*60}")

    # 终端汇总
    print(f"\n  IC: {ic_stats['mean_ic']:+.4f} | IR: {ic_stats['ir']:+.3f} | 正天数: {ic_stats['pos_pct']:.0f}%")
    print(f"\n  {'策略':<8} {'指标':<10} {'Baseline':>10} {'+外部':>10} {'变化':>8}")
    print("  " + "-" * 50)
    for strat in ["isir", "glm"]:
        b = ab_results["baseline"].get(strat, {})
        e = ab_results["enhanced"].get(strat, {})
        for m, l in [("win_rate","胜率%"), ("avg_return","均收益%"), ("cumulative","累积%")]:
            bv, ev = b.get(m,0), e.get(m,0)
            d = ev - bv
            print(f"  {strat.upper():<8} {l:<10} {bv:>10.1f} {ev:>10.1f} {d:>+8.1f}")

if __name__ == "__main__":
    main()
