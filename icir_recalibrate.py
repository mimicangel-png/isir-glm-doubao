#!/usr/bin/env python3
"""
ICIR 重标定
===========
用 85 交易日截面数据 + 5日前向收益，计算每个因子的信息系数(IC)，
自动得到34因子最优权重，对比原始手调权重 vs 数据标定权重的回测效果。

流程:
  1. 逐日计算34因子截面z值 + 5日前向收益
  2. 每个因子取 Spearman rank IC，多日平均 → ICIR
  3. 用 ICIR 做权重，重新跑回测
  4. 对比: 原始31因子 vs 重标定34因子

用法: python3 icir_recalibrate.py
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
from factor_ab_test import (
    fetch_index_klines, compute_all_factors_enhanced,
    NEW_FACTOR_WEIGHTS, NEW_FACTOR_HIGHER_BETTER,
)

# ================================================================
# ICIR 计算
# ================================================================

def compute_icir(klines, extra_today, sectors, index_klines, backtest_days=85, fwd_days=5):
    """
    计算所有因子的 ICIR (信息系数 × 信息比率)

    返回:
      icir_v3: {factor_name: icir_value}  (正值=高好, 负值=低好)
      ic_stats: {factor_name: {mean_ic, std_ic, ir, positive_days_pct}}
    """
    print("  计算 ICIR...")

    all_dates = set()
    for k in klines.values():
        for b in k:
            all_dates.add(b["date"])
    dates = sorted(all_dates)

    start_idx = max(60, len(dates) - backtest_days)
    test_dates = dates[start_idx:]

    # 需要留 fwd_days 的前向收益空间
    cutoff = len(dates) - fwd_days
    test_dates = [d for d in test_dates if dates.index(d) < cutoff]
    print(f"  ICIR计算区间: {test_dates[0]} ~ {test_dates[-1]} ({len(test_dates)}日, 前向{fwd_days}日)")

    # 所有因子名
    all_factors = sorted(eng.ICIR_V3.keys()) + list(NEW_FACTOR_WEIGHTS.keys())

    # 逐日存储因子IC
    daily_ics = {f: [] for f in all_factors}

    for di, date in enumerate(test_dates):
        if di % 20 == 0:
            print(f"  ICIR进度: {di}/{len(test_dates)}")

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
                    "name": extra_today.get(code, {}).get("name", code),
                    "price": last["close"],
                    "change_pct": (last["close"] / kb[-2]["close"] - 1) * 100,
                    "pe_ttm": 0, "pb": 0, "mcap": 0, "turnover": 0,
                    "vol_ratio": last["volume"] / avg5 if avg5 > 0 else 1.0
                }

        idx_klines_trunc = [b for b in index_klines if b["date"] <= date]

        # 计算34因子
        factor_data, _ = compute_all_factors_enhanced(
            day_klines, day_extra, {}, {}, sectors, idx_klines_trunc, {})

        if len(factor_data) < 100:
            continue

        # 计算前向收益
        fwd_returns = {}
        for code in factor_data:
            k = klines.get(code, [])
            fwd_bars = [b for b in k if b["date"] > date]
            if len(fwd_bars) >= fwd_days:
                entry_close = day_klines[code][-1]["close"]
                exit_close = fwd_bars[fwd_days - 1]["close"]
                if entry_close > 0:
                    fwd_returns[code] = (exit_close / entry_close - 1) * 100

        if len(fwd_returns) < 50:
            continue

        codes = sorted(factor_data.keys())
        fwd_arr = np.array([fwd_returns.get(c, 0) for c in codes])
        fwd_ranks = _rank_data(fwd_arr)

        for fname in all_factors:
            z_key = f"{fname}_z"
            z_vals = np.array([factor_data[c].get(z_key, 0) for c in codes])
            if np.std(z_vals) < 1e-10:
                daily_ics[fname].append(0.0)
                continue
            z_ranks = _rank_data(z_vals)
            ic = _spearman_ic(z_ranks, fwd_ranks)
            daily_ics[fname].append(ic)

    # 汇总
    icir_v3 = {}
    ic_stats = {}

    for fname in all_factors:
        ics = daily_ics[fname]
        if not ics:
            icir_v3[fname] = 0
            ic_stats[fname] = {"mean_ic": 0, "std_ic": 0, "ir": 0, "pos_pct": 0}
            continue

        mean_ic = float(np.mean(ics))
        std_ic = float(np.std(ics))
        ir = mean_ic / std_ic if std_ic > 1e-10 else 0
        pos_pct = sum(1 for x in ics if x > 0) / len(ics) * 100

        icir_v3[fname] = round(mean_ic, 4)
        ic_stats[fname] = {
            "mean_ic": round(mean_ic, 4),
            "std_ic": round(std_ic, 4),
            "ir": round(ir, 4),
            "pos_pct": round(pos_pct, 1),
        }

    return icir_v3, ic_stats

def _rank_data(arr):
    """计算rank (average rank for ties)"""
    sorter = np.argsort(arr)
    ranks = np.empty_like(sorter, dtype=float)
    ranks[sorter] = np.arange(len(arr), dtype=float)
    # handle ties
    unique_vals, counts = np.unique(arr, return_counts=True)
    for v, c in zip(unique_vals, counts):
        if c > 1:
            mask = arr == v
            ranks[mask] = np.mean(ranks[mask])
    return ranks

def _spearman_ic(x_ranks, y_ranks):
    """Spearman rank correlation"""
    n = len(x_ranks)
    if n < 5:
        return 0
    mx, my = np.mean(x_ranks), np.mean(y_ranks)
    dx = x_ranks - mx
    dy = y_ranks - my
    denom = np.sqrt(np.sum(dx**2) * np.sum(dy**2))
    if denom < 1e-10:
        return 0
    return float(np.sum(dx * dy) / denom)

# ================================================================
# 用给定权重跑回测
# ================================================================

def run_backtest_with_weights(klines, extra_today, sectors, index_klines,
                               icir_v3, icir_glm, factor_higher_better,
                               backtest_days=85, mode="enhanced"):
    """用指定的ICIR权重跑策略回测"""
    all_dates = set()
    for k in klines.values():
        for b in k:
            all_dates.add(b["date"])
    dates = sorted(all_dates)
    start_idx = max(60, len(dates) - backtest_days)
    test_dates = dates[start_idx:]

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
                    "name": extra_today.get(code, {}).get("name", code),
                    "price": last["close"],
                    "change_pct": (last["close"] / kb[-2]["close"] - 1) * 100,
                    "pe_ttm": 0, "pb": 0, "mcap": 0, "turnover": 0,
                    "vol_ratio": last["volume"] / avg5 if avg5 > 0 else 1.0
                }
        next_opens = {}
        for code in day_klines:
            fwd = [b for b in klines.get(code, []) if b["date"] > date]
            if fwd:
                next_opens[code] = fwd[0]["open"]
        idx_trunc = [b for b in index_klines if b["date"] <= date]
        daily_data[date] = {"klines": day_klines, "extra": day_extra, "next_opens": next_opens, "index_klines": idx_trunc}

    strategies = ["isir", "glm", "doubao"]
    max_positions = 30
    hold_days_max = 10
    stop_loss = -8.0
    take_profit = 15.0
    rank_collapse_pct = 0.5
    buy_count = 30

    weight_map = {"isir": icir_v3, "glm": icir_glm}
    strategy_results = {}

    for strat in strategies:
        positions = []
        closed_trades = []

        for date in sorted(daily_data.keys()):
            dd = daily_data[date]
            day_klines = dd["klines"]
            day_extra = dd["extra"]
            next_opens = dd["next_opens"]
            idx_klines = dd["index_klines"]

            if mode == "enhanced":
                factor_data, _ = compute_all_factors_enhanced(
                    day_klines, day_extra, {}, {}, sectors, idx_klines, {})
            else:
                factor_data, _ = eng.compute_all_factors(day_klines, day_extra, {}, {}, sectors)

            if len(factor_data) < 100:
                continue

            # 用给定权重计算排名
            all_fnames = sorted(weight_map.get(strat, icir_v3).keys()) if strat in weight_map else sorted(eng.ICIR_V3.keys())
            weights = weight_map.get(strat, eng.ICIR_V3)

            results = []
            for code, factors in factor_data.items():
                if strat == "doubao":
                    # 豆包用SS评分 (不变)
                    r = eng.compute_rankings(factor_data)
                    results = r
                    break

                score = sum(weights.get(f, 0) * factors.get(f"{f}_z", 0) for f in all_fnames)
                results.append({"code": code, "score": score})

            if strat != "doubao":
                results.sort(key=lambda x: x["score"], reverse=True)
                for i, r in enumerate(results):
                    r[f"{strat}_rank"] = i + 1
            else:
                for r in results:
                    r["doubao_rank"] = r.get("doubao_rank", 999)

            rank_key = f"{strat}_rank"
            rank_map = {r["code"]: r for r in results}
            n_total = len(results)
            current_top = {r["code"] for r in results if r.get(rank_key, 999) <= buy_count}

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
                    pos["exit_price"] = exit_price
                    pos["return_pct"] = exit_ret
                    pos["is_win"] = exit_ret > 0
                    pos["exit_reason"] = exit_reason
                    closed_trades.append(pos)
                else:
                    still_holding.append(pos)
            positions = still_holding

            existing_codes = {p["code"] for p in positions}
            available_slots = max_positions - len(positions)
            if available_slots > 0:
                new_candidates = []
                for code in current_top - existing_codes:
                    r = rank_map.get(code)
                    if r and code in next_opens and next_opens[code] > 0:
                        new_candidates.append((code, r.get(rank_key, 999)))
                new_candidates.sort(key=lambda x: x[1])
                for code, rank in new_candidates[:available_slots]:
                    entry_price = next_opens[code]
                    name = day_extra.get(code, {}).get("name", code)
                    positions.append({
                        "code": code, "name": name,
                        "entry_date": date, "entry_price": entry_price,
                        "entry_rank": rank, "hold_days": 0,
                    })

        last_date = sorted(daily_data.keys())[-1]
        for pos in positions:
            price = daily_data[last_date]["extra"].get(pos["code"], {}).get("price", 0)
            ret = round((price / pos["entry_price"] - 1) * 100, 2) if price > 0 else 0
            pos["exit_date"] = last_date
            pos["exit_price"] = price
            pos["return_pct"] = ret
            pos["is_win"] = ret > 0
            pos["exit_reason"] = "backtest_end"
            closed_trades.append(pos)

        total = len(closed_trades)
        wins = sum(1 for t in closed_trades if t["return_pct"] > 0)
        win_rate = round(wins / total * 100, 1) if total > 0 else 0
        avg_ret = round(float(np.mean([t["return_pct"] for t in closed_trades])), 2) if total > 0 else 0
        cum_ret = round(sum(t["return_pct"] for t in closed_trades), 1)
        median_ret = round(float(np.median([t["return_pct"] for t in closed_trades])), 2) if total > 0 else 0

        reason_stats = defaultdict(int)
        for t in closed_trades:
            reason_stats[t.get("exit_reason", "unknown")] += 1
        avg_hold = round(float(np.mean([t["hold_days"] for t in closed_trades])), 1) if total > 0 else 0

        strategy_results[strat] = {
            "total_trades": total,
            "win_rate": win_rate,
            "avg_return": avg_ret,
            "cumulative_return": cum_ret,
            "median_return": median_ret,
            "avg_hold_days": avg_hold,
            "reason_stats": dict(reason_stats),
        }
        print(f"  {strat.upper()}: {total}笔 | 胜率{win_rate}% | 均收益{avg_ret:+.1f}% | 累积{cum_ret:+.1f}% | 均持仓{avg_hold}天")

    return strategy_results

# ================================================================
# 报告生成
# ================================================================

def generate_report(icir_new, ic_stats, original_icir, baseline_results, recalibrated_results, output_path):
    html = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ICIR 重标定报告</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #f5f5f5; color: #333; padding: 20px; }
.container { max-width: 950px; margin: 0 auto; }
h1 { font-size: 22px; font-weight: 500; margin-bottom: 8px; }
.sub { font-size: 13px; color: #888; margin-bottom: 24px; }
.card { background: #fff; border-radius: 12px; padding: 24px; margin-bottom: 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }
h2 { font-size: 16px; font-weight: 500; margin-bottom: 16px; }
table { width: 100%; border-collapse: collapse; font-size: 13px; }
th { text-align: left; padding: 8px 6px; border-bottom: 2px solid #e0e0e0; font-weight: 500; color: #555; }
td { padding: 8px 6px; border-bottom: 1px solid #f0f0f0; }
tr:hover { background: #fafafa; }
.pos { color: #d32f2f; }
.neg { color: #2e7d32; }
.bar-bg { display: inline-block; height: 10px; background: #e0e0e0; border-radius: 2px; vertical-align: middle; position: relative; width: 80px; }
.bar-fill { position: absolute; height: 100%; border-radius: 2px; }
.tag { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 500; }
.tag-orig { background: #e3f2fd; color: #1565c0; }
.tag-new { background: #fce4ec; color: #c62828; }
.tag-new-factor { background: #e8f5e9; color: #2e7d32; }
.verdict { padding: 16px; border-radius: 8px; margin-top: 16px; font-size: 14px; }
.verdict-good { background: #e8f5e9; border-left: 4px solid #4caf50; }
.verdict-neutral { background: #fff3e0; border-left: 4px solid #ff9800; }
</style>
</head>
<body>
<div class="container">
<h1>ICIR 重标定报告</h1>
<div class="sub">34因子 · 85交易日 · 5日前向收益 · Spearman rank IC | 生成于 """ + datetime.now().strftime("%Y-%m-%d %H:%M") + """</div>
"""

    # === ICIR 对比表 ===
    html += '<div class="card"><h2>因子 ICIR 对比 (原始 vs 重标定)</h2>'
    html += '<table><tr><th>因子</th><th>标签</th><th>原始权重</th><th>新ICIR</th><th>方向</th><th>IC均值</th><th>IC标准差</th><th>IR</th><th>正IC天数%</th></tr>'

    all_factors = sorted(icir_new.keys())
    for fname in all_factors:
        orig_w = original_icir.get(fname, 0)
        new_w = icir_new.get(fname, 0)
        stats = ic_stats.get(fname, {})
        mean_ic = stats.get("mean_ic", 0)
        std_ic = stats.get("std_ic", 0)
        ir = stats.get("ir", 0)
        pos_pct = stats.get("pos_pct", 0)
        is_new = fname in NEW_FACTOR_WEIGHTS
        tag = '<span class="tag tag-new-factor">新增</span>' if is_new else ''
        direction = "高好" if new_w > 0 else ("低好" if new_w < 0 else "无")
        w_cls = "pos" if new_w > 0 else "neg"
        bar_w = min(abs(new_w) * 200, 80)
        bar_color = "#d32f2f" if new_w > 0 else "#2e7d32"
        html += f'<tr><td>{fname}</td><td>{tag}</td><td>{orig_w:.3f}</td><td class="{w_cls}">{new_w:+.4f}</td><td>{direction}</td><td>{mean_ic:+.4f}</td><td>{std_ic:.4f}</td><td>{ir:+.3f}</td><td>{pos_pct:.0f}%</td></tr>'

    html += '</table></div>'

    # === 回测对比 ===
    html += '<div class="card"><h2>回测对比: 原始31因子 vs 重标定34因子</h2>'
    html += '<table><tr><th>策略</th><th>指标</th>'
    html += '<th><span class="tag tag-orig">原始31因子</span></th>'
    html += '<th><span class="tag tag-new">重标定34因子</span></th>'
    html += '<th>变化</th></tr>'

    for strat in ["isir", "glm", "doubao"]:
        b = baseline_results.get(strat, {})
        e = recalibrated_results.get(strat, {})
        for metric, label in [("win_rate", "胜率%"), ("avg_return", "均收益%"), ("cumulative_return", "累积收益%"), ("total_trades", "交易笔数"), ("avg_hold_days", "均持仓天")]:
            bv = b.get(metric, 0)
            ev = e.get(metric, 0)
            delta = ev - bv if isinstance(ev, (int, float)) and isinstance(bv, (int, float)) else 0
            delta_cls = "pos" if delta > 0 else "neg" if delta < 0 else ""
            delta_str = f'{"+" if delta > 0 else ""}{delta:.1f}' if delta != 0 else "-"
            html += f'<tr><td>{strat.upper()}</td><td>{label}</td><td>{bv}</td><td>{ev}</td><td class="{delta_cls}">{delta_str}</td></tr>'

    html += '</table></div>'

    # === 结论 ===
    isir_wr_d = recalibrated_results.get("isir", {}).get("win_rate", 0) - baseline_results.get("isir", {}).get("win_rate", 0)
    glm_wr_d = recalibrated_results.get("glm", {}).get("win_rate", 0) - baseline_results.get("glm", {}).get("win_rate", 0)
    isir_ret_d = recalibrated_results.get("isir", {}).get("avg_return", 0) - baseline_results.get("isir", {}).get("avg_return", 0)
    glm_ret_d = recalibrated_results.get("glm", {}).get("avg_return", 0) - baseline_results.get("glm", {}).get("avg_return", 0)

    avg_wr_d = (isir_wr_d + glm_wr_d) / 2
    avg_ret_d = (isir_ret_d + glm_ret_d) / 2

    if avg_wr_d > 2 and avg_ret_d > 0.5:
        cls = "verdict-good"
        txt = f"重标定效果显著: ISIR/GLM平均胜率提升{avg_wr_d:+.1f}pp, 平均收益提升{avg_ret_d:+.1f}%。建议采用重标定权重。"
    elif avg_wr_d > 0.5 or avg_ret_d > 0:
        cls = "verdict-neutral"
        txt = f"重标定有轻微改善: 平均胜率{avg_wr_d:+.1f}pp, 平均收益{avg_ret_d:+.1f}%。可考虑采用但效果有限。"
    else:
        cls = "verdict-neutral"
        txt = f"重标定未带来改善: 平均胜率{avg_wr_d:+.1f}pp, 平均收益{avg_ret_d:+.1f}%。原始手调权重已接近最优。"

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
    print("  ICIR 重标定")
    print("  34因子 · 85交易日 · 5日前向收益")
    print("=" * 60)

    SELF_DIR = os.path.dirname(os.path.abspath(__file__))
    OUTPUT_DIR = os.path.join(SELF_DIR, "output")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. 加载数据
    print("\n[1/5] 加载数据...")
    db = StockDB()
    with open(os.path.join(SELF_DIR, "stock_codes.txt")) as f:
        codes = [line.strip() for line in f if line.strip()]
    klines = db.get_klines(codes, days=300)
    extra_today = db.get_extra_info(codes)
    sectors = sector_map.STOCK_SECTOR
    print(f"  股票: {len(klines)}只")

    # 2. 获取指数K线
    print("\n[2/5] 获取上证综指K线...")
    index_klines = fetch_index_klines(days=300)
    print(f"  上证综指: {len(index_klines)}条")

    # 3. 计算 ICIR
    print("\n[3/5] 计算 ICIR (34因子)...")
    t0 = time.time()
    icir_new, ic_stats = compute_icir(klines, extra_today, sectors, index_klines, backtest_days=85, fwd_days=5)
    print(f"  ICIR计算耗时: {time.time()-t0:.0f}s")

    # 保存ICIR结果
    icir_path = os.path.join(OUTPUT_DIR, "icir_recalibrated.json")
    with open(icir_path, "w") as f:
        json.dump({"icir_v3": icir_new, "ic_stats": ic_stats}, f, indent=2, ensure_ascii=False)
    print(f"  ICIR已保存: {icir_path}")

    # 打印ICIR排名
    print(f"\n  {'因子':<22} {'ICIR':>8} {'方向':>6} {'IC均值':>8} {'IR':>8} {'正天数':>6}")
    print("  " + "-" * 62)
    sorted_factors = sorted(icir_new.items(), key=lambda x: abs(x[1]), reverse=True)
    for fname, val in sorted_factors:
        stats = ic_stats.get(fname, {})
        direction = "高好" if val > 0 else "低好" if val < 0 else "无"
        is_new = " *" if fname in NEW_FACTOR_WEIGHTS else ""
        print(f"  {fname+is_new:<22} {val:>+8.4f} {direction:>6} {stats.get('mean_ic',0):>+8.4f} {stats.get('ir',0):>+8.3f} {stats.get('pos_pct',0):>5.0f}%")

    # 构建GLM权重 (翻转mfi和pct_52w)
    icir_glm_new = dict(icir_new)
    if "mfi" in icir_glm_new:
        icir_glm_new["mfi"] = -icir_glm_new["mfi"]
    if "pct_52w" in icir_glm_new:
        icir_glm_new["pct_52w"] = -icir_glm_new["pct_52w"]

    # 4. Baseline回测 (原始31因子)
    print("\n[4/5] Baseline回测 (原始手调31因子权重)...")
    t0 = time.time()
    baseline_results = run_backtest_with_weights(
        klines, extra_today, sectors, index_klines,
        eng.ICIR_V3, eng.ICIR_GLM, eng.FACTOR_HIGHER_BETTER,
        backtest_days=85, mode="baseline")
    print(f"  耗时: {time.time()-t0:.0f}s")

    # 5. 重标定回测 (34因子ICIR权重)
    print("\n[5/5] 重标定回测 (34因子ICIR权重)...")
    t0 = time.time()
    recalibrated_results = run_backtest_with_weights(
        klines, extra_today, sectors, index_klines,
        icir_new, icir_glm_new, eng.FACTOR_HIGHER_BETTER,
        backtest_days=85, mode="enhanced")
    print(f"  耗时: {time.time()-t0:.0f}s")

    # 6. 生成报告
    print("\n[6/6] 生成报告...")
    report_path = os.path.join(OUTPUT_DIR, "icir_recalibrate.html")
    generate_report(icir_new, ic_stats, eng.ICIR_V3, baseline_results, recalibrated_results, report_path)

    print(f"\n{'='*60}")
    print(f"  报告已生成: {report_path}")
    print(f"  ICIR数据: {icir_path}")
    print(f"{'='*60}")

    # 终端汇总
    print(f"\n{'='*60}")
    print(f"  汇总对比")
    print(f"{'='*60}")
    print(f"{'策略':<8} {'指标':<12} {'原始31':>10} {'重标定34':>10} {'变化':>10}")
    print("-" * 55)
    for strat in ["isir", "glm", "doubao"]:
        b = baseline_results.get(strat, {})
        e = recalibrated_results.get(strat, {})
        for metric, label in [("win_rate", "胜率%"), ("avg_return", "均收益%"), ("cumulative_return", "累积%")]:
            bv = b.get(metric, 0)
            ev = e.get(metric, 0)
            delta = ev - bv
            print(f"{strat.upper():<8} {label:<12} {bv:>10.1f} {ev:>10.1f} {delta:>+10.1f}")
    print()

if __name__ == "__main__":
    main()
