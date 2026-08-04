#!/usr/bin/env python3
"""
因子 A/B 测试
=============
对比 baseline (31因子) vs enhanced (31+3新因子) 在同一回测区间内的表现。

新增因子：
  1. rs_weighted  — 多周期加权相对强度 (3m/6m/12m vs 上证指数)
  2. close_location — 收盘在日内位置 (close-low)/(high-low)*100
  3. market_trend  — 市场趋势门控 (指数在50日均线之上=1, 之下=-1)

用法: python3 factor_ab_test.py
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

# ================================================================
# 新因子权重 (附加到现有 ICIR 权重之上)
# ================================================================

NEW_FACTOR_WEIGHTS = {
    "rs_weighted": 0.030,    # 相对强度，类似 ret_5d(0.021) 但更全面
    "close_location": 0.020, # 收盘位置，全新维度
    "market_trend": 0.015,   # 市场趋势门控，二元
}

NEW_FACTOR_HIGHER_BETTER = {
    "rs_weighted": True,
    "close_location": True,
    "market_trend": True,
}

# ================================================================
# 指数数据获取 (上证综指 sh000001)
# ================================================================

def fetch_index_klines(days=300):
    """从腾讯API获取上证综指K线 (使用kline端点，指数不支持fqkline)"""
    sym = "sh000001"
    url = f"https://web.ifzq.gtimg.cn/appstock/app/kline/kline?param={sym},day,,,{days},"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    resp = urllib.request.urlopen(req, timeout=15)
    data = json.loads(resp.read().decode("utf-8"))
    kls = data.get("data", {}).get(sym, {}).get("day", [])
    parsed = []
    for k in kls:
        try:
            parsed.append({"date": k[0], "open": float(k[1]), "close": float(k[2]),
                            "high": float(k[3]), "low": float(k[4]), "volume": float(k[5]) if len(k) > 5 else 0})
        except (IndexError, ValueError, TypeError):
            continue
    return parsed

# ================================================================
# 新因子计算
# ================================================================

def calc_rs_weighted(stock_closes, index_closes):
    """多周期加权相对强度: 0.40*3m + 0.30*6m + 0.30*12m"""
    def ret_over(closes, n):
        if len(closes) < n + 1 or closes[-n-1] == 0:
            return 0
        return (closes[-1] / closes[-n-1] - 1) * 100

    r3 = ret_over(stock_closes, 63)
    r6 = ret_over(stock_closes, 126)
    r12 = ret_over(stock_closes, 252)

    ir3 = ret_over(index_closes, 63)
    ir6 = ret_over(index_closes, 126)
    ir12 = ret_over(index_closes, 252)

    rel_3 = r3 - ir3
    rel_6 = r6 - ir6
    rel_12 = r12 - ir12

    return 0.40 * rel_3 + 0.30 * rel_6 + 0.30 * rel_12

def calc_close_location(high, low, close):
    """收盘在日内位置: (close-low)/(high-low)*100"""
    if high == low:
        return 50.0
    return (close - low) / (high - low) * 100

def calc_market_trend(index_closes):
    """市场趋势门控: 指数收盘在50日均线之上=1, 之下=-1"""
    if len(index_closes) < 50:
        return 1.0
    ma50 = sum(index_closes[-50:]) / 50
    return 1.0 if index_closes[-1] > ma50 else -1.0

# ================================================================
# 增强版因子计算
# ================================================================

def compute_all_factors_enhanced(klines, extra_info, fund_flows, events, sectors, index_klines, index_closes_by_date):
    """在原始因子基础上增加3个新因子"""
    # 先调用原始函数
    factor_data, return_data = eng.compute_all_factors(klines, extra_info, fund_flows, events, sectors)

    # 计算市场趋势 (全局，所有股票共用)
    latest_index_closes = [b["close"] for b in index_klines[-300:]] if len(index_klines) >= 50 else []
    market_trend = calc_market_trend(latest_index_closes) if latest_index_closes else 1.0

    # 为每只股票计算新因子
    for code, factors in factor_data.items():
        k = klines.get(code, [])
        if len(k) < 60:
            continue

        closes = [bar["close"] for bar in k]
        # RS加权
        idx_closes = latest_index_closes
        rs_val = calc_rs_weighted(closes, idx_closes) if len(idx_closes) >= 63 else 0
        factors["rs_weighted"] = rs_val

        # 收盘位置
        last_bar = k[-1]
        cl_val = calc_close_location(last_bar["high"], last_bar["low"], last_bar["close"])
        factors["close_location"] = cl_val

        # 市场趋势
        factors["market_trend"] = market_trend

    # 对新因子做截面标准化
    for fname in NEW_FACTOR_WEIGHTS:
        values = [factor_data[c].get(fname, 0) for c in factor_data]
        arr = np.array(values, dtype=float)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        arr = np.clip(arr, np.percentile(arr, 1), np.percentile(arr, 99))
        mean, std = np.mean(arr), np.std(arr)
        if std == 0 or np.isnan(std) or std < 1e-10:
            for c in factor_data:
                factor_data[c][f"{fname}_z"] = 0.0
        else:
            z = (arr - mean) / std
            if not NEW_FACTOR_HIGHER_BETTER.get(fname, True):
                z = -z
            for ci, c in enumerate(factor_data.keys()):
                factor_data[c][f"{fname}_z"] = float(z[ci])

    return factor_data, return_data

# ================================================================
# 增强版排名 (使用增强权重)
# ================================================================

def compute_rankings_enhanced(factor_data):
    """使用增强权重(31+3因子)计算排名"""
    # 保存原始权重
    orig_icir_v3 = dict(eng.ICIR_V3)
    orig_icir_glm = dict(eng.ICIR_GLM)
    orig_higher_better = dict(eng.FACTOR_HIGHER_BETTER)
    orig_labels = dict(eng.FACTOR_LABELS)

    # 添加新因子到权重
    for fname, w in NEW_FACTOR_WEIGHTS.items():
        eng.ICIR_V3[fname] = w
        eng.ICIR_GLM[fname] = w  # GLM不反转新因子方向
        eng.FACTOR_HIGHER_BETTER[fname] = NEW_FACTOR_HIGHER_BETTER[fname]

    eng.FACTOR_LABELS["rs_weighted"] = "相对强度"
    eng.FACTOR_LABELS["close_location"] = "收盘位置"
    eng.FACTOR_LABELS["market_trend"] = "市场趋势"

    try:
        results = eng.compute_rankings(factor_data)
    finally:
        # 恢复原始权重
        eng.ICIR_V3.clear()
        eng.ICIR_V3.update(orig_icir_v3)
        eng.ICIR_GLM.clear()
        eng.ICIR_GLM.update(orig_icir_glm)
        eng.FACTOR_HIGHER_BETTER.clear()
        eng.FACTOR_HIGHER_BETTER.update(orig_higher_better)
        eng.FACTOR_LABELS.clear()
        eng.FACTOR_LABELS.update(orig_labels)

    return results

# ================================================================
# 回测逻辑 (复用 strategy_backtest 的核心逻辑)
# ================================================================

def run_backtest(klines, extra_today, sectors, index_klines, backtest_days=85, mode="baseline"):
    """运行策略回测，返回三体系统计结果"""
    all_dates = set()
    for k in klines.values():
        for b in k:
            all_dates.add(b["date"])
    dates = sorted(all_dates)

    start_idx = max(60, len(dates) - backtest_days)
    test_dates = dates[start_idx:]
    print(f"  回测区间: {test_dates[0]} ~ {test_dates[-1]} ({len(test_dates)}个交易日)")

    # 预处理每日数据
    daily_data = {}
    for di, date in enumerate(test_dates):
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

        # 当日指数K线截断
        idx_klines_trunc = [b for b in index_klines if b["date"] <= date]

        daily_data[date] = {
            "klines": day_klines,
            "extra": day_extra,
            "next_opens": next_opens,
            "index_klines": idx_klines_trunc,
        }

    print(f"  有效交易日: {len(daily_data)}")

    # 回测参数
    strategies = ["isir", "glm", "doubao"]
    max_positions = 30
    hold_days_max = 10
    stop_loss = -8.0
    take_profit = 15.0
    rank_collapse_pct = 0.5
    buy_count = 30

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

            # 计算排名
            if mode == "enhanced":
                factor_data, _ = compute_all_factors_enhanced(
                    day_klines, day_extra, {}, {}, sectors, idx_klines, {})
                rankings = compute_rankings_enhanced(factor_data)
            else:
                factor_data, _ = eng.compute_all_factors(day_klines, day_extra, {}, {}, sectors)
                rankings = eng.compute_rankings(factor_data)

            if len(factor_data) < 100 or not rankings:
                continue

            rank_key = f"{strat}_rank"
            rank_map = {r["code"]: r for r in rankings}
            n_total = len(rankings)
            current_top = {r["code"] for r in rankings if r[rank_key] <= buy_count}

            # 检查退出
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

            # 买入新股
            existing_codes = {p["code"] for p in positions}
            available_slots = max_positions - len(positions)

            if available_slots > 0:
                new_candidates = []
                for code in current_top - existing_codes:
                    r = rank_map.get(code)
                    if r and code in next_opens and next_opens[code] > 0:
                        new_candidates.append((code, r[rank_key]))
                new_candidates.sort(key=lambda x: x[1])

                for code, rank in new_candidates[:available_slots]:
                    entry_price = next_opens[code]
                    name = day_extra.get(code, {}).get("name", code)
                    positions.append({
                        "code": code, "name": name,
                        "entry_date": date, "entry_price": entry_price,
                        "entry_rank": rank, "hold_days": 0,
                    })

        # 强制平仓
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

        # 统计
        total = len(closed_trades)
        wins = sum(1 for t in closed_trades if t["return_pct"] > 0)
        win_rate = round(wins / total * 100, 1) if total > 0 else 0
        avg_ret = round(float(np.mean([t["return_pct"] for t in closed_trades])), 2) if total > 0 else 0
        cum_ret = round(sum(t["return_pct"] for t in closed_trades), 1)
        median_ret = round(float(np.median([t["return_pct"] for t in closed_trades])), 2) if total > 0 else 0
        max_ret = round(max(t["return_pct"] for t in closed_trades), 2) if total > 0 else 0
        min_ret = round(min(t["return_pct"] for t in closed_trades), 2) if total > 0 else 0

        reason_stats = defaultdict(int)
        for t in closed_trades:
            reason_stats[t.get("exit_reason", "unknown")] += 1

        avg_hold = round(float(np.mean([t["hold_days"] for t in closed_trades])), 1) if total > 0 else 0

        strategy_results[strat] = {
            "total_trades": total,
            "win_rate": win_rate,
            "avg_return": avg_ret,
            "median_return": median_ret,
            "max_return": max_ret,
            "min_return": min_ret,
            "cumulative_return": cum_ret,
            "avg_hold_days": avg_hold,
            "reason_stats": dict(reason_stats),
        }

        print(f"  {strat.upper()}: {total}笔 | 胜率{win_rate}% | 均收益{avg_ret:+.1f}% | 累积{cum_ret:+.1f}% | 均持仓{avg_hold}天")

    return strategy_results

# ================================================================
# 固定持仓基准 (5/10/20日)
# ================================================================

def run_baseline_hold(klines, extra_today, sectors, index_klines, backtest_days=85, mode="baseline", hold_periods=[5,10,20]):
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
        idx_klines_trunc = [b for b in index_klines if b["date"] <= date]
        daily_data[date] = {"klines": day_klines, "extra": day_extra, "next_opens": next_opens, "index_klines": idx_klines_trunc}

    results = {}
    for strat in ["isir", "glm", "doubao"]:
        rank_key = f"{strat}_rank"
        for hold in hold_periods:
            returns = []
            for date in sorted(daily_data.keys()):
                dd = daily_data[date]
                if mode == "enhanced":
                    factor_data, _ = compute_all_factors_enhanced(dd["klines"], dd["extra"], {}, {}, sectors, dd["index_klines"], {})
                    rankings = compute_rankings_enhanced(factor_data)
                else:
                    factor_data, _ = eng.compute_all_factors(dd["klines"], dd["extra"], {}, {}, sectors)
                    rankings = eng.compute_rankings(factor_data)
                if len(factor_data) < 100 or not rankings:
                    continue
                top30 = [r for r in rankings if r[rank_key] <= 30]
                for r in top30:
                    code = r["code"]
                    entry = dd["next_opens"].get(code)
                    if not entry or entry <= 0:
                        continue
                    fwd_bars = [b for b in klines.get(code, []) if b["date"] > date]
                    if len(fwd_bars) >= hold:
                        exit_price = fwd_bars[hold - 1]["close"]
                        ret = round((exit_price / entry - 1) * 100, 2)
                        returns.append(ret)
            if returns:
                wr = round(sum(1 for r in returns if r > 0) / len(returns) * 100, 1)
                ar = round(float(np.mean(returns)), 2)
                results[f"{strat}_{hold}d"] = {"win_rate": wr, "avg_return": ar, "total": len(returns)}
                print(f"  {strat} {hold}d: 胜率{wr}% | 均收益{ar:+.1f}% | {len(returns)}笔")
    return results

# ================================================================
# 生成对比报告
# ================================================================

def generate_report(baseline_results, enhanced_results, baseline_hold, enhanced_hold, output_path):
    html = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>因子 A/B 测试报告</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #f5f5f5; color: #333; padding: 20px; }
.container { max-width: 900px; margin: 0 auto; }
h1 { font-size: 22px; font-weight: 500; margin-bottom: 8px; }
.sub { font-size: 13px; color: #888; margin-bottom: 24px; }
.card { background: #fff; border-radius: 12px; padding: 24px; margin-bottom: 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }
h2 { font-size: 16px; font-weight: 500; margin-bottom: 16px; }
table { width: 100%; border-collapse: collapse; font-size: 13px; }
th { text-align: left; padding: 10px 8px; border-bottom: 2px solid #e0e0e0; font-weight: 500; color: #555; }
td { padding: 10px 8px; border-bottom: 1px solid #f0f0f0; }
tr:hover { background: #fafafa; }
.pos { color: #d32f2f; font-weight: 500; }
.neg { color: #2e7d32; font-weight: 500; }
.delta-up { color: #d32f2f; font-size: 11px; }
.delta-down { color: #2e7d32; font-size: 11px; }
.tag { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 500; }
.tag-baseline { background: #e3f2fd; color: #1565c0; }
.tag-enhanced { background: #fce4ec; color: #c62828; }
.summary { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 12px; margin-bottom: 16px; }
.metric { text-align: center; padding: 16px; border-radius: 8px; background: #f5f5f5; }
.metric-val { font-size: 24px; font-weight: 500; }
.metric-label { font-size: 11px; color: #888; margin-top: 4px; }
.verdict { padding: 16px; border-radius: 8px; margin-top: 16px; font-size: 14px; }
.verdict-good { background: #e8f5e9; border-left: 4px solid #4caf50; }
.verdict-neutral { background: #fff3e0; border-left: 4px solid #ff9800; }
.verdict-bad { background: #ffebee; border-left: 4px solid #f44336; }
</style>
</head>
<body>
<div class="container">
<h1>因子 A/B 测试报告</h1>
<div class="sub">Baseline (31因子) vs Enhanced (31+3新因子: RS加权/收盘位置/市场趋势门控) | 生成于 """ + datetime.now().strftime("%Y-%m-%d %H:%M") + """</div>
"""

    # === 实战策略对比 ===
    html += '<div class="card"><h2>实战策略回测对比</h2><table><tr><th>策略</th><th>指标</th>'
    html += '<th><span class="tag tag-baseline">Baseline 31因子</span></th>'
    html += '<th><span class="tag tag-enhanced">Enhanced 34因子</span></th>'
    html += '<th>变化</th></tr>'

    for strat in ["isir", "glm", "doubao"]:
        b = baseline_results.get(strat, {})
        e = enhanced_results.get(strat, {})
        for metric, label in [("win_rate", "胜率%"), ("avg_return", "均收益%"), ("cumulative_return", "累积收益%"), ("total_trades", "交易笔数"), ("avg_hold_days", "均持仓天")]:
            bv = b.get(metric, 0)
            ev = e.get(metric, 0)
            delta = ev - bv if isinstance(ev, (int, float)) and isinstance(bv, (int, float)) else 0
            delta_cls = "delta-up" if delta > 0 else "delta-down" if delta < 0 else ""
            delta_str = f'{"+" if delta > 0 else ""}{delta:.1f}' if delta != 0 else "-"
            val_cls = "pos" if (metric in ["win_rate", "avg_return", "cumulative_return"] and ev > 0) else "neg" if ev < 0 else ""
            html += f'<tr><td>{strat.upper()}</td><td>{label}</td><td>{bv}</td><td class="{val_cls}">{ev}</td><td class="{delta_cls}">{delta_str}</td></tr>'

    html += '</table>'

    # 退出原因对比
    html += '<h2 style="margin-top:20px">退出原因分布</h2><table><tr><th>策略</th><th>原因</th>'
    html += '<th>Baseline</th><th>Enhanced</th><th>变化</th></tr>'
    all_reasons = set()
    for strat in ["isir", "glm", "doubao"]:
        all_reasons.update(baseline_results.get(strat, {}).get("reason_stats", {}).keys())
        all_reasons.update(enhanced_results.get(strat, {}).get("reason_stats", {}).keys())
    for strat in ["isir", "glm", "doubao"]:
        b_reasons = baseline_results.get(strat, {}).get("reason_stats", {})
        e_reasons = enhanced_results.get(strat, {}).get("reason_stats", {})
        for reason in sorted(all_reasons):
            bv = b_reasons.get(reason, 0)
            ev = e_reasons.get(reason, 0)
            delta = ev - bv
            html += f'<tr><td>{strat.upper()}</td><td>{reason}</td><td>{bv}</td><td>{ev}</td><td style="color:#888">{"+" if delta>0 else ""}{delta}</td></tr>'
    html += '</table></div>'

    # === 固定持仓基准对比 ===
    html += '<div class="card"><h2>固定持仓基准对比 (5/10/20日)</h2><table><tr><th>策略</th><th>持仓</th>'
    html += '<th><span class="tag tag-baseline">Baseline 胜率</span></th>'
    html += '<th><span class="tag tag-enhanced">Enhanced 胜率</span></th>'
    html += '<th>变化</th><th>BL均收益</th><th>EN均收益</th></tr>'
    for strat in ["isir", "glm", "doubao"]:
        for hold in [5, 10, 20]:
            key = f"{strat}_{hold}d"
            b = baseline_hold.get(key, {})
            e = enhanced_hold.get(key, {})
            bwr = b.get("win_rate", 0)
            ewr = e.get("win_rate", 0)
            delta = ewr - bwr
            delta_cls = "delta-up" if delta > 0 else "delta-down" if delta < 0 else ""
            html += f'<tr><td>{strat.upper()}</td><td>{hold}日</td><td>{bwr}%</td><td>{ewr}%</td><td class="{delta_cls}">{"+" if delta>0 else ""}{delta:.1f}pp</td><td>{b.get("avg_return", 0):+.1f}%</td><td>{e.get("avg_return", 0):+.1f}%</td></tr>'
    html += '</table></div>'

    # === 结论 ===
    isir_wr_delta = enhanced_results.get("isir", {}).get("win_rate", 0) - baseline_results.get("isir", {}).get("win_rate", 0)
    glm_wr_delta = enhanced_results.get("glm", {}).get("win_rate", 0) - baseline_results.get("glm", {}).get("win_rate", 0)
    isir_ret_delta = enhanced_results.get("isir", {}).get("avg_return", 0) - baseline_results.get("isir", {}).get("avg_return", 0)
    glm_ret_delta = enhanced_results.get("glm", {}).get("avg_return", 0) - baseline_results.get("glm", {}).get("avg_return", 0)

    avg_wr_delta = (isir_wr_delta + glm_wr_delta) / 2
    avg_ret_delta = (isir_ret_delta + glm_ret_delta) / 2

    if avg_wr_delta > 2 and avg_ret_delta > 0.5:
        verdict_cls = "verdict-good"
        verdict_text = f"新因子有显著正向贡献: ISIR/GLM平均胜率提升{avg_wr_delta:.1f}pp, 平均收益提升{avg_ret_delta:+.1f}%。建议正式纳入。"
    elif avg_wr_delta > 0.5 or avg_ret_delta > 0:
        verdict_cls = "verdict-neutral"
        verdict_text = f"新因子有轻微正向贡献: 平均胜率变化{avg_wr_delta:+.1f}pp, 平均收益变化{avg_ret_delta:+.1f}%。效果不显著，建议调整权重后重测。"
    else:
        verdict_cls = "verdict-bad"
        verdict_text = f"新因子未带来改善: 平均胜率变化{avg_wr_delta:+.1f}pp, 平均收益变化{avg_ret_delta:+.1f}%。当前权重配置下不建议纳入。"

    html += f'<div class="card"><h2>测试结论</h2><div class="verdict {verdict_cls}">{verdict_text}</div>'
    html += '<p style="margin-top:12px;font-size:12px;color:#888">注: 本测试使用相同数据、相同回测区间(82交易日)、相同买卖规则(T+1开盘执行, 止损-8%/止盈+15%/持仓10天/排名崩溃后50%)。唯一变量是因子集。新因子权重: RS=0.030, close_location=0.020, market_trend=0.015。未重新标定ICIR权重，仅增量叠加。</p></div>'

    html += '</div></body></html>'

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    return output_path

# ================================================================
# 主流程
# ================================================================

def main():
    print("=" * 60)
    print("  因子 A/B 测试")
    print("  Baseline (31因子) vs Enhanced (31+3新因子)")
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
    print(f"  股票: {len(klines)}只 | K线条数: {sum(len(v) for v in klines.values())}")

    # 2. 获取指数K线
    print("\n[2/5] 获取上证综指K线...")
    index_klines = fetch_index_klines(days=300)
    print(f"  上证综指: {len(index_klines)}条 | 日期范围: {index_klines[0]['date']} ~ {index_klines[-1]['date']}")

    # 3. Baseline 回测
    print("\n[3/5] Baseline 回测 (31因子)...")
    t0 = time.time()
    baseline_results = run_backtest(klines, extra_today, sectors, index_klines, backtest_days=85, mode="baseline")
    print(f"  耗时: {time.time()-t0:.0f}s")

    print("\n  Baseline 固定持仓基准...")
    baseline_hold = run_baseline_hold(klines, extra_today, sectors, index_klines, mode="baseline")

    # 4. Enhanced 回测
    print("\n[4/5] Enhanced 回测 (34因子)...")
    t0 = time.time()
    enhanced_results = run_backtest(klines, extra_today, sectors, index_klines, backtest_days=85, mode="enhanced")
    print(f"  耗时: {time.time()-t0:.0f}s")

    print("\n  Enhanced 固定持仓基准...")
    enhanced_hold = run_baseline_hold(klines, extra_today, sectors, index_klines, mode="enhanced")

    # 5. 生成报告
    print("\n[5/5] 生成对比报告...")
    report_path = os.path.join(OUTPUT_DIR, "factor_ab_test.html")
    generate_report(baseline_results, enhanced_results, baseline_hold, enhanced_hold, report_path)

    print(f"\n{'='*60}")
    print(f"  报告已生成: {report_path}")
    print(f"{'='*60}")

    # 终端打印汇总
    print(f"\n{'='*60}")
    print(f"  汇总对比")
    print(f"{'='*60}")
    print(f"{'策略':<8} {'指标':<12} {'Baseline':>10} {'Enhanced':>10} {'变化':>10}")
    print("-" * 55)
    for strat in ["isir", "glm", "doubao"]:
        b = baseline_results.get(strat, {})
        e = enhanced_results.get(strat, {})
        for metric, label in [("win_rate", "胜率%"), ("avg_return", "均收益%"), ("cumulative_return", "累积%")]:
            bv = b.get(metric, 0)
            ev = e.get(metric, 0)
            delta = ev - bv
            print(f"{strat.upper():<8} {label:<12} {bv:>10.1f} {ev:>10.1f} {delta:>+10.1f}")
    print()

if __name__ == "__main__":
    main()
