#!/usr/bin/env python3
"""
美股隔夜信号回测 (Yahoo Finance数据)
=====================================
用真正的美股隔夜收益率(非恒生代理)回测外部信号对A股的预测力。

数据源: Yahoo Finance API
  ^NDX  纳斯达克100 → 科技板块映射
  ^DJI  道琼斯      → 全市场风险偏好
  ^VIX  VIX         → 恐慌指数门控
  ^N225 日经225      → 亚太早盘信号

用法: python3 us_overnight_backtest.py
"""

import os, json, math, sys, time
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stock_db import StockDB
import sector_map
import unified_scoring_engine as eng

# ================================================================
# 板块 → 外部信号映射
# ================================================================

SECTOR_US_MAPPING = {
    "半导体/芯片":     {"^NDX": 0.6, "^DJI": 0.2, "^N225": 0.2},
    "AI/算力/通信":    {"^NDX": 0.6, "^DJI": 0.2, "^N225": 0.2},
    "电子/消费电子":   {"^NDX": 0.5, "^DJI": 0.3, "^N225": 0.2},
    "传媒/游戏":       {"^NDX": 0.4, "^DJI": 0.3, "^N225": 0.3},
    "新能源/电力":     {"^NDX": 0.3, "^DJI": 0.4, "^N225": 0.3},
    "金融/交通/基建":  {"^NDX": 0.1, "^DJI": 0.6, "^N225": 0.3},
    "智能制造":         {"^NDX": 0.2, "^DJI": 0.5, "^N225": 0.3},
    "医药生物":         {"^NDX": 0.1, "^DJI": 0.5, "^N225": 0.4},
    "消费":             {"^NDX": 0.1, "^DJI": 0.5, "^N225": 0.4},
    "化工/新材料":      {"^NDX": 0.0, "^DJI": 0.4, "^N225": 0.6},
}

# ================================================================
# Yahoo Finance 数据获取
# ================================================================

def fetch_yahoo_klines(symbol, range_str="1y"):
    """从Yahoo Finance获取历史K线"""
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?range={range_str}&interval=1d"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    resp = urllib.request.urlopen(req, timeout=15)
    data = json.loads(resp.read().decode("utf-8"))
    result = data.get("chart", {}).get("result", [{}])[0]
    timestamps = result.get("timestamp", [])
    quotes = result.get("indicators", {}).get("quote", [{}])[0]
    closes = quotes.get("close", [])

    klines = {}
    for t, c in zip(timestamps, closes):
        if c is not None:
            date_str = datetime.fromtimestamp(t).strftime("%Y-%m-%d")
            klines[date_str] = c
    return klines

def compute_daily_returns(klines_dict):
    """计算日收益率序列 {date: return_pct}"""
    dates = sorted(klines_dict.keys())
    returns = {}
    for i in range(1, len(dates)):
        d = dates[i]
        prev = dates[i - 1]
        prev_close = klines_dict[prev]
        if prev_close > 0:
            returns[d] = (klines_dict[d] / prev_close - 1) * 100
    return returns

def find_overnight_return(a_share_date, us_returns):
    """找A股交易日之前最近的美股交易日收益率
    美股日期D的收盘 = 北京时间D+1凌晨4点, A股D+1早上9:30开盘时可用
    所以对A股日期D, 只能用us_returns[D-1]及更早的数据, 不能用us_returns[D]
    """
    d = datetime.strptime(a_share_date, "%Y-%m-%d")
    for i in range(1, 6):
        prev_d = (d - timedelta(days=i)).strftime("%Y-%m-%d")
        if prev_d in us_returns:
            return us_returns[prev_d]
    return 0.0

# ================================================================
# sector_external 因子计算
# ================================================================

NEW_FACTOR = "us_overnight"

def compute_sector_external(sector, a_share_date, ndx_ret, dji_ret, n225_ret):
    """计算板块外部信号值"""
    weights = SECTOR_US_MAPPING.get(sector, {"^DJI": 0.3})
    return (ndx_ret * weights.get("^NDX", 0) +
            dji_ret * weights.get("^DJI", 0) +
            n225_ret * weights.get("^N225", 0))

# ================================================================
# ICIR 计算
# ================================================================

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

def compute_icir(klines, extra_today, sectors, ndx_ret, dji_ret, n225_ret, backtest_days=85, fwd_days=5):
    print("  计算IC...")
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
    daily_factors = []

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

        factor_data, _ = eng.compute_all_factors(day_klines, day_extra, {}, {}, sectors)

        # 计算美股隔夜信号
        ndx_r = find_overnight_return(date, ndx_ret)
        dji_r = find_overnight_return(date, dji_ret)
        n225_r = find_overnight_return(date, n225_ret)

        for code in factor_data:
            sector = sectors.get(code, "其他")
            ext_val = compute_sector_external(sector, date, ndx_r, dji_r, n225_r)
            factor_data[code][NEW_FACTOR] = ext_val

        # 截面标准化
        values = [factor_data[c].get(NEW_FACTOR, 0) for c in factor_data]
        arr = np.array(values, dtype=float)
        if np.std(arr) > 1e-10:
            z = (arr - np.mean(arr)) / np.std(arr)
            for ci, c in enumerate(factor_data.keys()):
                factor_data[c][f"{NEW_FACTOR}_z"] = float(z[ci])
        else:
            for c in factor_data:
                factor_data[c][f"{NEW_FACTOR}_z"] = 0.0

        # 前向收益
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

        ic = _spearman(_rank_data(z_vals), _rank_data(fwd_arr))
        daily_ics.append(ic)
        daily_factors.append({"date": date, "ndx": ndx_r, "dji": dji_r, "n225": n225_r, "ic": ic})

    if daily_ics:
        mean_ic = float(np.mean(daily_ics))
        std_ic = float(np.std(daily_ics))
        ir = mean_ic / std_ic if std_ic > 1e-10 else 0
        pos_pct = sum(1 for x in daily_ics if x > 0) / len(daily_ics) * 100
        return {
            "mean_ic": round(mean_ic, 4), "std_ic": round(std_ic, 4),
            "ir": round(ir, 4), "pos_pct": round(pos_pct, 1),
            "n_days": len(daily_ics), "daily": daily_factors,
        }
    return {"mean_ic": 0, "std_ic": 0, "ir": 0, "pos_pct": 0, "n_days": 0, "daily": []}

# ================================================================
# A/B 回测
# ================================================================

def run_ab_backtest(klines, extra_today, sectors, ndx_ret, dji_ret, n225_ret, backtest_days=85):
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
    results = {"baseline": {}, "enhanced": {}}
    ext_weight = 0.05  # 新因子权重

    for mode in ["baseline", "enhanced"]:
        print(f"\n  === {mode.upper()} ===")
        orig_v3 = dict(eng.ICIR_V3)
        orig_glm = dict(eng.ICIR_GLM)

        if mode == "enhanced":
            eng.ICIR_V3[NEW_FACTOR] = ext_weight
            eng.ICIR_GLM[NEW_FACTOR] = ext_weight

        try:
            for strat in strategies:
                positions = []
                closed_trades = []

                for date in sorted(daily_data.keys()):
                    dd = daily_data[date]
                    day_klines = dd["klines"]
                    day_extra = dd["extra"]
                    next_opens = dd["next_opens"]

                    factor_data, _ = eng.compute_all_factors(day_klines, day_extra, {}, {}, sectors)

                    if mode == "enhanced":
                        ndx_r = find_overnight_return(date, ndx_ret)
                        dji_r = find_overnight_return(date, dji_ret)
                        n225_r = find_overnight_return(date, n225_ret)
                        for code in factor_data:
                            sector = sectors.get(code, "其他")
                            factor_data[code][NEW_FACTOR] = compute_sector_external(sector, date, ndx_r, dji_r, n225_r)
                        # 标准化
                        values = [factor_data[c].get(NEW_FACTOR, 0) for c in factor_data]
                        arr = np.array(values, dtype=float)
                        if np.std(arr) > 1e-10:
                            z = (arr - np.mean(arr)) / np.std(arr)
                            for ci, c in enumerate(factor_data.keys()):
                                factor_data[c][f"{NEW_FACTOR}_z"] = float(z[ci])

                    if len(factor_data) < 100:
                        continue
                    rankings = eng.compute_rankings(factor_data)
                    if not rankings:
                        continue

                    rank_key = f"{strat}_rank"
                    rank_map = {r["code"]: r for r in rankings}
                    n_total = len(rankings)
                    current_top = {r["code"] for r in rankings if r.get(rank_key, 999) <= 30}

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
                        if pos["hold_days"] >= 10: exit_reason = "time"
                        elif ret <= -8.0: exit_reason = "stop"
                        elif ret >= 15.0: exit_reason = "profit"
                        elif current_rank > n_total * 0.5: exit_reason = "rank"
                        if exit_reason:
                            exit_price = next_opens.get(code, price)
                            exit_ret = round((exit_price / pos["entry_price"] - 1) * 100, 2)
                            pos["return_pct"] = exit_ret
                            closed_trades.append(pos)
                        else:
                            still_holding.append(pos)
                    positions = still_holding

                    existing = {p["code"] for p in positions}
                    available = 30 - len(positions)
                    if available > 0:
                        new_cands = []
                        for code in current_top - existing:
                            r = rank_map.get(code)
                            if r and code in next_opens and next_opens[code] > 0:
                                new_cands.append((code, r.get(rank_key, 999)))
                        new_cands.sort(key=lambda x: x[1])
                        for code, rank in new_cands[:available]:
                            positions.append({"code": code, "entry_date": date,
                                            "entry_price": next_opens[code], "hold_days": 0})

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
        finally:
            eng.ICIR_V3.clear()
            eng.ICIR_V3.update(orig_v3)
            eng.ICIR_GLM.clear()
            eng.ICIR_GLM.update(orig_glm)

    return results

# ================================================================
# 报告
# ================================================================

def generate_report(ic_stats, ab_results, today_snapshot, output_path):
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>美股隔夜信号回测</title>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family:-apple-system,sans-serif; background:#f5f5f5; color:#333; padding:20px; }}
.c {{ max-width:900px; margin:0 auto; }}
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
<h1>美股隔夜信号回测 (Yahoo Finance)</h1>
<div class="sub">纳斯达克+道琼斯+日经225 → A股板块映射 · {datetime.now().strftime("%Y-%m-%d %H:%M")}</div>

<div class="card">
<h2>us_overnight 因子 IC 统计</h2>
<table>
<tr><th>指标</th><th>值</th></tr>
<tr><td>IC 均值</td><td class="{"pos" if ic_stats["mean_ic"]>0 else "neg"}">{ic_stats["mean_ic"]:+.4f}</td></tr>
<tr><td>IC 标准差</td><td>{ic_stats["std_ic"]:.4f}</td></tr>
<tr><td>IR (IC/STD)</td><td>{ic_stats["ir"]:+.3f}</td></tr>
<tr><td>正 IC 天数</td><td>{ic_stats["pos_pct"]:.0f}%</td></tr>
<tr><td>有效天数</td><td>{ic_stats["n_days"]}</td></tr>
</table>
</div>

<div class="card">
<h2>A/B 回测: 原始31因子 vs 31+us_overnight</h2>
<table>
<tr><th>策略</th><th>指标</th><th>Baseline</th><th>+美股隔夜</th><th>变化</th></tr>
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

    # 今日快照
    html += '<div class="card"><h2>今日外围市场快照</h2><table>'
    html += '<tr><th>指数</th><th>最新收盘</th><th>涨跌</th><th>对A股影响</th></tr>'
    for name, info in today_snapshot.items():
        cls = "pos" if info["ret"] > 0 else "neg"
        html += f'<tr><td>{name}</td><td>{info["close"]:.1f}</td><td class="{cls}">{info["ret"]:+.2f}%</td><td>{info["impact"]}</td></tr>'
    html += '</table></div>'

    # 结论
    isir_d = ab_results["enhanced"].get("isir",{}).get("win_rate",0) - ab_results["baseline"].get("isir",{}).get("win_rate",0)
    glm_d = ab_results["enhanced"].get("glm",{}).get("win_rate",0) - ab_results["baseline"].get("glm",{}).get("win_rate",0)
    avg_d = (isir_d + glm_d) / 2
    ic = ic_stats["mean_ic"]

    if avg_d > 1 and ic > 0.005:
        cls = "v-good"
        txt = f"美股隔夜信号有效! IC={ic:+.4f}, 平均胜率提升{avg_d:+.1f}pp。建议纳入引擎。"
    elif avg_d > 0 or ic > 0:
        cls = "v-neutral"
        txt = f"美股隔夜信号有轻微正向: IC={ic:+.4f}, 平均胜率变化{avg_d:+.1f}pp。"
    else:
        cls = "v-neutral"
        txt = f"美股隔夜信号效果不明显: IC={ic:+.4f}, 平均胜率变化{avg_d:+.1f}pp。"

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
    print("  美股隔夜信号回测 (Yahoo Finance)")
    print("  ^NDX + ^DJI + ^N225 → A股板块映射")
    print("=" * 60)

    SELF_DIR = os.path.dirname(os.path.abspath(__file__))
    OUTPUT_DIR = os.path.join(SELF_DIR, "output")

    print("\n[1/5] 加载A股数据...")
    db = StockDB()
    with open(os.path.join(SELF_DIR, "stock_codes.txt")) as f:
        codes = [l.strip() for l in f if l.strip()]
    klines = db.get_klines(codes, days=300)
    extra_today = db.get_extra_info(codes)
    sectors = sector_map.STOCK_SECTOR
    print(f"  A股: {len(klines)}只")

    print("\n[2/5] 获取美股历史K线 (Yahoo Finance)...")
    ndx_data = fetch_yahoo_klines("^NDX")
    dji_data = fetch_yahoo_klines("^DJI")
    n225_data = fetch_yahoo_klines("^N225")
    vix_data = fetch_yahoo_klines("^VIX")

    ndx_ret = compute_daily_returns(ndx_data)
    dji_ret = compute_daily_returns(dji_data)
    n225_ret = compute_daily_returns(n225_data)

    print(f"  NDX: {len(ndx_data)}条 | DJI: {len(dji_data)}条 | N225: {len(n225_data)}条 | VIX: {len(vix_data)}条")
    print(f"  收益率: NDX {len(ndx_ret)}日 | DJI {len(dji_ret)}日 | N225 {len(n225_ret)}日")

    # 今日快照
    today = datetime.now().strftime("%Y-%m-%d")
    today_snapshot = {}
    for name, data, rets, impact in [
        ("纳斯达克100", ndx_data, ndx_ret, "科技板块情绪"),
        ("道琼斯", dji_data, dji_ret, "整体风险偏好"),
        ("日经225", n225_data, n225_ret, "亚太早盘信号"),
    ]:
        dates_sorted = sorted(data.keys())
        last_date = dates_sorted[-1]
        last_close = data[last_date]
        r = rets.get(last_date, 0)
        today_snapshot[name] = {"close": last_close, "ret": r, "impact": impact}

    vix_close = list(vix_data.values())[-1] if vix_data else 0
    today_snapshot["VIX"] = {"close": vix_close, "ret": 0, "impact": "恐慌指数" + ("(高位警戒)" if vix_close > 25 else "(正常)")}

    print(f"\n  今日外围快照:")
    for name, info in today_snapshot.items():
        print(f"    {name:<14} {info['close']:>10.1f}  {info['ret']:+.2f}%  {info['impact']}")

    print(f"\n[3/5] 计算 us_overnight IC...")
    ic_stats = compute_icir(klines, extra_today, sectors, ndx_ret, dji_ret, n225_ret, backtest_days=85)
    print(f"  IC={ic_stats['mean_ic']:+.4f} | IR={ic_stats['ir']:+.3f} | 正天数={ic_stats['pos_pct']:.0f}% | 天数={ic_stats['n_days']}")

    print(f"\n[4/5] A/B 回测...")
    ab_results = run_ab_backtest(klines, extra_today, sectors, ndx_ret, dji_ret, n225_ret, backtest_days=85)

    print(f"\n[5/5] 生成报告...")
    report_path = os.path.join(OUTPUT_DIR, "us_overnight.html")
    generate_report(ic_stats, ab_results, today_snapshot, report_path)

    print(f"\n{'='*60}")
    print(f"  报告: {report_path}")
    print(f"{'='*60}")

    print(f"\n  IC: {ic_stats['mean_ic']:+.4f} | IR: {ic_stats['ir']:+.3f} | 正天数: {ic_stats['pos_pct']:.0f}%")
    print(f"\n  {'策略':<8} {'指标':<10} {'Baseline':>10} {'+美股隔夜':>10} {'变化':>8}")
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
