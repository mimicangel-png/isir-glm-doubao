#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整长历史回放回测: 筹码类 + 资金流入类 指标的前向收益预测力。

设计(严格防前视偏差):
  信号日 T 的指标(可事前观察) → 前向 T+1..T+H 收益(事后验证)
  评分排名用 K 线逐日回放重算(与实时引擎同口径), 不用残缺的历史存档。

回测两类指标:
  [筹码类] conc70/conc90(集中度) / profit_rate(获利盘) / 成本偏离(close/avg_cost)
  [资金流入类] main_net_today(当日主力净流入) / main_net_5d / main_net_20d /
               inflow_rate / retail_inflow(散户流入) / jumbo_net(超大单)

数据源:
  - klines: 327天K线(2025-05~2026-09) 算因子+前向收益
  - chip: 筹码数据(2025-10~2026-09, 约220交易日)
  - fund_flows: 资金流(补全后~365天)
  - extra_info: 用于历史因子计算

用法:
  python replay_rank_backtest.py            # 全窗口回放
  python replay_rank_backtest.py --skip-replay  # 用缓存的历史排名(二次调试)
"""
import sqlite3, json, os, sys, glob, time
from collections import defaultdict
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(BASE, "output", "stock_cache.db")
OUTPUT = os.path.join(BASE, "output")
sys.path.insert(0, BASE)

import sector_map
from unified_scoring_engine import compute_all_factors, compute_rankings

# 持有期
HOLDS = [5, 10, 20]
# 分组比例
TOP_PCT = 0.3
BOTTOM_PCT = 0.3


def load_klines(conn):
    k = defaultdict(dict)
    for code, date, close in conn.execute("SELECT code,date,close FROM klines ORDER BY date"):
        k[code][date] = close
    return k


def load_klines_full(conn):
    """完整OHLCV, 用于回放因子"""
    k = defaultdict(list)
    for code, date, o, h, l, c, v in conn.execute(
            "SELECT code,date,open,high,low,close,volume FROM klines ORDER BY date"):
        k[code].append({"date": date, "open": o, "high": h, "low": l, "close": c, "volume": v})
    return k


def load_extra_info(conn):
    """extra_info -> {code: {pe,pb,mcap,turnover,vol_ratio,float_mcap,name}} 用最新一条"""
    ex = {}
    for code, date, name, price, chg, pe, pb, mcap, turnover, vr, fa, fm, zt, dt in conn.execute(
            "SELECT code,date,name,price,change_pct,pe_ttm,pb,mcap,turnover,vol_ratio,fetched_at,float_mcap,zt_price,dt_price FROM extra_info"):
        ex[code] = {"name": name, "price": price, "pe_ttm": pe, "pb": pb,
                    "mcap": mcap, "turnover": turnover, "vol_ratio": vr, "float_mcap": fm}
    return ex


def load_chip(conn):
    chip = defaultdict(dict)
    for code, date, avg, c70, c90, profit, close in conn.execute(
            "SELECT code,date,chip_avg_cost,conc_70,conc_90,profit_rate,close_price FROM chip"):
        chip[date][code] = {"avg_cost": avg, "conc70": c70, "conc90": c90,
                            "profit": profit, "close": close}
    return chip


def load_fundflow(conn):
    ff = defaultdict(dict)
    cols = {c[1] for c in conn.execute("PRAGMA table_info(fund_flows)")}
    has_retail = "retail_inflow" in cols
    sql = "SELECT code,date,main_net_today,jumbo_net"
    if has_retail:
        sql += ",retail_inflow,retail_outflow"
    sql += " FROM fund_flows"
    for row in conn.execute(sql):
        code, date = row[0], row[1]
        d = {"main_net_today": row[2], "jumbo_net": row[3]}
        if has_retail:
            d["retail_inflow"] = row[4]
            d["retail_outflow"] = row[5]
        ff[date][code] = d
    return ff


def fwd_return(klines, code, date, hold):
    dates = sorted(klines.get(code, {}).keys())
    if date not in dates:
        return None
    i = dates.index(date)
    if i + hold >= len(dates):
        return None
    entry = klines[code][date]
    exit_ = klines[code][dates[i + hold]]
    if entry <= 0:
        return None
    return (exit_ / entry - 1) * 100


def build_daily_rankings(conn, klines_full, extra_info, fundflow_by_date, dates):
    """逐日回放: 用 date 之前(含)的K线算因子, 生成当日排名"""
    # 缓存 fund_flows 为 {code: {date: {main_net_5d...}}} 的滚动结构
    # 简化: compute_all_factors 需要 fund_flows[code] = {main_net_5d, main_net_20d, inflow_rate}
    # 这里用当日 fundflow 的 main_net_today 做近似(历史5d/20d需滚动累加)
    # 为了严谨, 预计算每只股票每个日期的 main_net_5d/20d 滚动累加

    codes = list(klines_full.keys())
    # 预计算资金流滚动累加: main_net_5d = 当日及前4日累加
    all_ff_dates = sorted(fundflow_by_date.keys())
    ff_series = defaultdict(dict)  # code -> {date: main_net_today}
    for d in all_ff_dates:
        for code, v in fundflow_by_date[d].items():
            ff_series[code][d] = v.get("main_net_today") or 0

    # 每日排名结果
    daily_rank = {}  # date -> {code: {isir_rank, doubao_rank}}
    print(f"  回放 {len(dates)} 个交易日...")
    t0 = time.time()
    for di, date in enumerate(dates):
        if di % 50 == 0:
            print(f"    回放进度 {di}/{len(dates)} 耗时{time.time()-t0:.0f}s")
        # 当日及之前的K线(截断到 date)
        day_klines = {}
        for code in codes:
            bars = [b for b in klines_full[code] if b["date"] <= date]
            if len(bars) >= 60:
                day_klines[code] = bars
        if len(day_klines) < 100:
            daily_rank[date] = {}
            continue

        # 当日 extra_info (用历史反推, 简化用最新extra_info近似 pe/pb, mcap用K线反推)
        day_extra = {}
        for code in day_klines:
            ei = extra_info.get(code, {})
            tp = ei.get("price", 0)
            tm = ei.get("mcap", 0)
            shares = tm / tp if tp > 0 and tm > 0 else 0
            last = day_klines[code][-1]
            hist_mcap = last["close"] * shares if shares > 0 else 0
            day_extra[code] = {
                "name": ei.get("name", code),
                "price": last["close"],
                "pe_ttm": ei.get("pe_ttm", 0) or 0,
                "pb": ei.get("pb", 0) or 0,
                "mcap": hist_mcap,
                "turnover": ei.get("turnover", 0) or 0,
                "vol_ratio": ei.get("vol_ratio", 1) or 1,
            }

        # 当日资金流 (main_net_5d/20d 滚动累加)
        day_fund = {}
        for code in day_klines:
            # 找到 date 及之前最近的资金流日期, 累加5/20日
            series = ff_series.get(code, {})
            # 取 <= date 的资金流日期
            ff_dates = [d for d in series if d <= date]
            ff_dates.sort()
            if not ff_dates:
                day_fund[code] = {"main_net_5d": 0, "main_net_20d": 0, "inflow_rate": 0}
                continue
            last5 = ff_dates[-5:]
            last20 = ff_dates[-20:]
            m5 = sum(series[d] for d in last5)
            m20 = sum(series[d] for d in last20)
            day_fund[code] = {"main_net_5d": m5, "main_net_20d": m20, "inflow_rate": 0}

        try:
            factor_data, _ = compute_all_factors(day_klines, day_extra, day_fund, {}, sector_map.STOCK_SECTOR)
            if len(factor_data) < 100:
                daily_rank[date] = {}
                continue
            rankings = compute_rankings(factor_data)
            day = {}
            for r in rankings:
                day[r["code"]] = {"isir_rank": r["isir_rank"], "doubao_rank": r["doubao_rank"]}
            daily_rank[date] = day
        except Exception as e:
            daily_rank[date] = {}

    return daily_rank


def summarize(returns, label):
    if not returns:
        return f"【{label}】无样本"
    arr = np.array(returns)
    win = np.mean(arr > 0) * 100
    return (f"【{label}】n={len(arr)} | 均收益{arr.mean():+.2f}% | 中位{np.median(arr):+.2f}% "
            f"| 上涨{win:.1f}% | 最佳{arr.max():+.1f} 最差{arr.min():+.1f}")


def run_indicator_backtest(indicator_name, get_value, klines, valid_dates, chip=None, fundflow=None,
                           daily_rank=None, hold=10, higher_better=True):
    """通用指标回测: 按 get_value(code, date) 的值分位分组, 算前向收益"""
    top_ret = []
    bottom_ret = []
    all_ret = []
    # 可选: 过滤器模式(在 ISIR TOP30 候选里分组)
    filter_top_ret = []
    filter_bottom_ret = []

    for date in valid_dates:
        # 收集当日有指标值的股票
        values = []
        for code in klines:
            if date not in klines[code]:
                continue
            v = get_value(code, date)
            if v is not None:
                values.append((code, v))
        if len(values) < 30:
            continue
        values.sort(key=lambda x: x[1], reverse=higher_better)
        n = len(values)
        top_n = max(1, int(n * TOP_PCT))
        bottom_n = max(1, int(n * BOTTOM_PCT))
        top_group = values[:top_n]
        bottom_group = values[-bottom_n:]

        # 全池分组
        for code, _ in top_group:
            r = fwd_return(klines, code, date, hold)
            if r is not None:
                top_ret.append(r)
        for code, _ in bottom_group:
            r = fwd_return(klines, code, date, hold)
            if r is not None:
                bottom_ret.append(r)
        for code, _ in values:
            r = fwd_return(klines, code, date, hold)
            if r is not None:
                all_ret.append(r)

        # 过滤器模式: ISIR TOP30 候选内分组
        if daily_rank and date in daily_rank and daily_rank[date]:
            day_rank = daily_rank[date]
            cand = [c for c, _ in values if day_rank.get(c, {}).get("isir_rank", 9999) <= 30]
            if len(cand) >= 10:
                # 在候选内按指标值分半
                cand_values = [(c, get_value(c, date)) for c in cand]
                cand_values.sort(key=lambda x: x[1], reverse=higher_better)
                half = len(cand_values) // 2
                for c, _ in cand_values[:half]:
                    r = fwd_return(klines, c, date, hold)
                    if r is not None:
                        filter_top_ret.append(r)
                for c, _ in cand_values[half:]:
                    r = fwd_return(klines, c, date, hold)
                    if r is not None:
                        filter_bottom_ret.append(r)

    return {
        "top": top_ret, "bottom": bottom_ret, "all": all_ret,
        "filter_top": filter_top_ret, "filter_bottom": filter_bottom_ret,
    }


def main():
    skip_replay = "--skip-replay" in sys.argv
    conn = sqlite3.connect(DB)
    klines = load_klines(conn)
    klines_full = load_klines_full(conn)
    extra_info = load_extra_info(conn)
    chip = load_chip(conn)
    fundflow = load_fundflow(conn)

    # 交易日序列(以K线为准)
    all_dates = sorted({d for c in klines for d in klines[c]})
    chip_dates = sorted(chip.keys())
    ff_dates = sorted(fundflow.keys())
    print(f"K线: {all_dates[0]}~{all_dates[-1]} {len(all_dates)}日")
    print(f"筹码: {chip_dates[0]}~{chip_dates[-1]} {len(chip_dates)}日")
    print(f"资金流: {ff_dates[0]}~{ff_dates[-1]} {len(ff_dates)}日")

    # 回放评分(生成每日排名)
    rank_cache = os.path.join(OUTPUT, "replay_rank_cache.json")
    if skip_replay and os.path.exists(rank_cache):
        print("  使用缓存的历史排名")
        daily_rank = json.load(open(rank_cache))
    else:
        # 回放窗口: 用K线全历史(需留前向收益空间)
        replay_dates = all_dates[:-20]
        daily_rank = build_daily_rankings(conn, klines_full, extra_info, fundflow, replay_dates)
        # 缓存(rank是int, json可序列化)
        with open(rank_cache, "w") as f:
            json.dump(daily_rank, f)
        print(f"  已缓存历史排名到 {rank_cache}")

    # 回测窗口: 筹码交叉日期
    chip_valid = [d for d in chip_dates if d in all_dates]
    ff_valid = [d for d in ff_dates if d in all_dates]

    print(f"\n筹码可回测日期: {len(chip_valid)}日 ({chip_valid[0]}~{chip_valid[-1]})")
    print(f"资金流可回测日期: {len(ff_valid)}日 ({ff_valid[0]}~{ff_valid[-1]})")

    for hold in HOLDS:
        print(f"\n{'='*72}\n持有 {hold} 日\n{'='*72}")

        # ===== 筹码类指标 =====
        print(f"\n--- 筹码类指标 (窗口{chip_valid[0]}~{chip_valid[-1]}) ---")

        # 1. 筹码集中度 conc90 (越小越集中 => higher_better=False)
        def get_conc90(code, date):
            v = chip.get(date, {}).get(code, {})
            return v.get("conc90") if v.get("conc90") is not None else None
        r = run_indicator_backtest("conc90", get_conc90, klines, chip_valid,
                                   daily_rank=daily_rank, hold=hold, higher_better=False)
        print("  [筹码集中度 conc90(越小越集中)]")
        print("    " + summarize(r["top"], "最集中30%"))
        print("    " + summarize(r["bottom"], "最分散30%"))
        if r["top"] and r["bottom"]:
            print(f"    集中-分散收益差: {np.mean(r['top'])-np.mean(r['bottom']):+.2f}%")
        if r["filter_top"]:
            print("    " + summarize(r["filter_top"], "ISIR候选∩集中"))
            print("    " + summarize(r["filter_bottom"], "ISIR候选∩分散"))
            print(f"    过滤器收益差: {np.mean(r['filter_top'])-np.mean(r['filter_bottom']):+.2f}%")

        # 2. 获利盘比例 profit_rate (中段20-60%最佳, 两端差)
        def get_profit(code, date):
            v = chip.get(date, {}).get(code, {})
            return v.get("profit") if v.get("profit") is not None else None
        r = run_indicator_backtest("profit_rate", get_profit, klines, chip_valid,
                                   daily_rank=daily_rank, hold=hold, higher_better=True)
        print("  [获利盘比例 profit_rate]")
        print("    " + summarize(r["top"], "获利盘最高30%"))
        print("    " + summarize(r["bottom"], "获利盘最低30%"))

        # 3. 成本偏离 close/avg_cost - 1 (现价相对平均成本的偏离)
        def get_cost_dev(code, date):
            v = chip.get(date, {}).get(code, {})
            avg = v.get("avg_cost"); close = v.get("close")
            if avg and close and avg > 0:
                return (close / avg - 1) * 100
            return None
        r = run_indicator_backtest("cost_dev", get_cost_dev, klines, chip_valid,
                                   daily_rank=daily_rank, hold=hold, higher_better=True)
        print("  [成本偏离 close/avg_cost-1]")
        print("    " + summarize(r["top"], "现价远高于成本30%"))
        print("    " + summarize(r["bottom"], "现价低于成本30%"))

        # ===== 资金流入类指标 =====
        print(f"\n--- 资金流入类指标 (窗口{ff_valid[0]}~{ff_valid[-1]}) ---")

        # 1. 当日主力净流入 main_net_today
        def get_ff_today(code, date):
            v = fundflow.get(date, {}).get(code, {})
            return v.get("main_net_today") if v.get("main_net_today") is not None else None
        r = run_indicator_backtest("main_net_today", get_ff_today, klines, ff_valid,
                                   daily_rank=daily_rank, hold=hold, higher_better=True)
        print("  [当日主力净流入 main_net_today]")
        print("    " + summarize(r["top"], "净流入最多30%"))
        print("    " + summarize(r["bottom"], "净流出最多30%"))
        if r["top"] and r["bottom"]:
            print(f"    流入-流出收益差: {np.mean(r['top'])-np.mean(r['bottom']):+.2f}%")

        # 2. 散户流入 retail_inflow (散户热度)
        def get_retail(code, date):
            v = fundflow.get(date, {}).get(code, {})
            return v.get("retail_inflow") if v.get("retail_inflow") is not None else None
        r = run_indicator_backtest("retail_inflow", get_retail, klines, ff_valid,
                                   daily_rank=daily_rank, hold=hold, higher_better=True)
        print("  [散户流入 retail_inflow(散户热度)]")
        print("    " + summarize(r["top"], "散户流入最多30%"))
        print("    " + summarize(r["bottom"], "散户流入最少30%"))

        # 3. 超大单净流入 jumbo_net
        def get_jumbo(code, date):
            v = fundflow.get(date, {}).get(code, {})
            return v.get("jumbo_net") if v.get("jumbo_net") is not None else None
        r = run_indicator_backtest("jumbo_net", get_jumbo, klines, ff_valid,
                                   daily_rank=daily_rank, hold=hold, higher_better=True)
        print("  [超大单净流入 jumbo_net]")
        print("    " + summarize(r["top"], "超大单流入最多30%"))
        print("    " + summarize(r["bottom"], "超大单流出最多30%"))
        if r["top"] and r["bottom"]:
            print(f"    流入-流出收益差: {np.mean(r['top'])-np.mean(r['bottom']):+.2f}%")

    conn.close()


if __name__ == "__main__":
    main()
