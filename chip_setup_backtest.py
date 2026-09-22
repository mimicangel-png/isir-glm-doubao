#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
筹码集中度信号回测: 验证「资金拉升先决条件」指标是否有效。

核心问题:
  A方案(过滤器): 在 ISIR/SS分/超跌 选出的候选里, 筹码集中的 vs 筹码分散的, 前向收益差异?
  B方案(独立信号): 全池按筹码集中度排序, 最集中组的前向收益(基准对照)?

方法论约束(用户反复强调): 事前条件→事后验证, 禁止事后归因。
  信号 = 交易日T的筹码集中度(可事前观察), 前向收益 = T之后5/10/20日涨幅(事后)。

数据:
  - chip表: westock筹码数据(conc_70/conc_90越小越集中, profit_rate获利盘%)
  - klines表: 327天K线, 算前向收益
  - unified_*.json历史存档: 每天各股的 isir_rank/glm_rank/doubao_rank/rebound_valid
"""
import sqlite3, json, os, glob, re
from collections import defaultdict
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(BASE, "output", "stock_cache.db")
OUTPUT = os.path.join(BASE, "output")


def load_rank_history():
    """加载历史评分存档 -> {date: {code: {isir_rank, glm_rank, doubao_rank, rebound_valid}}}"""
    files = sorted(glob.glob(os.path.join(OUTPUT, "unified_2026-*.json")))
    history = {}
    for f in files:
        base = os.path.basename(f)
        # 排除历史/账本/时段后缀文件
        if re.search(r"rank_history|signal_history|trade_ledger|_\d{4}\.json", base):
            continue
        m = re.match(r"unified_(\d{4}-\d{2}-\d{2})\.json", base)
        if not m:
            continue
        date = m.group(1)
        try:
            d = json.load(open(f))
        except Exception:
            continue
        if not isinstance(d, list):
            continue
        day = {}
        for r in d:
            code = r.get("code")
            if not code:
                continue
            day[code] = {
                "isir_rank": r.get("isir_rank", 9999),
                "glm_rank": r.get("glm_rank", 9999),
                "doubao_rank": r.get("doubao_rank", 9999),
                "rebound_valid": bool(r.get("rebound_valid")),
                "rebound_score": r.get("rebound_score") or 0,
            }
        history[date] = day
    return history


def load_chip(conn):
    """chip表 -> {date: {code: {conc70, conc90, profit, avg_cost, close}}}"""
    chip = defaultdict(dict)
    for code, date, avg, c70, c90, profit, close in conn.execute(
            "SELECT code,date,chip_avg_cost,conc_70,conc_90,profit_rate,close_price FROM chip"):
        chip[date][code] = {
            "avg_cost": avg, "conc70": c70, "conc90": c90,
            "profit": profit, "close": close,
        }
    return chip


def load_klines(conn):
    """klines -> {code: {date: close}}"""
    k = defaultdict(dict)
    for code, date, close in conn.execute("SELECT code,date,close FROM klines ORDER BY date"):
        k[code][date] = close
    return k


def fwd_return(klines, code, date, hold):
    """date之后的第hold个交易日相对date的涨幅(%)"""
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


def summarize(returns, label):
    if not returns:
        return f"【{label}】无样本"
    arr = np.array(returns)
    win = np.mean(arr > 0) * 100
    return (f"【{label}】样本{len(arr)} | 均收益{arr.mean():+.2f}% | 中位{np.median(arr):+.2f}% "
            f"| 上涨概率{win:.1f}% | 最佳{arr.max():+.1f}% 最差{arr.min():+.1f}%")


def main():
    conn = sqlite3.connect(DB)
    history = load_rank_history()
    chip = load_chip(conn)
    klines = load_klines(conn)

    rank_dates = sorted(history.keys())
    print(f"评分存档: {len(rank_dates)}天 ({rank_dates[0]}~{rank_dates[-1]})")
    chip_dates = sorted(chip.keys())
    print(f"筹码数据: {len(chip_dates)}天 ({chip_dates[0]}~{chip_dates[-1]})")

    # 交叉日期(评分存档 ∩ 筹码数据)
    valid_dates = [d for d in rank_dates if d in chip]
    print(f"交叉可回测日期: {len(valid_dates)}天")

    # 筹码集中度定义: conc90 越小越集中。用全池当日 cross-section 分位划分
    # 集中 = conc90 处于当日最低 30% (数值小)
    for hold in [5, 10, 20]:
        print(f"\n{'='*70}\n持有 {hold} 日\n{'='*70}")

        # ===== B方案: 纯筹码集中(全池独立信号) =====
        b_top = []   # 筹码最集中(conc90最低30%)
        b_bottom = []  # 筹码最分散(conc90最高30%)
        for d in valid_dates:
            day_chip = chip[d]
            # 有conc90的股票
            stocks = [(c, v["conc90"]) for c, v in day_chip.items() if v.get("conc90") is not None]
            if len(stocks) < 30:
                continue
            stocks.sort(key=lambda x: x[1])  # 数值小=集中
            n = len(stocks)
            top30 = stocks[:int(n*0.3)]
            bottom30 = stocks[-int(n*0.3):]
            for c, _ in top30:
                r = fwd_return(klines, c, d, hold)
                if r is not None:
                    b_top.append(r)
            for c, _ in bottom30:
                r = fwd_return(klines, c, d, hold)
                if r is not None:
                    b_bottom.append(r)
        print("\n[B方案] 纯筹码集中(全池独立信号)")
        print("  " + summarize(b_top, "筹码最集中30%"))
        print("  " + summarize(b_bottom, "筹码最分散30%"))
        print(f"  集中-分散 收益差: {np.mean(b_top)-np.mean(b_bottom):+.2f}%" if b_top and b_bottom else "  无")

        # ===== A方案: 过滤器 (ISIR候选 ∩ 筹码集中) =====
        # ISIR TOP30 候选, 分筹码集中/分散
        for strat, rkey in [("ISIR", "isir_rank"), ("SS分", "doubao_rank")]:
            a_top = []    # 候选里筹码集中
            a_bottom = []  # 候选里筹码分散
            a_all = []    # 候选全部(基准)
            for d in valid_dates:
                day_rank = history.get(d, {})
                day_chip = chip.get(d, {})
                # 候选: rank <= 30
                cand = [c for c, r in day_rank.items() if r.get(rkey, 9999) <= 30]
                # 有筹码数据的候选
                cand_with_chip = [(c, day_chip[c].get("conc90")) for c in cand
                                  if c in day_chip and day_chip[c].get("conc90") is not None]
                if len(cand_with_chip) < 15:
                    continue
                cand_with_chip.sort(key=lambda x: x[1] if x[1] is not None else 9999)
                n = len(cand_with_chip)
                top_half = cand_with_chip[:max(1, n//2)]
                bottom_half = cand_with_chip[max(1, n//2):]
                for c, _ in top_half:
                    r = fwd_return(klines, c, d, hold)
                    if r is not None:
                        a_top.append(r)
                for c, _ in bottom_half:
                    r = fwd_return(klines, c, d, hold)
                    if r is not None:
                        a_bottom.append(r)
                for c, _ in cand_with_chip:
                    r = fwd_return(klines, c, d, hold)
                    if r is not None:
                        a_all.append(r)
            print(f"\n[A方案] {strat} TOP30候选 按筹码集中度分组")
            print("  " + summarize(a_top, "候选∩筹码集中"))
            print("  " + summarize(a_bottom, "候选∩筹码分散"))
            print("  " + summarize(a_all, "候选全部(基准)"))
            if a_top and a_bottom:
                print(f"  集中-分散 收益差: {np.mean(a_top)-np.mean(a_bottom):+.2f}%")

        # ===== 获利盘比例维度 =====
        print(f"\n[C参考] 获利盘比例(profit_rate)分组(全池)")
        c_low = []   # 获利盘低(20-60%, 套牢盘多, 抛压已释放)
        c_high = []  # 获利盘高(>80%, 随时兑现)
        for d in valid_dates:
            day_chip = chip[d]
            stocks = [(c, v["profit"]) for c, v in day_chip.items() if v.get("profit") is not None]
            if len(stocks) < 30:
                continue
            for c, p in stocks:
                r = fwd_return(klines, c, d, hold)
                if r is None:
                    continue
                if 20 <= p <= 60:
                    c_low.append(r)
                elif p > 80:
                    c_high.append(r)
        print("  " + summarize(c_low, "获利盘20-60%(套牢盘多)"))
        print("  " + summarize(c_high, "获利盘>80%(兑现压力大)"))

    conn.close()


if __name__ == "__main__":
    main()
