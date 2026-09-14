#!/usr/bin/env python3
"""
第五视图回测: 质量过滤超跌反弹 (超跌反弹 × 三体系认可度交叉)
==============================================================
假设: 超跌反弹标的若同时获得ISIR/GLM认可(排名前N), 反弹质量更高。

对比组:
  - pure   : 纯超跌反弹(第四体系原样, 不看质量)
  - cross100/150/200/250 : 超跌反弹 ∩ (ISIR或GLM排名 ≤ 阈值)
  - all    : 全池等权基线(每日所有有效因子股)

前向收益: 信号日收盘买入, 持有5/10/20日收盘卖出
分层: 按上证综指MA50多空状态分组(空头日的反弹才是本策略目标场景)

输出: 控制台报告 + output/quality_rebound_backtest.json (供引擎第五视图消费)

方法论约束:
  - 无前视偏差: 每日仅用当日及之前K线计算因子/排名/超跌分
  - 事前固定参数: 阈值组在运行前定义, 不做事后挑优
  - 股票池为当前450只(存在存活偏差, 与引擎自带回测同口径)
"""

import os, sys, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stock_db import StockDB
import sector_map
from unified_scoring_engine import (
    compute_all_factors, compute_rankings, compute_rebound_scores,
    fetch_index_klines,
)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
THRESHOLDS = [100, 150, 200, 250]   # ISIR/GLM最好名次阈值
HOLDS = [5, 10, 20]
REBOND_LOOKBACK = 80                # 超跌反弹评分用最近80根K线(需>=35, 覆盖MA60)


def group_key(r, threshold):
    """cross分组成员判定: 超跌反弹有效 且 ISIR或GLM排名≤阈值"""
    if not r.get("rebound_valid"):
        return False
    best = min(r.get("isir_rank", 9999), r.get("glm_rank", 9999))
    return best <= threshold


def main(backtest_days=180):
    print("=" * 64)
    print("  第五视图回测: 质量过滤超跌反弹 (超跌反弹×三体系认可)")
    print("  pure vs cross100/150/200/250 vs 全池基线")
    print("=" * 64)

    codes = sorted(sector_map.STOCK_SECTOR.keys())
    sectors = {c: sector_map.get_sector(c) for c in codes}
    db = StockDB()
    klines = db.get_klines(codes, days=300)
    extra_info = db.get_extra_info(codes, force_refresh=False)
    print(f"  K线{len(klines)}只 | 行情{len(extra_info)}只")

    # 上证综指(用于多空分层), 失败则降级为不分层
    idx_bars = None
    try:
        idx_bars = fetch_index_klines(days=300)
        print(f"  上证综指: {len(idx_bars)}根K线")
    except Exception as e:
        print(f"  [WARN] 指数获取失败, 跳过多空分层: {e}")

    all_dates = set()
    for k in klines.values():
        for b in k:
            all_dates.add(b["date"])
    dates = sorted(all_dates)
    test_dates = dates[-min(backtest_days, len(dates) - 60):]
    print(f"  回测区间: {test_dates[0]} ~ {test_dates[-1]} ({len(test_dates)}个交易日, 前{len(test_dates)-20}日发信号)")

    # 结果容器
    group_names = ["all", "pure", "pure_s40", "confirm", "cross100", "cross150", "cross150_confirm", "cross150_s40", "cross200", "cross250", "consensus_rb"]
    res = {g: {f"ret{h}": [] for h in HOLDS} for g in group_names}
    res_stage = {}          # (stage, hold) -> [returns]  纯超跌组按阶段细分
    day_counts = {g: 0 for g in group_names}
    signal_days = {g: set() for g in group_names}
    res_bear = {g: {f"ret{h}": [] for h in HOLDS} for g in group_names}
    res_bull = {g: {f"ret{h}": [] for h in HOLDS} for g in group_names}

    def market_state(date):
        """上证收盘 vs MA50: 1多头 -1空头 0未知"""
        if not idx_bars:
            return 0
        closes = [b["close"] for b in idx_bars if b["date"] <= date]
        if len(closes) < 50:
            return 0
        ma50 = sum(closes[-50:]) / 50
        return 1 if closes[-1] >= ma50 else -1

    # 股本缓存(今日市值反推, 同引擎run_backtest口径)
    shares_cache, float_shares_cache = {}, {}
    for code in klines:
        info = extra_info.get(code, {})
        tp, tm, tfm = info.get("price", 0), info.get("mcap", 0), info.get("float_mcap", 0)
        shares_cache[code] = tm / tp if tp > 0 and tm > 0 else 0
        float_shares_cache[code] = tfm / tp if tp > 0 and tfm > 0 else 0

    n_days = len(test_dates) - 20
    skip = 0
    _err_shown = False
    for di, date in enumerate(test_dates[:-20]):
        if di % 20 == 0:
            print(f"  进度: {di}/{n_days}")
        try:
            day_klines = {}
            for code, k in klines.items():
                bars = [b for b in k if b["date"] <= date]
                if len(bars) >= 60:
                    day_klines[code] = bars
            if len(day_klines) < 100:
                skip += 1
                continue

            day_extra = {}
            for code, k_bars in day_klines.items():
                last, prev = k_bars[-1], k_bars[-2]
                avg_vol_5 = sum(b["volume"] for b in k_bars[-6:-1]) / 5 if len(k_bars) >= 6 else last["volume"]
                vr = last["volume"] / avg_vol_5 if avg_vol_5 > 0 else 1.0
                fs = float_shares_cache.get(code, 0)
                day_extra[code] = {
                    "name": extra_info.get(code, {}).get("name", code),
                    "price": last["close"],
                    "change_pct": (last["close"] / prev["close"] - 1) * 100,
                    "pe_ttm": extra_info.get(code, {}).get("pe_ttm", 0) or 0,
                    "pb": extra_info.get(code, {}).get("pb", 0) or 0,
                    "mcap": last["close"] * shares_cache.get(code, 0),
                    "turnover": (last["volume"] * 100 / fs * 100) if fs > 0 else 0,
                    "vol_ratio": vr,
                }

            factor_data, _ = compute_all_factors(day_klines, day_extra, {}, {}, sectors)
            if len(factor_data) < 100:
                skip += 1
                continue
            rankings = compute_rankings(factor_data)
            if not rankings:
                skip += 1
                continue

            # 超跌反弹评分(截断K线提速)
            rb_klines = {c: b[-REBOND_LOOKBACK:] for c, b in day_klines.items()}
            compute_rebound_scores(rankings, rb_klines, day_extra)

            mkt = market_state(date)

            def record(gname, code):
                entry = day_klines[code][-1]["close"]
                fwd = [b for b in klines.get(code, []) if b["date"] > date]
                for h in HOLDS:
                    if len(fwd) >= h:
                        ret = (fwd[h - 1]["close"] / entry - 1) * 100
                        res[gname][f"ret{h}"].append(ret)
                        if mkt == -1:
                            res_bear[gname][f"ret{h}"].append(ret)
                        elif mkt == 1:
                            res_bull[gname][f"ret{h}"].append(ret)
                        day_counts[gname] += 1
                        signal_days[gname].add(date)

            for r in rankings:
                record("all", r["code"])
                if r.get("rebound_valid"):
                    record("pure", r["code"])
                    stage = r.get("rebound_stage", "")
                    entry = day_klines[r["code"]][-1]["close"]
                    fwd = [b for b in klines.get(r["code"], []) if b["date"] > date]
                    for h in HOLDS:
                        if len(fwd) >= h:
                            res_stage.setdefault((stage, f"ret{h}"), []).append(
                                (fwd[h - 1]["close"] / entry - 1) * 100)
                    if r.get("rebound_score", 0) >= 40:
                        record("pure_s40", r["code"])
                        if stage == "确认":
                            pass
                    if stage == "确认":
                        record("confirm", r["code"])
                    if r.get("consensus"):
                        record("consensus_rb", r["code"])
                    for t in THRESHOLDS:
                        if group_key(r, t):
                            record(f"cross{t}", r["code"])
                            if t == 150 and stage == "确认":
                                record("cross150_confirm", r["code"])
                            if t == 150 and r.get("rebound_score", 0) >= 40:
                                record("cross150_s40", r["code"])
        except Exception as e:
            skip += 1
            if not _err_shown:
                import traceback
                print(f"  [WARN] 首个异常({date}): {e}")
                traceback.print_exc()
                _err_shown = True
            continue

    print(f"  完成. 跳过{skip}天")

    # ============ 汇总 ============
    def stats(arr):
        a = np.array(arr) if arr else np.array([0.0])
        return {
            "n": len(arr),
            "win": round(float(np.mean(a > 0)) * 100, 1) if arr else 0,
            "avg": round(float(np.mean(a)), 2) if arr else 0,
            "med": round(float(np.median(a)), 2) if arr else 0,
            "max": round(float(np.max(a)), 1) if arr else 0,
            "min": round(float(np.min(a)), 1) if arr else 0,
        }

    summary = {"period": [test_dates[0], test_dates[-1]], "groups": {}}
    print()
    print("=" * 96)
    print(f"{'组':<12}{'样本':>7}{'5d胜率':>8}{'5d均收益':>9}{'10d胜率':>9}{'10d均收益':>10}{'20d胜率':>9}{'20d均收益':>10}{'20d中位':>9}")
    print("-" * 96)
    for g in group_names:
        s = {f"ret{h}": stats(res[g][f"ret{h}"]) for h in HOLDS}
        summary["groups"][g] = s
        summary["groups"][g]["active_days"] = len(signal_days[g])
        print(f"{g:<12}{s['ret5']['n']:>7}"
              f"{s['ret5']['win']:>7}%{s['ret5']['avg']:>8}%"
              f"{s['ret10']['win']:>8}%{s['ret10']['avg']:>9}%"
              f"{s['ret20']['win']:>8}%{s['ret20']['avg']:>9}%{s['ret20']['med']:>8}%")

    print()
    print("按市场状态分层 (策略目标场景=空头日):")
    print(f"{'组':<12}{'空头20d胜率':>12}{'空头20d均收益':>13}{'多头20d胜率':>12}{'多头20d均收益':>13}")
    print("-" * 76)
    summary["bear"] = {}
    for g in group_names:
        sb = stats(res_bear[g]["ret20"])
        sl = stats(res_bull[g]["ret20"])
        summary["bear"][g] = {"bear20": sb, "bull20": sl}
        print(f"{g:<12}{sb['win']:>11}%{sb['avg']:>12}%{sl['win']:>11}%{sl['avg']:>12}%")

    print()
    print("纯超跌组按阶段细分 (20日收益):")
    print(f"{'阶段':<10}{'样本':>8}{'20d胜率':>9}{'20d均收益':>10}")
    print("-" * 46)
    summary["stages"] = {}
    for stage in ["初现", "确认", "加速"]:
        key = (stage, "ret20")
        s = stats(res_stage.get(key, []))
        summary["stages"][stage] = s
        print(f"{stage:<10}{s['n']:>8}{s['win']:>8}%{s['avg']:>9}%")

    with open(os.path.join(OUTPUT_DIR, "quality_rebound_backtest.json"), "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n  结果已保存: output/quality_rebound_backtest.json")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--days", type=int, default=180)
    args = p.parse_args()
    main(args.days)
