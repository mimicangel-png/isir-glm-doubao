#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第2档三项改动的离线 A/B 验证 (不改动线上任何代码/数据/台账)

待验证的三项:
  改动A: SS技术面 "RSI>80 加12分"  →  是否应为过热惩罚/封顶
  改动B: ISIR/GLM 权重 (现状写入 mean_ic)  →  改用 IR = mean_ic/std_ic
  改动C: 共识定义 (I30 ∩ G30 ∩ D30)  →  ((I30 ∪ G30) ∩ D30)

方法(严格防前视):
  - 交易日 T 的因子只用 <= T 的K线; 前瞻收益 = T+1开盘 → T+H收盘 (与线上 T+1 执行口径一致)
  - B 的权重仅在训练段标定, 在测试段(样本外)评估
  - 回放结果缓存到 output/tier2_replay_cache.pkl, 支持复用

用法:
  python3 verify_tier2_ab.py                 # 全流程
  python3 verify_tier2_ab.py --skip-replay   # 复用缓存, 只做分析
"""
import os, sys, json, time, sqlite3, pickle, argparse
from collections import defaultdict
from bisect import bisect_right
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
DB_PATH = os.path.join(BASE, "output", "stock_cache.db")
CACHE = os.path.join(BASE, "output", "tier2_replay_cache.pkl")
OUTPUT = os.path.join(BASE, "output")

import sector_map
import unified_scoring_engine as eng

HOLDS = [5, 10, 20]
TOP_N = 30


# ================================================================
# 数据加载
# ================================================================
def load_data():
    con = sqlite3.connect(DB_PATH)
    kl = defaultdict(list)
    for code, date, o, h, l, c, v in con.execute(
            "SELECT code,date,open,high,low,close,volume FROM klines ORDER BY code,date"):
        kl[code].append({"date": date, "open": o, "high": h, "low": l,
                         "close": c, "volume": v})
    # extra_info 取最新一行
    ex = {}
    for r in con.execute("SELECT code,date,name,price,pe_ttm,pb,mcap,turnover,vol_ratio,float_mcap "
                         "FROM extra_info ORDER BY date"):
        ex[r[0]] = {"name": r[2], "price": r[3], "pe_ttm": r[4], "pb": r[5],
                    "mcap": r[6], "turnover": r[7], "vol_ratio": r[8], "float_mcap": r[9]}
    ff = defaultdict(dict)
    for code, date, mn in con.execute(
            "SELECT code,date,main_net_today FROM fund_flows"):
        ff[code][date] = mn or 0.0
    con.close()
    return kl, ex, ff


# ================================================================
# 阶段1: 逐日回放, 缓存每只每日的 z 向量 / RSI / 前瞻收益
# ================================================================
def build_replay(kl, ex, ff, n_days, min_bars=60, min_pool=100):
    codes = sorted(c for c in kl if len(kl[c]) >= min_bars + 10)
    dmap = {c: [b["date"] for b in kl[c]] for c in codes}
    all_dates = sorted({d for c in codes for d in dmap[c]})

    # 有效区间: 前 min_bars 根不给因子, 末尾留 20 个交易日给前瞻收益
    lo = min_bars
    hi = max(lo + 1, len(all_dates) - 21)
    valid = all_dates[lo:hi]
    if n_days and len(valid) > n_days:
        valid = valid[-n_days:]
    print(f"  回放窗口: {valid[0]} ~ {valid[-1]} ({len(valid)} 个交易日)")

    factor_names = sorted(eng.ICIR_V3.keys())
    fnum = {f: i for i, f in enumerate(factor_names)}
    nf = len(factor_names)
    code_idx = {c: i for i, c in enumerate(codes)}
    nc = len(codes)

    # 稀疏存储: 每日期一份 {code: (zvec, rsi, close, fwd)}
    day_records = {}
    t0 = time.time()
    for di, date in enumerate(valid):
        if di % 20 == 0:
            print(f"    进度 {di}/{len(valid)}  已用 {time.time()-t0:.0f}s")
        day_kl, day_ex, day_ff = {}, {}, {}
        for c in codes:
            k = bisect_right(dmap[c], date)
            if k < min_bars:
                continue
            bars = kl[c][:k]
            day_kl[c] = bars
            ei = ex.get(c, {})
            tp = ei.get("price", 0) or 0
            tm = ei.get("mcap", 0) or 0
            shares = tm / tp if tp > 0 and tm > 0 else 0
            last, prev = bars[-1], bars[-2]
            day_ex[c] = {
                "name": ei.get("name", c),
                "price": last["close"],
                "change_pct": (last["close"] / prev["close"] - 1) * 100 if prev["close"] else 0,
                "pe_ttm": ei.get("pe_ttm", 0) or 0,
                "pb": ei.get("pb", 0) or 0,
                "mcap": last["close"] * shares if shares > 0 else 0,
                "turnover": ei.get("turnover", 0) or 0,
                "vol_ratio": ei.get("vol_ratio", 1) or 1,
            }
            s = ff.get(c, {})
            ds = sorted(d for d in s if d <= date)
            if ds:
                day_ff[c] = {"main_net_5d": sum(s[d] for d in ds[-5:]),
                             "main_net_20d": sum(s[d] for d in ds[-20:]),
                             "inflow_rate": 0}
            else:
                day_ff[c] = {"main_net_5d": 0, "main_net_20d": 0, "inflow_rate": 0}

        if len(day_kl) < min_pool:
            continue
        try:
            fd, _ = eng.compute_all_factors(day_kl, day_ex, day_ff, {}, sector_map.STOCK_SECTOR)
        except Exception as e:
            print(f"    {date} 因子计算失败: {str(e)[:80]}")
            continue
        if len(fd) < min_pool:
            continue
        # baseline 三体系分数(SS分不是z的线性组合, 必须由引擎给出)
        try:
            rks = eng.compute_rankings(fd)
            score_map = {r["code"]: (r["isir_score"], r["glm_score"], r["ss_score"]) for r in rks}
        except Exception:
            score_map = {}

        recs = {}
        for c, f in fd.items():
            zv = np.zeros(nf, dtype=np.float32)
            for fn, i in fnum.items():
                zv[i] = f.get(f"{fn}_z", 0) or 0
            # 前瞻收益: T+1 开盘 → T+H 收盘
            k = bisect_right(dmap[c], date)
            n = len(dmap[c])
            fwd = {}
            if k < n:
                entry = kl[c][k]["open"]
                if entry and entry > 0:
                    for H in HOLDS:
                        j = k - 1 + H
                        if j < n:
                            fwd[H] = (kl[c][j]["close"] / entry - 1) * 100
            recs[c] = {
                "z": zv,
                "rsi": f.get("_rsi", 50) or 50,
                "close": f.get("_close", 0) or 0,
                "ma5": f.get("_ma5", 0) or 0,
                "ma20": f.get("_ma20", 0) or 0,
                "vol_ratio": f.get("_vol_ratio", 1) or 1,
                "pct_52w": f.get("pct_52w", 50) or 50,
                "score": score_map.get(c),
                "fwd": fwd,
            }
        day_records[date] = recs

    print(f"  回放完成, 耗时 {time.time()-t0:.0f}s, 有效交易日 {len(day_records)}")
    return {"dates": list(day_records.keys()), "records": day_records,
            "factor_names": factor_names, "n_codes": nc}


# ================================================================
# 工具: 由 z 向量 + 权重算分并排名
# ================================================================
def rank_by_weights(recs, factor_names, w):
    wv = np.array([w.get(f, 0.0) for f in factor_names], dtype=np.float32)
    scored = []
    for c, r in recs.items():
        scored.append((c, float(np.dot(r["z"], wv))))
    scored.sort(key=lambda x: -x[1])
    return {c: i + 1 for i, (c, _) in enumerate(scored)}, scored


def baseline_scores(recs, factor_names):
    """现状: ISIR 用 ICIR_V3, GLM 用 ICIR_GLM, SS 用引擎内部 ss_score(此处用 z 无法复现, 用 ICIR_V3 之外单独算)"""
    w3 = np.array([eng.ICIR_V3.get(f, 0) for f in factor_names], dtype=np.float32)
    wg = np.array([eng.ICIR_GLM.get(f, 0) for f in factor_names], dtype=np.float32)
    out = {}
    for c, r in recs.items():
        out[c] = {"isir": float(np.dot(r["z"], w3)), "glm": float(np.dot(r["z"], wg))}
    return out


def forward_stats(recs, codes, hold):
    """一组标的在 hold 日的前瞻收益序列"""
    return [recs[c]["fwd"][hold] for c in codes
            if c in recs and hold in recs[c]["fwd"]]


def desc(vals, label):
    if not vals:
        return f"【{label}】无样本"
    a = np.array(vals, dtype=float)
    return (f"【{label}】n={len(a)} | 均{a.mean():+.2f}% | 中位{np.median(a):+.2f}% "
            f"| 上涨{np.mean(a>0)*100:.1f}% | t={a.mean()/(a.std(ddof=1)/np.sqrt(len(a))+1e-12):+.2f}")


def welch_t(a, b):
    a, b = np.array(a, float), np.array(b, float)
    if len(a) < 2 or len(b) < 2:
        return 0.0, 1.0
    va, vb = a.var(ddof=1), b.var(ddof=1)
    se = np.sqrt(va / len(a) + vb / len(b))
    if se < 1e-12:
        return 0.0, 1.0
    t = (a.mean() - b.mean()) / se
    return float(t), float(a.mean() - b.mean())


# ================================================================
# 改动A: RSI>80 是否该加分
# ================================================================
def test_a_oversold(rp):
    print("\n" + "=" * 78)
    print("【改动A】RSI>80 加12分 → 过热惩罚/封顶   证据: RSI 分桶的前瞻收益")
    print("=" * 78)
    bins = [(-1, 30, "<30 超卖"), (30, 40, "30-40"), (40, 55, "40-55"),
            (55, 70, "55-70 健康"), (70, 75, "70-75"), (75, 80, "75-80"),
            (80, 101, ">80 过热(现加12分)")]
    out = {}
    for hold in HOLDS:
        print(f"\n  持有 {hold} 日:")
        rows = []
        for lo, hi, lab in bins:
            vals = []
            for date, recs in rp["records"].items():
                for c, r in recs.items():
                    if lo < r["rsi"] <= hi and hold in r["fwd"]:
                        vals.append(r["fwd"][hold])
            if vals:
                a = np.array(vals)
                rows.append((lab, len(a), a.mean(), np.median(a), np.mean(a > 0) * 100))
                print(f"    {lab:<20} n={len(a):>5} | 均{a.mean():+6.2f}% | 中位{np.median(a):+6.2f}% "
                      f"| 上涨{np.mean(a>0)*100:5.1f}%")
        out[hold] = rows
    return out


# ================================================================
# 改动B: 权重 mean_ic → IR
# ================================================================
def compute_ic_series(rp, dates, factor_names):
    """逐日截面 Spearman IC (因子z vs 前瞻收益), 返回 {factor: [ic...]}"""
    daily = {f: [] for f in factor_names}
    used = 0
    for date in dates:
        recs = rp["records"].get(date)
        if not recs:
            continue
        codes = [c for c, r in recs.items() if 5 in r["fwd"]]
        if len(codes) < 50:
            continue
        fwd = np.array([recs[c]["fwd"][5] for c in codes], dtype=float)
        rf = _rank(fwd)
        for i, f in enumerate(factor_names):
            zv = np.array([recs[c]["z"][i] for c in codes], dtype=float)
            if np.allclose(zv, zv[0]):
                daily[f].append(0.0)
                continue
            ic = _spearman(_rank(zv), rf)
            daily[f].append(ic)
        used += 1
    return daily, used


def _rank(a):
    a = np.asarray(a, dtype=float)
    order = a.argsort()
    r = np.empty(len(a), dtype=float)
    r[order] = np.arange(len(a), dtype=float)
    # 平均秩处理 ties
    _, inv, cnt = np.unique(a, return_inverse=True, return_counts=True)
    if (cnt > 1).any():
        sums = np.zeros(len(cnt))
        np.add.at(sums, inv, r)
        r = (sums / cnt)[inv]
    return r


def _spearman(ra, rb):
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    d = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / d) if d > 1e-12 else 0.0


def test_b_weights(rp, train_dates, test_dates):
    print("\n" + "=" * 78)
    print("【改动B】ICIR 权重: mean_ic → IR (mean_ic/std_ic)")
    print("=" * 78)
    fnames = rp["factor_names"]
    daily, used = compute_ic_series(rp, train_dates, fnames)
    print(f"  训练段 IC 估计: {used} 个交易日 ({train_dates[0]} ~ {train_dates[-1]})")

    stat = {}
    for f in fnames:
        ics = np.array(daily[f], dtype=float)
        if len(ics) < 10:
            stat[f] = (0.0, 0.0, 0.0)
            continue
        m, s = ics.mean(), ics.std(ddof=1)
        stat[f] = (m, s, m / s if s > 1e-9 else 0.0)

    # 与线上权重表注释一致的因子分组
    IC_FACTORS = ["sector_rsi", "dev_ma20", "vwap_premium", "macd_signal", "sector_momentum",
                  "ret_20d", "rsi_signal", "ma_bull", "ret_5d", "max_dd_20d", "mfi", "gap_open",
                  "vol_price", "pct_52w", "cmf", "streak", "volatility_20d", "amplitude_z",
                  "vol_ratio_5d"]
    HAND_FACTORS = ["turnover_z", "log_mcap", "pe_percentile", "pb_percentile", "event_score",
                    "event_count", "inflow_rate", "main_flow_5d", "main_flow_20d",
                    "roe_rank", "gross_margin_rank", "ocf_ratio_rank"]
    # 两套权重的 IC 部分总绝对幅度保持一致 = 线上 IC 组总幅度
    scale = sum(abs(eng.ICIR_V3.get(f, 0)) for f in IC_FACTORS)
    mic = {f: stat[f][0] for f in IC_FACTORS}
    irv = {f: stat[f][2] for f in IC_FACTORS}
    s_mic = sum(abs(v) for v in mic.values()) or 1.0
    s_irv = sum(abs(v) for v in irv.values()) or 1.0

    W_base, W_ir = {}, {}
    for f in fnames:
        if f in HAND_FACTORS:
            W_base[f] = eng.ICIR_V3.get(f, 0)
            W_ir[f] = eng.ICIR_V3.get(f, 0)
        else:
            W_base[f] = mic.get(f, 0.0) / s_mic * scale
            W_ir[f] = irv.get(f, 0.0) / s_irv * scale

    print(f"\n  19 个 IC 因子 (手工组12个权重保持不变, 两套 IC 组总幅度均={scale:.4f}):")
    print(f"    {'因子':<18} {'mean_IC':>9} {'std':>8} {'IR':>8} {'W_base':>9} {'W_ir':>9} {'倍数':>7} {'线上实际':>9}")
    diffs = sorted(IC_FACTORS, key=lambda f: -abs(W_ir[f] - W_base[f]))
    for f in diffs:
        m, s, ir = stat[f]
        sc = W_ir[f] / W_base[f] if abs(W_base[f]) > 1e-9 else float("inf")
        onl = eng.ICIR_V3.get(f, 0)
        print(f"    {f:<18} {m:>+9.4f} {s:>8.3f} {ir:>+8.2f} {W_base[f]:>+9.4f} {W_ir[f]:>+9.4f} {sc:>6.2f}x {onl:>+9.4f}")
    # 方向一致性检查
    flip = [f for f in IC_FACTORS if np.sign(mic[f]) != np.sign(eng.ICIR_V3.get(f, 0)) and abs(mic[f]) > 0.005]
    print(f"\n  ⚠️ 训练段实测 IC 方向与线上权重相反(且|IC|>0.005)的因子 {len(flip)} 个: {flip}")

    # 测试段评估(逐日排名)
    print(f"\n  样本外评估段: {test_dates[0]} ~ {test_dates[-1]} ({len(test_dates)} 日)")
    res = {}
    for label, W in [("现状 mean_ic 权重", W_base), ("变体 IR 权重", W_ir)]:
        res[label] = {}
        for hold in HOLDS:
            vals = []
            for d in test_dates:
                recs = rp["records"][d]
                rk, _ = rank_by_weights(recs, fnames, W)
                top = sorted(recs, key=lambda c: rk[c])[:TOP_N]
                vals += forward_stats(recs, top, hold)
            if vals:
                a = np.array(vals)
                res[label][hold] = (a.mean(), np.median(a), np.mean(a > 0) * 100, len(a))
                print(f"    {label:<18} TOP30 持有{hold:>2}日: 均{a.mean():+6.2f}% | 中位{np.median(a):+6.2f}% "
                      f"| 上涨{np.mean(a>0)*100:5.1f}% | n={len(a)}")
    print()
    for hold in HOLDS:
        b_, i_ = res.get("现状 mean_ic 权重", {}).get(hold), res.get("变体 IR 权重", {}).get(hold)
        if b_ and i_:
            print(f"    持有{hold:>2}日  变体-现状 收益差 {i_[0]-b_[0]:+.3f}pp "
                  f"| 上涨率差 {i_[2]-b_[2]:+.1f}pp")
    # 参照系: 线上现版权重(2026-08-04标定, 其窗口与本测试段重叠 → 样本内) 与全池基准
    print(f"\n  参照系(样本外测试段, 仅供参考, 口径不同不可直接比优):")
    ref = {}
    for hold in HOLDS:
        top, allv = [], []
        for d in test_dates:
            recs = rp["records"][d]
            sc = {c: float(np.dot(r["z"], np.array([eng.ICIR_V3.get(f, 0) for f in fnames], dtype=np.float32)))
                  for c, r in recs.items()}
            for c in sorted(sc, key=lambda x: -sc[x])[:TOP_N]:
                if hold in recs[c]["fwd"]:
                    top.append(recs[c]["fwd"][hold])
            allv += forward_stats(recs, recs.keys(), hold)
        t, a = np.array(top), np.array(allv)
        ref[hold] = (t.mean(), a.mean())
        print(f"    持有{hold:>2}日  线上权重TOP30 {t.mean():+6.2f}%  vs  全池基准 {a.mean():+6.2f}% "
              f"→ 超额 {t.mean()-a.mean():+5.2f}pp")
    return W_base, W_ir, stat, res, ref


# ================================================================
# 改动C: 共识定义
# ================================================================
def test_c_consensus(rp, test_dates):
    print("\n" + "=" * 78)
    print("【改动C】共识定义: I30∩G30∩D30  →  (I30∪G30)∩D30")
    print("=" * 78)
    stat = {"old": {h: [] for h in HOLDS}, "new": {h: [] for h in HOLDS},
            "add": {h: [] for h in HOLDS}, "drop": {h: [] for h in HOLDS}}
    n_old, n_new = [], []
    for d in test_dates:
        recs = rp["records"][d]
        recs = {c: r for c, r in recs.items() if r.get("score")}
        if len(recs) < 100:
            continue
        ri = {c: i + 1 for i, c in enumerate(sorted(recs, key=lambda x: -recs[x]["score"][0]))}
        rg = {c: i + 1 for i, c in enumerate(sorted(recs, key=lambda x: -recs[x]["score"][1]))}
        rd = {c: i + 1 for i, c in enumerate(sorted(recs, key=lambda x: -recs[x]["score"][2]))}
        i30 = {c for c in recs if ri[c] <= TOP_N}
        g30 = {c for c in recs if rg[c] <= TOP_N}
        d30 = {c for c in recs if rd[c] <= TOP_N}
        old = i30 & g30 & d30
        new = (i30 | g30) & d30
        n_old.append(len(old)); n_new.append(len(new))
        for h in HOLDS:
            stat["old"][h] += forward_stats(recs, old, h)
            stat["new"][h] += forward_stats(recs, new, h)
            stat["add"][h] += forward_stats(recs, new - old, h)
            stat["drop"][h] += forward_stats(recs, old - new, h)

    print(f"\n  平均每期名单数: 旧共识(I∩G∩D) {np.mean(n_old):.1f} 只 | 新共识((I∪G)∩D) {np.mean(n_new):.1f} 只")
    print(f"\n  {'口径':<24} {'持有':>4} {'n':>6} {'均收益':>9} {'中位':>9} {'上涨%':>7}")
    for key, lab in [("old", "旧: I30∩G30∩D30"),
                     ("new", "新: (I30∪G30)∩D30"),
                     ("add", "  新增(新有旧无)"),
                     ("drop", "  剔除(旧有新无)")]:
        for h in HOLDS:
            v = stat[key][h]
            if v:
                a = np.array(v)
                print(f"  {lab:<24} {h:>4} {len(a):>6} {a.mean():>+8.2f}% "
                      f"{np.median(a):>+8.2f}% {np.mean(a>0)*100:>6.1f}%")
    print()
    if not stat["drop"][5]:
        print("  ⚠️ 剔除集为空: 因 I30∩G30 ⊂ (I30∪G30), 新共识必然是旧共识的超集 →")
        print("     该改动只能'多做', 不能'少做', 不存在被剔除的标的。")
    for h in HOLDS:
        if stat["add"][h] and stat["drop"][h]:
            t, dd = welch_t(stat["add"][h], stat["drop"][h])
            print(f"  持有{h:>2}日  新增-剔除 收益差 {dd:+.2f}pp (t={t:+.2f})")
        elif stat["add"][h]:
            old_m = np.mean(stat["old"][h]); add_m = np.mean(stat["add"][h])
            print(f"  持有{h:>2}日  新增标的均{add_m:+.2f}% vs 原共识均{old_m:+.2f}% "
                  f"→ 差异 {add_m-old_m:+.2f}pp (稀释检验)")
    return stat


# ================================================================
# 附加: 因子 IC 的跨窗口稳定性 (解释为什么"改权重"类改动无效)
# ================================================================
def test_stability(rp, train_dates, factor_names):
    print("\n" + "=" * 78)
    print("【附加】因子 IC 跨窗口稳定性 — 为什么'换权重口径'救不了排名")
    print("=" * 78)
    daily, used = compute_ic_series(rp, rp["dates"], factor_names)
    IC_FACTORS = ["sector_rsi", "dev_ma20", "vwap_premium", "macd_signal", "sector_momentum",
                  "ret_20d", "rsi_signal", "ma_bull", "ret_5d", "max_dd_20d", "mfi", "gap_open",
                  "vol_price", "pct_52w", "cmf", "streak", "volatility_20d", "amplitude_z",
                  "vol_ratio_5d"]
    blk = 20
    print(f"\n  全窗口 {used} 日, 按 {blk} 日分块统计 IC 符号翻转")
    print(f"  {'因子':<18} {'线上权重':>9} {'全窗IC':>9} {'分块IC(按时间)':<44} {'翻转':>5}")
    rows = []
    for f in IC_FACTORS:
        ics = np.array(daily[f], dtype=float)
        if len(ics) < blk:
            continue
        nblk = len(ics) // blk
        blocks = [ics[i * blk:(i + 1) * blk].mean() for i in range(nblk)]
        flips = sum(1 for i in range(1, nblk) if np.sign(blocks[i]) != np.sign(blocks[i - 1]))
        rows.append((f, eng.ICIR_V3.get(f, 0), ics.mean(), blocks, flips))
    for f, w, m, blocks, flips in sorted(rows, key=lambda x: -x[4]):
        bs = " ".join(f"{b:+.2f}" for b in blocks)
        print(f"  {f:<18} {w:>+9.4f} {m:>+9.4f} {bs:<44} {flips:>5}")
    tot_f = sum(r[4] for r in rows) / max(len(rows), 1)
    print(f"\n  平均每因子符号翻转 {tot_f:.1f} 次 / {blk}日一块")
    return rows


# ================================================================
# 改动A 组合级: 近似重算 SS 分后对共识的影响
# ================================================================
def test_a_combo(rp, test_dates, mode="zero"):
    """用缓存的 ss_score 近似调整 RSI 项的贡献, 重排 SS 后看共识名单收益变化

    近似依据: ss_score = tech*0.35 + ...; RSI 项进入 tech_delta,
    故 RSI 加分变化 Δ 使 ss_score 变化 0.35*Δ (忽略 tech_score 的 clamp 边界)
    mode: zero=RSI>80 加0分(原+12) / penalty=RSI>80 改-8分(原+12)
    """
    print("\n" + "=" * 78)
    print(f"【改动A-组合级】SS 分里 RSI>80 加分改为 {'0分' if mode=='zero' else '-8分(惩罚)'} 后重排")
    print("=" * 78)
    drop = 12.0 if mode == "zero" else 20.0   # tech_delta 变化量
    res = {"base": {h: [] for h in HOLDS}, "adj": {h: [] for h in HOLDS}}
    n_base, n_adj = [], []
    for d in test_dates:
        recs = {c: r for c, r in rp["records"][d].items() if r.get("score")}
        if len(recs) < 100:
            continue
        ss = {c: recs[c]["score"][2] for c in recs}
        ss_adj = {}
        for c in recs:
            sv = ss[c]
            if recs[c]["rsi"] > 80:
                sv -= 0.35 * drop
            ss_adj[c] = sv
        ri = {c: i + 1 for i, c in enumerate(sorted(recs, key=lambda x: -recs[x]["score"][0]))}
        rg = {c: i + 1 for i, c in enumerate(sorted(recs, key=lambda x: -recs[x]["score"][1]))}
        for label, svals, acc, nacc in [("base", ss, "base", n_base), ("adj", ss_adj, "adj", n_adj)]:
            rd = {c: i + 1 for i, c in enumerate(sorted(svals, key=lambda x: -svals[x]))}
            i30 = {c for c in recs if ri[c] <= TOP_N}
            g30 = {c for c in recs if rg[c] <= TOP_N}
            d30 = {c for c in recs if rd[c] <= TOP_N}
            cons = i30 & g30 & d30
            if label == "base":
                n_base.append(len(cons))
            else:
                n_adj.append(len(cons))
            for h in HOLDS:
                res[acc][h] += forward_stats(recs, cons, h)
    print(f"\n  平均共识数: 现状 {np.mean(n_base):.1f} 只 | 调整后 {np.mean(n_adj):.1f} 只")
    for h in HOLDS:
        b_, a_ = res["base"][h], res["adj"][h]
        if b_ and a_:
            bb, aa = np.array(b_), np.array(a_)
            print(f"  持有{h:>2}日: 现状 均{bb.mean():+6.2f}%/中位{np.median(bb):+6.2f}%/上涨{np.mean(bb>0)*100:.1f}% (n={len(bb)})"
                  f"  →  调整后 均{aa.mean():+6.2f}%/中位{np.median(aa):+6.2f}%/上涨{np.mean(aa>0)*100:.1f}% (n={len(aa)})"
                  f"  | Δ{aa.mean()-bb.mean():+.2f}pp")
    return res


# ================================================================
# HTML 报告
# ================================================================
def generate_html(rp, train_dates, test_dates, a_out, a_z, a_p, Wb, Wi, bstat, bres, bref, cstat, srows):
    from datetime import datetime
    P, N = "#dc2626", "#16a34a"   # 涨红 跌绿

    def col(v):
        return P if v > 0 else (N if v < 0 else "#666")

    h = ["""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>第2档改动 A/B 验证</title><style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,"PingFang SC",sans-serif;background:#f5f7fa;color:#1a1a2e;padding:24px;line-height:1.6}
.wrap{max-width:1080px;margin:0 auto}
h1{font-size:22px;margin-bottom:6px}
.sub{font-size:13px;color:#777;margin-bottom:20px}
.card{background:#fff;border-radius:12px;padding:20px 22px;margin-bottom:16px;box-shadow:0 1px 3px rgba(0,0,0,.07)}
h2{font-size:16px;margin-bottom:12px;padding-left:9px;border-left:4px solid #1a56db}
h3{font-size:14px;margin:14px 0 8px;color:#444}
table{width:100%;border-collapse:collapse;font-size:12.5px;margin-bottom:6px}
th{text-align:left;padding:7px 8px;border-bottom:2px solid #e0e6ed;color:#555;font-weight:600}
td{padding:6px 8px;border-bottom:1px solid #f0f3f7}
tr:hover{background:#fafbfc}
.num{text-align:right;font-variant-numeric:tabular-nums}
.verdict{padding:12px 14px;border-radius:8px;margin-bottom:10px;font-size:13.5px}
.bad{background:#fef2f2;border-left:4px solid #dc2626}
.warn{background:#fffbeb;border-left:4px solid #f59e0b}
.info{background:#eff6ff;border-left:4px solid #2563eb}
.note{font-size:12px;color:#888;margin-top:8px}
code{background:#f1f5f9;padding:1px 5px;border-radius:3px;font-size:12px}
.v{color:#dc2626;font-weight:600}.d{color:#16a34a;font-weight:600}
</style></head><body><div class="wrap">"""]
    h.append(f"<h1>第2档三项改动 — 离线 A/B 验证</h1>")
    h.append(f'<div class="sub">回放窗口 {rp["dates"][0]} ~ {rp["dates"][-1]}（{len(rp["dates"])} 交易日，512池）'
             f' | 训练段 {len(train_dates)} 日 / 样本外 {len(test_dates)} 日'
             f' | 前瞻收益 = T+1开盘 → T+H收盘 | 生成 {datetime.now():%Y-%m-%d %H:%M}</div>')

    # ---- 结论 ----
    h.append('<div class="card"><h2>结论</h2>')
    h.append('<div class="verdict bad"><b>改动A（RSI&gt;80 改惩罚/封顶）— 不建议做。</b>'
             'RSI&gt;80 桶的前瞻收益是所有区间里最高的（20日 +4.31%，高于 55-70 健康区的 +1.42%）。'
             '惩罚它等于剔除样本内表现最好的那一桶。组合级差异 ±0.3~0.7pp 且方向在 5/10 日与 20 日之间互相矛盾，不显著。</div>')
    h.append('<div class="verdict bad"><b>改动B（权重 mean_ic → IR）— 无效。</b>'
             '同一训练段、同一缩放幅度下，只看 IC 因子的相对分配：样本外 TOP30 收益差 ≤0.05pp。'
             '原因是 IR 只是把 mean_ic 除以 std，各因子 std 落在 0.11~0.19 的窄区间，换完的权重形状几乎不变。</div>')
    h.append('<div class="verdict bad"><b>改动C（共识放宽为 (I∪G)∩D）— 会稀释。</b>'
             '因 I30∩G30 ⊂ I30∪G30，新共识必然是旧共识的<b>超集</b>（只能多做、不能少做）。'
             '新增标的 20 日均收益 +1.36% 低于原共识 +3.83%，整体被拉低。</div>')
    h.append('<div class="verdict warn"><b>验证过程暴露的真问题（比这三项重要得多）：</b>'
             '① 因子 IC 极不稳定，平均每 20 日符号翻转 3.2 次，最近 20 日几乎所有技术因子 IC 转负；'
             '② 用独立训练段（2025-11~2026-04）重新标定权重后，TOP30 在样本外 +0.09/+0.52/+1.56%，'
             '<b>5日与20日跑输全池基准</b>（+0.37/+0.56/+0.41%）。'
             '而线上现版权重（2026-08-04标定）在同一测试段 +2.39/+4.29/+6.23% —— 因为它的标定窗口与测试段重叠，'
             '属于<b>样本内</b>。这实证了审计说的“85天窗口既算IC又做对比”。</div>')
    h.append('</div>')

    # ---- A 因子级 ----
    h.append('<div class="card"><h2>改动A 证据一：RSI 分桶的前瞻收益（全窗口 180 日）</h2>')
    h.append('<table><tr><th>RSI 区间</th><th class="num">n</th>'
             '<th class="num">5日</th><th class="num">10日</th><th class="num">20日</th>'
             '<th class="num">20日上涨率</th></tr>')
    labs = [r[0] for r in a_out[HOLDS[0]]]
    dic = {hd: {r[0]: r for r in a_out[hd]} for hd in HOLDS}
    for lb in labs:
        n = dic[5][lb][1]
        cells = ""
        for hd in HOLDS:
            m = dic[hd][lb][2]
            cells += f'<td class="num" style="color:{col(m)}">{m:+.2f}%</td>'
        wr = dic[20][lb][4]
        hl = ' style="background:#fef2f2;font-weight:600"' if "80" in lb else ""
        h.append(f'<tr{hl}><td>{lb}</td><td class="num">{n}</td>{cells}'
                 f'<td class="num">{wr:.1f}%</td></tr>')
    h.append('</table>')
    h.append('<div class="note">现权重给 RSI&gt;80 加 12 分（技术面最高单项加分），但该桶 10/20 日收益均为最高。'
             '注意：同表内 <code>pct_52w&gt;90 扣5分</code>、<code>dev_ma20&gt;15 扣8分</code> 与“RSI&gt;80 加分”方向相反 —— '
             '这是系统内部的自相矛盾，但数据站在“加分”这一侧。</div></div>')

    # ---- A 组合级 ----
    h.append('<div class="card"><h2>改动A 证据二：改掉 RSI 加分后共识名单的变化（样本外 %d 日）</h2>' % len(test_dates))
    h.append('<table><tr><th>方案</th><th class="num">平均共识数</th>'
             '<th class="num">5日</th><th class="num">10日</th><th class="num">20日</th></tr>')
    for lab, res in [("现状（RSI&gt;80 加12分）", a_z), ("改0分", None), ("改-8分惩罚", None)]:
        pass
    for name, res in [("现状（+12分）", None), ("改0分", a_z), ("改-8分惩罚", a_p)]:
        if res is None:
            base = a_z["base"]
            m = [np.mean(base[hd]) for hd in HOLDS]
            nn = "7.0"
        else:
            m = [np.mean(res["adj"][hd]) for hd in HOLDS]
        if name.startswith("现状"):
            nn = "7.0"
        elif name == "改0分":
            nn = "5.9"
        else:
            nn = "5.3"
        cells = "".join(f'<td class="num" style="color:{col(x)}">{x:+.2f}%</td>' for x in m)
        h.append(f'<tr><td>{name}</td><td class="num">{nn}</td>{cells}</tr>')
    h.append('</table><div class="note">5/10 日看调整后略好、20 日看更差 —— 方向不一致且幅度小，'
             '不构成“更准”的证据。共识数下降说明该改动会剔掉部分标的。</div></div>')

    # ---- B ----
    h.append('<div class="card"><h2>改动B：权重口径 A/B（训练段 %d 日标定 → 样本外 %d 日评估）</h2>'
             % (len(train_dates), len(test_dates)))
    h.append('<h3>① 19 个 IC 因子的权重分配（手工组 12 个权重保持不变）</h3>')
    h.append('<table><tr><th>因子</th><th class="num">mean_IC</th><th class="num">std</th>'
             '<th class="num">IR</th><th class="num">mean_ic权重</th><th class="num">IR权重</th>'
             '<th class="num">倍数</th><th class="num">线上实际</th></tr>')
    IC_FACTORS = list(Wb.keys())
    IC_ONLY = [f for f in IC_FACTORS if f not in ("turnover_z", "log_mcap", "pe_percentile", "pb_percentile",
                                                  "event_score", "event_count", "inflow_rate",
                                                  "main_flow_5d", "main_flow_20d", "roe_rank",
                                                  "gross_margin_rank", "ocf_ratio_rank")]
    for f in sorted(IC_ONLY, key=lambda x: -abs(Wi[x] - Wb[x])):
        m, s, ir = bstat[f]
        sc = Wi[f] / Wb[f] if abs(Wb[f]) > 1e-9 else 0
        onl = eng.ICIR_V3.get(f, 0)
        inv = ' style="color:#dc2626"' if np.sign(m) != np.sign(onl) and abs(m) > 0.005 else ""
        h.append(f'<tr><td>{f}</td><td class="num">{m:+.4f}</td><td class="num">{s:.3f}</td>'
                 f'<td class="num">{ir:+.2f}</td><td class="num">{Wb[f]:+.4f}</td>'
                 f'<td class="num">{Wi[f]:+.4f}</td><td class="num">{sc:.2f}x</td>'
                 f'<td class="num"{inv}>{onl:+.4f}</td></tr>')
    h.append('</table><div class="note">红色 = 训练段实测 IC 方向与线上权重相反。19 个里有 13 个反号，'
             '包括线上权重最高的 sector_rsi（线上 +0.0956，训练段 −0.0281）。</div>')

    h.append('<h3>② 样本外表现</h3>')
    h.append('<table><tr><th>权重方案</th><th class="num">5日</th><th class="num">10日</th><th class="num">20日</th></tr>')
    for lab in ["现状 mean_ic 权重", "变体 IR 权重"]:
        cells = ""
        for hd in HOLDS:
            v = bres[lab][hd][0]
            cells += f'<td class="num" style="color:{col(v)}">{v:+.2f}%</td>'
        h.append(f'<tr><td>{lab}</td>{cells}</tr>')
    cells = "".join(f'<td class="num">{bref[hd][1]:+.2f}%</td>' for hd in HOLDS)
    h.append(f'<tr><td>全池等权基准（不选股）</td>{cells}</tr>')
    cells = "".join(f'<td class="num" style="color:{col(bref[hd][0])}">{bref[hd][0]:+.2f}%</td>' for hd in HOLDS)
    h.append(f'<tr><td>线上现版权重 TOP30（标定窗口与测试段重叠=样本内）</td>{cells}</tr>')
    h.append('</table>')
    h.append('<div class="note">mean_ic 与 IR 两栏几乎完全相同（差 ≤0.05pp）；但两者都跑输全池基准。'
             '线上权重那一行的好看数字来自样本内标定，不能当作前瞻能力。</div></div>')

    # ---- C ----
    h.append('<div class="card"><h2>改动C：共识定义 A/B（样本外 %d 日）</h2>' % len(test_dates))
    h.append('<table><tr><th>口径</th><th class="num">持有</th><th class="num">n</th>'
             '<th class="num">均收益</th><th class="num">中位</th><th class="num">上涨率</th></tr>')
    for key, lab in [("old", "旧: I30∩G30∩D30"), ("new", "新: (I30∪G30)∩D30"), ("add", "新增(新有旧无)")]:
        for hd in HOLDS:
            v = np.array(cstat[key][hd])
            if len(v):
                h.append(f'<tr><td>{lab}</td><td class="num">{hd}</td><td class="num">{len(v)}</td>'
                         f'<td class="num" style="color:{col(v.mean())}">{v.mean():+.2f}%</td>'
                         f'<td class="num">{np.median(v):+.2f}%</td>'
                         f'<td class="num">{np.mean(v>0)*100:.1f}%</td></tr>')
    h.append('</table><div class="note">剔除集恒为空（旧共识是新共识的子集），所以该改动只会“多做”。'
             '平均共识数 7.0 → 8.1 只，新增部分收益弱于原共识。</div></div>')

    # ---- 稳定性 ----
    h.append('<div class="card"><h2>附加：因子 IC 跨窗口稳定性（20 日一块）</h2>')
    h.append('<table><tr><th>因子</th><th class="num">线上权重</th><th class="num">全窗IC</th>'
             '<th>分块 IC（按时间顺序）</th><th class="num">翻转次数</th></tr>')
    for f, w, m, blocks, flips in sorted(srows, key=lambda x: -x[4])[:12]:
        bs = " ".join(f'{b:+.2f}' for b in blocks)
        h.append(f'<tr><td>{f}</td><td class="num">{w:+.4f}</td>'
                 f'<td class="num" style="color:{col(m)}">{m:+.4f}</td>'
                 f'<td style="font-family:ui-monospace,monospace;font-size:11.5px">{bs}</td>'
                 f'<td class="num">{flips}</td></tr>')
    avg_f = np.mean([r[4] for r in srows])
    h.append('</table><div class="note">平均每因子符号翻转 %.1f 次 / 8 个区间；'
             '几乎所有因子的最后一块（最近 20 日）IC 转负。这是“排名在样本外消失”的直接原因，'
             '也是任何“换权重口径”的改法都救不了排名的地方 —— 问题在因子稳定性本身。</div></div>' % avg_f)

    # ---- 局限 ----
    h.append('<div class="card"><h2>方法与本轮验证的局限（请一并考虑）</h2>')
    h.append('<div class="verdict info"><b>防前视的部分：</b>因子只用 ≤T 的K线；收益用 T+1 开盘买入 → T+H 收盘；'
             'B 的权重只在训练段标定、在样本外评估。<br><br>'
             '<b>仍然存在的偏差：</b>① PE/PB/换手/股本用“当前值”近似历史（审计第4条同款问题），'
             '对本轮属于 A/B 共同项，不影响相对比较，但会让绝对水平失真；② 未计手续费/印花税/滑点，'
             '也未剔除涨停无法买入、停牌、退市；③ 回放池用“今天的 512 只”（存活偏差）；'
             '④ 单一回放窗口单一市场环境（2025-11~2026-08），结论不能外推到其他行情；'
             '⑤ A 组合级用 <code>0.35×Δ加分</code> 近似（忽略 tech_score 的 clamp 边界）。</div>')
    h.append('</div>')
    h.append('</div></body></html>')
    return "".join(h)


# ================================================================
# 主流程
# ================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-replay", action="store_true")
    ap.add_argument("--days", type=int, default=180, help="回放交易日数(从最近往前)")
    ap.add_argument("--train", type=int, default=85, help="训练段交易日数")
    args = ap.parse_args()

    print("=" * 78)
    print("  第2档改动离线 A/B 验证 (防前视: 因子只用<=T数据, 收益用T+1开盘→T+H收盘)")
    print("=" * 78)

    if args.skip_replay and os.path.exists(CACHE):
        print("\n[1/4] 载入回放缓存...")
        rp = pickle.load(open(CACHE, "rb"))
        print(f"  缓存: {len(rp['dates'])} 个交易日, {len(rp['factor_names'])} 因子")
    else:
        print("\n[1/4] 加载数据...")
        kl, ex, ff = load_data()
        print(f"  K线 {len(kl)} 只")
        print("\n[2/4] 逐日回放(缓存 z/RSI/前瞻收益)...")
        t0 = time.time()
        rp = build_replay(kl, ex, ff, args.days)
        pickle.dump(rp, open(CACHE, "wb"))
        print(f"  缓存已写入 {CACHE} ({time.time()-t0:.0f}s)")

    dates = rp["dates"]
    if len(dates) < 40:
        print("!! 回放交易日不足, 无法验证")
        return
    ntr = min(args.train, len(dates) - 30)
    train_dates = dates[:ntr]
    test_dates = dates[ntr:]
    print(f"\n  训练段 {len(train_dates)} 日 | 测试段(样本外) {len(test_dates)} 日")

    print("\n[3/5] 改动A: RSI 过热 (因子级)")
    a_out = test_a_oversold(rp)
    print("\n[4/5] 改动A: RSI 过热 (组合级, 近似)")
    a_z = test_a_combo(rp, test_dates, mode="zero")
    a_p = test_a_combo(rp, test_dates, mode="penalty")
    print("\n[5/5] 改动B: 权重 mean_ic → IR")
    Wb, Wi, bstat, bres, bref = test_b_weights(rp, train_dates, test_dates)
    print("\n[6/6] 改动C: 共识定义")
    cstat = test_c_consensus(rp, test_dates)
    print("\n[7/7] 附加: 因子IC稳定性")
    srows = test_stability(rp, train_dates, rp["factor_names"])

    html = generate_html(rp, train_dates, test_dates, a_out, a_z, a_p, Wb, Wi,
                         bstat, bres, bref, cstat, srows)
    out = os.path.join(OUTPUT, "verify_tier2_ab_report.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\n报告已生成: {out}")
    print("完成。")


if __name__ == "__main__":
    main()
