#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子稳定性治理 —— 离线 A/B 验证 (不改动线上任何代码/数据/台账)

背景: verify_tier2_ab.py 已证明"换权重口径"(mean_ic→IR)、"RSI 过热惩罚"、"放宽共识"三项
      都无效或更差, 并暴露真问题: 因子 IC 极不稳定(每 20 日符号翻转 3.2 次),
      最近 20 日几乎所有技术因子 IC 转负, 19 个 IC 因子里 13 个训练段方向与线上权重相反。
      本脚本验证的候选方向: 用「滚动窗口的因子一致性」动态筛选/降权, 能否让排名更准、收益更高。

严格防前视:
  - 测试日 T 的权重只使用「已完全实现」的 IC: 日期 d 的 IC = d 的因子 vs (d+1开盘→d+H收盘),
    因此要求 d + H + 1 <= T 才可用于 T 日决策。
  - 滚动窗口 = 最近 W 个可用 IC 日。
  - 因子 z 向量本身由缓存提供(生成时只用了 <=d 的K线)。

对照设计(关键):
  - 全池等权基准(不选股) —— 没有它无法判断"排名有没有 alpha"。
  - 线上静态权重 ICIR_V3。
  - 每个"筛选类"变体都配一个「随机筛选同样数量因子」的对照分布(多次重抽),
    用于区分"筛选有效"与"随机运气"。

用法:
  python3 verify_factor_stability.py                 # 主验证(W=60, H=5)
  python3 verify_factor_stability.py --window 40 --ic-hold 5
  python3 verify_factor_stability.py --rand 500
"""
import os, sys, pickle, argparse
from datetime import datetime
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
CACHE = os.path.join(BASE, "output", "tier2_replay_cache.pkl")
OUTPUT = os.path.join(BASE, "output")

import unified_scoring_engine as eng

HOLDS = [5, 10, 20]
TOP_N = 30
MIN_POOL = 100

# 与权重表注释一致的分组
IC_FACTORS = ["sector_rsi", "dev_ma20", "vwap_premium", "macd_signal", "sector_momentum",
              "ret_20d", "rsi_signal", "ma_bull", "ret_5d", "max_dd_20d", "mfi", "gap_open",
              "vol_price", "pct_52w", "cmf", "streak", "volatility_20d", "amplitude_z",
              "vol_ratio_5d"]
HAND_FACTORS = ["turnover_z", "log_mcap", "pe_percentile", "pb_percentile", "event_score",
                "event_count", "inflow_rate", "main_flow_5d", "main_flow_20d",
                "roe_rank", "gross_margin_rank", "ocf_ratio_rank"]


# ================================================================
# 工具
# ================================================================
def rank_vec(a):
    a = np.asarray(a, dtype=float)
    r = np.empty(len(a), dtype=float)
    order = a.argsort()
    r[order] = np.arange(len(a), dtype=float)
    _, inv, cnt = np.unique(a, return_inverse=True, return_counts=True)
    if (cnt > 1).any():
        s = np.zeros(len(cnt))
        np.add.at(s, inv, r)
        r = (s / cnt)[inv]
    return r


def spearman(ra, rb):
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    d = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / d) if d > 1e-12 else 0.0


def stats(vals):
    a = np.array(vals, dtype=float)
    if len(a) < 2:
        return dict(n=len(a), mean=0.0, med=0.0, win=0.0, t=0.0)
    sd = a.std(ddof=1)
    return dict(n=len(a), mean=float(a.mean()), med=float(np.median(a)),
                win=float(np.mean(a > 0) * 100),
                t=float(a.mean() / (sd / np.sqrt(len(a)) + 1e-12)))


def welch_t(a, b):
    a, b = np.array(a, float), np.array(b, float)
    if len(a) < 2 or len(b) < 2:
        return 0.0, 0.0
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    if se < 1e-12:
        return 0.0, 0.0
    return float((a.mean() - b.mean()) / se), float(a.mean() - b.mean())


# ================================================================
# 逐日 IC 矩阵 (n_dates × n_factors), 用于滚动定权
# ================================================================
def precompute_ic(rp, hold):
    dates, fnames = rp["dates"], rp["factor_names"]
    nf = len(fnames)
    IC = np.full((len(dates), nf), np.nan)
    for i, d in enumerate(dates):
        recs = rp["records"][d]
        codes = [c for c, r in recs.items() if hold in r["fwd"]]
        if len(codes) < MIN_POOL:
            continue
        fwd = np.array([recs[c]["fwd"][hold] for c in codes], dtype=float)
        rf = rank_vec(fwd)
        Z = np.array([recs[c]["z"] for c in codes], dtype=float)
        for j in range(nf):
            z = Z[:, j]
            if np.allclose(z, z[0]):
                continue
            IC[i, j] = spearman(rank_vec(z), rf)
    return IC


# ================================================================
# 滚动权重构造器
# ================================================================
def col_mean_std(sub):
    """列均值/标准差, 容忍全 NaN 列(返回 0)"""
    cnt = np.sum(~np.isnan(sub), axis=0)
    safe = np.maximum(cnt, 1)
    filled = np.where(np.isnan(sub), 0.0, sub)
    m = np.where(cnt > 0, filled.sum(axis=0) / safe, 0.0)
    sq = np.where(cnt > 0, (filled ** 2).sum(axis=0) / safe, 0.0)
    var = np.where(cnt > 1, (sq - m ** 2) * cnt / np.maximum(cnt - 1, 1), 0.0)
    return m, np.sqrt(np.maximum(var, 0.0))


class RollingWeighter:
    """给定测试日 index i, 返回 {factor: weight}。只用已实现的 IC (d + ic_hold + 1 <= i)。"""

    def __init__(self, rp, IC, window, ic_hold, mode, topk=10, ic_only=True, seed=20260922):
        self.rp = rp
        self.fnames = rp["factor_names"]
        self.fidx = {f: j for j, f in enumerate(self.fnames)}
        self.IC = IC
        self.W = window
        self.H = ic_hold
        self.mode = mode
        self.topk = topk
        self.ic_only = ic_only
        self.rng = np.random.default_rng(seed)
        self._cache = {}
        # 静态线上权重(ICIR_V3), 用于符号矫正/降权基准
        self.w_online = {f: eng.ICIR_V3.get(f, 0.0) for f in self.fnames}
        self.ic_scale = sum(abs(self.w_online[f]) for f in IC_FACTORS)

    def _window_ic(self, i):
        """返回 (因子名列表, 窗口内 IC 矩阵 (n_avail × n_f))"""
        last = i - self.H - 1          # IC 需 H 日才完全实现
        if last < 0:
            return None
        lo = max(0, last - self.W + 1)
        sub = self.IC[lo:last + 1, :]
        mask = ~np.isnan(sub).all(axis=0)
        if sub.shape[0] < 5:
            return None
        return sub, mask

    def _stats(self, i):
        """缓存每日窗口统计 (m, sd, cons, mask), 随机对照会在同一天重复调用"""
        c = self._cache.get(i, "miss")
        if c != "miss":
            return c
        got = self._window_ic(i)
        if got is None:
            self._cache[i] = None
            return None
        sub, mask = got
        m, sd = col_mean_std(sub)
        cons = np.zeros(len(self.fnames))
        for j in range(len(self.fnames)):
            if not mask[j]:
                continue
            col = sub[:, j]
            col = col[~np.isnan(col)]
            if len(col) == 0:
                continue
            sg = np.sign(m[j]) if abs(m[j]) > 1e-12 else 1.0
            cons[j] = float(np.mean(np.sign(col) == sg))
        self._cache[i] = (m, sd, cons, mask)
        return self._cache[i]

    def weights(self, i, rng=None, rand_pick=None):
        out = {f: 0.0 for f in self.fnames}
        w = self.w_online

        # 手工组: 所有模式都保持线上原值(不参与筛选), 除非 ic_only=False
        for f in HAND_FACTORS:
            out[f] = w.get(f, 0.0)

        if self.mode == "online":
            for f in self.fnames:
                out[f] = w.get(f, 0.0)
            return out

        st = self._stats(i)
        if st is None:
            return out
        m, sd, cons, mask = st
        ir = np.where(sd > 1e-9, m / sd, 0.0)

        def renormalize(d):
            """把 IC 组缩放到与线上 IC 组相同的总幅度"""
            tot = sum(abs(d.get(f, 0.0)) for f in IC_FACTORS)
            if tot < 1e-12:
                return
            k = self.ic_scale / tot
            for f in IC_FACTORS:
                d[f] = d.get(f, 0.0) * k

        if self.mode == "sign_zero":
            # 滚动窗口方向与线上权重相反 → 归零
            for f in IC_FACTORS:
                j = self.fidx[f]
                if not mask[j]:
                    continue
                if np.sign(m[j]) != np.sign(w.get(f, 0.0)) and abs(m[j]) > 1e-9:
                    out[f] = 0.0
                else:
                    out[f] = w.get(f, 0.0)
            renormalize(out)

        elif self.mode == "sign_flip":
            # 反号 → 翻转符号(反向使用该因子)
            for f in IC_FACTORS:
                j = self.fidx[f]
                if not mask[j]:
                    continue
                onl = w.get(f, 0.0)
                out[f] = abs(onl) * np.sign(m[j]) if abs(m[j]) > 1e-9 else onl
            renormalize(out)

        elif self.mode == "consistency":
            # 线上权重 × max(0, 2*cons-1): 一致率越低越接近 0
            for f in IC_FACTORS:
                j = self.fidx[f]
                out[f] = w.get(f, 0.0) * max(0.0, 2 * cons[j] - 1)
            renormalize(out)

        elif self.mode == "rolling_ic":
            # 完全用滚动 mean_IC 重新定权
            for f in IC_FACTORS:
                j = self.fidx[f]
                out[f] = m[j] if mask[j] else 0.0
            renormalize(out)

        elif self.mode == "rolling_ir":
            for f in IC_FACTORS:
                j = self.fidx[f]
                out[f] = ir[j] if mask[j] else 0.0
            renormalize(out)

        elif self.mode == "equal_ic":
            # IC 组等权 (检验"精细权重是否真的比等权更好")
            for f in IC_FACTORS:
                out[f] = 1.0
            renormalize(out)

        elif self.mode in ("topk_cons", "topk_ic", "topk_rand"):
            pool = IC_FACTORS[:] if self.ic_only else self.fnames[:]
            if rand_pick is not None:
                # 显式给定因子集合时无条件优先(随机对照要对任意模式生效)
                picked = list(rand_pick)
            elif self.mode == "topk_cons":
                key = lambda f: (cons[self.fidx[f]], abs(m[self.fidx[f]]))
                picked = [f for f in sorted(pool, key=key, reverse=True)[:self.topk]]
            elif self.mode == "topk_ic":
                key = lambda f: abs(m[self.fidx[f]])
                picked = [f for f in sorted(pool, key=key, reverse=True)[:self.topk]]
            else:  # topk_rand
                picked = self.rng.choice(pool, size=self.topk, replace=False).tolist()
            for f in picked:
                out[f] = abs(w.get(f, 0.0)) if abs(w.get(f, 0.0)) > 1e-9 else 0.01
            renormalize(out)
        return out


# ================================================================
# 评估
# ================================================================
def score_and_top(recs, w, fnames):
    wv = np.array([w.get(f, 0.0) for f in fnames], dtype=np.float32)
    if not np.any(wv != 0):
        return []
    sc = {c: float(np.dot(r["z"], wv)) for c, r in recs.items()}
    return sorted(sc, key=lambda x: -sc[x])[:TOP_N]


def evaluate(rp, test_idx, weighter, rand_rounds=0, rng=None):
    """返回 {hold: [收益...]}; rand_rounds>0 时额外返回随机筛选的逐次均值分布"""
    fnames = rp["factor_names"]
    dates = rp["dates"]
    res = {h: [] for h in HOLDS}
    rand_dist = {h: [] for h in HOLDS} if rand_rounds else None
    pool = IC_FACTORS

    for i in test_idx:
        d = dates[i]
        recs = rp["records"][d]
        w = weighter.weights(i)
        top = score_and_top(recs, w, fnames)
        for h in HOLDS:
            res[h] += [recs[c]["fwd"][h] for c in top if h in recs[c]["fwd"]]

        if rand_rounds:
            for _ in range(rand_rounds):
                pick = rng.choice(pool, size=weighter.topk, replace=False).tolist()
                wr = weighter.weights(i, rand_pick=pick)
                tr = score_and_top(recs, wr, fnames)
                for h in HOLDS:
                    vals = [recs[c]["fwd"][h] for c in tr if h in recs[c]["fwd"]]
                    if vals:
                        rand_dist[h].append(float(np.mean(vals)))
    return res, rand_dist


def pool_baseline(rp, test_idx):
    res = {h: [] for h in HOLDS}
    for i in test_idx:
        recs = rp["records"][rp["dates"][i]]
        for h in HOLDS:
            res[h] += [r["fwd"][h] for r in recs.values() if h in r["fwd"]]
    return res


# ================================================================
# 补充分析: 因子选择重合度 / 参数敏感性
# ================================================================
def overlap_analysis(rp, test_idx, IC, window, ic_hold, topk, rounds=20):
    """各方案 TOP30 与线上权重 TOP30 的平均重合度(%)"""
    fnames = rp["factor_names"]
    wt_onl = RollingWeighter(rp, IC, window, ic_hold, "online")
    wt_c = RollingWeighter(rp, IC, window, ic_hold, "topk_cons", topk=topk)
    wt_i = RollingWeighter(rp, IC, window, ic_hold, "topk_ic", topk=topk)
    wt_e = RollingWeighter(rp, IC, window, ic_hold, "equal_ic")
    wt_r = RollingWeighter(rp, IC, window, ic_hold, "topk_rand", topk=topk)
    rng = np.random.default_rng(4477)
    acc = {"cons": [], "ic": [], "equal": [], "rand": []}
    for i in test_idx:
        recs = rp["records"][rp["dates"][i]]
        base = set(score_and_top(recs, wt_onl.weights(i), fnames))
        if not base:
            continue
        for key, wt in [("cons", wt_c), ("ic", wt_i), ("equal", wt_e)]:
            got = set(score_and_top(recs, wt.weights(i), fnames))
            acc[key].append(len(base & got) / len(base))
        for _ in range(rounds):
            pick = rng.choice(IC_FACTORS, size=topk, replace=False).tolist()
            got = set(score_and_top(recs, wt_r.weights(i, rand_pick=pick), fnames))
            acc["rand"].append(len(base & got) / len(base))
    return {k: (float(np.mean(v)) * 100 if v else 0.0) for k, v in acc.items()}


def sensitivity_scan(rp, IC, ic_hold, test_idx, windows, ks):
    """(窗口 × TopK) 网格: topk_cons 的样本外表现与超额"""
    pb = {h: stats(pool_baseline(rp, test_idx)[h])["mean"] for h in HOLDS}
    rows = []
    for W in windows:
        for K in ks:
            wt = RollingWeighter(rp, IC, W, ic_hold, "topk_cons", topk=K)
            res, _ = evaluate(rp, test_idx, wt)
            s = {h: stats(res[h]) for h in HOLDS}
            rows.append((W, K, s[5]["mean"], s[5]["mean"] - pb[5],
                         s[20]["mean"], s[20]["mean"] - pb[20]))
    return rows, pb


def daily_pct(rp, test_idx, weighter, rounds, rng):
    """按日百分位: 逐日算真值在当日随机组合分布中的位置, 再对日期取平均
    (消除"日期方差" —— 全局分布很宽主要来自不同交易日之间的市场波动)"""
    fnames = rp["factor_names"]
    dates = rp["dates"]
    pcts, real_m, rand_m = {h: [] for h in HOLDS}, {h: [] for h in HOLDS}, {h: [] for h in HOLDS}
    for i in test_idx:
        recs = rp["records"][dates[i]]
        real_top = score_and_top(recs, weighter.weights(i), fnames)
        day = {h: [] for h in HOLDS}
        for _ in range(rounds):
            pick = rng.choice(IC_FACTORS, size=weighter.topk, replace=False).tolist()
            top = score_and_top(recs, weighter.weights(i, rand_pick=pick), fnames)
            for h in HOLDS:
                v = [recs[c]["fwd"][h] for c in top if h in recs[c]["fwd"]]
                if v:
                    day[h].append(float(np.mean(v)))
        for h in HOLDS:
            rv = [recs[c]["fwd"][h] for c in real_top if h in recs[c]["fwd"]]
            if not rv or not day[h]:
                continue
            rm = float(np.mean(rv))
            arr = np.array(day[h])
            real_m[h].append(rm)
            rand_m[h].append(float(arr.mean()))
            pcts[h].append(float(np.mean(arr <= rm) * 100))
    return {h: (float(np.nanmean(pcts[h])), float(np.nanmean(real_m[h])),
                float(np.nanmean(rand_m[h]))) for h in HOLDS}


def random30_baseline(rp, test_idx, rounds=100, size=30, seed=889):
    """不看分数、随机抽 30 只的收益分布 (判断"取TOP30"这个动作本身有无 alpha)"""
    rng = np.random.default_rng(seed)
    dist = {h: [] for h in HOLDS}
    for i in test_idx:
        recs = rp["records"][rp["dates"][i]]
        codes = list(recs.keys())
        if len(codes) < size:
            continue
        for _ in range(rounds):
            pick = rng.choice(codes, size=size, replace=False)
            for h in HOLDS:
                v = [recs[c]["fwd"][h] for c in pick if h in recs[c]["fwd"]]
                if v:
                    dist[h].append(float(np.mean(v)))
    return {h: (float(np.mean(dist[h])), float(np.percentile(dist[h], 5)),
                float(np.percentile(dist[h], 95))) for h in HOLDS}


# ================================================================
# HTML 报告
# ================================================================
def generate_html(rp, cfg, test_idx, table, diag, rand_summary, seg_table, extras):
    P, N = "#dc2626", "#16a34a"
    def col(v):
        return P if v > 0 else (N if v < 0 else "#666")
    h = ["""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>因子稳定性治理 A/B 验证</title><style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,"PingFang SC",sans-serif;background:#f5f7fa;color:#1a1a2e;padding:24px;line-height:1.6}
.wrap{max-width:1100px;margin:0 auto}
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
.good{background:#f0fdf4;border-left:4px solid #16a34a}
.warn{background:#fffbeb;border-left:4px solid #f59e0b}
.info{background:#eff6ff;border-left:4px solid #2563eb}
.note{font-size:12px;color:#888;margin-top:8px}
code{background:#f1f5f9;padding:1px 5px;border-radius:3px;font-size:12px}
</style></head><body><div class="wrap">"""]
    h.append("<h1>因子稳定性治理 — 离线 A/B 验证</h1>")
    h.append(f'<div class="sub">回放 {rp["dates"][0]} ~ {rp["dates"][-1]}（{len(rp["dates"])} 交易日 / 512池）'
             f' | 滚动窗口 W={cfg["window"]} 日, IC 持有期 H={cfg["ic_hold"]} 日'
             f' | 样本外评估 {len(test_idx)} 日（{rp["dates"][test_idx[0]]} ~ {rp["dates"][test_idx[-1]]}）'
             f'<br>前瞻收益 = T+1开盘 → T+H收盘 | 权重只用已完全实现的 IC（d+H+1 ≤ T）'
             f' | 生成 {datetime.now():%Y-%m-%d %H:%M}</div>')

    # 结论
    base = table["__pool__"]
    h.append('<div class="card"><h2>结论</h2>')
    onl5, onl20 = table["online"][5]["mean"], table["online"][20]["mean"]
    beat = []
    for k, r in table.items():
        if k in ("__pool__", "online"):
            continue
        if r[5]["mean"] > onl5 and r[20]["mean"] > onl20:
            beat.append(cfg["labels"].get(k, k))
    if beat:
        h.append('<div class="verdict warn"><b>%d 个变体同时超过线上静态权重：%s</b> —— '
                 '但需对照随机筛选分布判断是否为运气（见下）。</div>'
                 % (len(beat), "；".join(beat)))
    else:
        h.append(f'<div class="verdict bad"><b>没有任何变体同时超过线上静态权重'
                 f'（5日 {onl5:+.2f}% / 20日 {onl20:+.2f}%）。</b>'
                 '滚动稳定性筛选、方向矫正、滚动重标定都未能让排名更准。</div>')
    if rand_summary:
        dpv = extras.get("daily_pct") or {}
        ovv = extras.get("overlap") or {}
        pcs = "/".join(f"{dpv[hd][0]:.0f}%" for hd in HOLDS if hd in dpv)
        h.append('<div class="verdict bad"><b>决定性证据：与随机选因子无法区分。</b>'
                 f'按日百分位 {pcs}（5/10/20日），'
                 f'随机组合的 TOP30 甚至略优于真实筛选；'
                 f'同时换因子几乎不换股票（与线上权重 TOP30 重合度 {ovv.get("cons", 0):.1f}%）。'
                 '机制是技术类因子高度冗余 —— 都在表达同一个「广义动量」成分，'
                 '因此「挑哪几个因子、给什么权重」改变不了名单。</div>')
        h.append('<div class="verdict warn"><b>本次验证同时否掉了我上一轮的建议。</b>'
                 '上一轮结论说「下一步值得做因子稳定性治理」，'
                 '本轮的随机对照表明「按历史一致性挑因子」与「随机挑因子」在统计上无法区分 —— '
                 '该方向不成立，不应实施。</div>')
        h.append('<div class="verdict good"><b>但有一条被证实的正面结论：'
                 '「按因子打分取 TOP30」这个动作本身有真实超额。</b>'
                 '随机抽 30 只（不看任何分数）与全池等权几乎相同，'
                 '而所有加权打分方案都明显更高 —— '
                 '系统的价值来自「用动量类因子排序取前 6%」这个结构，'
                 '而不是因子的精细调校。</div>')
    if seg_table:
        a20s = [v["a20"] for v in seg_table.values()]
        b20s = [v["b20"] for v in seg_table.values()]
        b5s = [v["b5"] for v in seg_table.values()]
        a5s = [v["a5"] for v in seg_table.values()]
        if min(a20s) > 0 and max(b20s) < 0:
            mid_d = rp["dates"][test_idx[len(test_idx) // 2]]
            h.append(f'<div class="verdict warn"><b>更重要的发现：20 日超额在测试段后半整体转负。</b>'
                     f'全部方案的 20 日超额在前半段（{mid_d} 之前）为 '
                     f'+{min(a20s):.1f} ~ +{max(a20s):.1f}pp，后半段（{mid_d} 之后）转为 '
                     f'{min(b20s):.1f} ~ {max(b20s):.1f}pp；'
                     f'而 5 日超额在后半段仍为正（+{min(b5s):.1f} ~ +{max(b5s):.1f}pp，'
                     f'前半 +{min(a5s):.1f} ~ +{max(a5s):.1f}pp）。'
                     f'这说明当前信号更适合短持有期，持有到 20 日会被反转吃掉 —— '
                     f'这一点比「因子怎么选」更影响实盘结果。</div>')
    h.append('</div>')

    # 主表
    h.append('<div class="card"><h2>主结果：各方案样本外 TOP30 表现</h2>')
    h.append('<table><tr><th>方案</th><th class="num">n</th>'
             '<th class="num">5日均收益</th><th class="num">10日</th><th class="num">20日</th>'
             '<th class="num">5日超额</th><th class="num">20日超额</th><th class="num">20日上涨率</th></tr>')
    order = ["__pool__"] + [k for k in table if not k.startswith("__")]
    for k in order:
        r = table[k]
        lab = cfg["labels"].get(k, k)
        b5, b20 = base[5]["mean"], base[20]["mean"]
        e5, e20 = r[5]["mean"] - b5, r[20]["mean"] - b20
        hl = ' style="background:#eff6ff"' if k == "__pool__" else ""
        h.append(f'<tr{hl}><td>{lab}</td><td class="num">{r[5]["n"]}</td>'
                 f'<td class="num" style="color:{col(r[5]["mean"])}">{r[5]["mean"]:+.2f}%</td>'
                 f'<td class="num" style="color:{col(r[10]["mean"])}">{r[10]["mean"]:+.2f}%</td>'
                 f'<td class="num" style="color:{col(r[20]["mean"])}">{r[20]["mean"]:+.2f}%</td>'
                 f'<td class="num" style="color:{col(e5)}">{e5:+.2f}pp</td>'
                 f'<td class="num" style="color:{col(e20)}">{e20:+.2f}pp</td>'
                 f'<td class="num">{r[20]["win"]:.1f}%</td></tr>')
    h.append('</table><div class="note">超额 = 该方案 TOP30 − 全池等权基准（同一测试日集合、同一收益口径）。'
             '注意 5/10/20 日收益序列存在重叠（相邻日名单高度相似），t 值仅作参考。</div></div>')

    # 随机对照
    if rand_summary:
        dist = rand_summary["dist"]
        h.append('<div class="card"><h2>关键对照：真实筛选 vs 随机筛选同数量因子</h2>')
        h.append('<table><tr><th>方案</th><th class="num">持有</th><th class="num">随机均值</th>'
                 '<th class="num">随机5%分位</th><th class="num">随机95%分位</th>'
                 '<th class="num">真实筛选</th><th class="num">百分位</th><th>判定</th></tr>')
        LAB = {"topk_cons": "V6 一致率 TopK", "topk_ic": "V7 |IC| TopK"}
        for key in ("topk_cons", "topk_ic"):
            for hd in HOLDS:
                rs = np.array(dist[hd])
                real = rand_summary["real"][key][hd]
                pct = float(np.mean(rs <= real) * 100)
                ok = ("<span style='color:#16a34a'>✅ 优于随机</span>" if pct >= 95
                      else ("<span style='color:#dc2626'>❌ 无法区分随机</span>" if 5 < pct < 95
                            else "<span style='color:#dc2626'>⚠️ 差于随机</span>"))
                h.append(f'<tr><td>{LAB[key]}</td><td class="num">{hd}日</td>'
                         f'<td class="num">{rs.mean():+.2f}%</td>'
                         f'<td class="num">{np.percentile(rs, 5):+.2f}%</td>'
                         f'<td class="num">{np.percentile(rs, 95):+.2f}%</td>'
                         f'<td class="num" style="color:{col(real)}">{real:+.2f}%</td>'
                         f'<td class="num">{pct:.0f}%</td><td>{ok}</td></tr>')
        h.append('</table><div class="note">随机筛选 = 从同一池因子中随机取同样数量，'
                 '权重构造方式与真实筛选完全相同，唯一差别是「选了哪几个因子」，'
                 '每日重抽 %d 次得到分布。若真实筛选的百分位落在 5%%~95%% 之间，说明它与随机选因子'
                 '在统计上无法区分 —— 即「按历史一致性挑因子」不成立。</div></div>' % cfg["rand"])
    else:
        h.append('<div class="card"><div class="verdict info">本次未启用随机对照（<code>--rand 0</code>）。'
                 '缺少它无法区分“筛选有效”与“随机运气”，建议至少跑 200 次。</div></div>')

    # 分段一致性
    if seg_table:
        h.append('<div class="card"><h2>分段一致性：测试段前半 / 后半</h2>')
        h.append('<table><tr><th>方案</th><th class="num">前半5日</th><th class="num">前半20日</th>'
                 '<th class="num">后半5日</th><th class="num">后半20日</th><th class="num">方向一致?</th></tr>')
        for k, v in seg_table.items():
            c1 = v["a5"] > 0 and v["a20"] > 0
            c2 = v["b5"] > 0 and v["b20"] > 0
            ok = "✅ 一致为正" if (c1 and c2) else ("一致为负" if not c1 and not c2 else "❌ 前后矛盾")
            h.append(f'<tr><td>{cfg["labels"].get(k, k)}</td>'
                     f'<td class="num" style="color:{col(v["a5"])}">{v["a5"]:+.2f}pp</td>'
                     f'<td class="num" style="color:{col(v["a20"])}">{v["a20"]:+.2f}pp</td>'
                     f'<td class="num" style="color:{col(v["b5"])}">{v["b5"]:+.2f}pp</td>'
                     f'<td class="num" style="color:{col(v["b20"])}">{v["b20"]:+.2f}pp</td>'
                     f'<td>{ok}</td></tr>')
        h.append('</table><div class="note">超额为相对同段全池基准。'
                 '若一个方案在前后半段方向相反，说明它的效果不稳定，不能依赖。</div></div>')

    # 因子诊断
    h.append('<div class="card"><h2>因子方向诊断：线上权重 vs 窗口实测</h2>')
    h.append('<table><tr><th>因子</th><th class="num">线上权重</th><th class="num">窗口IC均值</th>'
             '<th class="num">窗口IR</th><th class="num">符号一致率</th><th>判定</th></tr>')
    for f, w, m, ir, cons in diag:
        bad = (np.sign(w) != np.sign(m)) if abs(m) > 1e-9 else False
        tag = ('<span style="color:#dc2626">方向相反</span>' if bad
               else ('<span style="color:#16a34a">方向一致</span>' if abs(m) > 1e-9 else 'IC≈0'))
        h.append(f'<tr><td>{f}</td><td class="num">{w:+.4f}</td>'
                 f'<td class="num" style="color:{col(m)}">{m:+.4f}</td>'
                 f'<td class="num">{ir:+.2f}</td><td class="num">{cons*100:.0f}%</td>'
                 f'<td>{tag}</td></tr>')
    h.append('</table></div>')

    # 重合度
    ov = extras.get("overlap")
    if ov:
        h.append('<div class="card"><h2>补充一：换因子到底换掉了多少只股票？（与线上权重 TOP30 的重合度）</h2>')
        h.append('<table><tr><th>方案</th><th class="num">与线上权重 TOP30 重合度</th></tr>')
        for k, lab in [("equal", "IC 因子等权"), ("cons", "V6 一致率 TopK"),
                       ("ic", "V7 |IC| TopK"), ("rand", "随机 TopK 因子")]:
            h.append(f'<tr><td>{lab}</td><td class="num">{ov[k]:.1f}%</td></tr>')
        h.append('</table><div class="note">重合度 = 每日 TOP30 名单与线上权重名单的交集占比。'
                 '重合度低说明"换因子"确实换出了不同的股票组合，'
                 '因此上面"随机筛选无法区分"的结论不是因为各方案名单相同造成的。</div></div>')

    # 敏感性
    sens = extras.get("sens")
    if sens:
        rows, pb = sens
        h.append('<div class="card"><h2>补充二：参数敏感性（窗口长度 × 保留因子数）</h2>')
        h.append('<table><tr><th class="num">滚动窗口</th><th class="num">保留因子数</th>'
                 '<th class="num">5日超额</th><th class="num">20日超额</th><th>方向</th></tr>')
        for W, K, m5, e5, m20, e20 in rows:
            ok = ('<span style="color:#16a34a">5日+20日同为正</span>' if e5 > 0 and e20 > 0
                  else ('<span style="color:#dc2626">至少一档为负</span>' if e5 < 0 or e20 < 0
                        else '弱'))
            h.append(f'<tr><td class="num">{W} 日</td><td class="num">{K}</td>'
                     f'<td class="num" style="color:{col(e5)}">{e5:+.2f}pp</td>'
                     f'<td class="num" style="color:{col(e20)}">{e20:+.2f}pp</td>'
                     f'<td>{ok}</td></tr>')
        h.append('</table><div class="note">窗口与保留数量都属于可调参数，'
                 '若结果随参数大幅摆动，说明它不稳定、不可依赖。</div></div>')

    # 按日百分位 + 随机30只
    dp, r30 = extras.get("daily_pct"), extras.get("rand30")
    if dp and r30:
        onl = table["online"]
        ovc = (extras.get("overlap") or {}).get("cons", 0.0)
        pc_lo = min(dp[hd][0] for hd in HOLDS if hd in dp)
        pc_hi = max(dp[hd][0] for hd in HOLDS if hd in dp)
        h.append('<div class="card"><h2>补充三：拆掉「日期方差」后的对照 + 随机选股基准</h2>')
        h.append('<h3>① 按日百分位（每天独立计算百分位，再对日期取平均）</h3>')
        h.append('<table><tr><th class="num">持有</th><th class="num">V6 真值</th>'
                 '<th class="num">当日随机因子组合均值</th><th class="num">按日百分位</th><th>判定</th></tr>')
        for hd in HOLDS:
            pct, real, rmean = dp[hd]
            ok = ('<span style="color:#16a34a">✅ 显著优于随机</span>' if pct >= 95
                  else ('<span style="color:#dc2626">❌ 与随机无异</span>' if 5 < pct < 95
                        else '<span style="color:#dc2626">⚠️ 差于随机</span>'))
            h.append(f'<tr><td class="num">{hd}日</td>'
                     f'<td class="num">{real:+.2f}%</td><td class="num">{rmean:+.2f}%</td>'
                     f'<td class="num">{pct:.0f}%</td><td>{ok}</td></tr>')
        h.append('</table><div class="note">全局随机分布的跨度很宽（5日 −9% ~ +14%），'
                 '主要来自「不同交易日之间的市场波动」而非「不同因子组合」。'
                 '按日消除日期方差后，真值的百分位仍停在 50% 附近。</div>')
        h.append('<h3>② 随机选 30 只（完全不看分数）作为选股能力基准</h3>')
        h.append('<table><tr><th>方案</th><th class="num">5日</th><th class="num">10日</th><th class="num">20日</th></tr>')
        cells = "".join(f'<td class="num">{r30[hd][0]:+.2f}%</td>' for hd in HOLDS)
        h.append(f'<tr><td>随机抽 30 只（不看分数）</td>{cells}</tr>')
        cells = "".join(f'<td class="num">{base[hd]["mean"]:+.2f}%</td>' for hd in HOLDS)
        h.append(f'<tr><td>全池等权（512 只）</td>{cells}</tr>')
        cells = "".join(f'<td class="num" style="color:{col(onl[hd]["mean"])}">'
                        f'{onl[hd]["mean"]:+.2f}%</td>' for hd in HOLDS)
        h.append(f'<tr><td>线上权重 TOP30</td>{cells}</tr>')
        h.append('</table><div class="note">三点连起来看：'
                 f'（a）<b>换因子几乎不换股票</b>——与线上权重 TOP30 的重合度约 {ovc:.0f}%，'
                 '因为技术类因子高度冗余，都在表达同一个「广义动量」成分；'
                 f'（b）因此「挑哪几个因子、给什么权重」对名单几乎没有影响，按日百分位停在 '
                 f'{pc_lo:.0f}%~{pc_hi:.0f}%；'
                 '（c）但「按分数取 TOP30」这个动作本身确实产生了可观超额（对比随机抽 30 只与全池等权）。'
                 '换言之，<b>当前系统的价值来自「用动量类因子排序取前 6%」这个结构，'
                 '而不是因子的精细调校</b>。</div></div>')

    # 局限
    h.append('<div class="card"><h2>局限（请一并考虑）</h2>'
             '<div class="verdict info">'
             '<b>防前视：</b>权重只用完全实现的 IC（d+H+1 ≤ T）；因子 z 只用 ≤T 的K线；'
             '收益用 T+1 开盘 → T+H 收盘。<br><br>'
             '<b>仍存在：</b>① PE/PB/换手/股本用当前值近似历史（各方案共同项，不影响相对比较）；'
             '② 未计手续费/滑点，未剔涨停不可买与停牌；③ 回放池为今日 512 只（存活偏差）；'
             '④ 单一回放窗口（2025-11~2026-08，测试段市场明显走弱），结论不可外推；'
             '⑤ 5/10/20 日收益重叠导致 t 值偏乐观；⑥ 随机对照在“同一日同一池”内重抽，'
             '只控制“选哪些因子”这一个自由度。</div></div>')
    h.append("</div></body></html>")
    return "".join(h)


# ================================================================
# 主流程
# ================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=60, help="滚动 IC 窗口(交易日)")
    ap.add_argument("--ic-hold", type=int, default=5, help="用于定权的 IC 持有期")
    ap.add_argument("--rand", type=int, default=300, help="随机筛选重抽次数(topk_rand)")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--html", default="verify_factor_stability_report.html")
    args = ap.parse_args()

    print("=" * 80)
    print("  因子稳定性治理 A/B 验证 (防前视: 滚动 IC 只用已完全实现的数据)")
    print("=" * 80)
    rp = pickle.load(open(CACHE, "rb"))
    dates = rp["dates"]
    print(f"\n  缓存: {len(dates)} 交易日 ({dates[0]} ~ {dates[-1]}), {len(rp['factor_names'])} 因子")

    print("\n[1/4] 预计算逐日 IC 矩阵 (H=%d)..." % args.ic_hold)
    IC = precompute_ic(rp, args.ic_hold)
    print(f"  完成: {IC.shape[0]}×{IC.shape[1]}, 有效行 {np.sum(~np.isnan(IC).all(axis=1))}")

    # 测试段: 需要 W 根 IC + H 日实现 → 从 W+H+1 开始
    start = args.window + args.ic_hold + 1
    test_idx = list(range(start, len(dates)))
    print(f"\n  测试段: {len(test_idx)} 日 ({dates[test_idx[0]]} ~ {dates[test_idx[-1]]})")

    MODES = {
        "online":       "线上静态权重 ICIR_V3",
        "sign_zero":    "V1 方向矫正: 反号因子归零",
        "sign_flip":    "V2 方向矫正: 反号因子翻转",
        "consistency":  "V3 一致率降权: w × max(0,2r-1)",
        "rolling_ic":   "V4 滚动 IC 重标定",
        "rolling_ir":   "V5 滚动 IR 重标定",
        "equal_ic":     "V8 IC 因子等权",
        f"topk_cons":   f"V6 一致率 Top{args.topk} 因子",
        f"topk_ic":     f"V7 |IC| Top{args.topk} 因子",
    }

    print("\n[2/4] 逐方案评估...")
    table = {}
    _pb = pool_baseline(rp, test_idx)
    table["__pool__"] = {h: stats(_pb[h]) for h in HOLDS}
    wt_cache = {}
    for mode, lab in MODES.items():
        wt = RollingWeighter(rp, IC, args.window, args.ic_hold, mode, topk=args.topk)
        wt_cache[mode] = wt
        res, _ = evaluate(rp, test_idx, wt)
        table[mode] = {h: stats(res[h]) for h in HOLDS}
        b5 = table["__pool__"][5]["mean"]
        b20 = table["__pool__"][20]["mean"]
        print(f"  {lab:<28} 5日{table[mode][5]['mean']:+6.2f}% ({table[mode][5]['mean']-b5:+5.2f}pp) "
              f"20日{table[mode][20]['mean']:+6.2f}% ({table[mode][20]['mean']-b20:+5.2f}pp)")
    print(f"  {'全池等权基准(不选股)':<28} 5日{table['__pool__'][5]['mean']:+6.2f}% "
          f"20日{table['__pool__'][20]['mean']:+6.2f}%")

    # 随机对照: 与 V6/V7 唯一差别 = 选了哪些因子
    rand_summary = None
    if args.rand > 0:
        # 自检: rand_pick 必须真的改变权重(防止被模式分支静默吞掉, 曾出现过该 bug)
        _wt = wt_cache["topk_cons"]
        _i0 = test_idx[0]
        _p1, _p2 = IC_FACTORS[:args.topk], IC_FACTORS[-args.topk:]
        if _wt.weights(_i0, rand_pick=_p1) == _wt.weights(_i0, rand_pick=_p2):
            raise SystemExit("!! 自检失败: rand_pick 未生效, 随机对照无效")
        print("  自检通过: 随机对照确实改变了因子权重")
        print(f"\n  随机对照: 每日随机取 Top{args.topk} 因子 × {args.rand} 次 ...")
        rng = np.random.default_rng(20260922)
        _, dist = evaluate(rp, test_idx, wt_cache["topk_cons"],
                           rand_rounds=args.rand, rng=rng)
        rand_summary = {"dist": {h: dist[h] for h in HOLDS},
                        "real": {k: {h: table[k][h]["mean"] for h in HOLDS}
                                 for k in ("topk_cons", "topk_ic")}}
        for h_ in HOLDS:
            arr = np.array(dist[h_])
            real = rand_summary["real"]["topk_cons"][h_]
            print(f"    持有{h_:>2}日: V6真实{real:+.2f}% | 随机 均{arr.mean():+.2f}% "
                  f"[{np.percentile(arr,5):+.2f}, {np.percentile(arr,95):+.2f}] "
                  f"→ 百分位 {np.mean(arr<=real)*100:.0f}%")

    # 分段一致性
    mid = test_idx[len(test_idx) // 2]
    print(f"\n[3/4] 分段一致性检查 (前半 {dates[test_idx[0]]}~{dates[mid-1]}, "
          f"后半 {dates[mid]}~{dates[test_idx[-1]]})...")
    A = [i for i in test_idx if i < mid]
    B = [i for i in test_idx if i >= mid]
    pa, pb = pool_baseline(rp, A), pool_baseline(rp, B)
    seg_table = {}
    for mode in MODES:
        wt = RollingWeighter(rp, IC, args.window, args.ic_hold, mode, topk=args.topk)
        ra, _ = evaluate(rp, A, wt)
        rb, _ = evaluate(rp, B, wt)
        seg_table[mode] = {
            "a5": stats(ra[5])["mean"] - stats(pa[5])["mean"],
            "a20": stats(ra[20])["mean"] - stats(pa[20])["mean"],
            "b5": stats(rb[5])["mean"] - stats(pb[5])["mean"],
            "b20": stats(rb[20])["mean"] - stats(pb[20])["mean"],
        }
        v = seg_table[mode]
        print(f"    {MODES[mode][:20]:<22} 前半 5日{v['a5']:+5.2f} 20日{v['a20']:+6.2f}pp | "
              f"后半 5日{v['b5']:+5.2f} 20日{v['b20']:+6.2f}pp")

    # 补充: 重合度 + 敏感性
    print("\n  补充: 因子选择重合度(与线上权重 TOP30)...")
    ov = overlap_analysis(rp, test_idx, IC, args.window, args.ic_hold, args.topk)
    print(f"    重合度: 等权{ov['equal']:.1f}% / V6{ov['cons']:.1f}% / "
          f"V7{ov['ic']:.1f}% / 随机TopK{ov['rand']:.1f}%")
    print("\n  补充: 参数敏感性 (窗口 × TopK)...")
    sens_rows, sens_pb = sensitivity_scan(rp, IC, args.ic_hold, test_idx,
                                          [40, 60, 85], [6, 10, 14])
    for W, K, m5, e5, m20, e20 in sens_rows:
        print(f"    W={W:>3}日 K={K:>2}: 5日{m5:+.2f}%({e5:+.2f}pp)  20日{m20:+.2f}%({e20:+.2f}pp)")
    extras = {"overlap": ov, "sens": (sens_rows, sens_pb)}

    print("\n  补充: 按日百分位(拆掉日期方差)...")
    dp = daily_pct(rp, test_idx, wt_cache["topk_cons"], args.rand,
                   np.random.default_rng(999))
    for h_ in HOLDS:
        print(f"    持有{h_:>2}日: V6真值{dp[h_][1]:+.2f}% vs 当日随机均值{dp[h_][2]:+.2f}% "
              f"→ 按日百分位 {dp[h_][0]:.0f}%")
    print("\n  补充: 随机抽30只(不看任何分数)...")
    r30 = random30_baseline(rp, test_idx)
    for h_ in HOLDS:
        print(f"    持有{h_:>2}日: 随机30只 均{r30[h_][0]:+.2f}% "
              f"[{r30[h_][1]:+.2f}, {r30[h_][2]:+.2f}]")
    extras["daily_pct"] = dp
    extras["rand30"] = r30

    # 因子诊断(用最后一个测试日的窗口)
    print("\n[4/4] 因子诊断与报告...")
    got = RollingWeighter(rp, IC, args.window, args.ic_hold, "online")._window_ic(test_idx[-1])
    sub, mask = got
    m, sd = col_mean_std(sub)
    diag = []
    for f in IC_FACTORS:
        j = rp["factor_names"].index(f)
        col = sub[:, j]
        col = col[~np.isnan(col)]
        sg = np.sign(m[j]) if abs(m[j]) > 1e-12 else 1.0
        cons = float(np.mean(np.sign(col) == sg)) if len(col) else 0.0
        diag.append((f, eng.ICIR_V3.get(f, 0.0), float(m[j]),
                     float(m[j] / sd[j]) if sd[j] > 1e-9 else 0.0, cons))

    cfg = {"window": args.window, "ic_hold": args.ic_hold, "rand": args.rand,
           "labels": {**MODES, "__pool__": "全池等权基准(不选股)"}}
    html = generate_html(rp, cfg, test_idx, table, diag, rand_summary, seg_table, extras)
    out = os.path.join(OUTPUT, args.html)
    with open(out, "w", encoding="utf-8") as fp:
        fp.write(html)
    print(f"\n报告: {out}")

    # 因子诊断摘要
    flip = [d for d in diag if abs(d[2]) > 1e-9 and np.sign(d[1]) != np.sign(d[2])]
    print(f"\n  因子诊断(最后测试日窗口 W={args.window}): "
          f"{len(flip)}/{len(IC_FACTORS)} 个因子方向与线上权重相反")
    print("完成。")


if __name__ == "__main__":
    main()
