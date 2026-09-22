#!/usr/bin/env python3
"""
统一评分引擎 v2.0
================
整合 ISIR / GLM / 豆包 三套评分，每日生成综合决策报告。

v2.0 新增：
  - 点击展开查看因子评分明细 (各因子加权贡献+SS评分拆解)
  - 共识标记增强 (ISIR∩GLM∩豆包高亮)
  - 5日/10日/20日涨跌列
  - 纵览视图 + 按板块视图
  - 每个策略独立的买入/卖出信号追踪 + 累积收益计算

用法: python3 unified_scoring_engine.py
"""

import os, json, math, time, sys
from datetime import datetime, timedelta
from collections import defaultdict, OrderedDict
import numpy as np

from stock_db import StockDB
import sector_map

# ================================================================
# Configuration
# ================================================================

SELF_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SELF_DIR, "output")
STOCK_CODES_FILE = os.path.join(SELF_DIR, "stock_codes.txt")
os.makedirs(OUTPUT_DIR, exist_ok=True)

HISTORY_FILE = os.path.join(OUTPUT_DIR, "unified_rank_history.json")
TRADE_FILE = os.path.join(OUTPUT_DIR, "unified_trade_ledger.json")
SIGNAL_FILE = os.path.join(OUTPUT_DIR, "unified_signal_history.json")
HISTORY_DAYS = 5
TOP_N = 30
BUY_COUNT = TOP_N
HOLD_DAYS = 10  # 固定持仓周期
STOP_LOSS_PCT = -8.0   # 止损线
TAKE_PROFIT_PCT = 15.0  # 止盈线
MAX_POSITIONS = 30      # 最大持仓数

# 第五视图: 质量反弹 = 超跌反弹(第四体系) ∩ (ISIR或GLM排名≤QR_THRESHOLD)
# 回测依据(2026-09-14, 180日): 交叉组20日胜率55.5%/均收益+6.4% vs 纯超跌49.8%/+2.2% vs 全池48.1%/+2.1%
QR_THRESHOLD = 150
QR_BACKTEST_FILE = os.path.join(OUTPUT_DIR, "quality_rebound_backtest.json")

# ================================================================
# ICIR Weights — 精确来自 scoring_engine_icir.py / v3_vs_glm_tracker.py
# ================================================================

# ICIR v3.1 — 2026-08-04 重标定
# 有回测数据的因子用 Spearman rank IC 做权重 (85日5日前向收益)
# 无回测数据的因子 (IC=0) 保留原始手调权重
ICIR_V3 = {
    # --- 有IC数据: 使用标定值 ---
    "sector_rsi": 0.0956,      # IC最高, 正天数72%, 原0.015→大幅提升
    "dev_ma20": 0.0897,        # 原0.024→提升
    "vwap_premium": 0.0889,    # 原0.022→提升
    "macd_signal": 0.0734,    # 原0.026→提升
    "sector_momentum": 0.0688, # 原0.012→提升
    "ret_20d": 0.0681,        # 原0.004→大幅提升
    "rsi_signal": 0.0652,    # 原0.028→提升
    "ma_bull": 0.0559,       # 原0.029→提升
    "ret_5d": 0.0554,        # 原0.021→提升
    "max_dd_20d": 0.042,     # 原0.052→略降
    "mfi": 0.0399,           # 原0.153→大幅下降
    "gap_open": 0.0328,      # 原0.072→下降
    "vol_price": 0.0277,     # 原0.025→略升
    "pct_52w": 0.0248,       # 原0.091→大幅下降
    "cmf": 0.0189,           # 原0.025→略降
    "streak": 0.0123,        # 原0.020→略降
    "volatility_20d": 0.0079, # 原0.003→略升
    "amplitude_z": -0.0221,   # 原0.005→方向反转! 高振幅不利
    "vol_ratio_5d": -0.0142,  # 原0.023→方向反转! 高量比不利
    # --- 无IC数据(回测中缺行情/资金流): 保留原始权重 ---
    "turnover_z": 0.451,
    "log_mcap": 0.162,
    "pe_percentile": 0.078,
    "pb_percentile": 0.078,
    "event_score": 0.020,
    "event_count": 0.018,
    "inflow_rate": 0.010,
    "main_flow_5d": 0.008,
    "main_flow_20d": 0.006,
    "roe_rank": 0.001,
    "gross_margin_rank": 0.001,
    "ocf_ratio_rank": 0.001,
}

ICIR_GLM = dict(ICIR_V3)
ICIR_GLM["mfi"] = -ICIR_V3["mfi"]      # GLM 反转 mfi
ICIR_GLM["pct_52w"] = -ICIR_V3["pct_52w"]  # GLM 反转 pct_52w

FACTOR_HIGHER_BETTER = {
    "turnover_z": True, "log_mcap": True, "mfi": True, "pct_52w": True,
    "pe_percentile": False, "pb_percentile": False, "gap_open": True,
    "max_dd_20d": True, "ma_trend": True, "rsi_signal": True,
    "macd_signal": True, "cmf": True, "vol_price": True,
    "vol_ratio_5d": True, "ret_5d": True, "ret_20d": True,
    "dev_ma20": True, "streak": True, "vol_up_days": True,
    "amplitude_z": True, "vwap_premium": True,
    "main_flow_5d": True, "main_flow_20d": True, "inflow_rate": True,
    "ma_bull": True, "sector_rsi": True, "sector_momentum": True,
    "event_score": True, "event_count": True,
    "roe_rank": True, "gross_margin_rank": True, "ocf_ratio_rank": True,
    "volatility_20d": False,
}

FACTOR_LABELS = {
    "turnover_z":"换手率异常","log_mcap":"市值规模","mfi":"MFI资金流","pct_52w":"52周位置",
    "pe_percentile":"PE分位","pb_percentile":"PB分位","gap_open":"跳空缺口",
    "max_dd_20d":"20日回撤","ma_trend":"均线趋势","rsi_signal":"RSI信号",
    "macd_signal":"MACD","cmf":"CMF资金流","vol_price":"量价共振",
    "vol_ratio_5d":"5日量比","ret_5d":"5日涨幅","ret_20d":"20日涨幅",
    "dev_ma20":"MA20偏离","streak":"连涨天数","vol_up_days":"放量天数",
    "amplitude_z":"振幅","vwap_premium":"VWAP溢价",
    "main_flow_5d":"5日主力净流","main_flow_20d":"20日主力净流","inflow_rate":"主力流入占比",
    "ma_bull":"多头排列","sector_rsi":"行业RSI","sector_momentum":"板块动量",
    "event_score":"事件评分","event_count":"事件数量",
    "roe_rank":"ROE排名","gross_margin_rank":"毛利率排名","ocf_ratio_rank":"经营现金流排名",
    "volatility_20d":"20日波动",
}

# ================================================================
# Market Trend Gate — 指数 vs 50日均线, 熊市自动减仓
# ================================================================

def fetch_index_klines(days=300):
    """从腾讯API获取上证综指K线 (指数不支持fqkline, 用kline端点)"""
    import urllib.request
    sym = "sh000001"
    url = f"https://web.ifzq.gtimg.cn/appstock/app/kline/kline?param={sym},day,,,{days},"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        resp = urllib.request.urlopen(req, timeout=15)
        data = json.loads(resp.read().decode("utf-8"))
        kls = data.get("data", {}).get(sym, {}).get("day", [])
        parsed = []
        for k in kls:
            try:
                parsed.append({"date": k[0], "close": float(k[2]), "high": float(k[3]), "low": float(k[4])})
            except (IndexError, ValueError, TypeError):
                continue
        return parsed
    except Exception:
        return []

def calc_market_trend(index_klines):
    """计算市场趋势: 指数收盘 vs 50日均线
    返回: (trend: 1=多头/-1=空头, ma50, close, label)
    """
    if not index_klines or len(index_klines) < 50:
        return 1, 0, 0, "数据不足(默认多头)"
    closes = [b["close"] for b in index_klines]
    ma50 = sum(closes[-50:]) / 50
    close = closes[-1]
    if close > ma50:
        pct = (close / ma50 - 1) * 100
        return 1, round(ma50, 1), round(close, 1), f"多头(指数{pct:+.1f}%>MA50)"
    else:
        pct = (close / ma50 - 1) * 100
        return -1, round(ma50, 1), round(close, 1), f"空头(指数{pct:+.1f}%<MA50)"

# ================================================================
# VCP (波动收缩形态) 检测 — Minervini
# ================================================================

VCP_SWING_WINDOW = 3
VCP_CONTRACTION_RATIO = 0.85
VCP_T1_MIN = 0.03
VCP_T1_MAX = 0.50
VCP_MIN_CONTRACTIONS = 2
VCP_LOOKBACK = 200

def _find_swings(highs, lows, window=3):
    """识别 Swing High 和 Swing Low"""
    n = len(highs)
    swings = []
    for i in range(window, n - window):
        is_high = all(highs[i] >= highs[i + j] for j in range(-window, window + 1) if j != 0)
        if is_high:
            swings.append((i, "H", highs[i]))
        is_low = all(lows[i] <= lows[i + j] for j in range(-window, window + 1) if j != 0)
        if is_low:
            swings.append((i, "L", lows[i]))
    swings.sort(key=lambda x: x[0])
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

def _detect_vcp_pattern(swings, closes, volumes):
    """检测 VCP 收缩形态, 返回 {is_vcp, pivot, breakout, dry_up, quality, contractions}"""
    result = {"is_vcp": False, "pivot": 0, "breakout": False, "dry_up": 1.0, "quality": 0, "n_contractions": 0}
    if len(swings) < 5:
        return result

    # 提取 H-L 对
    pairs = []
    i = 0
    while i < len(swings) - 1:
        if swings[i][1] == "H":
            j = i + 1
            while j < len(swings) and swings[j][1] != "L":
                j += 1
            if j < len(swings) and swings[i][2] > 0 and swings[j][2] < swings[i][2]:
                depth = (swings[i][2] - swings[j][2]) / swings[i][2]
                pairs.append({"depth": depth, "high": swings[i][2], "low": swings[j][2]})
            i = j + 1
        else:
            i += 1

    if len(pairs) < VCP_MIN_CONTRACTIONS:
        return result

    # pivot = 最近的 swing high
    pivot = None
    for s in reversed(swings):
        if s[1] == "H":
            pivot = s[2]
            break
    if pivot is None:
        return result

    # 从末尾往前找最长的逐次收紧子序列
    current_seq = [pairs[-1]]
    for i in range(len(pairs) - 2, -1, -1):
        prev_depth = current_seq[0]["depth"]
        curr_depth = pairs[i]["depth"]
        if curr_depth > 0 and curr_depth >= prev_depth * 0.5:
            if curr_depth <= prev_depth / VCP_CONTRACTION_RATIO + 0.01:
                current_seq.insert(0, pairs[i])
            else:
                break
        else:
            break

    if len(current_seq) < VCP_MIN_CONTRACTIONS:
        if len(pairs) >= 2 and pairs[-1]["depth"] < pairs[-2]["depth"] * VCP_CONTRACTION_RATIO:
            current_seq = pairs[-2:]
        else:
            return result

    t1 = current_seq[0]["depth"]
    if t1 < VCP_T1_MIN or t1 > VCP_T1_MAX:
        return result

    # 量能枯竭
    n = len(volumes)
    vol_50d = np.mean(volumes[-50:]) if n >= 50 else np.mean(volumes)
    vol_10d = np.mean(volumes[-10:]) if n >= 10 else np.mean(volumes)
    dry_up = vol_10d / vol_50d if vol_50d > 0 else 1.0

    # 突破检测
    breakout = closes[-1] > pivot
    breakout_vol = volumes[-1] / vol_50d if vol_50d > 0 else 0

    # 质量评分
    score = min(len(current_seq) * 10, 40)
    score += max(0, 20 - current_seq[-1]["depth"] * 100)
    if dry_up < 0.30: score += 20
    elif dry_up < 0.50: score += 15
    elif dry_up < 0.70: score += 10
    if breakout and breakout_vol >= 1.5: score += 20
    elif breakout: score += 10

    result["is_vcp"] = True
    result["pivot"] = round(pivot, 2)
    result["breakout"] = breakout
    result["dry_up"] = round(dry_up, 3)
    result["quality"] = round(score, 1)
    result["n_contractions"] = len(current_seq)
    result["breakout_vol"] = round(breakout_vol, 2)
    return result

def scan_vcp_signals(klines):
    """扫描所有股票的 VCP 形态, 返回 {code: vcp_info}"""
    vcp_map = {}
    stage2_count = 0
    for code, k in klines.items():
        if len(k) < VCP_LOOKBACK:
            continue
        closes = [b["close"] for b in k]
        highs = [b["high"] for b in k]
        lows = [b["low"] for b in k]
        volumes = [b["volume"] for b in k]

        # Stage 2 快速过滤
        n = len(closes)
        if n < 200:
            continue
        price = closes[-1]
        ma50 = np.mean(closes[-50:])
        ma150 = np.mean(closes[-150:])
        ma200 = np.mean(closes[-200:])
        if not (price > ma50 and ma150 > ma200 and price > ma200):
            continue
        high_52w = max(highs[-250:]) if n >= 250 else max(highs)
        low_52w = min(lows[-250:]) if n >= 250 else min(lows)
        # Stage2 趋势硬过滤: 距低点涨幅不足15%直接剔除(趋势未确立)
        if (price - low_52w) / low_52w < 0.15:
            continue
        # 距高点: ≤35%不扣分 | 35%-65%软扣分 | >65%硬剔除(偏离前高太远)
        dist_high = (high_52w - price) / high_52w if high_52w > 0 else 0
        if dist_high > 0.65:
            continue
        stage2_count += 1

        swings = _find_swings(highs, lows, VCP_SWING_WINDOW)
        vcp = _detect_vcp_pattern(swings, closes, volumes)
        if vcp["is_vcp"]:
            # 距高点软扣分: 35%-50%扣0-10分, 50%-65%扣10-20分
            if dist_high > 0.35:
                penalty = min(20, (dist_high - 0.35) / 0.30 * 20)
                vcp["quality"] = round(max(0, vcp["quality"] - penalty), 1)
                vcp["dist_high_penalty"] = round(penalty, 1)
            vcp["dist_high"] = round(dist_high, 3)
            vcp_map[code] = vcp

    breakout_count = sum(1 for v in vcp_map.values() if v["breakout"])
    print(f"  Stage2通过: {stage2_count}只 | VCP形态: {len(vcp_map)}只 | 突破确认: {breakout_count}只")
    return vcp_map

# ================================================================
# 外围市场门控 — 美股隔夜 + VIX
# ================================================================

def fetch_global_markets():
    """获取外围市场数据 (美股指数+VIX, Yahoo Finance)"""
    import urllib.request as ur
    result = {}
    sources = {
        "^NDX": ("纳斯达克100", "科技板块情绪"),
        "^DJI": ("道琼斯", "整体风险偏好"),
        "^VIX": ("VIX恐慌指数", "市场恐慌度"),
        "^N225": ("日经225", "亚太早盘风向"),
        "^KS11": ("韩国KOSPI", "亚太半导体风向"),
    }
    for symbol, (name, impact) in sources.items():
        try:
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?range=5d&interval=1d"
            req = ur.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            resp = ur.urlopen(req, timeout=10)
            data = json.loads(resp.read().decode("utf-8"))
            result_data = data.get("chart", {}).get("result", [{}])[0]
            quotes = result_data.get("indicators", {}).get("quote", [{}])[0]
            closes = [c for c in quotes.get("close", []) if c is not None]
            if len(closes) >= 2:
                last_close = closes[-1]
                prev_close = closes[-2]
                ret = (last_close / prev_close - 1) * 100 if prev_close > 0 else 0
                if symbol == "^VIX":
                    gate = "danger" if last_close > 30 else ("warning" if last_close > 20 else "normal")
                else:
                    gate = "danger" if ret < -3 else ("warning" if ret < -2 else "normal")
                result[name] = {"close": round(last_close, 1), "overnight_ret": round(ret, 2), "impact": impact, "gate": gate}
        except Exception:
            result[name] = {"close": 0, "overnight_ret": 0, "impact": impact, "gate": "unknown"}
    return result

def compute_global_gate(global_markets):
    """外围门控: 0=正常 1=警戒 2=危险"""
    ndx_ret = global_markets.get("纳斯达克100", {}).get("overnight_ret", 0)
    dji_ret = global_markets.get("道琼斯", {}).get("overnight_ret", 0)
    vix_close = global_markets.get("VIX恐慌指数", {}).get("close", 0)

    level = 0
    pos_adjust = 1.0
    gates = []

    if vix_close > 30:
        level = max(level, 2); pos_adjust = min(pos_adjust, 0.5)
        gates.append(f"VIX={vix_close:.0f}>30 恐慌高位,仓位减半")
    elif vix_close > 20:
        level = max(level, 1); pos_adjust = min(pos_adjust, 0.75)
        gates.append(f"VIX={vix_close:.0f}>20 波动加剧")

    if ndx_ret < -3:
        level = max(level, 2); pos_adjust = min(pos_adjust, 0.0)
        gates.append(f"纳指{ndx_ret:+.1f}%<-3%暴跌,不开新仓")
    elif ndx_ret < -2:
        level = max(level, 1); pos_adjust = min(pos_adjust, 0.5)
        gates.append(f"纳指{ndx_ret:+.1f}%<-2%大跌,仓位减半")

    if dji_ret < -2:
        level = max(level, 1); pos_adjust = min(pos_adjust, 0.5)
        gates.append(f"道指{dji_ret:+.1f}%<-2%风险偏好下降")

    if level == 0:
        if ndx_ret > 1 and dji_ret > 1:
            interp = f"外围隔夜大涨(纳指{ndx_ret:+.1f}%,道指{dji_ret:+.1f}%),A股今日大概率高开,科技板块偏正面"
        elif ndx_ret > 0 and dji_ret > 0:
            interp = f"外围隔夜小涨(纳指{ndx_ret:+.1f}%,道指{dji_ret:+.1f}%),情绪中性偏正面"
        elif ndx_ret < -1 or dji_ret < -1:
            interp = f"外围隔夜下跌(纳指{ndx_ret:+.1f}%,道指{dji_ret:+.1f}%),A股可能低开,注意止损"
        else:
            interp = f"外围隔夜平稳(纳指{ndx_ret:+.1f}%,道指{dji_ret:+.1f}%),无明显方向信号"
    elif level == 1:
        interp = f"外围警戒: {'; '.join(gates)}. 建议缩减仓位,谨慎操作"
    else:
        interp = f"外围危险: {'; '.join(gates)}. 强烈防御,不开新仓,浮亏果断止损"

    vix_note = f"VIX={vix_close:.0f}" + ("(恐慌)" if vix_close > 25 else ("(正常)" if vix_close < 18 else "(偏高)"))
    label = f"外围门控: {'危险' if level==2 else '警戒' if level==1 else '正常'} | {vix_note}"
    return level, pos_adjust, label, interp

# ================================================================
# Technical Indicators
# ================================================================

def calc_ma(closes, period):
    if len(closes) < period: return closes[-1] if closes else 0
    return sum(closes[-period:]) / period

def calc_ema(closes, period):
    if len(closes) < 2: return closes[-1] if closes else 0
    k = 2 / (period + 1)
    ema = closes[0]
    for c in closes[1:]: ema = c * k + ema * (1 - k)
    return ema

def calc_rsi(closes, period=14):
    if len(closes) < period+1: return 50
    gains = sum(max(closes[i]-closes[i-1],0) for i in range(len(closes)-period,len(closes)))
    losses = sum(max(closes[i-1]-closes[i],0) for i in range(len(closes)-period,len(closes)))
    if losses == 0: return 100
    return 100 - 100 / (1 + gains / losses)

def calc_macd(closes, fast=12, slow=26, signal=9):
    if len(closes) < slow:
        return 0, 0, 0
    # 增量EMA计算，避免重复遍历
    kf = 2 / (fast + 1); ks = 2 / (slow + 1); kd = 2 / (signal + 1)
    ema_fast = closes[0]; ema_slow = closes[0]
    difs = []
    for i in range(1, len(closes)):
        ema_fast = closes[i] * kf + ema_fast * (1 - kf)
        if i >= slow - 1:
            ema_slow = closes[i] * ks + ema_slow * (1 - ks)
            difs.append(ema_fast - ema_slow)
    if not difs:
        return 0, 0, 0
    dif = difs[-1]
    dea = difs[0]
    for d in difs[1:]:
        dea = d * kd + dea * (1 - kd)
    return dif, dea, 2 * (dif - dea)

def calc_cmf(highs,lows,closes,volumes,period=20):
    if len(closes) < period: return 0
    mfv, tv = 0,0
    for i in range(len(closes)-period, len(closes)):
        h,l,c,v = highs[i],lows[i],closes[i],volumes[i]
        mfv += ((c-l)-(h-c))/(h-l)*v if h!=l else 0; tv += v
    return mfv/tv if tv>0 else 0

def calc_mfi(highs,lows,closes,volumes,period=14):
    if len(closes) < period+1: return 50
    pos, neg = 0,0
    for i in range(len(closes)-period, len(closes)):
        tp = (highs[i]+lows[i]+closes[i])/3
        tp_prev = (highs[i-1]+lows[i-1]+closes[i-1])/3
        mf = tp*volumes[i]
        if tp>tp_prev: pos+=mf
        elif tp<tp_prev: neg+=mf
    return 100 if neg==0 else 100-100/(1+pos/neg)

def calc_max_dd(closes, period=20):
    if len(closes) < period: return 0
    peak = closes[-period]; max_dd = 0
    for i in range(len(closes)-period, len(closes)):
        peak = max(peak, closes[i])
        max_dd = max(max_dd, (peak-closes[i])/peak)
    return -max_dd

# ================================================================
# Factor Computation
# ================================================================

def compute_all_factors(klines, extra_info, fund_flows, events, sectors):
    all_codes = sorted(klines.keys())
    factor_data = {}
    return_data = {}  # 5d/10d/20d returns

    # 预计算每个板块的聚合指标(避免O(K^2)重复计算)
    sector_aggs = {}
    for sector_name in set(sectors.values()):
        s_codes = [c for c in all_codes if sectors.get(c) == sector_name and c in klines and len(klines[c]) >= 15]
        if not s_codes: continue
        s_rsis = []; s_moms = []
        for sc in s_codes:
            sc_c = [b["close"] for b in klines[sc][-15:]]
            if len(sc_c) >= 15:
                sg = sum(max(sc_c[i]-sc_c[i-1],0) for i in range(1,15))
                sl = sum(max(sc_c[i-1]-sc_c[i],0) for i in range(1,15))
                s_rsis.append(100 if sl==0 and sg>0 else (50 if sl==0 else 100-100/(1+sg/sl)))
            sc_c6 = [b["close"] for b in klines[sc][-6:]]
            if len(sc_c6) >= 6 and sc_c6[0] > 0:
                s_moms.append((sc_c6[-1]/sc_c6[0]-1)*100)
        sector_aggs[sector_name] = {
            "rsi": (sum(s_rsis)/len(s_rsis) - 50) if s_rsis else 0,
            "momentum": sum(s_moms)/len(s_moms) if s_moms else 0,
        }

    for code in all_codes:
        k = klines.get(code, [])
        if len(k) < 60: continue
        closes = [bar["close"] for bar in k]
        highs = [bar["high"] for bar in k]
        lows = [bar["low"] for bar in k]
        volumes = [bar["volume"] for bar in k]
        opens = [bar["open"] for bar in k]

        close = closes[-1] if closes else 0
        extra = extra_info.get(code, {})

        # 多周期涨跌
        ret_5d = (closes[-1]/closes[-6]-1)*100 if len(closes)>=6 and closes[-6]>0 else 0
        ret_10d = (closes[-1]/closes[-11]-1)*100 if len(closes)>=11 and closes[-11]>0 else 0
        ret_20d_pct = (closes[-1]/closes[-21]-1)*100 if len(closes)>=21 and closes[-21]>0 else 0
        return_data[code] = {"ret_5d":ret_5d,"ret_10d":ret_10d,"ret_20d":ret_20d_pct}

        ma5,ma10,ma20 = calc_ma(closes,5), calc_ma(closes,10), calc_ma(closes,20)
        ma_trend = ((ma5/ma10-1)+(ma10/ma20-1))*100 if ma10 and ma20 else 0
        ma_bull = 1.0 if (ma5>ma10>ma20) else (0.5 if ma5>ma10 else -1.0)
        rsi_val = calc_rsi(closes,14)
        rsi_signal = (rsi_val-50)/15.0
        dif,dea,bar = calc_macd(closes)
        macd_signal = (dif-dea)/close*1000 if close else 0
        vol_ratio = extra.get("vol_ratio",1) or 1
        chg_pct = (close/closes[-2]-1)*100 if len(closes)>=2 else 0
        vol_price = vol_ratio * (1 if chg_pct>0 else -1) * min(abs(chg_pct),10)/10
        dev_ma20 = (close/ma20-1)*100 if ma20 else 0
        high_52w = max(highs[-250:]) if len(highs)>=20 else max(highs)
        low_52w = min(lows[-250:]) if len(lows)>=20 else min(lows)
        pct_52w = (close-low_52w)/(high_52w-low_52w)*100 if high_52w!=low_52w else 50

        avg_vol_5 = sum(volumes[-6:-1])/5 if len(volumes)>=6 else volumes[-1]
        vol_ratio_5d = volumes[-1]/avg_vol_5 if avg_vol_5>0 else 1

        streak = 0
        for i in range(len(closes)-1,0,-1):
            if closes[i]>closes[i-1]: streak+=1
            else: break
        streak = min(streak, 10)
        streak_dn = 0
        for i in range(len(closes)-1,0,-1):
            if closes[i]<closes[i-1]: streak_dn+=1
            else: break
        streak_dn = min(streak_dn, 10)

        gap_open = (opens[-1]/closes[-2]-1)*100 if len(closes)>=2 and closes[-2] else 0
        turnover_z = extra.get("turnover",0) or 0
        amplitude_z = (highs[-1]/lows[-1]-1)*100 if lows[-1] else 0

        cmf_val = calc_cmf(highs,lows,closes,volumes,20)
        mfi_val = calc_mfi(highs,lows,closes,volumes,14)
        vwap20 = sum(closes[i]*volumes[i] for i in range(len(closes)-20,len(closes))) / max(sum(volumes[-20:]),1) if len(closes)>=20 else close
        vwap_premium = (close/vwap20-1)*100 if vwap20 else 0

        vol_up_days = sum(1 for i in range(-5, 0) if abs(i)<len(volumes) and abs(i-1)<len(volumes) and volumes[i]>volumes[i-1])/5.0 if len(volumes)>=7 else 0.5

        flow = fund_flows.get(code,{})
        main_flow_5d = flow.get("main_net_5d",0) or 0
        main_flow_20d = flow.get("main_net_20d",0) or 0
        inflow_rate = flow.get("inflow_rate",0) or 0

        pe = extra.get("pe_ttm",0) or 0; pb = extra.get("pb",0) or 0
        mcap = extra.get("mcap",0) or 0
        # PE/PB分位: 用真实PE/PB，低值更好
        pe_percentile = pe if pe > 0 else 50  # 无PE时给中性值50
        pb_percentile = pb if pb > 0 else 50
        log_mcap = math.log(max(mcap,1e8))

        sector = sectors.get(code,"其他")
        # 从预计算的板块聚合数据中取值
        sector_agg = sector_aggs.get(sector, {})
        sector_rsi = sector_agg.get("rsi", 0)
        sector_momentum = sector_agg.get("momentum", 0)

        evts = events.get(code,[])
        event_score = sum(e.get("base_score",0) for e in evts if e.get("base_score"))
        event_count = len(evts)

        if len(closes)>=21:
            dr = [(closes[i]/closes[i-1]-1)*100 for i in range(len(closes)-20,len(closes))]
            volatility_20d = float(np.std(dr)) if dr else 0
        else:
            volatility_20d = 0
        max_dd_20d = calc_max_dd(closes,20)

        factor_data[code] = {
            "ma_trend":ma_trend,"ma_bull":ma_bull,"rsi_signal":rsi_signal,
            "macd_signal":macd_signal,"vol_price":vol_price,"dev_ma20":dev_ma20,
            "pct_52w":pct_52w,"vol_ratio_5d":vol_ratio_5d,"ret_5d":ret_5d,
            "ret_20d":ret_20d_pct,"streak":streak,"gap_open":gap_open,
            "turnover_z":turnover_z,"amplitude_z":amplitude_z,"cmf":cmf_val,
            "mfi":mfi_val,"vwap_premium":vwap_premium,"vol_up_days":vol_up_days,
            "main_flow_5d":main_flow_5d,"main_flow_20d":main_flow_20d,
            "inflow_rate":inflow_rate,"pe_percentile":pe_percentile,
            "pb_percentile":pb_percentile,"log_mcap":log_mcap,
            "roe_rank":50,"gross_margin_rank":50,"ocf_ratio_rank":50,
            "sector_rsi":sector_rsi,"sector_momentum":sector_momentum,
            "event_score":event_score,"event_count":event_count,
            "volatility_20d":volatility_20d,"max_dd_20d":max_dd_20d,
            # 原始指标值 (用于详情解读)
            "_close":close,"_ma5":ma5,"_ma10":ma10,"_ma20":ma20,
            "_rsi":rsi_val,"_dif":dif,"_dea":dea,"_vol_ratio":vol_ratio,
            "_cmf":cmf_val,"_main5d":main_flow_5d,"_turnover":turnover_z,
            "_streak_dn":streak_dn,
        }

    # 截面标准化
    factor_names = sorted(ICIR_V3.keys())
    for fname in factor_names:
        values = [factor_data[c].get(fname,0) for c in factor_data]
        arr = np.array(values,dtype=float)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)  # 防止NaN传播
        arr = np.clip(arr,np.percentile(arr,1),np.percentile(arr,99))
        mean,std = np.mean(arr),np.std(arr)
        if std == 0 or np.isnan(std) or std < 1e-10: continue
        z = (arr-mean)/std
        if not FACTOR_HIGHER_BETTER.get(fname,True): z = -z
        for ci,code in enumerate(factor_data.keys()):
            factor_data[code][f"{fname}_z"] = float(z[ci])

    return factor_data, return_data

# ================================================================
# Ranking + Signal Generation
# ================================================================

def compute_rankings(factor_data):
    factor_names = sorted(ICIR_V3.keys())
    results = []

    for code, factors in factor_data.items():
        isir_score = sum(ICIR_V3.get(f,0)*factors.get(f"{f}_z",0) for f in factor_names)
        glm_score = sum(ICIR_GLM.get(f,0)*factors.get(f"{f}_z",0) for f in factor_names)

        # SS评分 — 完整恢复 stock-scoring 原始算法 (V9 回测驱动权重)
        # 技术35% + 资金55% + 信息5% + 事件5%
        # 修复日期: 2026-08-05 (此前迁移时丢失80%指标导致严重撞分)
        close = factors.get("_close",0)
        ma5_v = factors.get("_ma5",0); ma10_v = factors.get("_ma10",0); ma20_v = factors.get("_ma20",0)
        rsi_val = factors.get("_rsi",50)
        dif = factors.get("_dif",0); dea = factors.get("_dea",0)
        vol_ratio = factors.get("_vol_ratio",1)
        vol_up_days = factors.get("vol_up_days",0.5)
        pct_52w = factors.get("pct_52w",50)
        dev_ma20 = factors.get("dev_ma20",0)
        gap_open = factors.get("gap_open",0)
        streak = factors.get("streak",0)
        cmf_raw = factors.get("cmf",0)
        mfi_val = factors.get("mfi",50)
        vwap_premium = factors.get("vwap_premium",0)
        main_5d = factors.get("main_flow_5d",0)
        main_20d = factors.get("main_flow_20d",0)
        inflow_rate = factors.get("inflow_rate",0)
        log_mcap = factors.get("log_mcap",25)
        amplitude_z = factors.get("amplitude_z",0)
        event_score_raw = factors.get("event_score",0)
        ret_5d = factors.get("ret_5d",0)
        price_up = close > ma5_v  # 用价格vs MA5近似今日方向(原始用c[-1]>c[-2])

        # ===== 技术面 (35%) — 6类指标 (原始stock-scoring完整版) =====
        tech_delta = 0
        # 1) 均线排列
        if ma5_v and ma10_v and ma20_v:
            if ma5_v > ma10_v > ma20_v:
                tech_delta += 15
            elif ma5_v < ma10_v < ma20_v:
                tech_delta -= 10
        # 2) MACD (与均线多头重叠时降权到2分, 避免双重计分)
        if dif and dea and dif > dea and dif > 0:
            if ma5_v and ma10_v and ma20_v and ma5_v > ma10_v > ma20_v:
                tech_delta += 2
            else:
                tech_delta += 5
        elif dif and dif < 0:
            tech_delta -= 3
        # 3) RSI四级细分
        if 40 <= rsi_val <= 55:
            tech_delta -= 3
        if rsi_val > 80:
            tech_delta += 12
        elif rsi_val > 75:
            tech_delta += 10
        elif rsi_val < 30:
            tech_delta -= 8
        # 4) 量价关系四级 (放量上涨+8/缩量下跌-10/放量回调+5/巨量下跌-8)
        if vol_ratio > 1.5 and price_up:
            tech_delta += 8
        elif not price_up:
            if vol_ratio < 0.7:
                tech_delta -= 10
            elif 1.1 <= vol_ratio <= 1.5:
                tech_delta += 5
            elif vol_ratio > 1.5:
                tech_delta -= 8
        # 5) MA20偏离度
        if ma20_v:
            if 2 < dev_ma20 < 8:
                tech_delta += 8
            elif dev_ma20 > 15:
                tech_delta -= 8
            elif -5 < dev_ma20 < -2:
                tech_delta -= 5
        # 6) 52周位置
        if pct_52w < 30:
            tech_delta -= 8
        elif pct_52w > 90:
            tech_delta -= 5
        tech_score = max(5, min(95, 50 + tech_delta))

        # ===== 资金面 (55%) — 8+类指标 (原始stock-scoring完整版) =====
        cap_delta = 0
        # 1) CMF (2档严格阈值0.15)
        if cmf_raw > 0.15:
            cap_delta += 12
        elif cmf_raw < -0.15:
            cap_delta -= 10
        # 2) MFI
        if mfi_val < 40:
            cap_delta -= 5
        # 3) VWAP20突破 (与均线多头重叠时降权)
        if vwap_premium > 3:
            if ma5_v and ma10_v and ma20_v and ma5_v > ma10_v > ma20_v:
                cap_delta += 1
            else:
                cap_delta += 3
        # 4) 持续放量 (近5日4天以上放量)
        if vol_up_days >= 0.8:
            cap_delta += 10
        # 5) 换手率异常 (量比20日)
        if vol_ratio > 3:
            if price_up:
                cap_delta -= 5  # 巨量突破(回测T+10夏普低)
            else:
                cap_delta -= 6  # 巨量出货
        elif vol_ratio > 2 and price_up:
            cap_delta += 4
        # 6) 振幅异常
        if amplitude_z > 5:
            if price_up:
                cap_delta += 5
            else:
                cap_delta -= 5
        # 7) 市值分层
        mcap_val = math.exp(log_mcap) if log_mcap else 0
        if mcap_val > 1e11:
            cap_delta += 8
        elif 0 < mcap_val < 5e9:
            cap_delta -= 8
        # 8) 主力资金净流向
        if main_5d > 0 and main_20d > 0:
            cap_delta += 8
        elif main_5d > 0 and inflow_rate > 0.5:
            cap_delta += 5
        if main_5d < 0 and main_20d < 0:
            cap_delta -= 8
        elif main_5d < 0:
            cap_delta -= 5
        capital_score = max(5, min(95, 50 + cap_delta))

        # ===== 信息面 (5%) — 5类指标 (原始stock-scoring完整版) =====
        info_delta = 0
        # 1) 3日涨跌幅 (用ret_5d近似)
        if ret_5d > 8:
            info_delta += 15
        elif ret_5d < -8:
            info_delta -= 12
        # 2) 跳空缺口
        if abs(gap_open) > 3:
            if gap_open > 0:
                info_delta += 15
            else:
                info_delta -= 5
        # 3) 量比
        if vol_ratio > 3:
            info_delta -= 3
        elif vol_ratio > 2:
            info_delta += 5
        # 4) 三连涨
        if streak >= 3:
            info_delta += 8
        # 5) 冲高回落封顶
        if gap_open > 2 and not price_up:
            info_delta = min(info_delta, 10)
        info_score = max(5, min(95, 50 + info_delta))

        # ===== 事件评分 (5%) — 衰减映射到0-50 =====
        event_norm = 25 + max(-25, min(25, event_score_raw))

        # ===== 最终加权 (V9: 技术35% + 资金55% + 信息5% + 事件5%) =====
        ss_score = tech_score * 0.35 + capital_score * 0.55 + info_score * 0.05 + event_norm * 0.05

        # Top factor contributions
        contributions = []
        for f in factor_names:
            z = factors.get(f"{f}_z",0)
            w_isir = ICIR_V3.get(f,0)
            contrib = w_isir * z
            contributions.append({"factor":f,"label":FACTOR_LABELS.get(f,f),"weight":w_isir,"z_score":round(z,3),"contribution":round(contrib,4)})
        contributions.sort(key=lambda x:abs(x["contribution"]),reverse=True)

        results.append({
            "code":code,"isir_score":round(isir_score,3),"glm_score":round(glm_score,3),
            "ss_score":round(ss_score,1),
            "ss_tech":round(tech_score,1),"ss_capital":round(capital_score,1),"ss_info":round(info_score,1),
            "ss_event":round(event_norm,1),
            "top_factors":contributions[:8],
            "indicator":{
                "close":factors.get("_close",0),"ma5":factors.get("_ma5",0),
                "ma10":factors.get("_ma10",0),"ma20":factors.get("_ma20",0),
                "rsi":factors.get("_rsi",50),"dif":factors.get("_dif",0),
                "dea":factors.get("_dea",0),"vol_ratio":factors.get("_vol_ratio",1),
                "cmf":factors.get("_cmf",0),"main5d":factors.get("_main5d",0),
                "turnover":factors.get("_turnover",0),
                "pct_52w":factors.get("pct_52w",50),"streak":factors.get("streak",0),
                "streak_dn":factors.get("_streak_dn",0),
                "event_score":factors.get("event_score",0),"event_count":factors.get("event_count",0),
            },
        })

    results.sort(key=lambda x:x["isir_score"],reverse=True)
    for i,r in enumerate(results): r["isir_rank"]=i+1
    results.sort(key=lambda x:x["glm_score"],reverse=True)
    for i,r in enumerate(results): r["glm_rank"]=i+1
    results.sort(key=lambda x:x["ss_score"],reverse=True)
    for i,r in enumerate(results): r["doubao_rank"]=i+1

    isir_top = set(r["code"] for r in results if r["isir_rank"]<=TOP_N)
    glm_top = set(r["code"] for r in results if r["glm_rank"]<=TOP_N)
    doubao_top = set(r["code"] for r in results if r["doubao_rank"]<=TOP_N)
    consensus = isir_top & glm_top & doubao_top

    for r in results:
        r["consensus"] = r["code"] in consensus
        r["in_isir_top"] = r["code"] in isir_top
        r["in_glm_top"] = r["code"] in glm_top
        r["in_doubao_top"] = r["code"] in doubao_top

    return results


# ================================================================
# 超跌反弹评分 (第四体系，独立标记，不参与三体系共识)
# ================================================================

def compute_rebound_scores(rankings, klines, extra_info):
    """
    对所有股票计算超跌反弹趋势评分，注入到rankings中。
    超跌反弹是逆向策略维度，与三体系(顺势)互补，独立标记不参与共识。

    评分 = 超跌(30) + 反弹(35) + 趋势确认(20) + 风控(15)
    硬性门槛: 超跌分≥5 且 总分≥20 才标记为有效超跌反弹信号
    """
    from oversold_rebound_engine import score_oversold_rebound_trend

    rebound_list = []
    for r in rankings:
        code = r["code"]
        kl = klines.get(code)
        if not kl or len(kl) < 35:
            r["rebound_score"] = 0
            r["rebound_stage"] = ""
            r["rebound_valid"] = False
            continue

        extra = extra_info.get(code, {})
        # 构造stock字典
        prev_close = kl[-2]["close"] if len(kl) >= 2 else (extra.get("price",0) or 0)
        price = extra.get("price",0) or kl[-1]["close"]
        stock = {
            "code": code,
            "name": extra.get("name", code),
            "price": price,
            "market": _get_market_name(code),
            "mcap": extra.get("mcap", 0) or 0,
            "amount": (kl[-1].get("volume",0) or 0) * price,
            "pct": (price / prev_close - 1) * 100 if prev_close > 0 else 0,
            "open": kl[-1].get("open", price),
            "high": kl[-1].get("high", price),
            "low": kl[-1].get("low", price),
            "prev_close": prev_close,
        }

        result = score_oversold_rebound_trend(stock, kl)
        if result and result["score"] >= 20 and result["oversold"] >= 5:
            r["rebound_score"] = result["score"]
            r["rebound_oversold"] = result["oversold"]
            r["rebound_signal"] = result["signal"]
            r["rebound_trend"] = result["trend"]
            r["rebound_risk"] = result["risk"]
            r["rebound_stage"] = result["rebound_stage"]
            r["rebound_valid"] = True
            r["rebound_atr"] = result["atr_advice"]
            r["rebound_indicators"] = result["indicators"]
            rebound_list.append(r)
        else:
            r["rebound_score"] = result["score"] if result else 0
            r["rebound_stage"] = result["rebound_stage"] if result else ""
            r["rebound_valid"] = False

    # 排名
    rebound_list.sort(key=lambda x: x["rebound_score"], reverse=True)
    for i, r in enumerate(rebound_list):
        r["rebound_rank"] = i + 1

    return len(rebound_list)


def _get_market_name(code):
    if code.startswith("30"): return "创业板"
    elif code.startswith("68"): return "科创板"
    elif code.startswith(("8", "4", "92")): return "北交所"
    elif code.startswith("6"): return "沪主板"
    return "深主板"


# ================================================================
# History & Trade Tracking
# ================================================================

def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE) as f: return json.load(f)
        except (json.JSONDecodeError, IOError):
            print(f"  [WARN] 历史文件损坏，重置")
    return {}

def save_history(rankings, extra_info, date_str):
    history = load_history()
    for r in rankings:
        code = r["code"]
        if code not in history: history[code] = {"isir":[],"glm":[],"doubao":[],"ss":[]}
        history[code]["name"] = extra_info.get(code,{}).get("name",code)
        for k in ["isir","glm","doubao"]:
            history[code][k].append({"date":date_str,"rank":r[f"{k}_rank"]})
            history[code][k] = history[code][k][-HISTORY_DAYS:]
        history[code]["ss"].append({"date":date_str,"score":r["ss_score"]})
        history[code]["ss"] = history[code]["ss"][-HISTORY_DAYS:]
    with open(HISTORY_FILE,"w") as f: json.dump(history,f,ensure_ascii=False)
    return history

def get_trend(code, rank_type, history):
    if code not in history: return 0, "→"
    records = history[code].get(rank_type,[])
    if len(records)<2: return 0, "→"
    diff = records[-2]["rank"] - records[-1]["rank"]
    if diff>10: return diff, "↑↑"
    elif diff>3: return diff, "↑"
    elif diff<-10: return diff, "↓↓"
    elif diff<-3: return diff, "↓"
    return diff, "→"

def get_ss_trend(code, history):
    if code not in history: return 0, "→"
    records = history[code].get("ss",[])
    if len(records)<2: return 0, "→"
    diff = records[-1]["score"]-records[-2]["score"]
    arrow = "↑" if diff>3 else ("↓" if diff<-3 else "→")
    return round(diff,1), arrow

def load_trades():
    if os.path.exists(TRADE_FILE):
        try:
            with open(TRADE_FILE) as f: return json.load(f)
        except (json.JSONDecodeError, IOError):
            print(f"  [WARN] 交易账本损坏，重置")
    return {"isir":{"open":[],"closed":[],"cumulative_return":0,"win_count":0,"total_count":0},
            "glm":{"open":[],"closed":[],"cumulative_return":0,"win_count":0,"total_count":0},
            "doubao":{"open":[],"closed":[],"cumulative_return":0,"win_count":0,"total_count":0}}

def load_signals():
    if os.path.exists(SIGNAL_FILE):
        try:
            with open(SIGNAL_FILE) as f: return json.load(f)
        except (json.JSONDecodeError, IOError):
            print(f"  [WARN] 信号历史损坏，重置")
    return []

def update_trades(rankings, extra_info, date_str, market_trend=1, global_pos_adjust=1.0):
    trades = load_trades()
    signals = load_signals()
    today_signals = {}
    n_total = len(rankings)
    rank_map = {r["code"]: r for r in rankings}

    # 综合仓位上限: 上证门控 × 外围门控
    base_max = MAX_POSITIONS if market_trend >= 0 else MAX_POSITIONS // 2
    max_positions = max(1, int(base_max * global_pos_adjust))

    for strategy in ["isir","glm","doubao"]:
        rank_key = f"{strategy}_rank"
        current_top = {r["code"] for r in rankings if r[rank_key] <= BUY_COUNT}
        current_top_info = {r["code"]: r for r in rankings if r[rank_key] <= BUY_COUNT}
        today_signals[strategy] = {}

        still_open = []
        for trade in trades[strategy]["open"]:
            code = trade["code"]
            price = extra_info.get(code,{}).get("price",0)
            current_ret = round((price/trade["entry_price"]-1)*100,2) if price > 0 else 0
            trade["hold_days"] += 1
            trade["current_price"] = price
            trade["current_return"] = current_ret

            current_rank = rank_map.get(code,{}).get(rank_key, n_total)
            exit_reason = None
            if trade["hold_days"] >= HOLD_DAYS:
                exit_reason = f"时间到期({HOLD_DAYS}天)"
            elif current_ret <= STOP_LOSS_PCT:
                exit_reason = f"止损({current_ret:+.1f}%≤{STOP_LOSS_PCT}%)"
            elif current_ret >= TAKE_PROFIT_PCT:
                exit_reason = f"止盈({current_ret:+.1f}%≥{TAKE_PROFIT_PCT}%)"
            elif current_rank > n_total * 0.5:
                exit_reason = f"排名崩溃(#{current_rank})"
            elif market_trend < 0 and current_ret < 0:
                exit_reason = f"空头减仓({current_ret:+.1f}%)"

            if exit_reason:
                # 信号日收盘触发，记录信号价；实际成交价在次日确认
                trade["exit_date"] = date_str
                trade["exit_price"] = price  # 信号价(收盘)
                trade["signal_price"] = price
                trade["return_pct"] = current_ret
                trade["is_win"] = current_ret > 0
                trade["status"] = "closed"
                trade["exit_reason"] = exit_reason
                trades[strategy]["closed"].append(trade)
                trades[strategy]["cumulative_return"] += current_ret
                trades[strategy]["total_count"] += 1
                if current_ret > 0: trades[strategy]["win_count"] += 1
                today_signals[strategy][code] = "sell"
                signals.append({
                    "date": date_str, "strategy": strategy, "code": code,
                    "name": trade.get("name",""), "signal": "sell",
                    "price": price, "rank": current_rank,
                    "return_pct": current_ret, "entry_date": trade.get("entry_date",""),
                    "entry_price": trade.get("entry_price",0),
                    "reason": exit_reason
                })
            elif code in current_top:
                still_open.append(trade)
                today_signals[strategy][code] = "hold"
            else:
                still_open.append(trade)
                today_signals[strategy][code] = "watch"

        # New entries → 买入信号（仅在有空位时买入，排除今日卖出的股票避免反复买卖）
        existing_codes = {t["code"] for t in still_open}
        sold_today = {s["code"] for s in signals if s.get("strategy")==strategy and s.get("signal")=="sell" and s.get("date")==date_str}
        available_slots = max_positions - len(still_open)
        if available_slots > 0:
            new_candidates = sorted(
                [(code, current_top_info[code][rank_key]) for code in current_top - existing_codes - sold_today],
                key=lambda x: x[1]
            )
            for code, rank in new_candidates[:available_slots]:
                price = extra_info.get(code,{}).get("price",0)
                if price > 0:
                    name = extra_info.get(code,{}).get("name",code)
                    entry_rank = current_top_info[code][rank_key]
                    entry_rank_pct = entry_rank / n_total if n_total > 0 else 0.15
                    still_open.append({
                        "code":code,"name":name,
                        "entry_date":date_str,"entry_price":price,
                        "entry_rank":entry_rank,
                        "entry_rank_pct":round(entry_rank_pct,4),
                        "hold_days":0,"current_price":price,"current_return":0,"status":"open"
                    })
                today_signals[strategy][code] = "buy"
                signals.append({
                    "date": date_str, "strategy": strategy, "code": code,
                    "name": name, "signal": "buy",
                    "price": price, "rank": current_top_info[code][rank_key],
                    "return_pct": 0, "entry_date": date_str, "entry_price": price
                })

        trades[strategy]["open"] = still_open

    # 信号历史只保留最近500条
    signals = signals[-500:]

    with open(TRADE_FILE,"w") as f: json.dump(trades,f,ensure_ascii=False,indent=2)
    with open(SIGNAL_FILE,"w") as f: json.dump(signals,f,ensure_ascii=False,indent=2)
    return trades, today_signals, signals

# ================================================================
# 市场宽度 & 盘面解读（借鉴 market-breadth, Wyckoff-Analysis 等项目）
# ================================================================

def compute_market_overview(klines, extra_info, rankings, index_klines=None, market_trend=None, market_trend_label=""):
    """计算市场宽度、指数状态、盘面解读"""
    result = {}
    if market_trend is not None:
        result["market_trend"] = {"trend": market_trend, "label": market_trend_label}

    # 1. 市场宽度：MA20以上占比
    above_ma20 = 0
    total_valid = 0
    new_high_5d = 0
    new_low_5d = 0
    for code, k in klines.items():
        if len(k) < 22:
            continue
        closes = [bar["close"] for bar in k]
        ma20 = sum(closes[-20:]) / 20 if len(closes) >= 20 else closes[-1]
        total_valid += 1
        if closes[-1] > ma20:
            above_ma20 += 1
        # 5日新高/新低（排除今日，对比前4日）
        if len(k) >= 5:
            if closes[-1] >= max(bar["high"] for bar in k[-5:-1]):
                new_high_5d += 1
            if closes[-1] <= min(bar["low"] for bar in k[-5:-1]):
                new_low_5d += 1

    if total_valid > 0:
        bread_pct = round(above_ma20 / total_valid * 100, 1)
        new_high_pct = round(new_high_5d / total_valid * 100, 1)
        new_low_pct = round(new_low_5d / total_valid * 100, 1)
    else:
        bread_pct = 0; new_high_pct = 0; new_low_pct = 0

    # 市场水温判断（借鉴 Wyckoff-Analysis 水温逻辑）
    if bread_pct >= 70:
        temp = "🔥 强势市场"; temp_color = "#dc2626"; temp_advice = "仓位可激进，追高需谨慎"
    elif bread_pct >= 50:
        temp = "🟡 中性偏强"; temp_color = "#d97706"; temp_advice = "正常仓位，精选个股"
    elif bread_pct >= 30:
        temp = "⚪ 中性偏弱"; temp_color = "#6b7280"; temp_advice = "半仓操作，控制风险"
    elif bread_pct >= 15:
        temp = "❄️ 弱势市场"; temp_color = "#059669"; temp_advice = "轻仓观望，等待信号"
    else:
        temp = "💀 极弱市场"; temp_color = "#047857"; temp_advice = "空仓保命，不操作就是赢"

    result["breadth"] = {
        "above_ma20_pct": bread_pct, "total": total_valid,
        "new_high_5d_pct": new_high_pct, "new_low_5d_pct": new_low_pct,
        "temperature": temp, "temp_color": temp_color, "temp_advice": temp_advice,
    }

    # 2. RSI整体水位
    rsi_values = []
    for code, k in klines.items():
        if len(k) < 15: continue
        closes = [bar["close"] for bar in k[-15:]]
        gains = sum(max(closes[i]-closes[i-1], 0) for i in range(1, len(closes)))
        losses = sum(max(closes[i-1]-closes[i], 0) for i in range(1, len(closes)))
        if losses == 0:
            rsi_values.append(100 if gains > 0 else 50)
        else:
            rsi_values.append(100 - 100 / (1 + gains/losses))

    if rsi_values:
        avg_rsi = np.mean(rsi_values)
        overbought = sum(1 for r in rsi_values if r > 70)
        oversold = sum(1 for r in rsi_values if r < 30)
        if avg_rsi > 65:
            rsi_status = f"🟠 整体偏热 (RSI={avg_rsi:.0f})，超买{overbought}只"
        elif avg_rsi < 35:
            rsi_status = f"🟢 整体偏冷 (RSI={avg_rsi:.0f})，超卖{oversold}只"
        else:
            rsi_status = f"⚪ 中性 (RSI={avg_rsi:.0f})"
    else:
        rsi_status = "数据不足"

    result["rsi_overview"] = rsi_status

    # 3. 板块强弱排名
    sector_perf = defaultdict(list)
    for r in rankings:
        s = sector_map.get_sector(r["code"])
        extra = extra_info.get(r["code"],{})
        chg = extra.get("change_pct",0) or 0
        sector_perf[s].append(chg)

    sector_strength = []
    for s, chgs in sector_perf.items():
        if len(chgs) < 3: continue
        avg_chg = np.mean(chgs)
        up_ratio = sum(1 for c in chgs if c > 0) / len(chgs) * 100
        sector_strength.append({"name": s, "avg_chg": round(avg_chg,2), "up_ratio": round(up_ratio,1), "count": len(chgs)})

    sector_strength.sort(key=lambda x: x["avg_chg"], reverse=True)
    result["sector_strength"] = sector_strength[:6]  # Top 6

    # 4. 涨跌统计
    up_count = sum(1 for r in rankings for c,u in [(r["code"], extra_info.get(r["code"],{}))] if u.get("change_pct",0) > 0)
    down_count = sum(1 for r in rankings for c,u in [(r["code"], extra_info.get(r["code"],{}))] if u.get("change_pct",0) < 0)
    flat_count = len(rankings) - up_count - down_count
    result["advance"] = {"up": up_count, "down": down_count, "flat": flat_count}

    # 5. 涨停/跌停统计 — 修复(2026-09-14): 原固定9.5%阈值会把创业板/科创板20cm、北交所30cm的
    #    未涨停大涨(如+10%)误计为涨停。现按板块规则分阈值: 主板10%/创业板科创板20%/北交所30%。
    #    优先用腾讯精确涨停价(zt_price/dt_price)判断, 无涨停价时回退到分板阈值。
    def _limit_pct(code):
        if code.startswith(("30", "68")):
            return 20.0
        if code.startswith(("8", "4", "92")):
            return 30.0
        return 10.0

    limit_up = 0
    limit_down = 0
    for r in rankings:
        code = r["code"]
        extra = extra_info.get(code, {})
        chg = extra.get("change_pct", 0) or 0
        price = extra.get("price", 0) or 0
        zt = extra.get("zt_price", 0) or 0
        dt = extra.get("dt_price", 0) or 0
        if zt > 0 and price > 0:
            if price >= zt - 0.005:
                limit_up += 1
            if price <= dt + 0.005:
                limit_down += 1
        else:
            thr = _limit_pct(code) * 0.98
            if chg > thr:
                limit_up += 1
            elif chg < -thr:
                limit_down += 1
    result["limits"] = {"up": limit_up, "down": limit_down}

    # 6. 成交量
    total_vol_ratio = np.mean([extra_info.get(r["code"],{}).get("vol_ratio",1) or 1 for r in rankings]) if rankings else 1
    if total_vol_ratio > 1.3:
        vol_status = f"📊 放量 ({total_vol_ratio:.1f}x)"
    elif total_vol_ratio < 0.7:
        vol_status = f"📊 缩量 ({total_vol_ratio:.1f}x)"
    else:
        vol_status = f"📊 平量 ({total_vol_ratio:.1f}x)"
    result["volume_status"] = vol_status

    return result

# ================================================================
# 180-Day Backtest
# ================================================================

def run_backtest(klines, extra_info, sectors, backtest_days=180):
    """180天回测：对比ISIR/GLM/豆包三套体系的胜率和收益"""
    all_dates = set()
    for k in klines.values():
        for bar in k: all_dates.add(bar["date"])
    dates = sorted(all_dates)
    if len(dates) < 90: return None

    # 取最近 backtest_days 个交易日
    # 实际上数据只有130天，所以用可用日期
    test_dates = dates[-min(backtest_days, len(dates)-60):]
    print(f"  回测区间: {test_dates[0]} ~ {test_dates[-1]} ({len(test_dates)}个交易日)")

    strategies = ["isir","glm","doubao"]
    results = {s:{"trades":[],"returns_5d":[],"returns_10d":[],"returns_20d":[]} for s in strategies}

    skip_count = 0
    for di, date in enumerate(test_dates[:-20]):  # 留20天做前向收益
        if di % 20 == 0: print(f"  回测进度: {di}/{len(test_dates)-20}")
        try:
            # 取该日之前的数据计算因子
            day_klines = {}
            for code, k in klines.items():
                day_bars = [b for b in k if b["date"] <= date]
                if len(day_bars) >= 60:
                    day_klines[code] = day_bars

            if len(day_klines) < 100: 
                skip_count += 1
                continue

            # 用历史K线推算extra_info，避免前视偏差
            # 预计算每只股票的股本(从今日市值/今日收盘价反推)
            # 修复(2026-09-14): ①mcap已改为总市值(原为流通市值) ②新增流通股本用于换手率
            # ③腾讯K线volume单位是手, 换算为股需×100(原直接相除导致换手率低100倍)
            shares_cache = {}
            float_shares_cache = {}
            for code in day_klines:
                today_info = extra_info.get(code, {})
                tp = today_info.get("price", 0)
                tm = today_info.get("mcap", 0)          # 总市值(元)
                tfm = today_info.get("float_mcap", 0)   # 流通市值(元)
                if tp > 0 and tm > 0:
                    shares_cache[code] = tm / tp        # 总股本(股)
                else:
                    shares_cache[code] = 0
                if tp > 0 and tfm > 0:
                    float_shares_cache[code] = tfm / tp # 流通股本(股)
                else:
                    float_shares_cache[code] = 0

            day_extra = {}
            for code in day_klines:
                k_bars = day_klines.get(code, [])
                if k_bars and len(k_bars) >= 2:
                    last = k_bars[-1]; prev = k_bars[-2]
                    avg_vol_5 = sum(b["volume"] for b in k_bars[-6:-1]) / 5 if len(k_bars) >= 6 else last["volume"]
                    vr = last["volume"] / avg_vol_5 if avg_vol_5 > 0 else 1.0
                    # 用股本反推历史市值和换手率
                    shares = shares_cache.get(code, 0)
                    float_shares = float_shares_cache.get(code, 0)
                    hist_mcap = last["close"] * shares if shares > 0 else 0
                    # volume单位是手(100股/手); 换手率=成交量(股)/流通股本×100
                    hist_turnover = (last["volume"] * 100 / float_shares * 100) if float_shares > 0 else 0
                    today_info = extra_info.get(code, {})
                    day_extra[code] = {
                        "name": today_info.get("name", code),
                        "price": last["close"],
                        "change_pct": (last["close"]/prev["close"]-1)*100,
                        "pe_ttm": today_info.get("pe_ttm", 0) or 0,  # 用今日PE近似(有偏差但优于0)
                        "pb": today_info.get("pb", 0) or 0,
                        "mcap": hist_mcap,           # 从总股本反推的历史总市值
                        "turnover": hist_turnover,   # 从流通股本反推(手→股换算修复)
                        "vol_ratio": vr,
                    }

            factor_data, _ = compute_all_factors(day_klines, day_extra, {}, {}, sectors)
            if len(factor_data) < 100: 
                skip_count += 1
                continue

            rankings = compute_rankings(factor_data)
            if not rankings: continue

            # 取每个策略的TOP30
            for strat in strategies:
                rank_key = f"{strat}_rank"
                top_stocks = sorted(rankings, key=lambda x: x[rank_key])[:30]

                # 记录每笔"交易"：买入后5/10/20天的收益
                for r in top_stocks:
                    code = r["code"]
                    entry_bars = [b for b in klines.get(code,[]) if b["date"] <= date]
                    if len(entry_bars) < 2: continue
                    entry_price = entry_bars[-1]["close"]

                    # 前向收益
                    fwd_bars = [b for b in klines.get(code,[]) if b["date"] > date]
                    for hold, label in [(5,"returns_5d"),(10,"returns_10d"),(20,"returns_20d")]:
                        if len(fwd_bars) >= hold:
                            exit_price = fwd_bars[hold-1]["close"]
                            ret = (exit_price/entry_price - 1) * 100
                            results[strat][label].append(ret)
                            results[strat]["trades"].append({
                                "code":code,"date":date,"entry":entry_price,
                                "hold":hold,"exit":exit_price,"return":round(ret,2)
                            })
        except Exception as e:
            skip_count += 1
            continue

    # 汇总
    summary = {}
    best_strat = None
    best_score = -1
    for strat in strategies:
        r5 = np.array(results[strat]["returns_5d"]) if results[strat]["returns_5d"] else np.array([0])
        r10 = np.array(results[strat]["returns_10d"]) if results[strat]["returns_10d"] else np.array([0])
        r20 = np.array(results[strat]["returns_20d"]) if results[strat]["returns_20d"] else np.array([0])

        win5 = np.mean(r5 > 0) * 100
        win10 = np.mean(r10 > 0) * 100
        win20 = np.mean(r20 > 0) * 100
        avg20 = np.mean(r20)
        max_r20 = np.max(r20)
        min_r20 = np.min(r20)

        # 综合评分：胜率权重0.6 + 平均收益权重0.4
        composite = win20 * 0.5 + avg20 * 3  # 不截断负值，让亏损策略区分开

        summary[strat] = {
            "total_trades": len(results[strat]["trades"]),
            "win_5d": round(win5,1),"win_10d": round(win10,1),"win_20d": round(win20,1),
            "avg_20d": round(avg20,2),"max_20d": round(max_r20,2),"min_20d": round(min_r20,2),
            "composite": round(composite,1),
        }

        if composite > best_score:
            best_score = composite
            best_strat = strat

    print(f"  跳过天数: {skip_count}")
    return summary, best_strat

# ================================================================
# HTML Report Generation v2.0
# ================================================================

def _build_market_overview(mkt, rankings, extra_info, n_consensus, n_total, top_n, bt_html, consensus_html):
    """构建市场全景解读面板"""
    if not mkt:
        return ""

    bread = mkt.get("breadth",{})
    adv = mkt.get("advance",{"up":0,"down":0,"flat":0})
    limits = mkt.get("limits",{"up":0,"down":0})
    rsi_ov = mkt.get("rsi_overview","")
    vol_s = mkt.get("volume_status","")
    sectors = mkt.get("sector_strength",[])

    # 板块强弱条
    sector_bars = ""
    for s in sectors:
        color = "#dc2626" if s["avg_chg"] > 1 else ("#059669" if s["avg_chg"] < -1 else "#999")
        bar_w = min(100, abs(s["avg_chg"]) * 15)
        sector_bars += f"""<div class="sector-bar-row">
<span class="sector-bar-name">{s['name']}</span>
<span class="sector-bar-val" style="color:{color}">{s['avg_chg']:+.1f}%</span>
<span style="font-size:10px;color:#999">({s['up_ratio']:.0f}%涨)</span>
</div>"""

    return f"""
<div class="market-overview">
  <div class="mo-grid">
    <!-- 市场宽度 -->
    <div class="mo-card" style="border-color:{bread.get('temp_color','#999')}">
      <div class="mo-title">市场宽度</div>
      <div class="mo-big" style="color:{bread.get('temp_color','#999')}">{bread.get('above_ma20_pct',0)}%</div>
      <div class="mo-sub">站上MA20 ({bread.get('total',0)}只)</div>
      <div class="mo-breadth-bar"><div class="mo-breadth-fill" style="width:{bread.get('above_ma20_pct',0)}%;background:{bread.get('temp_color','#999')}"></div></div>
      <div style="margin-top:6px;font-size:12px;font-weight:600">{bread.get('temperature','-')}</div>
      <div style="font-size:11px;color:var(--text-secondary)">{bread.get('temp_advice','-')}</div>
      <div style="font-size:10px;color:#999;margin-top:4px">5日新高{bread.get('new_high_5d_pct',0)}% | 新低{bread.get('new_low_5d_pct',0)}%</div>
    </div>

    <!-- 涨跌统计 -->
    <div class="mo-card">
      <div class="mo-title">涨跌统计</div>
      <div style="display:flex;gap:12px;justify-content:center;margin:8px 0">
        <div style="text-align:center"><div style="color:#dc2626;font-size:22px;font-weight:700">{adv.get('up',0)}</div><div style="font-size:10px;color:#999">上涨</div></div>
        <div style="text-align:center"><div style="color:#16a34a;font-size:22px;font-weight:700">{adv.get('down',0)}</div><div style="font-size:10px;color:#999">下跌</div></div>
        <div style="text-align:center"><div style="color:#999;font-size:22px;font-weight:700">{adv.get('flat',0)}</div><div style="font-size:10px;color:#999">平收</div></div>
      </div>
      <div style="font-size:11px;color:var(--text-secondary)">涨跌比 {adv.get('up',0)}:{adv.get('down',0)}</div>
      <div style="font-size:11px">涨停 🚀{limits.get('up',0)} 跌停 💀{limits.get('down',0)}</div>
      <div style="font-size:11px">{vol_s}</div>
    </div>

    <!-- RSI水位 -->
    <div class="mo-card">
      <div class="mo-title">RSI & 信号</div>
      <div style="font-size:13px;line-height:1.8">{rsi_ov}</div>
      <div style="margin-top:8px;font-size:12px;font-weight:600;color:var(--primary)">ISIR主评分 👑</div>
      <div style="font-size:11px;color:#999">共识TOP{top_n}: {n_consensus}只 | 总计{n_total}只</div>
    </div>

    <!-- 市场趋势门控 -->
    <div class="mo-card" style="{"border-color:#dc2626" if mkt.get("market_trend",{}).get("trend",1) < 0 else ""}">
      <div class="mo-title">市场趋势门控</div>
      <div class="mo-big" style="color:{"#dc2626" if mkt.get("market_trend",{}).get("trend",1) < 0 else "#16a34a"}">{"空头" if mkt.get("market_trend",{}).get("trend",1) < 0 else "多头"}</div>
      <div class="mo-sub">{mkt.get("market_trend",{}).get("label","-")}</div>
      <div style="margin-top:6px;font-size:11px;color:{"#dc2626" if mkt.get("market_trend",{}).get("trend",1) < 0 else "var(--text-secondary)"}">{"仓位上限减半→15只, 浮亏强制减仓" if mkt.get("market_trend",{}).get("trend",1) < 0 else "仓位上限30只, 正常操作"}</div>
    </div>

    <!-- 外围市场门控 -->
    <div class="mo-card" style="{"border-color:#dc2626" if mkt.get("global_gate",{}).get("level",0) >= 2 else ("border-color:#f59e0b" if mkt.get("global_gate",{}).get("level",0) == 1 else "")}">
      <div class="mo-title">外围市场隔夜</div>
      <div class="mo-big" style="color:{"#dc2626" if mkt.get("global_gate",{}).get("level",0) >= 2 else ("#f59e0b" if mkt.get("global_gate",{}).get("level",0) == 1 else "#16a34a")}">{"危险" if mkt.get("global_gate",{}).get("level",0) >= 2 else ("警戒" if mkt.get("global_gate",{}).get("level",0) == 1 else "正常")}</div>
      <div style="font-size:11px;margin-top:4px">""" + "\n".join([f'{gname}: {ginfo["overnight_ret"]:+.2f}%' if gname != "VIX恐慌指数" else f'VIX: {ginfo["close"]}' for gname, ginfo in mkt.get("global_markets",{}).items()]) + f"""</div>
      <div style="margin-top:6px;font-size:11px;color:var(--text-secondary)">{mkt.get("global_gate",{}).get("interpretation","")}</div>
    </div>
    <div class="mo-card">
      <div class="mo-title">板块强弱</div>
      {sector_bars}
    </div>
  </div>
</div>
{bt_html}
<div class="consensus-panel" style="display:{'block' if n_consensus>0 else 'none'};margin-top:12px">
<h3>共识TOP{top_n} — ISIR ∩ GLM ∩ SS分排名 ({n_consensus}只)</h3>
<div class="consensus-list">{consensus_html}</div>
</div>"""


def _build_signal_history(signal_history, trades=None):
    """构建历史操作建议记录面板 — 按股票视角"""
    # 收集所有已结算交易
    all_trades = []
    if trades:
        for strat in ["isir","glm","doubao"]:
            for t in trades.get(strat,{}).get("closed",[]):
                t_copy = dict(t)
                t_copy["strategy"] = strat
                all_trades.append(t_copy)

    if not all_trades:
        return """<div class="panel" id="panel-signals">
<div class="section-title">历史操作记录</div>
<div style="padding:40px;text-align:center;color:var(--text-secondary)">
<p style="font-size:16px">📝 暂无历史操作记录</p>
<p style="font-size:12px;margin-top:8px">运行回测或每天运行引擎后，交易记录将自动显示</p>
</div></div>"""

    # 按股票分组
    stock_trades = defaultdict(list)
    for t in all_trades:
        stock_trades[t["code"]].append(t)

    # 统计每只股票
    stock_stats = []
    for code, trades_list in stock_trades.items():
        name = trades_list[0].get("name", code)
        n = len(trades_list)
        wins = sum(1 for t in trades_list if t.get("return_pct",0) > 0)
        total_ret = sum(t.get("return_pct",0) for t in trades_list)
        avg_ret = total_ret / n if n > 0 else 0
        best = max(t.get("return_pct",0) for t in trades_list) if trades_list else 0
        worst = min(t.get("return_pct",0) for t in trades_list) if trades_list else 0
        # 按策略分布
        strat_counts = defaultdict(int)
        for t in trades_list:
            strat_counts[t.get("strategy","")] += 1
        stock_stats.append({
            "code": code, "name": name, "n": n, "wins": wins,
            "win_rate": round(wins/n*100,1) if n > 0 else 0,
            "total_ret": round(total_ret,1), "avg_ret": round(avg_ret,1),
            "best": round(best,1), "worst": round(worst,1),
            "strats": dict(strat_counts),
            "trades": sorted(trades_list, key=lambda x: x.get("entry_date","")),
        })

    # 按总收益降序
    stock_stats.sort(key=lambda x: x["total_ret"], reverse=True)

    # 总体统计
    total_trades = len(all_trades)
    total_wins = sum(1 for t in all_trades if t.get("return_pct",0) > 0)
    overall_wr = round(total_wins/total_trades*100,1) if total_trades > 0 else 0
    total_return = round(sum(t.get("return_pct",0) for t in all_trades),1)
    best_stock = stock_stats[0] if stock_stats else None
    worst_stock = stock_stats[-1] if stock_stats else None

    # 构建表格行
    rows = ""
    for si, s in enumerate(stock_stats[:100]):  # 前100只
        code = s["code"]
        ret_cls = "positive" if s["total_ret"] > 0 else "negative"
        wr_cls = "positive" if s["win_rate"] >= 50 else ("negative" if s["win_rate"] < 40 else "")
        strat_tags = ""
        for sk, sv in s["strats"].items():
            sc = {"isir":"var(--isir-c)","glm":"var(--glm-c)","doubao":"var(--doubao-c)"}.get(sk,"#999")
            strat_tags += f'<span style="color:{sc};font-size:10px;font-weight:600">{sk.upper()}×{sv}</span> '

        # 展开详情
        detail_rows = ""
        for t in s["trades"]:
            tret = t.get("return_pct",0)
            tret_cls = "positive" if tret > 0 else "negative"
            strat = t.get("strategy","")
            sc = {"isir":"var(--isir-c)","glm":"var(--glm-c)","doubao":"var(--doubao-c)"}.get(strat,"#999")
            reason = t.get("exit_reason","")
            detail_rows += f"""<tr>
<td style="color:{sc};font-weight:600">{strat.upper()}</td>
<td>{t.get('entry_date','')}</td><td class="num">{t.get('entry_price',0):.2f}</td>
<td>{t.get('exit_date','')}</td><td class="num">{t.get('exit_price',0):.2f}</td>
<td class="num {tret_cls}">{tret:+.1f}%</td>
<td class="num">{t.get('hold_days',0)}天</td>
<td style="font-size:11px;color:var(--text-secondary)">{reason}</td></tr>"""

        rows += f"""<tr style="cursor:pointer" onclick="toggleRow('sig-{code}')">
<td><strong>{code}</strong></td>
<td>{s['name']}</td>
<td class="num">{s['n']}</td>
<td class="num {wr_cls}">{s['win_rate']}%</td>
<td class="num {ret_cls}">{s['total_ret']:+.1f}%</td>
<td class="num">{s['avg_ret']:+.1f}%</td>
<td class="num positive">{s['best']:+.1f}%</td>
<td class="num negative">{s['worst']:+.1f}%</td>
<td>{strat_tags}</td>
</tr>
<tr id="sig-{code}" class="detail-row" style="display:none">
<td colspan="9" style="padding:0">
<div style="padding:10px 16px;background:#f8fafc">
<table style="font-size:11px"><thead><tr>
<th>策略</th><th>买入日</th><th>买入价</th><th>卖出日</th><th>卖出价</th><th>收益</th><th>持仓</th><th>退出原因</th>
</tr></thead><tbody>{detail_rows}</tbody></table>
</div></td></tr>"""

    best_html = f'<span class="positive">{best_stock["name"]}({best_stock["total_ret"]:+.1f}%)</span>' if best_stock else "—"
    worst_html = f'<span class="negative">{worst_stock["name"]}({worst_stock["total_ret"]:+.1f}%)</span>' if worst_stock else "—"

    return f"""
<div class="panel" id="panel-signals">
<div class="section-title">历史操作记录 — 按股票视角</div>
<div class="trade-summary" style="margin-bottom:16px">
<div class="ts-card"><div class="ts-value" style="color:var(--primary)">{total_trades}</div><div class="ts-label">总交易笔数</div></div>
<div class="ts-card"><div class="ts-value" style="color:var(--primary)">{overall_wr}%</div><div class="ts-label">总胜率</div></div>
<div class="ts-card"><div class="ts-value {'positive' if total_return>0 else 'negative'}">{total_return:+.1f}%</div><div class="ts-label">累积收益</div></div>
<div class="ts-card"><div class="ts-value" style="color:#16a34a">{len(stock_stats)}</div><div class="ts-label">涉及股票</div></div>
<div class="ts-card"><div class="ts-value" style="font-size:14px;line-height:1.4">🏆{best_html}<br>💀{worst_html}</div><div class="ts-label">最佳/最差</div></div>
</div>
<div class="table-wrap"><table><thead><tr>
<th>代码</th><th>名称</th><th>交易次数</th><th>胜率</th><th>总收益</th><th>均收益</th><th>最佳</th><th>最差</th><th>策略分布</th>
</tr></thead><tbody>{rows}</tbody></table></div>
<div style="padding:8px;font-size:11px;color:var(--text-secondary)">点击行展开查看该股票的每笔交易明细</div>
</div>"""


def _build_qrebound_rows(qr_list, extra_info):
    """第五视图: 质量反弹面板表格行 (超跌反弹 ∩ ISIR/GLM前QR_THRESHOLD)"""
    rows = ""
    for i, r in enumerate(qr_list[:40]):
        code = r["code"]
        extra = extra_info.get(code, {})
        name = extra.get("name", code)
        price = extra.get("price", 0) or r.get("rebound_indicators", {}).get("close", 0)
        ind = r.get("rebound_indicators", {})
        atr = r.get("rebound_atr", {})
        stage = r.get("rebound_stage", "")
        stage_color = {"加速": "#dc2626", "确认": "#f59e0b", "初现": "#6b7280"}.get(stage, "#6b7280")
        sector = sector_map.get_sector(code)
        best_rank = min(r.get("isir_rank", 9999), r.get("glm_rank", 9999))

        tags = ""
        if r.get("consensus"): tags += '<span class="badge consensus">共识</span>'
        if r.get("in_isir_top"): tags += '<span class="tag top-isir">ISIR</span>'
        if r.get("in_glm_top"): tags += '<span class="tag top-glm">GLM</span>'
        vcp = r.get("vcp_status", "")
        if vcp == "突破": tags += '<span class="tag top-vcp-breakout">VCP突破</span>'
        elif vcp == "预突破": tags += '<span class="tag top-vcp-pre">VCP预突破</span>'
        if r.get("fund_confirm"): tags += '<span class="tag top-fund">资金确认</span>'
        if not tags: tags = '<span style="color:#ccc;font-size:11px">—</span>'

        rows += f"""<tr>
<td>{i+1}</td>
<td class="code-col">{code}</td>
<td>{name} <small style="color:#999">{sector}</small></td>
<td class="num">{price:.2f}</td>
<td class="num negative">{ind.get('ret_5d',0):+.1f}%</td>
<td class="num">{ind.get('rsi',50):.0f}</td>
<td class="num" style="color:#dc2626;font-weight:700">{r.get('rebound_score',0):.0f}</td>
<td><span style="background:{stage_color};color:white;padding:2px 8px;border-radius:8px;font-size:10px;font-weight:600">{stage}</span></td>
<td class="num" style="color:var(--isir-c)">#{r.get('isir_rank','-')}</td>
<td class="num" style="color:var(--glm-c)">#{r.get('glm_rank','-')}</td>
<td class="num" style="font-weight:700">#{best_rank}</td>
<td class="num" style="color:#dc2626">{atr.get('stop_price',0):.2f}</td>
<td class="num" style="color:#059669">{atr.get('tp_price',0):.2f}</td>
<td>{tags}</td>
</tr>"""
    if not rows:
        rows = f'<tr><td colspan="14" style="text-align:center;padding:30px;color:#999">今日无质量反弹信号 (超跌反弹 ∩ ISIR/GLM前{QR_THRESHOLD}名 交集体为空 — 黄金坑未出现)</td></tr>'
    return rows


def _build_qrebound_backtest_html(qr_backtest):
    """第五视图: 质量反弹回测对比表"""
    if not qr_backtest:
        return '<div style="background:#fef2f2;border:1px solid #fecaca;border-radius:8px;padding:10px 14px;font-size:12px;color:#991b1b">回测数据缺失: 请运行 quality_rebound_backtest.py 生成 output/quality_rebound_backtest.json</div>'
    groups = qr_backtest.get("groups", {})
    bear = qr_backtest.get("bear", {})
    rows_html = ""
    label_map = {"all": "全池基线", "pure": "纯超跌反弹", "pure_s40": "纯超跌·高分≥40", "confirm": "纯超跌·确认阶段", "cross100": "交叉(前100)", "cross150": "交叉(前150)⭐", "cross150_confirm": "交叉(前150)·确认", "cross200": "交叉(前200)", "cross250": "交叉(前250)"}
    for g in ["all", "pure", "pure_s40", "confirm", "cross100", "cross150", "cross150_confirm", "cross200", "cross250"]:
        if g not in groups: continue
        s20 = groups[g].get("ret20", {})
        b = bear.get(g, {}).get("bear20", {})
        bl = bear.get(g, {}).get("bull20", {})
        hl = ' style="background:#f0fdfa;font-weight:600"' if g == "cross150" else ""
        rows_html += f"""<tr{hl}>
<td>{label_map.get(g,g)}</td>
<td class="num">{s20.get('n',0)}</td>
<td class="num">{s20.get('win',0)}%</td>
<td class="num" style="font-weight:700;color:{'#dc2626' if s20.get('avg',0)>0 else '#16a34a'}">{s20.get('avg',0):+.2f}%</td>
<td class="num">{s20.get('med',0):+.2f}%</td>
<td class="num">{b.get('win',0)}% / {b.get('avg',0):+.1f}%</td>
<td class="num">{bl.get('win',0)}% / {bl.get('avg',0):+.1f}%</td>
</tr>"""
    period = qr_backtest.get("period", ["", ""])
    stages = qr_backtest.get("stages", {})
    stage_str = " | ".join(f"{k}:{v.get('win',0)}%/{v.get('avg',0):+.1f}%" for k, v in stages.items()) if stages else ""
    return f"""<div style="background:#f0fdfa;border:1px solid #99f6e4;border-radius:8px;padding:12px 16px;margin-bottom:10px">
<h3 style="color:#0f766e;font-size:15px;margin-bottom:8px">回测验证: 质量交叉 vs 纯超跌 vs 全池 ({period[0]}~{period[1]}, 180交易日)</h3>
<div class="table-wrap" style="border:none"><table><thead><tr>
<th>组</th><th>样本</th><th>20日胜率</th><th>20日均收益</th><th>20日中位</th><th>空头日(胜率/均收益)</th><th>多头日(胜率/均收益)</th>
</tr></thead><tbody>{rows_html}</tbody></table></div>
<div style="font-size:11px;color:var(--text-secondary);margin-top:8px">
* 信号日收盘买入持有20日 | 阶段细分(纯超跌组20日): {stage_str} | 注意: 5日持有无优势(45-49%), 本策略需20日耐心 | 均值被右尾拉高, 以中位数辅助判断 | 存活偏差与信号自相关同引擎回测口径
</div>
</div>"""


def build_html(rankings, extra_info, history, trades, return_data, date_str, freshness, backtest_summary=None, best_strat=None, mkt_overview=None, today_signals=None, signal_history=None, qr_backtest=None):
    n_total = len(rankings)
    n_consensus = sum(1 for r in rankings if r["consensus"])

    def arrow_cls(diff):
        if diff>3: return "arrow-up"
        elif diff<-3: return "arrow-down"
        return "arrow-flat"

    def build_panel_rows(sorted_r, strategy=None):
        sig_map = today_signals.get(strategy, {}) if (today_signals and strategy) else {}
        rows = ""
        for r in sorted_r:
            code = r["code"]; extra = extra_info.get(code,{})
            name = extra.get("name",code); price = extra.get("price",0) or 0
            chg_pct = extra.get("change_pct",0) or 0; rets = return_data.get(code,{})
            sector = sector_map.get_sector(code)
            isir_diff,isir_arrow = get_trend(code,"isir",history)
            glm_diff,glm_arrow = get_trend(code,"glm",history)
            doubao_diff,doubao_arrow = get_trend(code,"doubao",history)
            ss_diff,ss_arrow = get_ss_trend(code,history)
            consensus_badge = '<span class="badge consensus">共识</span>' if r["consensus"] else ""
            # 买卖信号标识
            sig = sig_map.get(code, "")
            if sig == "buy": sig_badge = '<span class="badge sig-buy">🟢买入</span>'
            elif sig == "sell": sig_badge = '<span class="badge sig-sell">🔴卖出</span>'
            elif sig == "hold": sig_badge = '<span class="badge sig-hold">✅持有</span>'
            elif sig == "watch": sig_badge = '<span class="badge sig-watch">⏸关注</span>'
            else: sig_badge = ""
            top_tags = ""
            if r["in_isir_top"]: top_tags += '<span class="tag top-isir">ISIR</span>'
            if r["in_glm_top"]: top_tags += '<span class="tag top-glm">GLM</span>'
            # VCP 标签
            vcp_status = r.get("vcp_status", "")
            vcp_badge = ""
            if vcp_status == "突破":
                vcp_badge = '<span class="tag top-vcp-breakout" title="VCP波动收缩形态突破确认">VCP突破</span>'
            elif vcp_status == "预突破":
                vcp_badge = '<span class="tag top-vcp-pre" title="VCP形态形成中,接近pivot">VCP预突破</span>'
            top_tags += vcp_badge
            # 资金确认标签 (加分项, 与VCP并列)
            if r.get("fund_confirm"):
                ff_today = (r.get("fund_today", 0) or 0) / 1e8
                top_tags += f'<span class="tag top-fund" title="当日主力净流入{ff_today:+.2f}亿且5日累计为正(真实持续流入)">资金确认</span>'
            # 超跌反弹标记 (第四体系)
            if r.get("rebound_valid"):
                rb_stage = r.get("rebound_stage", "")
                rb_score = r.get("rebound_score", 0)
                top_tags += f'<span class="tag top-rebound" title="超跌反弹{rb_stage}: 总分{rb_score:.0f} (超跌{r.get("rebound_oversold",0)}/反弹{r.get("rebound_signal",0)}/趋势{r.get("rebound_trend",0)})">超跌{rb_stage}</span>'
            row_class = "row-consensus" if r["consensus"] else ""

            # === 可读技术面解读 ===
            ind = r.get("indicator",{})
            close_v = ind.get("close",0)
            ma5_v = ind.get("ma5",0); ma10_v = ind.get("ma10",0); ma20_v = ind.get("ma20",0)
            rsi_v = ind.get("rsi",50)
            dif_v = ind.get("dif",0); dea_v = ind.get("dea",0)
            cmf_v = ind.get("cmf",0); main5d_v = ind.get("main5d",0)
            vol_r = ind.get("vol_ratio",1); turnover_v = ind.get("turnover",0)
            pct52 = ind.get("pct_52w",50); streak_v = ind.get("streak",0)
            evt_s = ind.get("event_score",0)

            # 均线解读
            ma_items = []
            for label, mav in [("5日",ma5_v),("10日",ma10_v),("20日",ma20_v)]:
                if mav > 0:
                    above = close_v > mav
                    pct = (close_v/mav-1)*100
                    color = "#059669" if above else "#dc2626"
                    ma_items.append(f'<span style="color:{color}">{label}线 {pct:+.1f}%</span>')
            ma_status = f'<span style="color:{"#059669" if close_v>ma5_v>ma10_v>ma20_v else "#d97706"}">{"🟢 多头排列" if close_v>ma5_v>ma10_v>ma20_v else ("🔴 空头排列" if close_v<ma5_v<ma10_v<ma20_v else "🟡 均线交叉")}</span>'

            # RSI解读
            if rsi_v > 70: rsi_text = f'<span style="color:#dc2626;font-weight:700">🔴 超买区 ({rsi_v:.0f})</span>'
            elif rsi_v < 30: rsi_text = f'<span style="color:#059669;font-weight:700">🟢 超卖区 ({rsi_v:.0f})</span>'
            elif rsi_v > 60: rsi_text = f'<span style="color:#d97706">🟡 偏强 ({rsi_v:.0f})</span>'
            elif rsi_v < 40: rsi_text = f'<span style="color:#6b7280">⚪ 偏弱 ({rsi_v:.0f})</span>'
            else: rsi_text = f'<span style="color:#6b7280">⚪ 中性 ({rsi_v:.0f})</span>'

            # MACD解读
            if dif_v > 0 and dif_v > dea_v: macd_text = '<span style="color:#059669">🟢 MACD金叉 多头</span>'
            elif dif_v < 0 and dif_v < dea_v: macd_text = '<span style="color:#dc2626">🔴 MACD死叉 空头</span>'
            elif dif_v > 0: macd_text = '<span style="color:#d97706">🟡 MACD多头但柱缩</span>'
            else: macd_text = '<span style="color:#6b7280">⚪ MACD空头但柱缩</span>'

            # 成交量
            if vol_r > 1.5: vol_text = f'<span style="color:#dc2626">📊 放量 {vol_r:.1f}x</span>'
            elif vol_r < 0.5: vol_text = f'<span style="color:#059669">📊 缩量 {vol_r:.1f}x</span>'
            else: vol_text = f'<span style="color:#6b7280">📊 平量 {vol_r:.1f}x</span>'

            # 资金面
            if cmf_v > 0.05: cap_text = f'<span style="color:#059669">💰 资金流入 CMF={cmf_v:.3f}</span>'
            elif cmf_v < -0.05: cap_text = f'<span style="color:#dc2626">💰 资金流出 CMF={cmf_v:.3f}</span>'
            else: cap_text = f'<span style="color:#6b7280">💰 资金平衡 CMF={cmf_v:.3f}</span>'

            main5_text = ""
            if main5d_v > 0: main5_text = f'<span style="color:#059669">主力5日净流入 {main5d_v/1e8:.1f}亿</span>'
            elif main5d_v < -1e8: main5_text = f'<span style="color:#dc2626">主力5日净流出 {abs(main5d_v)/1e8:.1f}亿</span>'

            # 位置
            if pct52 > 80: pos_text = f'<span style="color:#dc2626">⚠️ 52周高位 ({pct52:.0f}%)</span>'
            elif pct52 < 20: pos_text = f'<span style="color:#059669">💎 52周低位 ({pct52:.0f}%)</span>'
            else: pos_text = f'<span style="color:#6b7280">52周位置 {pct52:.0f}%</span>'

            # 连涨连跌
            streak_up = ind.get("streak",0)
            streak_dn = ind.get("streak_dn",0)
            if streak_up >= 3: streak_text = f'<span style="color:#dc2626">🔥 连涨{streak_up}天</span>'
            elif streak_dn >= 3: streak_text = f'<span style="color:#059669">❄️ 连跌{streak_dn}天</span>'
            elif streak_up > 0: streak_text = f'<span style="color:#d97706">连涨{streak_up}天</span>'
            elif streak_dn > 0: streak_text = f'<span style="color:#6b7280">连跌{streak_dn}天</span>'
            else: streak_text = '<span style="color:#999">平收</span>'

            # 换手率
            turn_text = f'换手率 {turnover_v:.1f}%' if turnover_v else ""

            # 信息面
            info_text = ""
            if evt_s > 10: info_text = f'<span style="color:#059669">📰 利好事件 +{evt_s:.0f}</span>'
            elif evt_s < -5: info_text = f'<span style="color:#dc2626">📰 利空事件 {evt_s:.0f}</span>'

            detail_html = f"""
<tr id="d-{code}" class="detail-row" style="display:none">
<td colspan="12"><div class="detail-card">
<div class="detail-grid">
  <div class="detail-box">
    <div class="detail-title">📈 技术面</div>
    <div class="indicator-list">
      <div class="ind-item">{ma_status}</div>
      <div class="ind-sub">{" | ".join(ma_items)}</div>
      <div class="ind-item">{rsi_text}</div>
      <div class="ind-item">{macd_text}</div>
      <div class="ind-item">{vol_text} &nbsp; {turn_text}</div>
      <div class="ind-item">{pos_text} &nbsp; {streak_text}</div>
    </div>
  </div>
  <div class="detail-box">
    <div class="detail-title">💰 资金面 & 信息面</div>
    <div class="indicator-list">
      <div class="ind-item">{cap_text}</div>
      <div class="ind-item">{main5_text}</div>
      <div class="ind-item">{info_text if info_text else '<span style="color:#6b7280">📰 近期无重大事件</span>'}</div>
      <div style="margin-top:10px;padding:8px;background:#f1f5f9;border-radius:6px">
        <strong>SS评分拆解:</strong> 技术{r.get('ss_tech',0):.0f}×0.35 + 资金{r.get('ss_capital',0):.0f}×0.55 + 信息{r.get('ss_info',0):.0f}×0.10 = <strong style="color:var(--primary)">{r['ss_score']:.1f}</strong>
      </div>
      <div style="margin-top:4px;font-size:11px;color:var(--text-secondary)">
        ISIR分:{r['isir_score']:.4f} | GLM分:{r['glm_score']:.4f} | 换手:{turnover_v:.1f}%
      </div>
    </div>
  </div>
</div></div></td></tr>"""

            rows += f"""<tr class="{row_class}" onclick="toggleRow('d-{code}')" style="cursor:pointer">
<td>{code}</td><td>{name}<br><small style="color:#999">{sector}</small></td>
<td class="num">{price:.2f}</td>
<td class="num {'positive' if chg_pct>0 else 'negative' if chg_pct<0 else ''}">{chg_pct:+.2f}%</td>
<td class="num {'positive' if rets.get('ret_5d',0)>0 else 'negative' if rets.get('ret_5d',0)<0 else ''}">{rets.get('ret_5d',0):+.1f}%</td>
<td class="num {'positive' if rets.get('ret_10d',0)>0 else 'negative' if rets.get('ret_10d',0)<0 else ''}">{rets.get('ret_10d',0):+.1f}%</td>
<td class="num {'positive' if rets.get('ret_20d',0)>0 else 'negative' if rets.get('ret_20d',0)<0 else ''}">{rets.get('ret_20d',0):+.1f}%</td>
<td class="num ss-col">{r['ss_score']:.1f}<span class="{arrow_cls(ss_diff)}" style="font-size:10px">{ss_arrow}</span></td>
<td class="num rank-col"><span class="rank-num c-isir">#{r['isir_rank']}</span><span class="{arrow_cls(isir_diff)}">{isir_arrow}</span></td>
<td class="num rank-col"><span class="rank-num c-glm">#{r['glm_rank']}</span><span class="{arrow_cls(glm_diff)}">{glm_arrow}</span></td>
<td>{sig_badge}{consensus_badge}{top_tags}</td>
</tr>{detail_html}"""
        return rows

    # Build each panel
    rankings_ss = sorted(rankings, key=lambda x:x["ss_score"], reverse=True)
    rankings_isir = sorted(rankings, key=lambda x:x["isir_rank"])
    rankings_glm = sorted(rankings, key=lambda x:x["glm_rank"])
    rankings_doubao = sorted(rankings, key=lambda x:x["doubao_rank"])

    ss_rows = build_panel_rows(rankings_ss, None)
    isir_rows = build_panel_rows(rankings_isir, "isir")
    glm_rows = build_panel_rows(rankings_glm, "glm")
    doubao_rows = build_panel_rows(rankings_doubao, "doubao")

    # 超跌反弹面板行 (第四体系)
    rebound_valid = [r for r in rankings if r.get("rebound_valid")]
    rebound_valid.sort(key=lambda x: x.get("rebound_score", 0), reverse=True)

    # 第五视图: 质量反弹 (超跌反弹 ∩ ISIR/GLM前QR_THRESHOLD)
    qr_list = [r for r in rankings if r.get("rebound_valid")
               and min(r.get("isir_rank", 9999), r.get("glm_rank", 9999)) <= QR_THRESHOLD]
    qr_list.sort(key=lambda x: x.get("rebound_score", 0), reverse=True)
    qrebound_rows = _build_qrebound_rows(qr_list, extra_info)
    qrebound_bt_html = _build_qrebound_backtest_html(qr_backtest)

    # Sector view rows
    sector_groups = defaultdict(list)
    for r in rankings:
        sector_groups[sector_map.get_sector(r["code"])].append(r)
    sector_html = ""
    for s in sector_map.SECTORS:
        stocks = sector_groups.get(s,[])
        if not stocks: continue
        stocks_ss = sorted(stocks, key=lambda x:x["ss_score"],reverse=True)
        top3_html = ""
        for i,r in enumerate(stocks_ss[:5]):
            code = r["code"]; name = extra_info.get(code,{}).get("name",code)
            chg = extra_info.get(code,{}).get("change_pct",0) or 0
            top3_html += f'<div class="sector-stock"><span>{code}</span><span>{name}</span><span class="{"positive" if chg>0 else "negative" if chg<0 else ""}">{chg:+.1f}%</span></div>'
        sector_html += f"""<div class="sector-block" id="sec-{s}">
<div class="sector-header" onclick="toggleSector('{s}')">📊 {s} ({len(stocks)}只) | ISIR入榜:{sum(1 for r in stocks if r["in_isir_top"])} | GLM入榜:{sum(1 for r in stocks if r["in_glm_top"])} | 共识:{sum(1 for r in stocks if r["consensus"])}</div>
<div class="sector-body">{build_panel_rows(stocks_ss)}</div>
</div>"""

    # Trade ledger HTML per strategy
    def trade_ledger_html(strategy):
        t = trades[strategy]
        open_len = len(t["open"])
        closed = t["closed"][-50:]  # last 50 closed trades
        cum_ret = round(t["cumulative_return"],2)
        win_rate = round(t["win_count"]/t["total_count"]*100,1) if t["total_count"]>0 else 0
        # Active positions table
        active_rows = ""
        for tr in sorted(t["open"], key=lambda x:abs(x.get("current_return",0)), reverse=True):
            ret_cls = "positive" if tr.get("current_return",0)>0 else "negative"
            active_rows += f'<tr><td>{tr["code"]}</td><td>{tr["name"]}</td><td>{tr["entry_date"]}</td><td class="num">{tr["entry_price"]:.2f}</td><td class="num">{tr["current_price"]:.2f}</td><td class="num {ret_cls}">{tr["current_return"]:+.1f}%</td><td class="num">{tr["hold_days"]}天</td></tr>'

        closed_rows = ""
        for tr in reversed(closed[-20:]):
            ret_cls = "positive" if tr.get("return_pct",0)>0 else "negative"
            cls_tag = "✅" if tr.get("is_win") else "❌"
            closed_rows += f'<tr><td>{tr["code"]}</td><td>{tr.get("name","")}</td><td>{tr["entry_date"]}</td><td class="num">{tr["entry_price"]:.2f}</td><td>{tr.get("exit_date","")}</td><td class="num">{tr.get("exit_price",0):.2f}</td><td class="num {ret_cls}">{tr.get("return_pct",0):+.1f}%</td><td>{cls_tag}</td></tr>'

        return f"""
<div class="trade-panel">
  <div class="trade-summary">
    <div class="ts-card"><div class="ts-value">{cum_ret:+.1f}%</div><div class="ts-label">累积收益</div></div>
    <div class="ts-card"><div class="ts-value">{t['total_count']}</div><div class="ts-label">已结算</div></div>
    <div class="ts-card"><div class="ts-value">{win_rate:.0f}%</div><div class="ts-label">胜率</div></div>
    <div class="ts-card"><div class="ts-value">{open_len}</div><div class="ts-label">当前持仓</div></div>
  </div>
  <div class="detail-title" style="margin-top:16px">当前持仓</div>
  <table><thead><tr><th>代码</th><th>名称</th><th>入场日</th><th>入场价</th><th>现价</th><th>浮盈</th><th>持仓天</th></tr></thead><tbody>{active_rows}</tbody></table>
  <div class="detail-title" style="margin-top:16px">最近结算记录</div>
  <table><thead><tr><th>代码</th><th>名称</th><th>入场日</th><th>入场价</th><th>退出日</th><th>退出价</th><th>收益</th><th></th></tr></thead><tbody>{closed_rows}</tbody></table>
</div>"""

    # Consensus section
    consensus_list = [r for r in rankings if r["consensus"]]
    consensus_html = "".join(f'<span class="consensus-item">{r["code"]} {extra_info.get(r["code"],{}).get("name","")}</span>' for r in consensus_list)

    timestr = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    sector_options = "".join(f'<option value="{s}">{s}</option>' for s in sector_map.SECTORS)

    n_isir_top = sum(1 for r in rankings if r["in_isir_top"])
    n_glm_top = sum(1 for r in rankings if r["in_glm_top"])
    n_doubao_top = sum(1 for r in rankings if r["in_doubao_top"])

    # 回测结果
    bt_html = ""
    if backtest_summary and best_strat:
        strat_names = {"isir":"ISIR","glm":"GLM","doubao":"SS分排名"}
        bt_rows = ""
        for strat in ["isir","glm","doubao"]:
            s = backtest_summary[strat]
            crown = " 👑主评分" if strat == best_strat else ""
            bg = "background:#fef3c7;border:2px solid var(--consensus-c)" if strat==best_strat else ""
            bt_rows += f"""<div class="ts-card" style="{bg}">
<div class="ts-value" style="color:{ 'var(--isir-c)' if strat=='isir' else 'var(--glm-c)' if strat=='glm' else 'var(--doubao-c)' }">{backtest_summary[strat]['win_20d']}%</div>
<div class="ts-label">{strat_names[strat]} 20日胜率{crown}</div>
<div style="font-size:10px;color:var(--text-secondary);margin-top:4px">均收益{s['avg_20d']:+.1f}% | 最大{s['max_20d']:+.1f}% | {s['total_trades']}笔</div>
</div>"""
        bt_html = f"""<div class="consensus-panel" style="background:#f0f9ff;border-color:var(--primary);margin-bottom:16px">
<h3 style="color:var(--primary)">📊 180天回测对比 — 主评分: {strat_names.get(best_strat,'-')} 👑</h3>
<div class="trade-summary">{bt_rows}</div>
<div style="font-size:11px;color:var(--text-secondary);margin-top:8px">* 综合评分 = 20日胜率×0.6 + 平均收益×2 | 回测区间180交易日 | TOP30信号买入模拟</div>
</div>"""

    # Tab ordering: best strategy first, then the rest
    all_tabs = [("overview","纵览","")]
    if best_strat:
        all_tabs.append((best_strat, {"isir":"ISIR","glm":"GLM","doubao":"SS分排名"}[best_strat], f' style="color:{ {"isir":"var(--isir-c)","glm":"var(--glm-c)","doubao":"var(--doubao-c)"}[best_strat] }"'))
    for s in ["isir","glm","doubao"]:
        if s != best_strat:
            all_tabs.append((s, {"isir":"ISIR","glm":"GLM","doubao":"SS分排名"}[s], ""))
    all_tabs.append(("sector","板块",""))
    all_tabs.append(("signals","操作记录",""))
    all_tabs.append(("ss","SS分",""))
    all_tabs.append(("qrebound","质量反弹",' style="color:#0d9488"'))

    tab_btns = ""
    for i, (tid, label, extra_style) in enumerate(all_tabs):
        active = "active" if i==0 else ""
        cls = "sector-tab" if tid=="sector" else ""
        tab_btns += f'<button class="tab-btn {cls} {active}" onclick="switchTab(\'{tid}\')"{extra_style}>{label}</button>\n'

    html = f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>统一评分决策日报 v2.0 {date_str}</title>
<style>
:root{{--bg:#f5f7fa;--card-bg:#fff;--text:#1a1a2e;--text-secondary:#666;--border:#e0e5ec;--primary:#1a56db;--isir-c:#2563eb;--glm-c:#7c3aed;--doubao-c:#059669;--consensus-c:#dc2626;--positive:#dc2626;--negative:#16a34a;--rank-up:#dc2626;--rank-down:#16a34a;}}
*{{margin:0;padding:0;box-sizing:border-box}}body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:var(--bg);color:var(--text);line-height:1.6}}
.container{{max-width:1500px;margin:0 auto;padding:20px}}
.header{{background:linear-gradient(135deg,#1a1a2e,#16213e);color:white;padding:30px 40px;border-radius:16px;margin-bottom:20px}}
.header h1{{font-size:26px;margin-bottom:6px}}.header .meta{{font-size:14px;opacity:.8}}
.dashboard{{display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin-bottom:20px}}
.dash-card{{background:var(--card-bg);border-radius:12px;padding:16px;text-align:center;border:1px solid var(--border)}}
.dash-card .value{{font-size:32px;font-weight:700}}.dash-card .label{{font-size:12px;color:var(--text-secondary);margin-top:4px}}
.c-isir{{color:var(--isir-c)}}.c-glm{{color:var(--glm-c)}}.c-doubao{{color:var(--doubao-c)}}.c-cons{{color:var(--consensus-c)}}
.consensus-panel{{background:#fff7ed;border:2px solid var(--consensus-c);border-radius:12px;padding:20px;margin-bottom:20px}}
.consensus-panel h3{{color:var(--consensus-c);margin-bottom:12px;font-size:18px}}
.consensus-list{{display:flex;flex-wrap:wrap;gap:8px}}
.consensus-item{{background:var(--consensus-c);color:white;padding:4px 10px;border-radius:6px;font-size:13px;font-weight:500}}
.tab-bar{{display:flex;gap:0;background:var(--card-bg);border-radius:12px 12px 0 0;border:1px solid var(--border);border-bottom:none;overflow:hidden;margin-bottom:0;margin-top:16px}}
.tab-btn{{flex:1;padding:13px 8px;text-align:center;cursor:pointer;font-size:14px;font-weight:600;border:none;background:transparent;color:var(--text-secondary);transition:all .2s;border-bottom:3px solid transparent}}
.tab-btn:hover{{background:#f8fafc;color:var(--text)}}
.tab-btn.active{{background:white;border-bottom-color:var(--primary);color:var(--primary)}}
.tab-btn.sector-tab.active{{border-bottom-color:#f59e0b;color:#f59e0b}}
.panel{{display:none}}.panel.active{{display:block}}
.filter-bar{{background:var(--card-bg);padding:10px 16px;border:1px solid var(--border);border-top:none;display:flex;flex-wrap:wrap;gap:10px;align-items:center;font-size:13px}}
.filter-bar select,.filter-bar input{{padding:5px 10px;border:1px solid var(--border);border-radius:6px;font-size:13px}}
.table-wrap{{background:var(--card-bg);border-radius:0 0 12px 12px;overflow-x:auto;border:1px solid var(--border);border-top:none}}
table{{width:100%;border-collapse:collapse;font-size:12px}}
thead{{background:#f1f5f9;position:sticky;top:0}}
th{{padding:8px 5px;text-align:left;font-weight:600;color:var(--text-secondary);cursor:pointer;white-space:nowrap;font-size:11px}}
td{{padding:6px 5px;border-bottom:1px solid var(--border)}}
tr:hover{{background:#f8fafc}}
tr.row-consensus{{background:#fef3c7!important}}
tr.row-consensus:hover{{background:#fde68a!important}}
.num{{text-align:right;font-variant-numeric:tabular-nums;font-size:12px}}
.positive{{color:var(--positive)!important}}.negative{{color:var(--negative)!important}}
.rank-num{{font-weight:700;font-size:13px}}
.rank-col{{min-width:55px}}
.arrow-up{{color:var(--rank-up);font-weight:bold;font-size:11px}}
.arrow-down{{color:var(--rank-down);font-weight:bold;font-size:11px}}
.arrow-flat{{color:#999;font-size:11px}}
.badge{{padding:2px 6px;border-radius:8px;font-size:10px;font-weight:600;white-space:nowrap}}
.badge.consensus{{background:var(--consensus-c);color:white;font-size:11px}}
.badge.sig-buy{{background:#16a34a;color:white;font-size:10px;margin-right:2px}}
.badge.sig-sell{{background:#dc2626;color:white;font-size:10px;margin-right:2px}}
.badge.sig-hold{{background:#2563eb;color:white;font-size:10px;margin-right:2px}}
.badge.sig-watch{{background:#d97706;color:white;font-size:10px;margin-right:2px}}
.tag{{padding:1px 5px;border-radius:4px;font-size:9px;font-weight:600;margin-right:2px}}
.tag.top-isir{{background:#dbeafe;color:var(--isir-c)}}.tag.top-glm{{background:#ede9fe;color:var(--glm-c)}}.tag.top-doubao{{background:#d1fae5;color:var(--doubao-c)}}.tag.top-vcp-breakout{{background:#fee2e2;color:#dc2626;font-weight:600}}.tag.top-vcp-pre{{background:#fef3c7;color:#d97706}}.tag.top-rebound{{background:#fce7f3;color:#be185d;font-weight:600}}.tag.top-fund{{background:#cffafe;color:#0891b2;font-weight:600}}
.ss-col{{color:var(--primary);font-weight:600;font-size:13px}}
.section-title{{font-size:18px;font-weight:700;margin:28px 0 12px;color:var(--text)}}
.detail-row td{{padding:0}}
.detail-card{{padding:14px 20px;background:#f8fafc;border-top:2px dashed var(--border)}}
.detail-grid{{display:grid;grid-template-columns:1fr 1fr;gap:20px}}
.detail-box{{background:white;border-radius:8px;padding:12px;border:1px solid var(--border);min-width:300px}}
.detail-title{{font-size:14px;font-weight:700;margin-bottom:10px;color:var(--primary)}}
.indicator-list{{display:flex;flex-direction:column;gap:6px}}
.ind-item{{font-size:13px;padding:3px 0;border-bottom:1px solid #f1f5f9}}
.ind-sub{{font-size:11px;color:var(--text-secondary);padding-left:8px}}
.factor-table{{width:100%;font-size:11px}}
.factor-table th,.factor-table td{{padding:3px 6px;border-bottom:1px solid #eee}}
.sector-block{{margin-bottom:16px;background:var(--card-bg);border-radius:10px;border:1px solid var(--border);overflow:hidden}}
.sector-header{{background:linear-gradient(135deg,#fff7ed,#fef3c7);padding:12px 16px;font-weight:700;font-size:14px;cursor:pointer}}
.sector-body{{padding:0;display:block}}
.sector-stock{{display:inline-flex;gap:8px;padding:2px 8px;background:#f1f5f9;border-radius:4px;margin:2px;font-size:11px}}
.trade-panel{{margin-top:20px}}
.trade-summary{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:10px}}
.ts-card{{background:var(--card-bg);border-radius:8px;padding:14px;text-align:center;border:1px solid var(--border)}}
.ts-value{{font-size:24px;font-weight:700}}.ts-label{{font-size:11px;color:var(--text-secondary);margin-top:2px}}
.footer{{text-align:center;padding:20px;color:var(--text-secondary);font-size:12px}}
.market-overview{{margin-bottom:16px}}
.mo-grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}}
.mo-card{{background:var(--card-bg);border-radius:10px;padding:14px;border:1px solid var(--border)}}
.mo-card:first-child{{border-width:2px}}
.mo-title{{font-size:12px;font-weight:700;color:var(--text-secondary);margin-bottom:6px;text-transform:uppercase;letter-spacing:.5px}}
.mo-big{{font-size:36px;font-weight:800;line-height:1}}
.mo-sub{{font-size:11px;color:var(--text-secondary);margin-top:2px}}
.mo-breadth-bar{{height:6px;background:#e5e7eb;border-radius:3px;margin-top:6px;overflow:hidden}}
.mo-breadth-fill{{height:100%;border-radius:3px;transition:width .5s}}
.sector-bar-row{{display:flex;align-items:center;gap:6px;padding:2px 0;font-size:12px}}
.sector-bar-name{{min-width:70px;font-weight:500}}
.sector-bar-val{{font-weight:600;min-width:50px}}
@media(max-width:768px){{.dashboard{{grid-template-columns:repeat(3,1fr)}}.detail-grid{{grid-template-columns:1fr}}.mo-grid{{grid-template-columns:repeat(2,1fr)}}}}
</style></head><body><div class="container">
<div class="header"><h1>统一评分决策日报 v3.5</h1><div class="meta">日期:{date_str} | 股票池:{n_total}只 | K线:{freshness.get('kline_latest','-')} | 行情:{freshness.get('extra_latest','-')}</div></div>

<!-- 市场全景解读面板 -->
{_build_market_overview(mkt_overview, rankings, extra_info, n_consensus, n_total, TOP_N, bt_html if backtest_summary else "", consensus_html)}

<div class="tab-bar">
{tab_btns}</div>

<div class="panel active" id="panel-overview"><div class="filter-bar">
<label>板块:</label><select onchange="filterPanel('overview')" id="sector-overview"><option value="all">全部</option>{sector_options}</select>
<label>仅共识:</label><input type="checkbox" id="consensus-overview" onchange="filterPanel('overview')">
<label>搜索:</label><input type="text" id="search-overview" placeholder="代码/名称" oninput="filterPanel('overview')" style="width:120px">
</div><div class="table-wrap"><table><thead><tr>
<th>代码</th><th>名称/板块</th><th>价格</th><th>日涨跌</th><th>5日</th><th>10日</th><th>20日</th><th>SS分</th><th>ISIR</th><th>GLM</th><th>共识</th>
</tr></thead><tbody id="body-overview">{ss_rows}</tbody></table></div></div>

<div class="panel" id="panel-ss"><div class="filter-bar">
<label>板块:</label><select onchange="filterPanel('ss')" id="sector-ss"><option value="all">全部</option>{sector_options}</select>
<label>仅共识:</label><input type="checkbox" id="consensus-ss" onchange="filterPanel('ss')">
<label>搜索:</label><input type="text" id="search-ss" placeholder="代码/名称" oninput="filterPanel('ss')" style="width:120px">
<small style="color:var(--text-secondary);margin-left:auto">按SS分从高到低排列</small>
</div><div class="table-wrap"><table><thead><tr>
<th>代码</th><th>名称/板块</th><th>价格</th><th>日涨跌</th><th>5日</th><th>10日</th><th>20日</th><th>SS分↓</th><th>ISIR</th><th>GLM</th><th>共识</th>
</tr></thead><tbody id="body-ss">{ss_rows}</tbody></table></div></div>

<div class="panel" id="panel-isir"><div class="filter-bar">
<label>板块:</label><select onchange="filterPanel('isir')" id="sector-isir"><option value="all">全部</option>{sector_options}</select>
<label>仅共识:</label><input type="checkbox" id="consensus-isir" onchange="filterPanel('isir')">
<label>搜索:</label><input type="text" id="search-isir" placeholder="代码/名称" oninput="filterPanel('isir')" style="width:120px">
<small style="color:var(--text-secondary);margin-left:auto">按ISIR排名</small>
</div><div class="table-wrap"><table><thead><tr>
<th>代码</th><th>名称/板块</th><th>价格</th><th>日涨跌</th><th>5日</th><th>10日</th><th>20日</th><th>SS分</th><th>ISIR↓</th><th>GLM</th><th>共识</th>
</tr></thead><tbody id="body-isir">{isir_rows}</tbody></table></div>
{trade_ledger_html('isir')}</div>

<div class="panel" id="panel-glm"><div class="filter-bar">
<label>板块:</label><select onchange="filterPanel('glm')" id="sector-glm"><option value="all">全部</option>{sector_options}</select>
<label>仅共识:</label><input type="checkbox" id="consensus-glm" onchange="filterPanel('glm')">
<label>搜索:</label><input type="text" id="search-glm" placeholder="代码/名称" oninput="filterPanel('glm')" style="width:120px">
<small style="color:var(--text-secondary);margin-left:auto">按GLM排名</small>
</div><div class="table-wrap"><table><thead><tr>
<th>代码</th><th>名称/板块</th><th>价格</th><th>日涨跌</th><th>5日</th><th>10日</th><th>20日</th><th>SS分</th><th>ISIR</th><th>GLM↓</th><th>共识</th>
</tr></thead><tbody id="body-glm">{glm_rows}</tbody></table></div>
{trade_ledger_html('glm')}</div>

<div class="panel" id="panel-doubao"><div class="filter-bar">
<label>板块:</label><select onchange="filterPanel('doubao')" id="sector-doubao"><option value="all">全部</option>{sector_options}</select>
<label>仅共识:</label><input type="checkbox" id="consensus-doubao" onchange="filterPanel('doubao')">
<label>搜索:</label><input type="text" id="search-doubao" placeholder="代码/名称" oninput="filterPanel('doubao')" style="width:120px">
<small style="color:var(--text-secondary);margin-left:auto">按SS分排名</small>
</div><div class="table-wrap"><table><thead><tr>
<th>代码</th><th>名称/板块</th><th>价格</th><th>日涨跌</th><th>5日</th><th>10日</th><th>20日</th><th>SS分</th><th>ISIR</th><th>GLM</th><th>共识</th>
</tr></thead><tbody id="body-doubao">{doubao_rows}</tbody></table></div>
{trade_ledger_html('doubao')}</div>

<!-- 历史操作建议记录面板 -->
{_build_signal_history(signal_history, trades)}

<!-- 第五视图: 质量反弹面板 (超跌反弹 × 三体系认可度交叉; 用户2026-09-14决定替代第四体系独立展示) -->
<div class="panel" id="panel-qrebound">
<div class="filter-bar">
<label>板块:</label><select onchange="filterPanel('qrebound')" id="sector-qrebound"><option value="all">全部</option>{sector_options}</select>
<label>搜索:</label><input type="text" id="search-qrebound" placeholder="代码/名称" oninput="filterPanel('qrebound')" style="width:120px">
<small style="color:#0d9488;margin-left:auto">质量反弹 = 超跌反弹 ∩ ISIR/GLM前{QR_THRESHOLD}名 | 跌得深 且 质地未坏</small>
</div>
<div style="background:#ecfdf5;border:1px solid #99f6e4;border-radius:8px;padding:10px 14px;margin-bottom:10px;font-size:12px;color:#065f46">
<strong>第五视图·质量反弹:</strong> 第四体系(超跌反弹)不看质量的缺陷修正——纯超跌名单常被弱势股污染(深跌的多数是基本面走坏者)。
本视图做交叉过滤: 超跌反弹有效 <strong>且</strong> ISIR或GLM排名前{QR_THRESHOLD}(三体系认可质地未坏), 定位"错杀的好票"。
<span style="color:#6b7280">逻辑: 均值回归(错杀修复), 与三体系的动量延续方向相反, 与第四体系互补。适用: 事件冲击后的科技链/板块错杀。</span>
</div>
<div style="margin-bottom:10px"><div class="dash-card" style="text-align:left;padding:14px 18px">
<div style="font-size:13px;font-weight:700;color:#0d9488">今日质量反弹标的: {len(qr_list)}只 (超跌反弹{len(rebound_valid)}只 × 三体系认可, 交集{len(qr_list)}只)</div>
</div></div>
<div style="margin:-6px 0 12px 0">{qrebound_bt_html}</div>
<div class="table-wrap" style="margin-bottom:12px"><table><thead><tr>
<th>#</th><th>代码</th><th>名称/板块</th><th>现价</th><th>5日</th><th>RSI</th><th>超跌分↓</th><th>阶段</th><th>ISIR</th><th>GLM</th><th>最好</th><th>ATR止损价</th><th>反弹目标</th><th>三体系</th>
</tr></thead><tbody id="body-qrebound">{qrebound_rows}</tbody></table></div>
</div>

<div class="panel" id="panel-sector"><div class="section-title">按板块纵览</div>
{sector_html}</div>

<div class="footer"><p>统一评分引擎 v3.5 | 数据:腾讯财经 | 生成于 {timestr}</p>
<p style="margin-top:3px;font-size:11px">ISIR=33因子ICIR原始加权 | GLM=mfi/pct_52w方向反转 | SS分=传统技术评分(独立) | 质量反弹=超跌反弹∩ISIR/GLM前150(第五视图,2026-09-14起替代第四体系独立展示) | 共识=ISIR∩GLM∩SS分排名 | 点击行展开因子明细 | 每策略含交易账本</p></div></div>
<script>
function switchTab(n){{document.querySelectorAll('.panel').forEach(p=>p.classList.remove('active'));document.querySelectorAll('.tab-btn').forEach(b=>b.classList.remove('active'));document.getElementById('panel-'+n).classList.add('active');[...document.querySelectorAll('.tab-btn')].find(b=>b.textContent.includes(n=='overview'?'纵览':n=='sector'?'板块':n.toUpperCase())||b.onclick.toString().includes("'"+n+"'"))?.classList.add('active')}}
function toggleRow(id){{var r=document.getElementById(id);if(r)r.style.display=r.style.display==='none'?'table-row':'none'}}
function filterPanel(name){{var s=document.getElementById('sector-'+name)?.value;var c=document.getElementById('consensus-'+name)?.checked;var q=(document.getElementById('search-'+name)?.value||'').toLowerCase();var rows=document.querySelectorAll('#body-'+name+' tr');rows.forEach(function(row){{if(row.classList.contains('detail-row'))return;var cells=row.getElementsByTagName('td');if(cells.length<11)return;var code=cells[0].textContent.trim();var nm=cells[1].textContent.trim();var badge=cells[10].textContent.trim();var show=true;if(s&&s!=='all'&&!cells[1].textContent.includes(s))show=false;if(c&&!badge.includes('共识'))show=false;if(q&&!code.toLowerCase().includes(q)&&!nm.toLowerCase().includes(q))show=false;row.style.display=show?'':'none';var next=row.nextElementSibling;if(next&&next.classList.contains('detail-row'))next.style.display=show?(next.style.display):'none'}})}}
function toggleSector(s){{var b=document.getElementById('sec-'+s)?.querySelector('.sector-body');if(b)b.style.display=b.style.display==='none'?'block':'none'}}
</script></body></html>"""
    return html

# ================================================================
# Main
# ================================================================

def main():
    print("="*60)
    print("  统一评分引擎 v3.5")
    print("  ISIR | GLM | SS分排名 | 质量反弹(第五视图,替代第四体系展示) + 信号追踪")
    print("  ICIR重标定 + VCP形态 + 外围市场门控 + 超跌反弹趋势(作第五视图输入)")
    print("="*60)

    if not os.path.exists(STOCK_CODES_FILE):
        codes = sorted(sector_map.STOCK_SECTOR.keys())
        with open(STOCK_CODES_FILE,"w") as f: f.write("\n".join(codes))
        print(f"  从sector_map生成股票池: {len(codes)}只")
    else:
        with open(STOCK_CODES_FILE) as f: codes = [l.strip() for l in f if l.strip()]
        print(f"  股票池: {len(codes)}只")

    db = StockDB()
    freshness = db.check_data_freshness(codes)
    print(f"  DB: K线最新 {freshness['kline_latest']} 行情最新 {freshness['extra_latest']}")

    print(f"\n  [1/4] 拉取数据...")
    klines = db.get_klines(codes, days=300)
    extra_info = db.get_extra_info(codes, force_refresh=True)
    fund_flows = db.get_fund_flows(codes)
    print(f"  K线:{len(klines)} | 行情:{len(extra_info)} | 资金流:{len(fund_flows)}")

    print(f"\n  [2/4] 计算33因子 + 三套排名...")
    sectors = {code: sector_map.get_sector(code) for code in codes}
    factor_data, return_data = compute_all_factors(klines, extra_info, fund_flows, {}, sectors)
    print(f"  有效因子: {len(factor_data)}只")

    rankings = compute_rankings(factor_data)
    n_consensus = sum(1 for r in rankings if r["consensus"])

    # VCP 形态扫描
    print(f"\n  [VCP扫描] 检测波动收缩形态...")
    vcp_map = scan_vcp_signals(klines)
    # 标注到排名数据
    for r in rankings:
        vcp = vcp_map.get(r["code"])
        if vcp:
            r["vcp_status"] = "突破" if vcp["breakout"] else "预突破"
            r["vcp_info"] = vcp
        else:
            r["vcp_status"] = ""
            r["vcp_info"] = None

    # 资金确认标签 (加分项, 与VCP并列, 不参与共识排名)
    # 判定: 当日主力净流入>0 且 5日累计净流入>0 (真实持续流入, 避免单日脉冲)
    # 回测依据: 主力/超大单净流入为弱正向信号(20日+0.8%), 详见 2026-09-21 长历史回测
    for r in rankings:
        ff = fund_flows.get(r["code"], {})
        m_today = ff.get("main_net_today", 0) or 0
        m_5d = ff.get("main_net_5d", 0) or 0
        r["fund_confirm"] = bool(m_today > 0 and m_5d > 0)
        r["fund_today"] = m_today
        r["fund_5d"] = m_5d

    # 超跌反弹评分 (第四体系，独立标记，不参与三体系共识)
    print(f"\n  [超跌反弹] 扫描超跌反弹趋势信号...")
    n_rebound = compute_rebound_scores(rankings, klines, extra_info)
    print(f"  超跌反弹有效: {n_rebound}只 (超跌分≥5+总分≥20)")

    # 第五视图: 质量反弹 (超跌反弹 ∩ ISIR/GLM前QR_THRESHOLD, 定位错杀好票)
    qr_list = [r for r in rankings if r.get("rebound_valid")
               and min(r.get("isir_rank", 9999), r.get("glm_rank", 9999)) <= QR_THRESHOLD]
    qr_list.sort(key=lambda x: x.get("rebound_score", 0), reverse=True)
    print(f"  [质量反弹] 第五视图: {len(qr_list)}只 (超跌反弹{n_rebound}只 ∩ ISIR/GLM前{QR_THRESHOLD}名)")
    qr_backtest = None
    if os.path.exists(QR_BACKTEST_FILE):
        try:
            with open(QR_BACKTEST_FILE) as f: qr_backtest = json.load(f)
        except (json.JSONDecodeError, IOError):
            pass

    # 市场趋势门控
    print(f"\n  [市场趋势门控] 获取上证综指...")
    index_klines = fetch_index_klines(days=300)
    mkt_trend, mkt_ma50, mkt_idx_close, mkt_trend_label = calc_market_trend(index_klines)
    print(f"  上证综指: {mkt_idx_close} | MA50: {mkt_ma50} | {mkt_trend_label}")
    if mkt_trend < 0:
        print(f"  ⚠️ 空头市场: 仓位上限 {MAX_POSITIONS}→{MAX_POSITIONS//2}, 浮亏持仓强制减仓")

    # 外围市场门控
    print(f"\n  [外围市场门控] 获取美股隔夜数据...")
    global_markets = fetch_global_markets()
    global_level, global_pos_adj, global_label, global_interp = compute_global_gate(global_markets)
    for gname, ginfo in global_markets.items():
        ret_str = f"{ginfo['overnight_ret']:+.2f}%" if gname != "VIX恐慌指数" else f"close={ginfo['close']}"
        print(f"  {gname}: {ret_str} | {ginfo['impact']} | 门控={ginfo['gate']}")
    print(f"  → {global_label}")
    print(f"  → {global_interp}")
    # 综合仓位上限 (上证门控 × 外围门控)
    effective_max = int(MAX_POSITIONS * (0.5 if mkt_trend < 0 else 1.0) * global_pos_adj)
    if effective_max < MAX_POSITIONS:
        print(f"  ⚠️ 综合仓位上限: {MAX_POSITIONS}→{effective_max}")

    print(f"\n  [3/4] 信号追踪 + 交易更新...")
    date_str = datetime.now().strftime("%Y-%m-%d")
    history = save_history(rankings, extra_info, date_str)
    trades, today_signals, signal_history = update_trades(rankings, extra_info, date_str, market_trend=mkt_trend, global_pos_adjust=global_pos_adj)

    for strat in ["isir","glm","doubao"]:
        t = trades[strat]
        sigs = today_signals.get(strat,{})
        n_buy = sum(1 for v in sigs.values() if v=="buy")
        n_sell = sum(1 for v in sigs.values() if v=="sell")
        n_hold = sum(1 for v in sigs.values() if v=="hold")
        print(f"  {strat}: 持仓{len(t['open'])}只 | 今日🟢买入{n_buy} 🔴卖出{n_sell} ✅持有{n_hold} | 累积收益{t['cumulative_return']:+.1f}% | 已结算{t['total_count']}笔 | 胜率{round(t['win_count']/t['total_count']*100,1) if t['total_count']>0 else 0:.0f}%")
    print(f"  历史信号总数: {len(signal_history)}")

    # 6. 市场全景解读
    print(f"\n  [盘面解读] 计算市场宽度...")
    mkt_overview = compute_market_overview(klines, extra_info, rankings, index_klines, mkt_trend, mkt_trend_label)
    mkt_overview["global_markets"] = global_markets
    mkt_overview["global_gate"] = {"level": global_level, "label": global_label, "interpretation": global_interp, "pos_adjust": global_pos_adj}
    print(f"  MA20以上占比: {mkt_overview['breadth']['above_ma20_pct']}% | {mkt_overview['breadth']['temperature']}")
    print(f"  涨跌比: {mkt_overview['advance']['up']}:{mkt_overview['advance']['down']} | 涨停{mkt_overview['limits']['up']} 跌停{mkt_overview['limits']['down']}")

    print(f"\n  [回测] 180天回测对比...")
    bt_summary, best_strat = run_backtest(klines, extra_info, sectors, backtest_days=180)
    if bt_summary:
        for strat in ["isir","glm","doubao"]:
            s = bt_summary[strat]
            crown = " 👑" if strat==best_strat else ""
            print(f"  {strat}{crown}: 20日胜率{s['win_20d']}% | 均收益{s['avg_20d']:+.1f}% | 最大{s['max_20d']:+.1f}%")
    else:
        best_strat = "isir"

    print(f"\n  [4/4] 生成报告... 共识TOP{TOP_N}: {n_consensus}只")
    n_vcp_breakout = sum(1 for r in rankings if r.get("vcp_status") == "突破")
    n_vcp_pre = sum(1 for r in rankings if r.get("vcp_status") == "预突破")
    n_fund_confirm = sum(1 for r in rankings if r.get("fund_confirm"))
    print(f"  VCP: 突破{n_vcp_breakout}只 | 预突破{n_vcp_pre}只 | 资金确认{n_fund_confirm}只")
    html = build_html(rankings, extra_info, history, trades, return_data, date_str, freshness, bt_summary, best_strat, mkt_overview, today_signals, signal_history, qr_backtest)
    html_path = os.path.join(OUTPUT_DIR, f"unified_{date_str}.html")
    with open(html_path, "w", encoding="utf-8") as f: f.write(html)

    json_path = os.path.join(OUTPUT_DIR, f"unified_{date_str}.json")
    with open(json_path, "w") as f: json.dump(rankings, f, ensure_ascii=False, indent=2)

    consensus_list = [r for r in rankings if r["consensus"]]
    rebound_list = [r for r in rankings if r.get("rebound_valid")]
    print(f"\n  {'='*60}")
    print(f"  ✅ 报告已生成!")
    print(f"     HTML: {html_path}")
    print(f"     JSON: {json_path}")
    print(f"     共识TOP{TOP_N}: {n_consensus} 只")
    if consensus_list:
        print(f"  共识标的:")
        for i,r in enumerate(consensus_list,1):
            name = extra_info.get(r["code"],{}).get("name","")
            vcp_tag = f" VCP{r['vcp_status']}" if r.get("vcp_status") else ""
            rb_tag = f" 超跌{r.get('rebound_stage','')}" if r.get("rebound_valid") else ""
            fund_tag = " 资金确认" if r.get("fund_confirm") else ""
            print(f"     {i}. {r['code']} {name} | ISIR#{r['isir_rank']} GLM#{r['glm_rank']} SS分排名#{r['doubao_rank']}{vcp_tag}{fund_tag}{rb_tag}")
    if rebound_list:
        print(f"  超跌反弹标的(第四体系,仅作第五视图输入): {len(rebound_list)}只")
    if qr_list:
        print(f"  质量反弹标的·第五视图({len(qr_list)}只):")
        for i,r in enumerate(qr_list[:5],1):
            name = extra_info.get(r["code"],{}).get("name","")
            atr = r.get("rebound_atr",{})
            print(f"     {i}. {r['code']} {name} | 超跌分{r['rebound_score']:.0f}[{r['rebound_stage']}] "
                  f"ISIR#{r['isir_rank']} GLM#{r['glm_rank']} | 止损{atr.get('stop_price',0):.2f} 目标{atr.get('tp_price',0):.2f}")
    print(f"  {'='*60}")
    db.stats()
    return html_path

if __name__ == "__main__":
    main()
