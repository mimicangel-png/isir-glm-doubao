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

# ================================================================
# ICIR Weights — 精确来自 scoring_engine_icir.py / v3_vs_glm_tracker.py
# ================================================================

ICIR_V3 = {
    "turnover_z": 0.451, "log_mcap": 0.162, "mfi": 0.153, "pct_52w": 0.091,
    "pe_percentile": 0.078, "pb_percentile": 0.078, "gap_open": 0.072,
    "max_dd_20d": 0.052, "ma_bull": 0.029, "rsi_signal": 0.028,
    "macd_signal": 0.026, "cmf": 0.025, "vol_price": 0.025,
    "dev_ma20": 0.024, "vol_ratio_5d": 0.023, "vwap_premium": 0.022,
    "ret_5d": 0.021, "streak": 0.020, "event_score": 0.020,
    "event_count": 0.018, "sector_rsi": 0.015, "sector_momentum": 0.012,
    "inflow_rate": 0.010, "main_flow_5d": 0.008, "main_flow_20d": 0.006,
    "amplitude_z": 0.005, "ret_20d": 0.004, "volatility_20d": 0.003,
    "roe_rank": 0.001, "gross_margin_rank": 0.001, "ocf_ratio_rank": 0.001,
}

ICIR_GLM = dict(ICIR_V3)
ICIR_GLM["mfi"] = -0.153
ICIR_GLM["pct_52w"] = -0.091

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

        # SS评分 — 完全对齐 scoring_engine_icir.py 源码算法
        # 技术面: 基分50, 离散加分/减分
        rsi_val = factors.get("_rsi",50)  # 直接用存储的原始RSI值
        tech_delta = 0
        # MA多头排列: MA5>MA10>MA20
        ma5_v = factors.get("_ma5",0); ma10_v = factors.get("_ma10",0); ma20_v = factors.get("_ma20",0)
        if ma5_v > ma10_v > ma20_v: tech_delta += 15
        elif ma5_v < ma10_v < ma20_v: tech_delta -= 10  # 完整空头排列才扣分
        # MACD: DIF>0 且 DIF>DEA
        dif = factors.get("_dif",0); dea = factors.get("_dea",0)
        if dif > 0 and dif > dea: tech_delta += 5
        # RSI (A股阈值调整: 40-60中性, >85超买危险, <30超卖机会)
        if rsi_val > 85: tech_delta -= 5   # 超买回调风险
        elif rsi_val < 30: tech_delta += 8  # 超卖反弹机会
        elif rsi_val < 40: tech_delta -= 3  # 偏弱
        tech_score = max(5, min(95, 50 + tech_delta))

        # 资金面: 基分50, CMF加减分
        cmf_raw = factors.get("cmf",0)
        cap_delta = 0
        if cmf_raw > 0.1: cap_delta += 8
        elif cmf_raw > 0: cap_delta += 3
        elif cmf_raw < -0.1: cap_delta -= 8
        elif cmf_raw < 0: cap_delta -= 3  # 对称: 微流出也扣分
        capital_score = max(5, min(95, 50 + cap_delta))

        # 信息面: 固定50
        info_score = 50

        ss_score = tech_score * 0.35 + capital_score * 0.55 + info_score * 0.10

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

def update_trades(rankings, extra_info, date_str):
    trades = load_trades()
    signals = load_signals()
    today_signals = {}
    n_total = len(rankings)
    rank_map = {r["code"]: r for r in rankings}

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
        available_slots = MAX_POSITIONS - len(still_open)
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

def compute_market_overview(klines, extra_info, rankings):
    """计算市场宽度、指数状态、盘面解读"""
    result = {}

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

    # 5. 涨停/跌停统计（涨跌>9.5%近似）
    limit_up = sum(1 for r in rankings if abs(extra_info.get(r["code"],{}).get("change_pct",0)) > 9.5 and extra_info.get(r["code"],{}).get("change_pct",0) > 0)
    limit_down = sum(1 for r in rankings if abs(extra_info.get(r["code"],{}).get("change_pct",0)) > 9.5 and extra_info.get(r["code"],{}).get("change_pct",0) < 0)
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
            shares_cache = {}
            for code in day_klines:
                today_info = extra_info.get(code, {})
                tp = today_info.get("price", 0)
                tm = today_info.get("mcap", 0)
                if tp > 0 and tm > 0:
                    shares_cache[code] = tm / tp  # 股本(股)
                else:
                    shares_cache[code] = 0

            day_extra = {}
            for code in day_klines:
                k_bars = day_klines.get(code, [])
                if k_bars and len(k_bars) >= 2:
                    last = k_bars[-1]; prev = k_bars[-2]
                    avg_vol_5 = sum(b["volume"] for b in k_bars[-6:-1]) / 5 if len(k_bars) >= 6 else last["volume"]
                    vr = last["volume"] / avg_vol_5 if avg_vol_5 > 0 else 1.0
                    # 用股本反推历史市值和换手率
                    shares = shares_cache.get(code, 0)
                    hist_mcap = last["close"] * shares if shares > 0 else 0
                    hist_turnover = (last["volume"] / shares * 100) if shares > 0 else 0
                    today_info = extra_info.get(code, {})
                    day_extra[code] = {
                        "name": today_info.get("name", code),
                        "price": last["close"],
                        "change_pct": (last["close"]/prev["close"]-1)*100,
                        "pe_ttm": today_info.get("pe_ttm", 0) or 0,  # 用今日PE近似(有偏差但优于0)
                        "pb": today_info.get("pb", 0) or 0,
                        "mcap": hist_mcap,           # 从股本反推
                        "turnover": hist_turnover,   # 从股本反推
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

    <!-- 板块强弱 -->
    <div class="mo-card">
      <div class="mo-title">板块强弱</div>
      {sector_bars}
    </div>
  </div>
</div>
{bt_html}
<div class="consensus-panel" style="display:{'block' if n_consensus>0 else 'none'};margin-top:12px">
<h3>共识TOP{top_n} — ISIR ∩ GLM ∩ 豆包 ({n_consensus}只)</h3>
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


def build_html(rankings, extra_info, history, trades, return_data, date_str, freshness, backtest_summary=None, best_strat=None, mkt_overview=None, today_signals=None, signal_history=None):
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
            if r["in_doubao_top"]: top_tags += '<span class="tag top-doubao">豆包</span>'
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
<td class="num rank-col"><span class="rank-num c-doubao">#{r['doubao_rank']}</span><span class="{arrow_cls(doubao_diff)}">{doubao_arrow}</span></td>
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
        strat_names = {"isir":"ISIR","glm":"GLM","doubao":"豆包"}
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
        all_tabs.append((best_strat, {"isir":"ISIR","glm":"GLM","doubao":"豆包"}[best_strat], f' style="color:{ {"isir":"var(--isir-c)","glm":"var(--glm-c)","doubao":"var(--doubao-c)"}[best_strat] }"'))
    for s in ["isir","glm","doubao"]:
        if s != best_strat:
            all_tabs.append((s, {"isir":"ISIR","glm":"GLM","doubao":"豆包"}[s], ""))
    all_tabs.append(("sector","板块",""))
    all_tabs.append(("signals","操作记录",""))
    all_tabs.append(("ss","SS分",""))

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
.tag.top-isir{{background:#dbeafe;color:var(--isir-c)}}.tag.top-glm{{background:#ede9fe;color:var(--glm-c)}}.tag.top-doubao{{background:#d1fae5;color:var(--doubao-c)}}
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
<div class="header"><h1>统一评分决策日报 v2.0</h1><div class="meta">日期:{date_str} | 股票池:{n_total}只 | K线:{freshness.get('kline_latest','-')} | 行情:{freshness.get('extra_latest','-')}</div></div>

<!-- 市场全景解读面板 -->
{_build_market_overview(mkt_overview, rankings, extra_info, n_consensus, n_total, TOP_N, bt_html if backtest_summary else "", consensus_html)}

<div class="tab-bar">
{tab_btns}</div>

<div class="panel active" id="panel-overview"><div class="filter-bar">
<label>板块:</label><select onchange="filterPanel('overview')" id="sector-overview"><option value="all">全部</option>{sector_options}</select>
<label>仅共识:</label><input type="checkbox" id="consensus-overview" onchange="filterPanel('overview')">
<label>搜索:</label><input type="text" id="search-overview" placeholder="代码/名称" oninput="filterPanel('overview')" style="width:120px">
</div><div class="table-wrap"><table><thead><tr>
<th>代码</th><th>名称/板块</th><th>价格</th><th>日涨跌</th><th>5日</th><th>10日</th><th>20日</th><th>SS分</th><th>ISIR</th><th>GLM</th><th>豆包</th><th>共识</th>
</tr></thead><tbody id="body-overview">{ss_rows}</tbody></table></div></div>

<div class="panel" id="panel-ss"><div class="filter-bar">
<label>板块:</label><select onchange="filterPanel('ss')" id="sector-ss"><option value="all">全部</option>{sector_options}</select>
<label>仅共识:</label><input type="checkbox" id="consensus-ss" onchange="filterPanel('ss')">
<label>搜索:</label><input type="text" id="search-ss" placeholder="代码/名称" oninput="filterPanel('ss')" style="width:120px">
<small style="color:var(--text-secondary);margin-left:auto">按SS分从高到低排列</small>
</div><div class="table-wrap"><table><thead><tr>
<th>代码</th><th>名称/板块</th><th>价格</th><th>日涨跌</th><th>5日</th><th>10日</th><th>20日</th><th>SS分↓</th><th>ISIR</th><th>GLM</th><th>豆包</th><th>共识</th>
</tr></thead><tbody id="body-ss">{ss_rows}</tbody></table></div></div>

<div class="panel" id="panel-isir"><div class="filter-bar">
<label>板块:</label><select onchange="filterPanel('isir')" id="sector-isir"><option value="all">全部</option>{sector_options}</select>
<label>仅共识:</label><input type="checkbox" id="consensus-isir" onchange="filterPanel('isir')">
<label>搜索:</label><input type="text" id="search-isir" placeholder="代码/名称" oninput="filterPanel('isir')" style="width:120px">
<small style="color:var(--text-secondary);margin-left:auto">按ISIR排名</small>
</div><div class="table-wrap"><table><thead><tr>
<th>代码</th><th>名称/板块</th><th>价格</th><th>日涨跌</th><th>5日</th><th>10日</th><th>20日</th><th>SS分</th><th>ISIR↓</th><th>GLM</th><th>豆包</th><th>共识</th>
</tr></thead><tbody id="body-isir">{isir_rows}</tbody></table></div>
{trade_ledger_html('isir')}</div>

<div class="panel" id="panel-glm"><div class="filter-bar">
<label>板块:</label><select onchange="filterPanel('glm')" id="sector-glm"><option value="all">全部</option>{sector_options}</select>
<label>仅共识:</label><input type="checkbox" id="consensus-glm" onchange="filterPanel('glm')">
<label>搜索:</label><input type="text" id="search-glm" placeholder="代码/名称" oninput="filterPanel('glm')" style="width:120px">
<small style="color:var(--text-secondary);margin-left:auto">按GLM排名</small>
</div><div class="table-wrap"><table><thead><tr>
<th>代码</th><th>名称/板块</th><th>价格</th><th>日涨跌</th><th>5日</th><th>10日</th><th>20日</th><th>SS分</th><th>ISIR</th><th>GLM↓</th><th>豆包</th><th>共识</th>
</tr></thead><tbody id="body-glm">{glm_rows}</tbody></table></div>
{trade_ledger_html('glm')}</div>

<div class="panel" id="panel-doubao"><div class="filter-bar">
<label>板块:</label><select onchange="filterPanel('doubao')" id="sector-doubao"><option value="all">全部</option>{sector_options}</select>
<label>仅共识:</label><input type="checkbox" id="consensus-doubao" onchange="filterPanel('doubao')">
<label>搜索:</label><input type="text" id="search-doubao" placeholder="代码/名称" oninput="filterPanel('doubao')" style="width:120px">
<small style="color:var(--text-secondary);margin-left:auto">按豆包排名</small>
</div><div class="table-wrap"><table><thead><tr>
<th>代码</th><th>名称/板块</th><th>价格</th><th>日涨跌</th><th>5日</th><th>10日</th><th>20日</th><th>SS分</th><th>ISIR</th><th>GLM</th><th>豆包↓</th><th>共识</th>
</tr></thead><tbody id="body-doubao">{doubao_rows}</tbody></table></div>
{trade_ledger_html('doubao')}</div>

<!-- 历史操作建议记录面板 -->
{_build_signal_history(signal_history, trades)}

<div class="panel" id="panel-sector"><div class="section-title">按板块纵览</div>
{sector_html}</div>

<div class="footer"><p>统一评分引擎 v2.0 | 数据:腾讯财经 | 生成于 {timestr}</p>
<p style="margin-top:3px;font-size:11px">ISIR=33因子ICIR原始加权 | GLM=mfi/pct_52w方向反转 | 豆包=SS传统评分(35/55/10) | 共识=三者∩ | 点击行展开因子明细 | 每策略含交易账本</p></div></div>
<script>
function switchTab(n){{document.querySelectorAll('.panel').forEach(p=>p.classList.remove('active'));document.querySelectorAll('.tab-btn').forEach(b=>b.classList.remove('active'));document.getElementById('panel-'+n).classList.add('active');[...document.querySelectorAll('.tab-btn')].find(b=>b.textContent.includes(n=='overview'?'纵览':n=='sector'?'板块':n.toUpperCase())||b.onclick.toString().includes("'"+n+"'"))?.classList.add('active')}}
function toggleRow(id){{var r=document.getElementById(id);if(r)r.style.display=r.style.display==='none'?'table-row':'none'}}
function filterPanel(name){{var s=document.getElementById('sector-'+name)?.value;var c=document.getElementById('consensus-'+name)?.checked;var q=(document.getElementById('search-'+name)?.value||'').toLowerCase();var rows=document.querySelectorAll('#body-'+name+' tr');rows.forEach(function(row){{if(row.classList.contains('detail-row'))return;var cells=row.getElementsByTagName('td');if(cells.length<12)return;var code=cells[0].textContent.trim();var nm=cells[1].textContent.trim();var badge=cells[11].textContent.trim();var show=true;if(s&&s!=='all'&&!cells[1].textContent.includes(s))show=false;if(c&&!badge.includes('共识'))show=false;if(q&&!code.toLowerCase().includes(q)&&!nm.toLowerCase().includes(q))show=false;row.style.display=show?'':'none';var next=row.nextElementSibling;if(next&&next.classList.contains('detail-row'))next.style.display=show?(next.style.display):'none'}})}}
function toggleSector(s){{var b=document.getElementById('sec-'+s)?.querySelector('.sector-body');if(b)b.style.display=b.style.display==='none'?'block':'none'}}
</script></body></html>"""
    return html

# ================================================================
# Main
# ================================================================

def main():
    print("="*60)
    print("  统一评分引擎 v2.0")
    print("  ISIR | GLM | 豆包 + 信号追踪 + 收益计算")
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
    klines = db.get_klines(codes, days=130)
    extra_info = db.get_extra_info(codes, force_refresh=True)
    fund_flows = db.get_fund_flows(codes)
    print(f"  K线:{len(klines)} | 行情:{len(extra_info)} | 资金流:{len(fund_flows)}")

    print(f"\n  [2/4] 计算33因子 + 三套排名...")
    sectors = {code: sector_map.get_sector(code) for code in codes}
    factor_data, return_data = compute_all_factors(klines, extra_info, fund_flows, {}, sectors)
    print(f"  有效因子: {len(factor_data)}只")

    rankings = compute_rankings(factor_data)
    n_consensus = sum(1 for r in rankings if r["consensus"])

    print(f"\n  [3/4] 信号追踪 + 交易更新...")
    date_str = datetime.now().strftime("%Y-%m-%d")
    history = save_history(rankings, extra_info, date_str)
    trades, today_signals, signal_history = update_trades(rankings, extra_info, date_str)

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
    mkt_overview = compute_market_overview(klines, extra_info, rankings)
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
    html = build_html(rankings, extra_info, history, trades, return_data, date_str, freshness, bt_summary, best_strat, mkt_overview, today_signals, signal_history)
    html_path = os.path.join(OUTPUT_DIR, f"unified_{date_str}.html")
    with open(html_path, "w", encoding="utf-8") as f: f.write(html)

    json_path = os.path.join(OUTPUT_DIR, f"unified_{date_str}.json")
    with open(json_path, "w") as f: json.dump(rankings, f, ensure_ascii=False, indent=2)

    consensus_list = [r for r in rankings if r["consensus"]]
    print(f"\n  {'='*60}")
    print(f"  ✅ 报告已生成!")
    print(f"     HTML: {html_path}")
    print(f"     JSON: {json_path}")
    print(f"     共识TOP{TOP_N}: {n_consensus} 只")
    if consensus_list:
        print(f"  共识标的:")
        for i,r in enumerate(consensus_list,1):
            name = extra_info.get(r["code"],{}).get("name","")
            print(f"     {i}. {r['code']} {name} | ISIR#{r['isir_rank']} GLM#{r['glm_rank']} 豆包#{r['doubao_rank']}")
    print(f"  {'='*60}")
    db.stats()
    return html_path

if __name__ == "__main__":
    main()
