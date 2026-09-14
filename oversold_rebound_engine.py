#!/usr/bin/env python3
"""
超跌反弹趋势选股引擎 v2.0
全A股扫描 → 超跌初筛 → 反弹信号 → 趋势确认 → ATR动态止损建议 → 独立HTML报告

v2.0 核心升级 (vs v1.0):
  1. 评分体系: 超跌30% + 反弹35% + 趋势确认20% + 风控15% (v1.0是超跌40%+反弹35%+风控25%)
  2. 新增"趋势确认"维度: 站上MA5 / MA5拐头 / MACD金叉 / KDJ金叉 / 突破5日高点
  3. ATR动态止损止盈建议 (从ML项目移植): 高波动股给更多空间，低波动股更紧
  4. 反弹阶段标注: 初现 / 确认 / 加速
  5. 增强量价配合: 量价背离 + 放量阳线 + OBV趋势
  6. 威廉指标 + CCI 经典超跌指标

持仓周期: T+3~5 短线
"""

import os, json, sqlite3, urllib.request, math, time
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

SELF_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SELF_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================================================================
# 全A股市场数据获取
# ================================================================

def _safe_float(v, default=0):
    """安全转float，处理'-'等非数字"""
    try:
        return float(v) if v not in ("-", "", None) else default
    except (ValueError, TypeError):
        return default


def fetch_all_a_shares():
    """从东方财富API获取全A股列表+实时行情"""
    markets = [
        ("m:0+t:6", "深主板"),
        ("m:0+t:80", "创业板"),
        ("m:1+t:2", "沪主板"),
        ("m:1+t:23", "科创板"),
        ("m:0+t:81+s:2048", "北交所"),
    ]
    all_stocks = []
    for market_filter, market_name in markets:
        url = (f"https://push2.eastmoney.com/api/qt/clist/get"
               f"?pn=1&pz=10000&po=1&np=1&fltt=2&invt=2"
               f"&fid=f3&fs={market_filter}"
               f"&fields=f12,f14,f2,f3,f4,f5,f6,f15,f16,f17,f18,f9,f23,f20")
        success = False
        for attempt in range(3):
            try:
                req = urllib.request.Request(url, headers={
                    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
                    "Referer": "https://quote.eastmoney.com",
                    "Accept": "application/json, text/plain, */*",
                })
                resp = urllib.request.urlopen(req, timeout=20)
                data = json.loads(resp.read().decode("utf-8"))
                items = data.get("data", {}).get("diff", []) or []
                for item in items:
                    code = item.get("f12", "")
                    name = item.get("f14", "")
                    price = _safe_float(item.get("f2", 0))
                    if not code or price <= 0:
                        continue
                    if "ST" in name or "*ST" in name or "退" in name:
                        continue
                    all_stocks.append({
                        "code": code, "name": name, "price": price,
                        "pct": _safe_float(item.get("f3", 0)),
                        "amount": _safe_float(item.get("f6", 0)),
                        "high": _safe_float(item.get("f15", 0)),
                        "low": _safe_float(item.get("f16", 0)),
                        "open": _safe_float(item.get("f17", 0)),
                        "prev_close": _safe_float(item.get("f18", 0)),
                        "pe": _safe_float(item.get("f9", 0)),
                        "pb": _safe_float(item.get("f23", 0)),
                        "mcap": _safe_float(item.get("f20", 0)),
                        "market": market_name,
                    })
                print(f"  [{market_name}] {len(items)}只")
                success = True
                break
            except Exception as e:
                if attempt < 2:
                    time.sleep(1)
                else:
                    print(f"  [{market_name}] 获取失败: {e}")
    return all_stocks


def fetch_pool_stocks():
    """Fallback: 从主项目446只池获取实时行情（腾讯API，sandbox可用）"""
    from stock_db import StockDB
    pool_path = os.path.join(SELF_DIR, "stock_codes.txt")
    with open(pool_path, "r") as f:
        codes = [line.strip() for line in f if line.strip()]
    print(f"  [446池] 加载股票池: {len(codes)}只")

    db = StockDB()
    extra = db.get_extra_info(codes)
    klines_all = db.get_klines(codes, days=80)

    all_stocks = []
    for code in codes:
        info = extra.get(code, {})
        klines = klines_all.get(code, [])
        if not info or not klines or len(klines) < 30:
            continue
        price = info.get("price", 0)
        if price <= 0:
            continue
        name = info.get("name", code)
        if "ST" in name or "退" in name:
            continue
        # 从K线推算今日涨跌幅和成交额
        today_k = klines[-1] if klines else {}
        prev_close = klines[-2]["close"] if len(klines) >= 2 else price
        pct = (price / prev_close - 1) * 100 if prev_close > 0 else 0
        # 板块
        if code.startswith("30"):
            market = "创业板"
        elif code.startswith("68"):
            market = "科创板"
        elif code.startswith(("8", "4", "92")):
            market = "北交所"
        elif code.startswith("6"):
            market = "沪主板"
        else:
            market = "深主板"
        all_stocks.append({
            "code": code, "name": name, "price": price,
            "pct": pct, "amount": today_k.get("volume", 0) * price,
            "high": today_k.get("high", price), "low": today_k.get("low", price),
            "open": today_k.get("open", price), "prev_close": prev_close,
            "pe": info.get("pe_ttm", 0), "pb": info.get("pb", 0),
            "mcap": info.get("mcap", 0), "market": market,
        })
    return all_stocks, klines_all


def fetch_kline(code, days=80):
    """获取单只股票K线（腾讯API）"""
    prefix = "sh" if code.startswith(("6", "5")) else "sz"
    if code.startswith(("8", "4", "92")):
        prefix = "bj"
    url = (f"http://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
           f"?param={prefix}{code},day,,,{days},qfq")
    try:
        resp = urllib.request.urlopen(url, timeout=8)
        data = json.loads(resp.read().decode("utf-8"))
        kdata = data.get("data", {}).get(f"{prefix}{code}", {})
        klines_raw = kdata.get("day") or kdata.get("qfqday") or []
        klines = []
        for k in klines_raw:
            if len(k) >= 6:
                klines.append({
                    "date": k[0], "open": float(k[1]), "high": float(k[2]),
                    "low": float(k[3]), "close": float(k[4]),
                    "volume": float(k[5]) if k[5] else 0
                })
        return klines
    except:
        return []


def batch_fetch_klines(codes, days=80, workers=20):
    """批量获取K线"""
    results = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(fetch_kline, c, days): c for c in codes}
        done = 0
        for future in as_completed(futures):
            code = futures[future]
            done += 1
            if done % 50 == 0:
                print(f"  K线: {done}/{len(codes)}")
            klines = future.result()
            if len(klines) >= 30:
                results[code] = klines
    return results


# ================================================================
# 技术指标计算
# ================================================================

def calc_ma(closes, n):
    if len(closes) < n: return None
    return sum(closes[-n:]) / n

def calc_rsi(closes, n=14):
    if len(closes) < n + 1: return 50
    gains = [max(0, closes[i] - closes[i-1]) for i in range(-n, 0)]
    losses = [max(0, closes[i-1] - closes[i]) for i in range(-n, 0)]
    avg_gain = sum(gains) / n
    avg_loss = sum(losses) / n
    if avg_loss == 0: return 100
    rs = avg_gain / avg_loss
    return 100 - 100 / (1 + rs)

def calc_ema(values, n):
    if len(values) < n: return None
    k = 2 / (n + 1)
    ema = values[-n]
    for v in values[-n+1:]:
        ema = v * k + ema * (1 - k)
    return ema

def calc_atr(highs, lows, closes, n=14):
    """ATR(14) - Average True Range"""
    if len(closes) < n + 1: return 0
    trs = []
    for i in range(-n, 0):
        tr1 = highs[i] - lows[i]
        tr2 = abs(highs[i] - closes[i-1])
        tr3 = abs(lows[i] - closes[i-1])
        trs.append(max(tr1, tr2, tr3))
    return sum(trs) / n

def calc_kdj(highs, lows, closes, n=9, m1=3, m2=3):
    """KDJ指标，返回 (K, D, J)"""
    if len(closes) < n: return 50, 50, 50
    low_n = min(lows[-n:])
    high_n = max(highs[-n:])
    if high_n == low_n: return 50, 50, 50
    rsv = (closes[-1] - low_n) / (high_n - low_n) * 100
    # 简化：用前一日K/D做EMA平滑
    if len(closes) >= n + 1:
        prev_low_n = min(lows[-n-1:-1])
        prev_high_n = max(highs[-n-1:-1])
        if prev_high_n != prev_low_n:
            prev_rsv = (closes[-2] - prev_low_n) / (prev_high_n - prev_low_n) * 100
        else:
            prev_rsv = 50
    else:
        prev_rsv = 50
    k_prev = prev_rsv  # 简化近似
    k = (m1 - 1) / m1 * k_prev + 1 / m1 * rsv
    d = (m2 - 1) / m2 * 50 + 1 / m2 * k  # 简化
    j = 3 * k - 2 * d
    return k, d, j

def calc_kdj_series(highs, lows, closes, n=9, m1=3, m2=3):
    """计算KDJ序列（返回最近2日的K,D,J用于判断金叉）"""
    if len(closes) < n + 1: return None
    k_values, d_values = [], []
    rsv_list = []
    for end_idx in range(n, len(closes) + 1):
        h_window = highs[end_idx-n:end_idx]
        l_window = lows[end_idx-n:end_idx]
        if max(h_window) == min(l_window):
            rsv_list.append(50)
        else:
            rsv_list.append((closes[end_idx-1] - min(l_window)) / (max(h_window) - min(l_window)) * 100)
    # EMA平滑
    k = rsv_list[0]
    d = 50
    for i, rsv in enumerate(rsv_list):
        k = (m1 - 1) / m1 * k + 1 / m1 * rsv
        d = (m2 - 1) / m2 * d + 1 / m2 * k
        if i >= len(rsv_list) - 2:
            k_values.append(k)
            d_values.append(d)
    j_values = [3*k_values[i] - 2*d_values[i] for i in range(len(k_values))]
    if len(k_values) >= 2:
        return k_values[-1], d_values[-1], j_values[-1], k_values[-2], d_values[-2], j_values[-2]
    return k_values[-1] if k_values else 50, d_values[-1] if d_values else 50, j_values[-1] if j_values else 50, 50, 50, 50

def calc_williams_r(highs, lows, closes, n=14):
    """威廉指标 %R"""
    if len(closes) < n: return -50
    high_n = max(highs[-n:])
    low_n = min(lows[-n:])
    if high_n == low_n: return -50
    return (high_n - closes[-1]) / (high_n - low_n) * -100

def calc_obv(closes, volumes):
    """OBV趋势（最近5日变化率）"""
    if len(closes) < 6: return 0
    obv = 0
    obv_list = [0]
    for i in range(1, len(closes)):
        if closes[i] > closes[i-1]:
            obv += volumes[i]
        elif closes[i] < closes[i-1]:
            obv -= volumes[i]
        obv_list.append(obv)
    if obv_list[-6] != 0:
        return (obv_list[-1] - obv_list[-6]) / abs(obv_list[-6]) * 100
    return 0


# ================================================================
# 超跌反弹趋势评分 v2.0
# ================================================================

def score_oversold_rebound_trend(stock, klines):
    """
    超跌反弹趋势评分 v2.0
    = 超跌程度(30%) + 反弹信号(35%) + 趋势确认(20%) + 风险控制(15%)

    v2.0核心改进：新增"趋势确认"维度，不是找跌狠的，而是找跌狠后趋势开始反转的。

    返回: {score, oversold, signal, trend, risk, factors, indicators, atr_advice, rebound_stage}
    """
    if len(klines) < 35:
        return None

    closes = [k["close"] for k in klines]
    highs = [k["high"] for k in klines]
    lows = [k["low"] for k in klines]
    opens = [k["open"] for k in klines]
    volumes = [k["volume"] for k in klines]

    code = stock["code"]
    name = stock["name"]
    price = stock["price"]
    market = stock["market"]
    mcap = stock["mcap"]
    amount = stock.get("amount", 0)

    # 板块涨跌幅限制
    if code.startswith("30"):
        limit_pct = 20
    elif code.startswith("68"):
        limit_pct = 20
    elif code.startswith(("8", "4", "92")):
        limit_pct = 30
    else:
        limit_pct = 10

    factors = []
    def add(dim, name_str, delta, detail=""):
        factors.append({"dim": dim, "name": name_str, "delta": delta, "detail": detail})

    # ========== 1. 超跌程度 (30分) ==========
    oversold_score = 0

    # 5日跌幅
    ret_5d = (closes[-1] / closes[-6] - 1) * 100 if len(closes) >= 6 else 0
    if ret_5d < -15:
        oversold_score += 12; add("超跌", "5日暴跌", 12, f"5日{ret_5d:.1f}%")
    elif ret_5d < -10:
        oversold_score += 10; add("超跌", "5日大跌", 10, f"5日{ret_5d:.1f}%")
    elif ret_5d < -8:
        oversold_score += 8; add("超跌", "5日显著下跌", 8, f"5日{ret_5d:.1f}%")
    elif ret_5d < -5:
        oversold_score += 5; add("超跌", "5日下跌", 5, f"5日{ret_5d:.1f}%")

    # 10日跌幅
    ret_10d = (closes[-1] / closes[-11] - 1) * 100 if len(closes) >= 11 else 0
    if ret_10d < -20:
        oversold_score += 7; add("超跌", "10日深跌", 7, f"10日{ret_10d:.1f}%")
    elif ret_10d < -15:
        oversold_score += 5; add("超跌", "10日大跌", 5, f"10日{ret_10d:.1f}%")
    elif ret_10d < -10:
        oversold_score += 3; add("超跌", "10日下跌", 3, f"10日{ret_10d:.1f}%")

    # RSI超卖
    rsi = calc_rsi(closes, 14)
    if rsi < 20:
        oversold_score += 8; add("超跌", "RSI极度超卖", 8, f"RSI={rsi:.1f}")
    elif rsi < 25:
        oversold_score += 7; add("超跌", "RSI严重超卖", 7, f"RSI={rsi:.1f}")
    elif rsi < 30:
        oversold_score += 5; add("超跌", "RSI超卖", 5, f"RSI={rsi:.1f}")
    elif rsi < 35:
        oversold_score += 3; add("超跌", "RSI偏弱", 3, f"RSI={rsi:.1f}")

    # 威廉指标超卖
    wr = calc_williams_r(highs, lows, closes, 14)
    if wr < -80:
        oversold_score += 3; add("超跌", "威廉超卖", 3, f"WR={wr:.0f}")

    oversold_score = min(30, oversold_score)

    # ========== 2. 反弹信号 (35分) ==========
    signal_score = 0

    # 下影线
    today_body = abs(closes[-1] - opens[-1])
    today_range = highs[-1] - lows[-1]
    lower_shadow = min(opens[-1], closes[-1]) - lows[-1]
    if today_range > 0:
        if lower_shadow > today_body * 2 and lower_shadow > today_range * 0.3:
            signal_score += 8; add("信号", "长下影线", 8, f"下影线占{lower_shadow/today_range*100:.0f}%")
        if today_body < today_range * 0.3:
            signal_score += 5; add("信号", "十字星", 5, f"实体仅{today_body/today_range*100:.0f}%")

    # 缩量止跌
    if len(volumes) >= 6:
        avg_vol_5 = sum(volumes[-6:-1]) / 5
        if avg_vol_5 > 0:
            vol_ratio = volumes[-1] / avg_vol_5
            today_ret = (closes[-1] / closes[-2] - 1) * 100 if len(closes) >= 2 else 0
            yest_ret = (closes[-2] / closes[-3] - 1) * 100 if len(closes) >= 3 else 0
            if vol_ratio < 0.7 and today_ret > yest_ret:
                signal_score += 8; add("信号", "缩量止跌", 8, f"量比{vol_ratio:.2f}")
            elif vol_ratio < 0.5:
                signal_score += 5; add("信号", "极度缩量", 5, f"量比{vol_ratio:.2f}")

    # RSI底背离
    if len(closes) >= 30:
        mid_idx = len(closes) - 15
        if closes[-1] < closes[mid_idx]:
            closes_mid = closes[:mid_idx+1]
            rsi_mid = calc_rsi(closes_mid, 14) if len(closes_mid) > 14 else 50
            if rsi > rsi_mid:
                signal_score += 10; add("信号", "RSI底背离", 10, f"RSI {rsi:.0f}>{rsi_mid:.0f}")

    # MACD绿柱缩短
    dif = calc_ema(closes, 12)
    dea = calc_ema(closes, 26)
    hist_now = None
    if dif is not None and dea is not None:
        hist_now = dif - dea
        dif_prev = calc_ema(closes[:-1], 12)
        dea_prev = calc_ema(closes[:-1], 26)
        if dif_prev is not None and dea_prev is not None:
            hist_prev = dif_prev - dea_prev
            if hist_now < 0 and hist_now > hist_prev:
                signal_score += 5; add("信号", "MACD绿柱缩短", 5, f"柱{hist_now:.4f}>{hist_prev:.4f}")

    # 量价背离（价格新低但量能萎缩）
    if len(volumes) >= 25:
        low_5d = min(lows[-5:])
        vol_20d_avg = sum(volumes[-20:]) / 20
        near_low = (closes[-1] - low_5d) / low_5d < 0.02 if low_5d > 0 else False
        vol_shrinking = volumes[-1] < vol_20d_avg * 0.7
        if near_low and vol_shrinking:
            signal_score += 5; add("信号", "量价背离", 5, "价新低+量萎缩")

    # 尾盘回升
    if today_range > 0:
        pos_in_range = (closes[-1] - lows[-1]) / today_range
        if pos_in_range > 0.5:
            signal_score += 4; add("信号", "尾盘回升", 4, f"日内{pos_in_range*100:.0f}%")

    # 放量阳线确认
    if len(volumes) >= 6:
        avg_vol_5 = sum(volumes[-6:-1]) / 5
        if avg_vol_5 > 0 and closes[-1] > opens[-1]:
            if volumes[-1] > avg_vol_5 * 1.5:
                signal_score += 5; add("信号", "放量阳线", 5, f"量比{volumes[-1]/avg_vol_5:.1f}")

    signal_score = min(35, signal_score)

    # ========== 3. 趋势确认 (20分) ← v2.0新增核心维度 ==========
    trend_score = 0

    # 站上MA5
    ma5 = calc_ma(closes, 5)
    if ma5 and closes[-1] > ma5:
        trend_score += 5; add("趋势", "站上MA5", 5, f"价{closes[-1]:.2f}>MA5 {ma5:.2f}")

    # MA5拐头向上
    if len(closes) >= 10:
        ma5_now = sum(closes[-5:]) / 5
        ma5_prev = sum(closes[-6:-1]) / 5
        if ma5_now > ma5_prev:
            trend_score += 4; add("趋势", "MA5拐头", 4, f"MA5 {ma5_now:.2f}>{ma5_prev:.2f}")

    # MACD金叉或绿柱连续缩短
    if hist_now is not None:
        if hist_now > 0:
            trend_score += 4; add("趋势", "MACD红柱", 4, f"DIF>DEA 金叉区间")
        elif hist_now < 0 and len(closes) >= 36:
            # 检查绿柱是否连续2日缩短
            dif_p2 = calc_ema(closes[:-2], 12)
            dea_p2 = calc_ema(closes[:-2], 26)
            if dif_p2 is not None and dea_p2 is not None:
                hist_p2 = dif_p2 - dea_p2
                dif_prev = calc_ema(closes[:-1], 12)
                dea_prev = calc_ema(closes[:-1], 26)
                hist_p1 = dif_prev - dea_prev if dif_prev and dea_prev else hist_now
                if hist_now > hist_p1 > hist_p2:
                    trend_score += 3; add("趋势", "MACD绿柱连缩", 3, "连续2日缩短")

    # KDJ金叉（J上穿D）
    kdj_result = calc_kdj_series(highs, lows, closes)
    if kdj_result:
        k_now, d_now, j_now, k_prev, d_prev, j_prev = kdj_result
        if j_prev <= d_prev and j_now > d_now:
            trend_score += 4; add("趋势", "KDJ金叉", 4, f"J{j_now:.0f}上穿D{d_now:.0f}")
        elif j_now > d_now and j_now > j_prev:
            trend_score += 2; add("趋势", "KDJ向上", 2, f"J{j_now:.0f}>D{d_now:.0f}")

    # 突破近5日高点
    if len(highs) >= 6:
        high_5d = max(highs[-6:-1])
        if closes[-1] > high_5d:
            trend_score += 3; add("趋势", "突破5日高", 3, f"收{closes[-1]:.2f}>5日高{high_5d:.2f}")

    trend_score = min(20, trend_score)

    # ========== 4. 风险控制 (15分) ==========
    risk_score = 0

    # 流动性
    if amount > 5e8:
        risk_score += 5; add("风控", "高流动性", 5, f"{amount/1e8:.1f}亿")
    elif amount > 2e8:
        risk_score += 4; add("风控", "良好流动性", 4, f"{amount/1e8:.1f}亿")
    elif amount > 5e7:
        risk_score += 2; add("风控", "基本流动性", 2, f"{amount/1e7:.0f}千万")

    # 市值分层
    if mcap > 0:
        mcap_yi = mcap / 1e8
        if 50 <= mcap_yi <= 500:
            risk_score += 4; add("风控", "中小盘", 4, f"{mcap_yi:.0f}亿")
        elif 500 <= mcap_yi <= 2000:
            risk_score += 2; add("风控", "中大盘", 2, f"{mcap_yi:.0f}亿")

    # 主板优先
    if limit_pct == 10:
        risk_score += 3; add("风控", "主板可控", 3, "±10%")
    elif limit_pct == 20:
        risk_score += 1; add("风控", "创/科创板", 1, "±20%")

    # MA60支撑
    ma20 = calc_ma(closes, 20)
    ma60 = calc_ma(closes, 60)
    if ma60 and ma60 > 0:
        dev_ma60 = (closes[-1] / ma60 - 1) * 100
        if -3 < dev_ma60 < 3:
            risk_score += 3; add("风控", "MA60支撑", 3, f"{dev_ma60:+.1f}%")
        elif -5 < dev_ma60 < 5:
            risk_score += 1; add("风控", "MA60附近", 1, f"{dev_ma60:+.1f}%")

    risk_score = min(15, risk_score)

    # ========== 最终评分 ==========
    total = oversold_score + signal_score + trend_score + risk_score

    # 排除一字跌停
    if stock.get("open", 0) == stock.get("high", 0) == stock.get("low", 0) == stock.get("price", 0):
        return None
    # 排除涨停
    if stock.get("pct", 0) >= limit_pct - 0.5:
        return None
    # v2.0核心门槛：超跌分必须>=5，确保只选真正超跌的股票（不是强势上涨股）
    if oversold_score < 5:
        return None

    # ========== ATR动态止损建议 (从ML项目移植) ==========
    atr = calc_atr(highs, lows, closes, 14)
    atr_pct = (atr / closes[-1]) if closes[-1] > 0 else 0.05
    atr_pct = max(0.01, min(atr_pct, 0.15))  # 限制1%-15%

    stop_loss = max(-1.5 * atr_pct, -0.08)   # 不超过-8%
    stop_loss = min(stop_loss, -0.05)          # 至少-5%
    take_profit = min(1.0 * atr_pct, 0.06)    # 不超过6%
    take_profit = max(take_profit, 0.03)       # 至少3%
    trailing_stop = take_profit                 # 移动止盈同止盈

    atr_advice = {
        "atr": round(atr, 3),
        "atr_pct": round(atr_pct * 100, 1),
        "stop_loss": round(stop_loss * 100, 1),
        "take_profit": round(take_profit * 100, 1),
        "trailing_stop": round(trailing_stop * 100, 1),
        "stop_price": round(closes[-1] * (1 + stop_loss), 2),
        "tp_price": round(closes[-1] * (1 + take_profit), 2),
    }

    # ========== 反弹阶段标注 ==========
    if trend_score >= 12:
        stage = "加速"
        stage_color = "#dc2626"
    elif trend_score >= 6:
        stage = "确认"
        stage_color = "#f59e0b"
    else:
        stage = "初现"
        stage_color = "#6b7280"

    # ========== OBV趋势 ==========
    obv_5d_chg = calc_obv(closes, volumes)

    return {
        "code": code, "name": name, "price": price,
        "market": market, "limit_pct": limit_pct,
        "score": round(total, 1),
        "oversold": oversold_score, "signal": signal_score,
        "trend": trend_score, "risk": risk_score,
        "factors": factors,
        "indicators": {
            "close": closes[-1], "ma5": ma5 or 0, "ma20": ma20 or 0, "ma60": ma60 or 0,
            "rsi": rsi, "wr": wr, "ret_5d": ret_5d, "ret_10d": ret_10d,
            "vol_ratio": volumes[-1] / (sum(volumes[-6:-1])/5) if len(volumes) >= 6 and sum(volumes[-6:-1]) > 0 else 1,
            "dev_ma20": (closes[-1] / ma20 - 1) * 100 if ma20 else 0,
            "amount": amount, "mcap": mcap,
            "pct_today": stock.get("pct", 0),
            "obv_5d": obv_5d_chg,
        },
        "atr_advice": atr_advice,
        "rebound_stage": stage,
        "stage_color": stage_color,
    }


# ================================================================
# HTML报告生成 v2.0
# ================================================================

def generate_html(results, date_str, scan_stats):
    """生成超跌反弹趋势选股HTML报告 v2.0"""
    timestr = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def build_rows(sorted_r):
        rows = ""
        for i, r in enumerate(sorted_r):
            ind = r["indicators"]
            atr = r["atr_advice"]
            pos_f = [f for f in r["factors"] if f["delta"] > 0]
            detail_html = ""
            for f in pos_f[:10]:
                detail_html += f'<span class="factor-tag dim-{f["dim"]}">{f["dim"]}:{f["name"]} +{f["delta"]}</span>'

            rows += f"""<tr class="{'row-top' if i < 10 else ''}" onclick="toggleRow('d-{r['code']}')">
<td>{i+1}</td><td class="code-col">{r['code']}</td><td>{r['name']}</td><td>{r['market']}</td>
<td class="num">{r['price']:.2f}</td>
<td class="num {'negative' if ind['pct_today']<0 else 'positive'}">{ind['pct_today']:+.2f}%</td>
<td class="num negative">{ind['ret_5d']:+.1f}%</td>
<td class="num">{ind['rsi']:.0f}</td>
<td class="num score-col">{r['score']:.0f}</td>
<td class="num">{r['oversold']}/30</td>
<td class="num">{r['signal']}/35</td>
<td class="num trend-col">{r['trend']}/20</td>
<td class="num">{r['risk']}/15</td>
<td><span class="stage-tag" style="background:{r['stage_color']}">{r['rebound_stage']}</span></td>
<td class="num">{atr['stop_loss']:.1f}%</td>
<td class="num">{atr['take_profit']:.1f}%</td>
</tr>
<tr class="detail-row" id="d-{r['code']}" style="display:none">
<td colspan="16"><div class="detail-card">
<div style="margin-bottom:6px"><strong>评分因子:</strong> {detail_html}</div>
<div class="indicator-grid">
<span>MA5={ind['ma5']:.2f}</span><span>MA20={ind['ma20']:.2f}(偏离{ind['dev_ma20']:+.1f}%)</span>
<span>MA60={ind['ma60']:.2f}</span><span>WR={ind['wr']:.0f}</span>
<span>10日{ind['ret_10d']:+.1f}%</span><span>量比{ind['vol_ratio']:.2f}</span>
<span>OBV5日{ind['obv_5d']:+.0f}%</span><span>成交{ind['amount']/1e8:.1f}亿</span>
<span>市值{ind['mcap']/1e8:.0f}亿</span><span>ATR={atr['atr_pct']:.1f}%</span>
</div>
<div class="atr-advice">
<strong>ATR动态风控:</strong> 止损价{atr['stop_price']:.2f}({atr['stop_loss']:.1f}%) |
止盈价{atr['tp_price']:.2f}({atr['take_profit']:.1f}%) |
移动止盈{atr['trailing_stop']:.1f}% |
建议持仓T+3~5
</div></div></td>
</tr>"""
        return rows

    top40 = sorted(results, key=lambda x: x["score"], reverse=True)[:40]
    rows = build_rows(top40)

    # 统计
    stage_counts = {"加速": 0, "确认": 0, "初现": 0}
    for r in top40:
        stage_counts[r["rebound_stage"]] = stage_counts.get(r["rebound_stage"], 0) + 1

    market_dist = {}
    for r in top40:
        m = r["market"]
        market_dist[m] = market_dist.get(m, 0) + 1
    market_html = " | ".join(f"{m}: {c}只" for m, c in sorted(market_dist.items(), key=lambda x: -x[1]))

    avg_rsi = sum(r["indicators"]["rsi"] for r in top40) / len(top40) if top40 else 50
    avg_score = sum(r["score"] for r in top40) / len(top40) if top40 else 0

    html = f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>超跌反弹趋势选股报告 {date_str}</title>
<style>
:root{{--bg:#f8fafc;--card-bg:#fff;--text:#1e293b;--text-sec:#64748b;--border:#e2e8f0;--primary:#dc2626;--positive:#dc2626;--negative:#16a34a;--accent:#f59e0b}}
*{{margin:0;padding:0;box-sizing:border-box}}body{{font-family:-apple-system,BlinkMacSystemFont,sans-serif;background:var(--bg);color:var(--text);line-height:1.6}}
.container{{max-width:1500px;margin:0 auto;padding:16px}}
.header{{background:linear-gradient(135deg,#7c2d12,#991b1b,#b91c1c);color:white;padding:28px 36px;border-radius:14px;margin-bottom:16px}}
.header h1{{font-size:22px;margin-bottom:4px}}.header .meta{{font-size:12px;opacity:.85}}
.header .badge{{display:inline-block;background:rgba(255,255,255,.2);padding:2px 10px;border-radius:12px;font-size:11px;margin-left:8px}}
.stats{{display:grid;grid-template-columns:repeat(7,1fr);gap:10px;margin-bottom:16px}}
.stat-card{{background:var(--card-bg);border-radius:10px;padding:14px 10px;text-align:center;border:1px solid var(--border)}}
.stat-card .val{{font-size:24px;font-weight:700;color:var(--primary)}}.stat-card .lbl{{font-size:10px;color:var(--text-sec);margin-top:3px}}
.table-wrap{{background:var(--card-bg);border-radius:12px;overflow-x:auto;border:1px solid var(--border)}}
table{{width:100%;border-collapse:collapse;font-size:11px}}
thead{{background:#fef2f2;position:sticky;top:0}}
th{{padding:9px 5px;text-align:left;font-weight:600;color:#991b1b;cursor:pointer;white-space:nowrap;font-size:10px}}
td{{padding:6px 5px;border-bottom:1px solid var(--border)}}
tr:hover{{background:#fff9f9}}tr.row-top{{background:#fff5f5!important}}
.num{{text-align:right;font-variant-numeric:tabular-nums}}
.code-col{{font-family:monospace;font-weight:600}}
.positive{{color:var(--positive)!important;font-weight:600}}.negative{{color:var(--negative)!important;font-weight:600}}
.score-col{{color:var(--primary);font-weight:700;font-size:13px}}
.trend-col{{color:var(--accent);font-weight:600}}
.stage-tag{{display:inline-block;color:white;padding:2px 8px;border-radius:8px;font-size:10px;font-weight:600}}
.factor-tag{{display:inline-block;padding:2px 6px;border-radius:4px;font-size:10px;margin:2px}}
.dim-超跌{{background:#fee2e2;color:#991b1b}}.dim-信号{{background:#fef3c7;color:#92400e}}
.dim-趋势{{background:#dbeafe;color:#1e40af}}.dim-风控{{background:#d1fae5;color:#065f46}}
.detail-row td{{padding:0}}.detail-card{{padding:10px 14px;background:#fffbeb;border-top:2px dashed var(--border)}}
.indicator-grid{{display:grid;grid-template-columns:repeat(5,1fr);gap:4px;font-size:11px;color:var(--text-sec);margin:6px 0}}
.indicator-grid span{{background:#f1f5f9;padding:3px 8px;border-radius:4px}}
.atr-advice{{font-size:11px;color:#7c2d12;background:#fff7ed;padding:6px 10px;border-radius:6px;margin-top:4px}}
.footer{{margin-top:16px;text-align:center;font-size:11px;color:var(--text-sec)}}
.legend{{display:flex;gap:12px;justify-content:center;margin:10px 0;font-size:11px}}
.legend span{{display:flex;align-items:center;gap:4px}}
.legend .dot{{width:10px;height:10px;border-radius:50%;display:inline-block}}
</style></head><body>
<div class="container">
<div class="header">
<h1>超跌反弹趋势选股报告 <span class="badge">v2.0</span></h1>
<div class="meta">{date_str} | 全A股扫描 | 持仓T+3~5 | 评分=超跌30+反弹35+趋势确认20+风控15 | ATR动态止损</div>
</div>

<div class="stats">
<div class="stat-card"><div class="val">{scan_stats['total']}</div><div class="lbl">全市场扫描</div></div>
<div class="stat-card"><div class="val">{scan_stats['screened']}</div><div class="lbl">超跌初筛</div></div>
<div class="stat-card"><div class="val">{len(results)}</div><div class="lbl">有效评分</div></div>
<div class="stat-card"><div class="val">{len(top40)}</div><div class="lbl">推荐TOP40</div></div>
<div class="stat-card"><div class="val">{avg_score:.0f}</div><div class="lbl">TOP40均分</div></div>
<div class="stat-card"><div class="val">{avg_rsi:.0f}</div><div class="lbl">TOP40均RSI</div></div>
<div class="stat-card"><div class="val">{stage_counts.get('确认',0)+stage_counts.get('加速',0)}</div><div class="lbl">趋势确认+加速</div></div>
</div>

<div class="legend">
<span><span class="dot" style="background:#dc2626"></span>加速({stage_counts.get('加速',0)})</span>
<span><span class="dot" style="background:#f59e0b"></span>确认({stage_counts.get('确认',0)})</span>
<span><span class="dot" style="background:#6b7280"></span>初现({stage_counts.get('初现',0)})</span>
</div>

<div class="table-wrap"><table><thead><tr>
<th>#</th><th>代码</th><th>名称</th><th>市场</th><th>现价</th><th>今日</th><th>5日</th><th>RSI</th>
<th>总分↓</th><th>超跌30</th><th>反弹35</th><th>趋势20</th><th>风控15</th>
<th>阶段</th><th>止损</th><th>止盈</th>
</tr></thead><tbody>{rows}</tbody></table></div>

<div style="margin-top:10px;font-size:11px;color:var(--text-sec)">
<strong>板块分布:</strong> {market_html}
</div>

<div class="footer">
<p>超跌反弹趋势引擎 v2.0 | 全A股扫描 | 生成于 {timestr}</p>
<p style="margin-top:3px">超跌=跌幅/RSI/威廉/偏离度 | 反弹=下影线/缩量/底背离/MACD/量价背离/放量阳线 | 趋势=MA5/拐头/MACD/KDJ/突破 | 风控=流动性/市值/主板/MA60</p>
<p style="margin-top:3px">ATR动态止损: 止损=1.5×ATR(限-5%~-8%) | 止盈=1.0×ATR(限3%~6%) | 移动止盈同止盈</p>
<p style="margin-top:3px;color:#999">⚠️ 超跌反弹为逆向策略，风险较高。阶段标注仅参考，非买卖建议。建议严格按ATR止损执行。</p>
</div>
</div>
<script>
function toggleRow(id){{var r=document.getElementById(id);if(r)r.style.display=r.style.display==='none'?'table-row':'none'}}
</script>
</body></html>"""
    return html


# ================================================================
# 主流程
# ================================================================

def main():
    print("=" * 60)
    print("  超跌反弹趋势选股引擎 v2.0")
    print("  全A股扫描 | 超跌初筛 → 反弹信号 → 趋势确认 → ATR止损")
    print("  评分 = 超跌(30) + 反弹(35) + 趋势确认(20) + 风控(15)")
    print("=" * 60)

    date_str = datetime.now().strftime("%Y-%m-%d")

    # [1/4] 全市场扫描（失败则fallback到446只池）
    print(f"\n  [1/4] 获取全A股实时行情...")
    all_stocks = fetch_all_a_shares()
    klines_preloaded = None
    if len(all_stocks) < 1000:
        print(f"  ⚠️ 全市场扫描仅{len(all_stocks)}只(需>1000)，fallback到446只精选池...")
        all_stocks, klines_preloaded = fetch_pool_stocks()
        scan_mode = "446精选池"
    else:
        scan_mode = "全A股"
    print(f"  股票池: {len(all_stocks)}只 ({scan_mode})")

    # [2/4] 超跌初筛（全市场模式才需要，446池直接全评）
    print(f"\n  [2/4] 超跌初筛...")
    if scan_mode == "全A股":
        screened = []
        for s in all_stocks:
            if s["amount"] < 1e7:
                continue
            if s["pct"] >= 0:
                continue
            if s["pct"] < -3:
                screened.append(s)
        print(f"  初筛通过: {len(screened)}只 (今日跌幅<-3%)")
    else:
        # 446池模式：跳过初筛，全部评分（量小速度快）
        screened = [s for s in all_stocks if s["amount"] >= 1e7]
        print(f"  跳过初筛(446池模式): {len(screened)}只直接评分")

    # [3/4] 获取K线 + 详细评分
    print(f"\n  [3/4] 获取K线 + 计算超跌反弹趋势评分...")
    if klines_preloaded:
        # 446池模式：K线已预加载
        klines_map = {c: klines_preloaded.get(c, []) for c in [s["code"] for s in screened]}
        klines_map = {c: k for c, k in klines_map.items() if len(k) >= 30}
        print(f"  K线(预加载): {len(klines_map)}/{len(screened)}")
    else:
        codes = [s["code"] for s in screened]
        klines_map = batch_fetch_klines(codes, days=80, workers=20)
        print(f"  K线获取: {len(klines_map)}/{len(codes)}")

    results = []
    for s in screened:
        kl = klines_map.get(s["code"])
        if not kl:
            continue
        r = score_oversold_rebound_trend(s, kl)
        if r and r["score"] >= 20:
            results.append(r)

    results.sort(key=lambda x: x["score"], reverse=True)
    print(f"  有效评分: {len(results)}只 (评分>=20)")

    # [4/4] 生成报告
    print(f"\n  [4/4] 生成报告...")
    top40 = results[:40]
    scan_stats = {
        "total": len(all_stocks),
        "screened": len(screened),
    }

    html = generate_html(results, date_str, scan_stats)
    html_path = os.path.join(OUTPUT_DIR, f"oversold_{date_str}.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)

    json_path = os.path.join(OUTPUT_DIR, f"oversold_{date_str}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results[:50], f, ensure_ascii=False, indent=2)

    print(f"\n  {'='*60}")
    print(f"  ✅ 报告已生成!")
    print(f"     HTML: {html_path}")
    print(f"     JSON: {json_path}")
    print(f"     超跌反弹趋势TOP10:")
    for i, r in enumerate(top40[:10]):
        ind = r["indicators"]
        atr = r["atr_advice"]
        print(f"     {i+1}. {r['code']} {r['name']} | 分{r['score']:.0f} "
              f"(超跌{r['oversold']}/反弹{r['signal']}/趋势{r['trend']}/风控{r['risk']}) "
              f"[{r['rebound_stage']}] "
              f"| 5日{ind['ret_5d']:+.1f}% RSI{ind['rsi']:.0f} "
              f"止损{atr['stop_loss']:.1f}% 止盈{atr['take_profit']:.1f}%")
    print(f"  {'='*60}")


if __name__ == "__main__":
    main()
