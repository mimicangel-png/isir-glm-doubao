#!/usr/bin/env python3
"""
超跌反弹评分引擎 v1.0
全A股扫描 → 超跌初筛 → 反弹信号评分 → 独立HTML报告

评分 = 超跌程度(40%) + 反弹信号(35%) + 风险控制(25%)
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

def fetch_all_a_shares():
    """从东方财富API获取全A股列表+实时行情"""
    # m:0=t:6 深主板, m:0=t:80 创业板, m:1=t:2 沪主板, m:1=t:23 科创板, m:0=t:81+s:2048 北交所
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
        try:
            req = urllib.request.Request(url, headers={
                "User-Agent": "Mozilla/5.0",
                "Referer": "https://quote.eastmoney.com"
            })
            resp = urllib.request.urlopen(req, timeout=15)
            data = json.loads(resp.read().decode("utf-8"))
            items = data.get("data", {}).get("diff", [])
            for item in items:
                code = item.get("f12", "")
                name = item.get("f14", "")
                price = item.get("f2", 0)
                pct = item.get("f3", 0)  # 今日涨跌幅
                amount = item.get("f6", 0)  # 成交额
                high = item.get("f15", 0)
                low = item.get("f16", 0)
                open_p = item.get("f17", 0)
                prev_close = item.get("f18", 0)
                pe = item.get("f9", 0)
                pb = item.get("f23", 0)
                mcap = item.get("f20", 0)  # 总市值

                if not code or price <= 0 or price == "-":
                    continue
                if "ST" in name or "*ST" in name or "退" in name:
                    continue

                all_stocks.append({
                    "code": code, "name": name, "price": float(price),
                    "pct": float(pct), "amount": float(amount),
                    "high": float(high), "low": float(low),
                    "open": float(open_p), "prev_close": float(prev_close),
                    "pe": float(pe) if pe != "-" else 0,
                    "pb": float(pb) if pb != "-" else 0,
                    "mcap": float(mcap) if mcap != "-" else 0,
                    "market": market_name,
                })
            print(f"  [{market_name}] {len(items)}只")
        except Exception as e:
            print(f"  [{market_name}] 获取失败: {e}")

    return all_stocks


def fetch_kline(code, days=60):
    """获取单只股票K线（腾讯API）"""
    prefix = "sh" if code.startswith(("6", "5")) else "sz"
    if code.startswith(("8", "4")):
        prefix = "bj"
    url = (f"http://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
           f"?param={prefix}{code},day,,, {days},qfq")
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


def batch_fetch_klines(codes, days=60, workers=20):
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

def calc_cmf(highs, lows, closes, volumes, n=20):
    if len(closes) < n: return 0
    mf_vol = []
    for i in range(-n, 0):
        if highs[i] == lows[i]:
            mf = 0
        else:
            mf = ((closes[i] - lows[i]) - (highs[i] - closes[i])) / (highs[i] - lows[i])
        mf_vol.append(mf * volumes[i])
    total_vol = sum(volumes[-n:])
    return sum(mf_vol) / total_vol if total_vol > 0 else 0

# ================================================================
# 超跌反弹评分
# ================================================================

def score_oversold_rebound(stock, klines):
    """
    超跌反弹评分
    = 超跌程度(40%) + 反弹信号(35%) + 风险控制(25%)
    返回: {score, oversold, signal, risk, factors, indicators}
    """
    if len(klines) < 30:
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

    # 板块涨跌幅限制
    if code.startswith("30"):  # 创业板 ±20%
        limit_pct = 20
    elif code.startswith("68"):  # 科创板 ±20%
        limit_pct = 20
    elif code.startswith(("8", "4")):  # 北交所 ±30%
        limit_pct = 30
    else:  # 主板 ±10%
        limit_pct = 10

    factors = []  # [{"dim":..., "name":..., "delta":..., "detail":...}]

    def add(dim, name, delta, detail=""):
        factors.append({"dim": dim, "name": name, "delta": delta, "detail": detail})

    # ========== 超跌程度 (40%) ==========
    oversold_score = 0

    # 5日跌幅
    if len(closes) >= 6:
        ret_5d = (closes[-1] / closes[-6] - 1) * 100
    else:
        ret_5d = 0
    if ret_5d < -15:
        oversold_score += 15
        add("超跌", "5日暴跌", 15, f"5日跌幅{ret_5d:.1f}% (<-15%)")
    elif ret_5d < -10:
        oversold_score += 12
        add("超跌", "5日大跌", 12, f"5日跌幅{ret_5d:.1f}% (-10%~-15%)")
    elif ret_5d < -8:
        oversold_score += 10
        add("超跌", "5日显著下跌", 10, f"5日跌幅{ret_5d:.1f}% (-8%~-10%)")
    elif ret_5d < -5:
        oversold_score += 6
        add("超跌", "5日下跌", 6, f"5日跌幅{ret_5d:.1f}% (-5%~-8%)")

    # 10日跌幅
    if len(closes) >= 11:
        ret_10d = (closes[-1] / closes[-11] - 1) * 100
        if ret_10d < -20:
            oversold_score += 9
            add("超跌", "10日深跌", 9, f"10日跌幅{ret_10d:.1f}% (<-20%)")
        elif ret_10d < -15:
            oversold_score += 7
            add("超跌", "10日大跌", 7, f"10日跌幅{ret_10d:.1f}% (-15%~-20%)")
        elif ret_10d < -10:
            oversold_score += 5
            add("超跌", "10日下跌", 5, f"10日跌幅{ret_10d:.1f}% (-10%~-15%)")

    # RSI超卖
    rsi = calc_rsi(closes, 14)
    if rsi < 20:
        oversold_score += 12
        add("超跌", "RSI极度超卖", 12, f"RSI={rsi:.1f} (<20)")
    elif rsi < 25:
        oversold_score += 10
        add("超跌", "RSI严重超卖", 10, f"RSI={rsi:.1f} (20-25)")
    elif rsi < 30:
        oversold_score += 8
        add("超跌", "RSI超卖", 8, f"RSI={rsi:.1f} (25-30)")
    elif rsi < 35:
        oversold_score += 5
        add("超跌", "RSI偏弱", 5, f"RSI={rsi:.1f} (30-35)")

    # 偏离MA20
    ma20 = calc_ma(closes, 20)
    if ma20 and ma20 > 0:
        dev_ma20 = (closes[-1] / ma20 - 1) * 100
        if dev_ma20 < -12:
            oversold_score += 8
            add("超跌", "严重偏离MA20", 8, f"偏离{dev_ma20:.1f}% (<-12%)")
        elif dev_ma20 < -8:
            oversold_score += 6
            add("超跌", "显著偏离MA20", 6, f"偏离{dev_ma20:.1f}% (-8%~-12%)")
        elif dev_ma20 < -5:
            oversold_score += 4
            add("超跌", "偏离MA20", 4, f"偏离{dev_ma20:.1f}% (-5%~-8%)")

    # 连跌天数
    streak_dn = 0
    for i in range(len(closes) - 1, 0, -1):
        if closes[i] < closes[i-1]:
            streak_dn += 1
        else:
            break
    if streak_dn >= 5:
        oversold_score += 5
        add("超跌", "连续下跌", 5, f"连跌{streak_dn}天")
    elif streak_dn >= 3:
        oversold_score += 3
        add("超跌", "连续下跌", 3, f"连跌{streak_dn}天")

    oversold_score = min(40, oversold_score)

    # ========== 反弹信号 (35%) ==========
    signal_score = 0

    # 下影线长度
    today_body = abs(closes[-1] - opens[-1])
    today_range = highs[-1] - lows[-1]
    lower_shadow = min(opens[-1], closes[-1]) - lows[-1]
    if today_range > 0:
        if lower_shadow > today_body * 2 and lower_shadow > today_range * 0.3:
            signal_score += 8
            add("信号", "长下影线", 8, f"下影线占振幅{lower_shadow/today_range*100:.0f}%")

        # 十字星/锤头
        if today_body < today_range * 0.3:
            signal_score += 6
            add("信号", "十字星", 6, f"实体仅占振幅{today_body/today_range*100:.0f}%")

    # 缩量止跌
    if len(volumes) >= 6:
        avg_vol_5 = sum(volumes[-6:-1]) / 5
        if avg_vol_5 > 0:
            vol_ratio = volumes[-1] / avg_vol_5
            today_ret = (closes[-1] / closes[-2] - 1) * 100 if len(closes) >= 2 else 0
            yest_ret = (closes[-2] / closes[-3] - 1) * 100 if len(closes) >= 3 else 0

            if vol_ratio < 0.7 and today_ret > yest_ret:
                signal_score += 8
                add("信号", "缩量止跌", 8, f"量比{vol_ratio:.2f},跌幅收窄{today_ret:.1f}%>{yest_ret:.1f}%")
            elif vol_ratio < 0.5:
                signal_score += 5
                add("信号", "极度缩量", 5, f"量比{vol_ratio:.2f} (<0.5)")

    # RSI底背离
    if len(closes) >= 30:
        rsi_now = calc_rsi(closes, 14)
        # 检查10-20天前是否有一个低点
        mid_idx = len(closes) - 15
        if closes[-1] < closes[mid_idx]:
            closes_mid = closes[:mid_idx+1]
            rsi_mid = calc_rsi(closes_mid, 14) if len(closes_mid) > 14 else 50
            if rsi_now > rsi_mid:
                signal_score += 10
                add("信号", "RSI底背离", 10, f"价格新低但RSI{rsi_now:.0f}>{rsi_mid:.0f}")

    # MACD绿柱缩短
    if len(closes) >= 35:
        dif = calc_ema(closes, 12)
        dea = calc_ema(closes, 26)
        if dif and dea:
            hist_now = dif - dea
            closes_prev = closes[:-1]
            dif_prev = calc_ema(closes_prev, 12)
            dea_prev = calc_ema(closes_prev, 26)
            if dif_prev and dea_prev:
                hist_prev = dif_prev - dea_prev
                if hist_now < 0 and hist_now > hist_prev:
                    signal_score += 5
                    add("信号", "MACD绿柱缩短", 5, f"柱体{hist_now:.4f}>{hist_prev:.4f}")

    # 收盘价回升至当日低点上方
    if today_range > 0:
        pos_in_range = (closes[-1] - lows[-1]) / today_range
        if pos_in_range > 0.5:
            signal_score += 5
            add("信号", "尾盘回升", 5, f"收盘位于日内{pos_in_range*100:.0f}%位置")

    signal_score = min(35, signal_score)

    # ========== 风险控制 (25%) ==========
    risk_score = 0

    # 流动性
    amount = stock.get("amount", 0)
    if amount > 5e8:  # >5亿
        risk_score += 8
        add("风控", "高流动性", 8, f"成交额{amount/1e8:.1f}亿")
    elif amount > 2e8:  # >2亿
        risk_score += 6
        add("风控", "良好流动性", 6, f"成交额{amount/1e8:.1f}亿")
    elif amount > 5e7:  # >5000万
        risk_score += 4
        add("风控", "基本流动性", 4, f"成交额{amount/1e7:.0f}千万")

    # 市值分层
    if mcap > 0:
        mcap_yi = mcap / 1e8
        if 50 <= mcap_yi <= 500:
            risk_score += 5
            add("风控", "中小盘", 5, f"市值{mcap_yi:.0f}亿")
        elif 500 <= mcap_yi <= 2000:
            risk_score += 3
            add("风控", "中大盘", 3, f"市值{mcap_yi:.0f}亿")
        elif mcap_yi > 2000:
            risk_score += 2
            add("风控", "大盘股", 2, f"市值{mcap_yi:.0f}亿")

    # 主板优先（±10%涨跌幅更可控）
    if limit_pct == 10:
        risk_score += 4
        add("风控", "主板可控", 4, "涨跌幅±10%")
    elif limit_pct == 20:
        risk_score += 2
        add("风控", "创/科创板", 2, "涨跌幅±20%")

    # 接近支撑位（MA60/MA120）
    ma60 = calc_ma(closes, 60)
    if ma60 and ma60 > 0:
        dev_ma60 = (closes[-1] / ma60 - 1) * 100
        if -3 < dev_ma60 < 3:
            risk_score += 5
            add("风控", "MA60支撑", 5, f"接近MA60({dev_ma60:+.1f}%)")
        elif -5 < dev_ma60 < 5:
            risk_score += 3
            add("风控", "MA60附近", 3, f"MA60偏离{dev_ma60:+.1f}%")

    # 波动率适中
    if len(closes) >= 21:
        daily_rets = [(closes[i] / closes[i-1] - 1) * 100 for i in range(-20, 0)]
        vol_20d = float(np.std(daily_rets)) if daily_rets else 0
        if 2 <= vol_20d <= 4:
            risk_score += 3
            add("风控", "波动适中", 3, f"20日波动率{vol_20d:.1f}%")

    risk_score = min(25, risk_score)

    # ========== 最终评分 ==========
    total = oversold_score * 0.40 + signal_score * 0.35 + risk_score * 0.25

    # 排除一字跌停（无反弹机会）
    if stock.get("open", 0) == stock.get("high", 0) == stock.get("low", 0) == stock.get("price", 0):
        return None

    # 排除涨停（不是超跌）
    if stock.get("pct", 0) >= limit_pct - 0.5:
        return None

    return {
        "code": code, "name": name, "price": price,
        "market": market, "limit_pct": limit_pct,
        "score": round(total, 1),
        "oversold": oversold_score, "signal": signal_score, "risk": risk_score,
        "factors": factors,
        "indicators": {
            "close": closes[-1], "ma20": ma20 or 0, "ma60": ma60 or 0,
            "rsi": rsi, "ret_5d": ret_5d, "ret_10d": ret_10d if len(closes) >= 11 else 0,
            "streak_dn": streak_dn, "vol_ratio": volumes[-1] / (sum(volumes[-6:-1])/5) if len(volumes) >= 6 and sum(volumes[-6:-1]) > 0 else 1,
            "dev_ma20": (closes[-1] / ma20 - 1) * 100 if ma20 else 0,
            "amount": amount, "mcap": mcap,
            "pct_today": stock.get("pct", 0),
        }
    }


# ================================================================
# HTML报告生成
# ================================================================

def generate_html(results, date_str, scan_stats):
    """生成超跌反弹独立HTML报告"""
    timestr = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 评分明细行
    def build_rows(sorted_r):
        rows = ""
        for i, r in enumerate(sorted_r):
            ind = r["indicators"]
            # 评分因子明细
            pos_f = [f for f in r["factors"] if f["delta"] > 0]
            detail_html = ""
            for f in pos_f[:8]:
                detail_html += f'<span class="factor-tag">{f["dim"]}:{f["name"]} +{f["delta"]}</span>'

            rows += f"""<tr class="{'row-top' if i < 10 else ''}" onclick="toggleRow('d-{r['code']}')">
<td>{i+1}</td><td>{r['code']}</td><td>{r['name']}</td><td>{r['market']}</td>
<td class="num">{r['price']:.2f}</td>
<td class="num {'negative' if ind['pct_today']<0 else 'positive'}">{ind['pct_today']:+.2f}%</td>
<td class="num negative">{ind['ret_5d']:+.1f}%</td>
<td class="num">{ind['rsi']:.0f}</td>
<td class="num score-col">{r['score']:.1f}</td>
<td class="num">{r['oversold']}/40</td>
<td class="num">{r['signal']}/35</td>
<td class="num">{r['risk']}/25</td>
</tr>
<tr class="detail-row" id="d-{r['code']}" style="display:none">
<td colspan="12"><div class="detail-card">
<div style="margin-bottom:6px"><strong>评分因子:</strong> {detail_html}</div>
<div style="font-size:12px;color:#666">
MA20={ind['ma20']:.2f} (偏离{ind['dev_ma20']:+.1f}%) | MA60={ind['ma60']:.2f} |
连跌{ind['streak_dn']}天 | 量比{ind['vol_ratio']:.2f} | 成交额{ind['amount']/1e8:.1f}亿 | 市值{ind['mcap']/1e8:.0f}亿 |
10日{ind['ret_10d']:+.1f}%
</div></div></td>
</tr>"""
        return rows

    top30 = sorted(results, key=lambda x: x["score"], reverse=True)[:30]
    rows = build_rows(top30)

    # 板块分布
    market_dist = {}
    for r in top30:
        m = r["market"]
        market_dist[m] = market_dist.get(m, 0) + 1
    market_html = " | ".join(f"{m}: {c}只" for m, c in sorted(market_dist.items(), key=lambda x: -x[1]))

    html = f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>超跌反弹评分报告 {date_str}</title>
<style>
:root{{--bg:#f5f7fa;--card-bg:#fff;--text:#1a1a2e;--text-sec:#666;--border:#e0e5ec;--primary:#dc2626;--positive:#dc2626;--negative:#16a34a}}
*{{margin:0;padding:0;box-sizing:border-box}}body{{font-family:-apple-system,BlinkMacSystemFont,sans-serif;background:var(--bg);color:var(--text);line-height:1.6}}
.container{{max-width:1400px;margin:0 auto;padding:20px}}
.header{{background:linear-gradient(135deg,#7f1d1d,#991b1b);color:white;padding:30px 40px;border-radius:16px;margin-bottom:20px}}
.header h1{{font-size:24px;margin-bottom:6px}}.header .meta{{font-size:13px;opacity:.8}}
.stats{{display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin-bottom:20px}}
.stat-card{{background:var(--card-bg);border-radius:12px;padding:16px;text-align:center;border:1px solid var(--border)}}
.stat-card .val{{font-size:28px;font-weight:700;color:var(--primary)}}.stat-card .lbl{{font-size:11px;color:var(--text-sec);margin-top:4px}}
.table-wrap{{background:var(--card-bg);border-radius:12px;overflow-x:auto;border:1px solid var(--border)}}
table{{width:100%;border-collapse:collapse;font-size:12px}}
thead{{background:#fef2f2;position:sticky;top:0}}
th{{padding:10px 6px;text-align:left;font-weight:600;color:#991b1b;cursor:pointer;white-space:nowrap;font-size:11px}}
td{{padding:7px 6px;border-bottom:1px solid var(--border)}}
tr:hover{{background:#fef9f9}}tr.row-top{{background:#fff5f5!important}}
.num{{text-align:right;font-variant-numeric:tabular-nums}}
.positive{{color:var(--positive)!important;font-weight:600}}.negative{{color:var(--negative)!important;font-weight:600}}
.score-col{{color:var(--primary);font-weight:700;font-size:14px}}
.factor-tag{{display:inline-block;background:#fee2e2;color:#991b1b;padding:2px 6px;border-radius:4px;font-size:10px;margin:2px}}
.detail-row td{{padding:0}}.detail-card{{padding:12px 16px;background:#fef9f9;border-top:2px dashed var(--border)}}
.footer{{margin-top:20px;text-align:center;font-size:11px;color:var(--text-sec)}}
</style></head><body>
<div class="container">
<div class="header">
<h1>超跌反弹评分报告</h1>
<div class="meta">{date_str} | 全A股扫描 | 持仓周期T+3~5 | 评分=超跌40%+反弹信号35%+风控25%</div>
</div>

<div class="stats">
<div class="stat-card"><div class="val">{scan_stats['total']}</div><div class="lbl">全市场扫描</div></div>
<div class="stat-card"><div class="val">{scan_stats['screened']}</div><div class="lbl">超跌初筛</div></div>
<div class="stat-card"><div class="val">{len(results)}</div><div class="lbl">有效评分</div></div>
<div class="stat-card"><div class="val">{len(top30)}</div><div class="lbl">推荐TOP30</div></div>
<div class="stat-card"><div class="val">{scan_stats['avg_rsi']:.0f}</div><div class="lbl">TOP30均RSI</div></div>
</div>

<div class="table-wrap"><table><thead><tr>
<th>#</th><th>代码</th><th>名称</th><th>市场</th><th>现价</th><th>今日</th><th>5日</th><th>RSI</th>
<th>反弹分↓</th><th>超跌(40)</th><th>信号(35)</th><th>风控(25)</th>
</tr></thead><tbody>{rows}</tbody></table></div>

<div style="margin-top:12px;font-size:12px;color:var(--text-sec)">
<strong>板块分布:</strong> {market_html}
</div>

<div class="footer">
<p>超跌反弹引擎 v1.0 | 全A股扫描 | 生成于 {timestr}</p>
<p style="margin-top:3px">超跌=跌幅/RSI/偏离度/连跌 | 信号=下影线/缩量/底背离/MACD | 风控=流动性/市值/主板/支撑位 | 点击行展开明细</p>
<p style="margin-top:3px;color:#999">⚠️ 超跌反弹为逆向策略，风险较高。建议设-5%止损，+5%~8%止盈，严格纪律。</p>
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
    print("  超跌反弹评分引擎 v1.0")
    print("  全A股扫描 | 超跌初筛 → 反弹信号评分 → 独立报告")
    print("  评分 = 超跌程度(40%) + 反弹信号(35%) + 风险控制(25%)")
    print("=" * 60)

    date_str = datetime.now().strftime("%Y-%m-%d")

    # [1/4] 全市场扫描
    print(f"\n  [1/4] 获取全A股实时行情...")
    all_stocks = fetch_all_a_shares()
    print(f"  全A股: {len(all_stocks)}只 (已排除ST/退市)")

    # [2/4] 超跌初筛
    print(f"\n  [2/4] 超跌初筛...")
    screened = []
    for s in all_stocks:
        # 排除条件
        if s["amount"] < 1e7:  # 成交额<1000万
            continue
        if s["pct"] >= 0:  # 今日非下跌
            continue
        # 初筛: 今日跌幅<-3% 或 5日跌幅<-5% (用pct近似,后续K线精确计算)
        if s["pct"] < -3:
            screened.append(s)

    print(f"  初筛通过: {len(screened)}只 (今日跌幅<-3%)")

    # [3/4] 获取K线 + 详细评分
    print(f"\n  [3/4] 获取K线 + 计算超跌反弹评分...")
    codes = [s["code"] for s in screened]
    klines_map = batch_fetch_klines(codes, days=60, workers=20)
    print(f"  K线获取: {len(klines_map)}/{len(codes)}")

    results = []
    for s in screened:
        kl = klines_map.get(s["code"])
        if not kl:
            continue
        r = score_oversold_rebound(s, kl)
        if r and r["score"] >= 20:  # 最低分阈值
            results.append(r)

    results.sort(key=lambda x: x["score"], reverse=True)
    print(f"  有效评分: {len(results)}只 (评分>=20)")

    # [4/4] 生成报告
    print(f"\n  [4/4] 生成报告...")
    top30 = results[:30]
    avg_rsi = sum(r["indicators"]["rsi"] for r in top30) / len(top30) if top30 else 50
    scan_stats = {
        "total": len(all_stocks),
        "screened": len(screened),
        "avg_rsi": avg_rsi,
    }

    html = generate_html(results, date_str, scan_stats)
    html_path = os.path.join(OUTPUT_DIR, f"oversold_{date_str}.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)

    # JSON输出
    json_path = os.path.join(OUTPUT_DIR, f"oversold_{date_str}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results[:50], f, ensure_ascii=False, indent=2)

    print(f"\n  {'='*60}")
    print(f"  ✅ 报告已生成!")
    print(f"     HTML: {html_path}")
    print(f"     JSON: {json_path}")
    print(f"     超跌反弹TOP30:")
    for i, r in enumerate(top30[:10]):
        ind = r["indicators"]
        print(f"     {i+1}. {r['code']} {r['name']} | 反弹分{r['score']:.1f} "
              f"(超跌{r['oversold']}/信号{r['signal']}/风控{r['risk']}) "
              f"| 5日{ind['ret_5d']:+.1f}% RSI{ind['rsi']:.0f}")
    print(f"  {'='*60}")


if __name__ == "__main__":
    main()
