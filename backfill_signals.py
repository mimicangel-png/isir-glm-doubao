#!/usr/bin/env python3
"""回溯14天：固定10天持仓周期，对齐回测方法。"""
import os, json, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stock_db import StockDB
import sector_map
from unified_scoring_engine import (
    compute_all_factors, compute_rankings, TOP_N, BUY_COUNT,
    OUTPUT_DIR, SELF_DIR, HISTORY_FILE, TRADE_FILE, SIGNAL_FILE
)

def run_backfill(days=14):
    print("="*60)
    print(f"  回溯 {days} 天 — 固定10天持仓周期")
    print("="*60)

    db = StockDB()
    with open(os.path.join(SELF_DIR, "stock_codes.txt")) as f:
        codes = [l.strip() for l in f if l.strip()]
    print(f"  股票池: {len(codes)}只")

    klines = db.get_klines(codes, days=130)
    extra_today = db.get_extra_info(codes, force_refresh=False)
    print(f"  K线: {len(klines)}只")

    all_dates = set()
    for k in klines.values():
        for b in k: all_dates.add(b["date"])
    dates = sorted(all_dates)
    test_dates = dates[-days:]
    print(f"  回溯: {test_dates[0]} ~ {test_dates[-1]}")

    trades = {s:{"open":[],"closed":[],"cumulative_return":0,"win_count":0,"total_count":0} for s in ["isir","glm","doubao"]}
    signals = []
    rank_history = {}
    sectors = {c: sector_map.get_sector(c) for c in codes}
    HOLD = 10

    print(f"\n  逐日模拟...")
    for di, date in enumerate(test_dates):
        day_klines = {c: [b for b in k if b["date"] <= date] for c, k in klines.items() if len([b for b in k if b["date"] <= date]) >= 60}
        if len(day_klines) < 100: continue

        day_extra = {}
        for code in day_klines:
            kb = day_klines[code]
            if len(kb) >= 2:
                last = kb[-1]
                avg5 = sum(b["volume"] for b in kb[-6:-1])/5 if len(kb)>=6 else last["volume"]
                day_extra[code] = {
                    "name": extra_today.get(code,{}).get("name",code),
                    "price": last["close"],
                    "change_pct": (last["close"]/kb[-2]["close"]-1)*100,
                    "pe_ttm":0,"pb":0,"mcap":0,"turnover":0,
                    "vol_ratio": last["volume"]/avg5 if avg5>0 else 1.0
                }

        factor_data, _ = compute_all_factors(day_klines, day_extra, {}, {}, sectors)
        if len(factor_data) < 100: continue
        rankings = compute_rankings(factor_data)
        if not rankings: continue
        rank_map = {r["code"]: r for r in rankings}

        for r in rankings:
            c = r["code"]
            if c not in rank_history: rank_history[c] = {"isir":[],"glm":[],"doubao":[],"ss":[]}
            rank_history[c]["name"] = day_extra.get(c,{}).get("name",c)
            for k in ["isir","glm","doubao"]:
                rank_history[c][k].append({"date":date,"rank":r[f"{k}_rank"]})
                rank_history[c][k] = rank_history[c][k][-5:]
            rank_history[c]["ss"].append({"date":date,"score":r["ss_score"]})
            rank_history[c]["ss"] = rank_history[c]["ss"][-5:]

        for strategy in ["isir","glm","doubao"]:
            rk = f"{strategy}_rank"
            cur_top = {r["code"] for r in rankings if r[rk] <= BUY_COUNT}
            cur_info = {r["code"]: r for r in rankings if r[rk] <= BUY_COUNT}

            still_open = []
            for trade in trades[strategy]["open"]:
                code = trade["code"]
                price = day_extra.get(code,{}).get("price",0)
                ret = round((price/trade["entry_price"]-1)*100,2) if price>0 else 0
                trade["hold_days"] += 1
                trade["current_price"] = price
                trade["current_return"] = ret

                if trade["hold_days"] >= HOLD:
                    trade["exit_date"] = date; trade["exit_price"] = price
                    trade["return_pct"] = ret; trade["is_win"] = ret > 0
                    trade["status"] = "closed"; trade["exit_reason"] = f"持仓到期({HOLD}天)"
                    trades[strategy]["closed"].append(trade)
                    trades[strategy]["cumulative_return"] += ret
                    trades[strategy]["total_count"] += 1
                    if ret > 0: trades[strategy]["win_count"] += 1
                    signals.append({"date":date,"strategy":strategy,"code":code,
                        "name":trade.get("name",""),"signal":"sell","price":price,
                        "rank":rank_map.get(code,{}).get(rk,0),"return_pct":ret,
                        "entry_date":trade.get("entry_date",""),"entry_price":trade.get("entry_price",0),
                        "reason":f"持仓到期({HOLD}天)"})
                else:
                    still_open.append(trade)

            existing = {t["code"] for t in still_open}
            for code in cur_top - existing:
                price = day_extra.get(code,{}).get("price",0)
                if price > 0:
                    name = day_extra.get(code,{}).get("name",code)
                    still_open.append({"code":code,"name":name,"entry_date":date,
                        "entry_price":price,"entry_rank":cur_info[code][rk],
                        "hold_days":0,"current_price":price,"current_return":0,"status":"open"})
                    signals.append({"date":date,"strategy":strategy,"code":code,
                        "name":name,"signal":"buy","price":price,
                        "rank":cur_info[code][rk],"return_pct":0,
                        "entry_date":date,"entry_price":price})

            trades[strategy]["open"] = still_open

        buys = sum(1 for s in signals if s["date"]==date and s["signal"]=="buy")
        sells = sum(1 for s in signals if s["date"]==date and s["signal"]=="sell")
        print(f"  [{di+1}/{len(test_dates)}] {date}: 买入{buys} 卖出{sells}")

    signals = signals[-500:]
    with open(TRADE_FILE,"w") as f: json.dump(trades,f,ensure_ascii=False,indent=2)
    with open(SIGNAL_FILE,"w") as f: json.dump(signals,f,ensure_ascii=False,indent=2)
    with open(HISTORY_FILE,"w") as f: json.dump(rank_history,f,ensure_ascii=False)

    total = len(signals)
    sbuys = [s for s in signals if s["signal"]=="buy"]
    ssells = [s for s in signals if s["signal"]=="sell"]
    wins = sum(1 for s in ssells if s.get("return_pct",0)>0)
    wr = round(wins/len(ssells)*100,1) if ssells else 0

    print(f"\n  {'='*60}")
    print(f"  ✅ 完成! 信号{total}条 (买入{len(sbuys)} 卖出{len(ssells)})")
    print(f"  卖出胜率: {wr}%")
    for s in ["isir","glm","doubao"]:
        t = trades[s]
        w = round(t["win_count"]/t["total_count"]*100,1) if t["total_count"]>0 else 0
        print(f"  {s}: 持仓{len(t['open'])} | 结算{t['total_count']}笔 | 胜率{w}% | 累积{t['cumulative_return']:+.1f}%")
    print(f"  {'='*60}")

if __name__ == "__main__":
    run_backfill(14)
