#!/usr/bin/env python3
"""
实战买卖策略回测
==================
策略规则：
  1. 买入：股票进入TOP30，T+1开盘价买入
  2. 仓位上限：30只（每个TOP30席位一个仓位）
  3. 退出条件（任一触发即卖）：
     a. 时间止盈：持仓10个交易日，收盘卖出
     b. 止损：浮亏 ≤ -8%，次日开盘卖出
     c. 止盈：浮盈 ≥ +15%，次日开盘卖出
     d. 排名崩溃：排名跌到后50%，次日开盘卖出
  4. 仓位管理：只在有空闲席位时才买入新股

回测方法：
  - 逐日遍历历史数据
  - T+1开盘价成交（避免当日收盘追高）
  - 每个策略独立回测
  - 输出：胜率、均收益、最大收益、累积收益、夏普比率
"""

import os, json, sys, math
from datetime import datetime
from collections import defaultdict
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stock_db import StockDB
import sector_map
from unified_scoring_engine import (
    compute_all_factors, compute_rankings,
    OUTPUT_DIR, SELF_DIR, BUY_COUNT, TRADE_FILE, SIGNAL_FILE
)

# ================================================================
# Strategy Parameters
# ================================================================

MAX_POSITIONS = 30       # 最大持仓数
HOLD_DAYS_MAX = 10       # 最长持仓天数
STOP_LOSS = -8.0         # 止损线 (%)
TAKE_PROFIT = 15.0       # 止盈线 (%)
RANK_COLLAPSE_PCT = 0.5  # 排名崩溃阈值（后50%）

STRATEGIES = ["isir", "glm", "doubao"]
HOLD_PERIODS = [5, 10, 20]  # 对比用：固定持仓周期基准


def run_strategy_backtest(klines, extra_today, sectors, backtest_days=85):
    """
    完整回测：实战策略 vs 固定持仓基准
    """
    # 获取交易日历
    all_dates = set()
    for k in klines.values():
        for b in k:
            all_dates.add(b["date"])
    dates = sorted(all_dates)
    
    # 留够前向收益空间
    start_idx = max(60, len(dates) - backtest_days)
    test_dates = dates[start_idx:]
    print(f"  回测区间: {test_dates[0]} ~ {test_dates[-1]} ({len(test_dates)}个交易日)")
    
    # 预构建每个交易日的K线快照和次日开盘价
    print(f"\n  预处理数据...")
    daily_data = {}  # {date: {klines, extra, next_opens}}
    for di, date in enumerate(test_dates):
        if di % 20 == 0:
            print(f"  {di}/{len(test_dates)}")
        
        day_klines = {}
        for code, k in klines.items():
            day_bars = [b for b in k if b["date"] <= date]
            if len(day_bars) >= 60:
                day_klines[code] = day_bars
        
        if len(day_klines) < 100:
            continue
        
        # 构建extra
        day_extra = {}
        for code in day_klines:
            kb = day_klines[code]
            if len(kb) >= 2:
                last = kb[-1]
                avg5 = sum(b["volume"] for b in kb[-6:-1]) / 5 if len(kb) >= 6 else last["volume"]
                day_extra[code] = {
                    "name": extra_today.get(code, {}).get("name", code),
                    "price": last["close"],
                    "change_pct": (last["close"] / kb[-2]["close"] - 1) * 100,
                    "pe_ttm": 0, "pb": 0, "mcap": 0, "turnover": 0,
                    "vol_ratio": last["volume"] / avg5 if avg5 > 0 else 1.0
                }
        
        # T+1开盘价
        next_opens = {}
        for code in day_klines:
            fwd = [b for b in klines.get(code, []) if b["date"] > date]
            if fwd:
                next_opens[code] = fwd[0]["open"]
        
        daily_data[date] = {
            "klines": day_klines,
            "extra": day_extra,
            "next_opens": next_opens,
        }
    
    print(f"  有效交易日: {len(daily_data)}")
    
    # ================================================================
    # 1. 实战策略回测
    # ================================================================
    
    strategy_results = {}
    
    for strat in STRATEGIES:
        print(f"\n  === 回测策略: {strat.upper()} ===")
        positions = []  # [{code, entry_date, entry_price, entry_rank, hold_days, ...}]
        closed_trades = []
        signals = []
        rank_key = f"{strat}_rank"
        
        for date in sorted(daily_data.keys()):
            dd = daily_data[date]
            day_klines = dd["klines"]
            day_extra = dd["extra"]
            next_opens = dd["next_opens"]
            
            # 计算当日排名
            factor_data, _ = compute_all_factors(day_klines, day_extra, {}, {}, sectors)
            if len(factor_data) < 100:
                continue
            rankings = compute_rankings(factor_data)
            if not rankings:
                continue
            rank_map = {r["code"]: r for r in rankings}
            n_total = len(rankings)
            
            current_top = {r["code"] for r in rankings if r[rank_key] <= BUY_COUNT}
            
            # === Step 1: 检查现有持仓的退出条件 ===
            still_holding = []
            for pos in positions:
                code = pos["code"]
                price = day_extra.get(code, {}).get("price", 0)
                if price <= 0:
                    still_holding.append(pos)
                    continue
                
                ret = (price / pos["entry_price"] - 1) * 100
                pos["hold_days"] += 1
                pos["current_price"] = price
                pos["current_return"] = round(ret, 2)
                
                current_rank = rank_map.get(code, {}).get(rank_key, n_total)
                
                # 退出判断
                exit_reason = None
                if pos["hold_days"] >= HOLD_DAYS_MAX:
                    exit_reason = f"时间到期({HOLD_DAYS_MAX}天)"
                elif ret <= STOP_LOSS:
                    exit_reason = f"止损({ret:+.1f}%≤{STOP_LOSS}%)"
                elif ret >= TAKE_PROFIT:
                    exit_reason = f"止盈({ret:+.1f}%≥{TAKE_PROFIT}%)"
                elif current_rank > n_total * RANK_COLLAPSE_PCT:
                    exit_reason = f"排名崩溃(#{current_rank}/{n_total})"
                
                if exit_reason:
                    # T+1开盘卖出
                    exit_price = next_opens.get(code, price)
                    exit_ret = round((exit_price / pos["entry_price"] - 1) * 100, 2)
                    pos["exit_date"] = date
                    pos["exit_price"] = exit_price
                    pos["return_pct"] = exit_ret
                    pos["is_win"] = exit_ret > 0
                    pos["exit_reason"] = exit_reason
                    closed_trades.append(pos)
                    signals.append({
                        "date": date, "strategy": strat, "code": code,
                        "name": pos.get("name", ""), "signal": "sell",
                        "price": exit_price, "rank": current_rank,
                        "return_pct": exit_ret,
                        "entry_date": pos["entry_date"],
                        "entry_price": pos["entry_price"],
                        "reason": exit_reason
                    })
                else:
                    still_holding.append(pos)
            
            positions = still_holding
            
            # === Step 2: 买入新股（只在有空位时）===
            existing_codes = {p["code"] for p in positions}
            available_slots = MAX_POSITIONS - len(positions)
            
            if available_slots > 0:
                # 按排名排序，取前 available_slots 个新股
                new_candidates = []
                for code in current_top - existing_codes:
                    r = rank_map.get(code)
                    if r and code in next_opens and next_opens[code] > 0:
                        new_candidates.append((code, r[rank_key]))
                
                new_candidates.sort(key=lambda x: x[1])  # 按排名升序
                
                for code, rank in new_candidates[:available_slots]:
                    entry_price = next_opens[code]  # T+1开盘买入
                    name = day_extra.get(code, {}).get("name", code)
                    pos = {
                        "code": code, "name": name,
                        "entry_date": date, "entry_price": entry_price,
                        "entry_rank": rank,
                        "hold_days": 0,
                        "current_price": entry_price,
                        "current_return": 0,
                    }
                    positions.append(pos)
                    signals.append({
                        "date": date, "strategy": strat, "code": code,
                        "name": name, "signal": "buy",
                        "price": entry_price, "rank": rank,
                        "return_pct": 0,
                        "entry_date": date,
                        "entry_price": entry_price,
                    })
        
        # 强制平仓剩余持仓
        last_date = sorted(daily_data.keys())[-1]
        for pos in positions:
            dd = daily_data[last_date]
            price = dd["extra"].get(pos["code"], {}).get("price", 0)
            ret = round((price / pos["entry_price"] - 1) * 100, 2) if price > 0 else 0
            pos["exit_date"] = last_date
            pos["exit_price"] = price
            pos["return_pct"] = ret
            pos["is_win"] = ret > 0
            pos["exit_reason"] = "回测结束平仓"
            closed_trades.append(pos)
        
        # 统计
        total = len(closed_trades)
        wins = sum(1 for t in closed_trades if t["return_pct"] > 0)
        win_rate = round(wins / total * 100, 1) if total > 0 else 0
        avg_ret = round(np.mean([t["return_pct"] for t in closed_trades]), 2) if total > 0 else 0
        max_ret = round(max([t["return_pct"] for t in closed_trades]), 2) if total > 0 else 0
        min_ret = round(min([t["return_pct"] for t in closed_trades]), 2) if total > 0 else 0
        cum_ret = round(sum(t["return_pct"] for t in closed_trades), 1)
        median_ret = round(np.median([t["return_pct"] for t in closed_trades]), 2) if total > 0 else 0
        
        # 退出原因统计
        reason_stats = defaultdict(int)
        for t in closed_trades:
            reason = t.get("exit_reason", "未知").split("(")[0]
            reason_stats[reason] += 1
        
        # 平均持仓天数
        avg_hold = round(np.mean([t["hold_days"] for t in closed_trades]), 1) if total > 0 else 0
        
        strategy_results[strat] = {
            "total_trades": total,
            "win_rate": win_rate,
            "avg_return": avg_ret,
            "median_return": median_ret,
            "max_return": max_ret,
            "min_return": min_ret,
            "cumulative_return": cum_ret,
            "avg_hold_days": avg_hold,
            "reason_stats": dict(reason_stats),
            "closed_trades": closed_trades[-50:],  # 最近50笔
            "signals": signals[-500:],
        }
        
        print(f"  交易: {total}笔 | 胜率: {win_rate}% | 均收益: {avg_ret:+.1f}% | "
              f"中位: {median_ret:+.1f}% | 最大: {max_ret:+.1f}% | 累积: {cum_ret:+.1f}%")
        print(f"  平均持仓: {avg_hold}天 | 退出原因: {dict(reason_stats)}")
    
    # ================================================================
    # 2. 固定持仓基准对比
    # ================================================================
    
    print(f"\n  === 固定持仓基准对比 ===")
    baseline_results = {}
    
    for strat in STRATEGIES:
        rank_key = f"{strat}_rank"
        for hold in HOLD_PERIODS:
            returns = []
            
            for date in sorted(daily_data.keys()):
                dd = daily_data[date]
                day_klines = dd["klines"]
                day_extra = dd["extra"]
                
                factor_data, _ = compute_all_factors(day_klines, day_extra, {}, {}, sectors)
                if len(factor_data) < 100:
                    continue
                rankings = compute_rankings(factor_data)
                if not rankings:
                    continue
                
                top30 = [r for r in rankings if r[rank_key] <= BUY_COUNT]
                
                for r in top30:
                    code = r["code"]
                    # T+1 open 买入
                    entry = dd["next_opens"].get(code)
                    if not entry or entry <= 0:
                        continue
                    
                    # hold天后 close 卖出
                    fwd_bars = [b for b in klines.get(code, []) if b["date"] > date]
                    if len(fwd_bars) >= hold:
                        exit_price = fwd_bars[hold - 1]["close"]
                        ret = round((exit_price / entry - 1) * 100, 2)
                        returns.append(ret)
            
            if returns:
                arr = np.array(returns)
                wr = round(np.mean(arr > 0) * 100, 1)
                avg = round(np.mean(arr), 2)
                baseline_results[f"{strat}_hold{hold}"] = {
                    "win_rate": wr, "avg_return": avg,
                    "total": len(returns)
                }
                print(f"  {strat} hold{hold}: {len(returns)}笔 | 胜率{wr}% | 均收益{avg:+.1f}%")
    
    # ================================================================
    # 3. 保存结果
    # ================================================================
    
    # 更新trade ledger和signal history
    trades_out = {}
    for strat in STRATEGIES:
        r = strategy_results[strat]
        trades_out[strat] = {
            "open": [],
            "closed": r["closed_trades"],
            "cumulative_return": r["cumulative_return"],
            "win_count": sum(1 for t in r["closed_trades"] if t["return_pct"] > 0),
            "total_count": r["total_trades"],
        }
    
    all_signals = []
    for strat in STRATEGIES:
        all_signals.extend(strategy_results[strat]["signals"])
    all_signals = sorted(all_signals, key=lambda x: x.get("date", ""))[-500:]
    
    with open(TRADE_FILE, "w") as f:
        json.dump(trades_out, f, ensure_ascii=False, indent=2)
    with open(SIGNAL_FILE, "w") as f:
        json.dump(all_signals, f, ensure_ascii=False, indent=2)
    
    # 保存策略回测结果
    bt_result = {
        "strategy": {s: {k: v for k, v in r.items() if k not in ("closed_trades", "signals")}
                      for s, r in strategy_results.items()},
        "baseline": baseline_results,
        "params": {
            "max_positions": MAX_POSITIONS,
            "hold_days_max": HOLD_DAYS_MAX,
            "stop_loss": STOP_LOSS,
            "take_profit": TAKE_PROFIT,
            "rank_collapse_pct": RANK_COLLAPSE_PCT,
            "entry_price": "T+1 open",
            "exit_price": "T+1 open (止损/止盈/排名) or close (时间到期)",
        },
        "date": datetime.now().strftime("%Y-%m-%d"),
    }
    bt_path = os.path.join(OUTPUT_DIR, "strategy_backtest.json")
    with open(bt_path, "w") as f:
        json.dump(bt_result, f, ensure_ascii=False, indent=2)
    
    # ================================================================
    # 4. 汇总输出
    # ================================================================
    
    print(f"\n{'='*70}")
    print(f"  实战策略回测结果")
    print(f"  策略: T+1开盘买入 | 最大{MAX_POSITIONS}仓位 | 止损{STOP_LOSS}% | 止盈+{TAKE_PROFIT}% | 持仓≤{HOLD_DAYS_MAX}天")
    print(f"{'='*70}")
    print(f"\n  {'策略':<8} {'交易笔数':>8} {'胜率':>6} {'均收益':>8} {'中位收益':>8} {'最大收益':>8} {'累积收益':>10} {'均持仓':>6}")
    print(f"  {'-'*70}")
    
    for strat in STRATEGIES:
        r = strategy_results[strat]
        print(f"  {strat:<8} {r['total_trades']:>8} {r['win_rate']:>5.1f}% {r['avg_return']:>+7.1f}% "
              f"{r['median_return']:>+7.1f}% {r['max_return']:>+7.1f}% {r['cumulative_return']:>+9.1f}% {r['avg_hold_days']:>5.1f}天")
    
    print(f"\n  对比基准 (固定持仓, T+1开盘买入):")
    print(f"  {'策略':<8} {'周期':>4} {'笔数':>6} {'胜率':>6} {'均收益':>8}")
    print(f"  {'-'*40}")
    for key, val in sorted(baseline_results.items()):
        parts = key.split("_hold")
        print(f"  {parts[0]:<8} {parts[1]+'天':>4} {val['total']:>6} {val['win_rate']:>5.1f}% {val['avg_return']:>+7.1f}%")
    
    # 选出最优策略
    best_strat = max(strategy_results.keys(), 
                     key=lambda s: strategy_results[s]["win_rate"] * 0.5 + max(0, strategy_results[s]["avg_return"]) * 2)
    print(f"\n  🏆 最优策略: {best_strat.upper()}")
    print(f"     胜率: {strategy_results[best_strat]['win_rate']}% | 均收益: {strategy_results[best_strat]['avg_return']:+.1f}%")
    
    print(f"\n  文件:")
    print(f"    {bt_path}")
    print(f"    {TRADE_FILE}")
    print(f"    {SIGNAL_FILE}")
    print(f"{'='*70}")
    
    return strategy_results, baseline_results, best_strat


def main():
    print("=" * 70)
    print("  实战买卖策略回测")
    print("  T+1开盘买入 | 30仓位上限 | 止损-8% | 止盈+15% | 持仓≤10天")
    print("=" * 70)
    
    db = StockDB()
    with open(os.path.join(SELF_DIR, "stock_codes.txt")) as f:
        codes = [l.strip() for l in f if l.strip()]
    print(f"  股票池: {len(codes)}只")
    
    klines = db.get_klines(codes, days=130)
    extra_today = db.get_extra_info(codes, force_refresh=False)
    print(f"  K线: {len(klines)}只")
    
    sectors = {c: sector_map.get_sector(c) for c in codes}
    
    strategy_results, baseline_results, best_strat = run_strategy_backtest(
        klines, extra_today, sectors, backtest_days=85
    )
    
    return best_strat


if __name__ == "__main__":
    main()
