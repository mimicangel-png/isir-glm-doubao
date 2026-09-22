#!/usr/bin/env python3
"""backfill_fundflow_history.py — 补拉科技板块历史资金流(60天)用于轮动回测
用 westock CLI 单代码 --start --end 拉逐日 MainNetFlow, 写 fund_flows 历史表。
"""
import os, sys, subprocess, sqlite3, time
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

SELF_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(SELF_DIR, "output", "stock_cache.db")
WESTOCK = os.path.expanduser("~/.local/bin/westock")

import sector_map
TECH = {"半导体/芯片", "AI/算力/通信", "电子/消费电子"}

def _to_symbol(code):
    if code.startswith(("6", "9", "58")):
        return "sh" + code
    if code.startswith(("8", "4", "92")):
        return "bj" + code
    return "sz" + code

def parse_flow(text):
    """解析历史资金流表 -> {date: main_net_today}"""
    out = {}
    lines = [l for l in text.splitlines() if l.strip().startswith("|")]
    if len(lines) < 3:
        return out
    header = [c.strip() for c in lines[0].strip("|").split("|")]
    if "MainNetFlow" not in header:
        return out
    mi = header.index("MainNetFlow")
    di = header.index("date") if "date" in header else 0
    for line in lines[2:]:
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < len(header):
            continue
        d = cells[di][:10]
        try:
            v = float(cells[mi])
        except (ValueError, IndexError):
            continue
        out[d] = v
    return out

def fetch_one(code, start, end):
    sym = _to_symbol(code)
    for attempt in (1, 2):
        try:
            r = subprocess.run([WESTOCK, "fund", "flow", sym, "--start", start, "--end", end],
                               capture_output=True, text=True, timeout=60)
            data = parse_flow(r.stdout)
            if data:
                return code, data
        except Exception:
            pass
        time.sleep(1)
    return code, {}

def main():
    codes = [c for c, s in sector_map.STOCK_SECTOR.items() if s in TECH]
    end = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=75)).strftime("%Y-%m-%d")  # 75自然日≈55交易日
    print(f"补拉 {len(codes)}只科技股资金流 {start}~{end}")

    got = 0
    rows = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = {ex.submit(fetch_one, c, start, end): c for c in codes}
        for f in as_completed(futures):
            code, data = f.result()
            if data:
                for d, v in data.items():
                    rows.append((code, d, v))
            got += 1
            if got % 50 == 0:
                print(f"  进度 {got}/{len(codes)}", file=sys.stderr)

    if not rows:
        print("❌ 未拉到任何数据")
        return 1

    with sqlite3.connect(DB_PATH) as conn:
        conn.executemany(
            "INSERT OR REPLACE INTO fund_flows(code,date,main_net_today,fetched_at) VALUES(?,?,?,datetime('now'))",
            [(c, d, v) for c, d, v in rows])
    ndays = len({d for _, d, _ in rows})
    print(f"✅ 入库 {len(rows)}条 | {got}只 | {ndays}个交易日")
    return 0

if __name__ == "__main__":
    sys.exit(main())
