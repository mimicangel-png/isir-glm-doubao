#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全池资金流历史回补脚本 — 拉到 365 天逐日数据，存完整字段(含散户)。

字段来源(westock CLI fund flow 区间查询逐日返回):
  MainNetFlow  当日主力净流入
  JumboNetFlow 超大单净流入
  MidNetFlow   大单净流入
  SmallNetFlow 小单净流入
  RetailInFlow 散户流入
  RetailOutFlow 散户流出
  MainInFlow   主力流入
  MainOutFlow  主力流出

用法:
  python backfill_fundflow_full.py --days 400    # 全池回补400自然日
  python backfill_fundflow_full.py --codes 300497,000001 --days 400
"""
import os, sys, subprocess, sqlite3, time
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

SELF_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(SELF_DIR, "output", "stock_cache.db")
WESTOCK = os.path.expanduser("~/.local/bin/westock")
CODES_FILE = os.path.join(SELF_DIR, "stock_codes.txt")

FIELDS = ["MainNetFlow", "JumboNetFlow", "MidNetFlow", "SmallNetFlow",
          "RetailInFlow", "RetailOutFlow", "MainInFlow", "MainOutFlow"]


def _to_symbol(code):
    if code.startswith(("6", "9", "58")):
        return "sh" + code
    if code.startswith(("8", "4", "92")):
        return "bj" + code
    return "sz" + code


def load_codes():
    codes = []
    with open(CODES_FILE) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                codes.append(line)
    return codes


def parse_flow(text):
    """解析区间查询逐日数据 -> {date: {field: value}}"""
    out = {}
    lines = [l for l in text.splitlines() if l.strip().startswith("|")]
    if len(lines) < 3:
        return out
    header = [c.strip() for c in lines[0].strip("|").split("|")]
    if "MainNetFlow" not in header:
        return out
    idx = {h: i for i, h in enumerate(header) if h in FIELDS or h == "date"}
    if "date" not in idx:
        return out
    for line in lines[2:]:
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < len(header):
            continue
        d = cells[idx["date"]][:10]
        row = {}
        for fld in FIELDS:
            try:
                row[fld] = float(cells[idx[fld]])
            except (ValueError, IndexError, KeyError):
                row[fld] = None
        out[d] = row
    return out


def fetch_one(code, start, end):
    sym = _to_symbol(code)
    for attempt in (1, 2):
        try:
            r = subprocess.run([WESTOCK, "fund", "flow", sym, "--start", start, "--end", end],
                               capture_output=True, text=True, timeout=90)
            data = parse_flow(r.stdout)
            if data:
                return code, data
        except Exception:
            pass
        time.sleep(0.5)
    return code, {}


def ensure_schema(conn):
    # 加散户/大单/小单字段(若不存在)
    cols = {c[1] for c in conn.execute("PRAGMA table_info(fund_flows)")}
    for fld in ["retail_inflow", "retail_outflow", "mid_net", "small_net",
                "main_inflow", "main_outflow"]:
        if fld not in cols:
            conn.execute(f"ALTER TABLE fund_flows ADD COLUMN {fld} REAL")
    conn.commit()


def main():
    args = sys.argv[1:]
    days = 400
    only_codes = None
    i = 0
    while i < len(args):
        if args[i] == "--days":
            days = int(args[i+1]); i += 2
        elif args[i] == "--codes":
            only_codes = args[i+1].split(","); i += 2
        else:
            i += 1

    codes = only_codes if only_codes else load_codes()
    end = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    print(f"回补 {len(codes)}只资金流 {start} ~ {end}")

    conn = sqlite3.connect(DB_PATH)
    ensure_schema(conn)

    rows = []
    got = 0
    fail = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=10) as ex:
        futures = {ex.submit(fetch_one, c, start, end): c for c in codes}
        for f in as_completed(futures):
            code, data = f.result()
            if data:
                for d, row in data.items():
                    rows.append((code, d, row["MainNetFlow"], row["JumboNetFlow"],
                                 row["MidNetFlow"], row["SmallNetFlow"],
                                 row["RetailInFlow"], row["RetailOutFlow"],
                                 row["MainInFlow"], row["MainOutFlow"]))
                got += 1
            else:
                fail += 1
            if (got + fail) % 50 == 0:
                print(f"  进度 {got+fail}/{len(codes)} 成功{got} 失败{fail} 耗时{time.time()-t0:.0f}s")

    if rows:
        conn.executemany(
            """INSERT OR REPLACE INTO fund_flows
               (code,date,main_net_today,jumbo_net,mid_net,small_net,
                retail_inflow,retail_outflow,main_inflow,main_outflow,fetched_at)
               VALUES(?,?,?,?,?,?,?,?,?,?,datetime('now'))""",
            rows)
        conn.commit()

    n = conn.execute("SELECT COUNT(*), COUNT(DISTINCT code), MIN(date), MAX(date) FROM fund_flows").fetchone()
    print(f"\n✅ 完成: 成功{got}只 失败{fail}只")
    print(f"   fund_flows表: {n[0]}行 {n[1]}只 {n[2]}~{n[3]}")
    conn.close()


if __name__ == "__main__":
    main()
