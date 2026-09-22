#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
拉取全池筹码数据 (westock CLI chip 接口) 写入 SQLite。

字段: chipAvgCost(平均成本) / chipConcentration70(70%筹码集中度,越小越集中)
      chipConcentration90(90%筹码集中度) / chipProfitRate(获利盘比例%) / closePrice

用法:
  python fetch_chip_data.py                 # 全池拉今日筹码快照
  python fetch_chip_data.py --history 60    # 全池拉近60个交易日筹码历史(回测用)
  python fetch_chip_data.py --codes sz300497 --history 60   # 只拉指定
"""
import subprocess, sqlite3, sys, re, os, json, time
from datetime import datetime, timedelta

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "stock_cache.db")
WESTOCK = os.path.expanduser("~/.local/bin/westock")
CODES_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stock_codes.txt")


def load_codes():
    """从 stock_codes.txt 读股票代码列表(纯数字)"""
    codes = []
    with open(CODES_FILE) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            codes.append(line)
    return codes


def to_westock_sym(code):
    """6位数字 -> sh/sz 前缀"""
    if code.startswith(("60", "68", "51", "58", "90", "11")):
        return "sh" + code
    return "sz" + code


def parse_chip_output(text):
    """解析 westock chip 输出的 markdown 表格 -> list of dict"""
    rows = []
    lines = [l for l in text.split("\n") if l.strip()]
    header = None
    for l in lines:
        if l.strip().startswith("|"):
            cells = [c.strip() for c in l.strip().strip("|").split("|")]
            if not header:
                header = cells
                continue
            if set(cells) == {"---"} or all(re.fullmatch(r"-+", c or "-") for c in cells):
                continue
            if len(cells) == len(header):
                d = dict(zip(header, cells))
                rows.append(d)
    return rows


def query_chip(sym, date=None):
    """查单只股票筹码, 返回 list of dict"""
    cmd = [WESTOCK, "chip", sym]
    if date:
        cmd += ["--date", date]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return parse_chip_output(out.stdout)
    except Exception as e:
        return []


def init_table(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chip (
            code TEXT, date TEXT, name TEXT,
            chip_avg_cost REAL, conc_70 REAL, conc_90 REAL,
            profit_rate REAL, close_price REAL,
            fetched_at TEXT,
            PRIMARY KEY (code, date)
        )
    """)
    conn.commit()


def upsert(conn, code, date, name, avg_cost, conc70, conc90, profit, close):
    conn.execute("""
        INSERT OR REPLACE INTO chip
        (code, date, name, chip_avg_cost, conc_70, conc_90, profit_rate, close_price, fetched_at)
        VALUES (?,?,?,?,?,?,?,?,?)
    """, (code, date, name, avg_cost, conc70, conc90, profit, close,
          datetime.now().strftime("%Y-%m-%d %H:%M:%S")))


def f(x):
    """安全转 float"""
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def main():
    conn = sqlite3.connect(DB_PATH)
    init_table(conn)

    args = sys.argv[1:]
    history_days = 0
    only_codes = None
    i = 0
    while i < len(args):
        if args[i] == "--history":
            history_days = int(args[i+1]); i += 2
        elif args[i] == "--codes":
            only_codes = args[i+1].split(","); i += 2
        else:
            i += 1

    codes = only_codes if only_codes else load_codes()
    print(f"股票数: {len(codes)} | 历史天数: {history_days}")

    total_ok = 0
    total_fail = 0
    t0 = time.time()

    for idx, code in enumerate(codes, 1):
        sym = to_westock_sym(code)
        if history_days > 0:
            # 历史区间查询
            end = datetime.now().strftime("%Y-%m-%d")
            start = (datetime.now() - timedelta(days=history_days * 1.6)).strftime("%Y-%m-%d")
            rows = query_chip(sym, date=None)
            # 用区间查询
            cmd = [WESTOCK, "chip", sym, "--start", start, "--end", end]
            try:
                out = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                rows = parse_chip_output(out.stdout)
            except Exception:
                rows = []
        else:
            rows = query_chip(sym)

        if rows:
            for r in rows:
                d = r.get("date", "")
                if not d:
                    continue
                upsert(conn, code, d, r.get("name", ""),
                       f(r.get("chipAvgCost")), f(r.get("chipConcentration70")),
                       f(r.get("chipConcentration90")), f(r.get("chipProfitRate")),
                       f(r.get("closePrice")))
            total_ok += 1
        else:
            total_fail += 1

        if idx % 20 == 0:
            conn.commit()
            el = time.time() - t0
            print(f"  进度 {idx}/{len(codes)} 成功{total_ok} 失败{total_fail} 耗时{el:.0f}s")

    conn.commit()
    n = conn.execute("SELECT COUNT(*), COUNT(DISTINCT code), MIN(date), MAX(date) FROM chip").fetchone()
    print(f"\n✅ 完成: 成功{total_ok}只 失败{total_fail}只")
    print(f"   chip表: {n[0]}行 {n[1]}只 日期{n[2]}~{n[3]}")
    conn.close()


if __name__ == "__main__":
    main()
