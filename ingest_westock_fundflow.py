#!/usr/bin/env python3
"""
westock(腾讯自选股)实时资金流入库脚本 — 盘中实时数据通道
用途: AI会话通过westock-mcp data_fund_flow批量拉取当日实时资金流存为JSON,
      本脚本解析写入stock_cache.db的fund_flows表, 统一评分引擎随即读到当日实时资金流因子。
      解决: SS分资金面55%权重盘中读昨日数据导致的排名滞后(如涨停当日豆包排名卡壳)。

输入文件: output/fundflow_westock.json
格式(-westock data_fund_flow逐次调用结果的合并, 每个元素为一次调用的data map):
[
  {"sz002222": {"code":"sz002222","data":[{"SecuCode":"sz002222","EndDate":"2026-09-16",
     "MainNetFlow":"228890151","MainNetFlow5D":"-212006230","MainNetFlow20D":"-446310553",
     "JumboNetFlow":"278145186","MainInflowCircRate":"0.71","ClosePrice":"69.13", ...}]}, ...},
  ...  # 多次调用合并成一个数组
]

字段映射(与fund_flows表口径一致):
  MainNetFlow     -> main_net_today  (当日实时主力净流入, 元)
  MainNetFlow5D   -> main_net_5d     (近5日累计, 元)
  MainNetFlow20D  -> main_net_20d    (近20日累计, 元)
  JumboNetFlow    -> jumbo_net       (当日超大单净流入, 元)
  MainInflowCircRate -> inflow_rate  (主力流入占流通市值比, %, westock已算好)
  EndDate         -> date

用法: python3 ingest_westock_fundflow.py [json路径] (默认 output/fundflow_westock.json)
"""

import os, sys, json, sqlite3

SELF_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(SELF_DIR, "output", "stock_cache.db")
DEFAULT_RAW = os.path.join(SELF_DIR, "output", "fundflow_westock.json")


def _f(v, default=0.0):
    try:
        return float(v) if v not in (None, "", "-", "--") else default
    except (ValueError, TypeError):
        return default


def _clean_code(secucode):
    """sz002222/sh600089 -> 002222/600089"""
    return str(secucode)[-6:]


def ingest(raw_path=DEFAULT_RAW):
    with open(raw_path, "r", encoding="utf-8") as f:
        batches = json.load(f)

    records = {}
    for batch in batches:
        if not isinstance(batch, dict):
            continue
        for key, node in batch.items():
            rows = (node or {}).get("data") or []
            for r in rows:
                date = str(r.get("EndDate", ""))[:10]
                code = _clean_code(r.get("SecuCode") or key)
                if not date or not code:
                    continue
                # 同日多次拉取以最新为准(后写覆盖)
                records[(code, date)] = {
                    "main_net_5d": _f(r.get("MainNetFlow5D")),
                    "main_net_20d": _f(r.get("MainNetFlow20D")),
                    "inflow_rate": _f(r.get("MainInflowCircRate")),
                    "jumbo_net": _f(r.get("JumboNetFlow")),
                    "main_net_today": _f(r.get("MainNetFlow")),
                }

    if not records:
        print("无有效数据可入库")
        return 0

    with sqlite3.connect(DB_PATH) as conn:
        for (code, date), r in records.items():
            conn.execute(
                "INSERT OR REPLACE INTO fund_flows(code,date,main_net_5d,main_net_20d,inflow_rate,jumbo_net,main_net_today,fetched_at) "
                "VALUES(?,?,?,?,?,?,?,datetime('now'))",
                (code, date, r["main_net_5d"], r["main_net_20d"],
                 r["inflow_rate"], r["jumbo_net"], r["main_net_today"]),
            )
            conn.execute(
                "INSERT OR REPLACE INTO fetch_log(code,date,data_type,status,updated_at) "
                "VALUES(?,?,?,'ok',datetime('now'))",
                (code, date, "fund_flow_westock"),
            )

    n = len(records)
    dates = sorted({d for _, d in records})
    print(f"✅ westock实时资金流入库完成: {n}只, 日期{','.join(dates)}")
    # 打印主力净流入/流出TOP3(当日)
    today_records = [(c, r) for (c, d), r in records.items() if d == dates[-1]]
    today_records.sort(key=lambda x: -x[1]["main_net_today"])
    for c, r in today_records[:3]:
        print(f"  净流入TOP {c}: 今日{r['main_net_today']/1e8:+.2f}亿 5日{r['main_net_5d']/1e8:+.2f}亿")
    for c, r in today_records[-3:]:
        print(f"  净流出TOP {c}: 今日{r['main_net_today']/1e8:+.2f}亿 5日{r['main_net_5d']/1e8:+.2f}亿")
    return n


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_RAW
    if not os.path.exists(path):
        print(f"输入文件不存在: {path}")
        sys.exit(1)
    ingest(path)
