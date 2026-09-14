#!/usr/bin/env python3
"""
资金流数据入库脚本 — 配合通达信MCP定时任务使用
用途: 自动化任务中AI通过tdx-connector拉取资金流数据存为JSON, 本脚本解析并写入stock_cache.db的fund_flows表,
      统一评分引擎随后即可读取资金流因子(main_flow_5d/main_flow_20d/inflow_rate)。

输入文件: output/fundflow_raw.json
格式:
{
  "zjlx": {  # tdx_api_data entry=TdxSharePCCW.tdxf10_gg_jyds fixedTag=zjlx 的逐股结果
    "301183": [ {"日期":"2026-09-11","主力净额金额(元)":239590976,"主力净额占比(%)":14.34,
                  "超大单净买入金额(元)":126378400,"大单净买入金额(元)":113212576,"收盘价":246.99}, ... ],
    ...
  },
  "screener": [ # tdx_screener "主力净流入排名/净流出排名" 的当日快照(与股票池取交集后)
    {"code":"600172","main_net_today":1245121152}, ...
  ]
}

口径(与原东财fflow实现一致):
  main_net_5d   = 近5个交易日主力净额合计(元)
  main_net_20d  = 近20个交易日主力净额合计(元)
  main_net_today= 最新交易日主力净额(元)
  jumbo_net     = 最新交易日超大单净买入(元)
  inflow_rate   = 最新交易日主力净额 / 流通市值 × 100 (%)

用法: python3 ingest_fund_flows.py [raw.json路径] (默认 output/fundflow_raw.json)
"""

import os, sys, json, sqlite3
from datetime import datetime

SELF_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(SELF_DIR, "output", "stock_cache.db")
DEFAULT_RAW = os.path.join(SELF_DIR, "output", "fundflow_raw.json")


def _f(v, default=0.0):
    try:
        return float(v) if v not in (None, "", "-", "--") else default
    except (ValueError, TypeError):
        return default


def _d(s, default=None):
    try:
        return datetime.strptime(str(s)[:10], "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return default


def ingest(raw_path=DEFAULT_RAW):
    with open(raw_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    today = datetime.now().strftime("%Y-%m-%d")
    zjlx = raw.get("zjlx", {})
    screener = raw.get("screener", [])

    # 流通市值(用于inflow_rate) — 从今日extra_info取
    float_mcap = {}
    if os.path.exists(DB_PATH):
        with sqlite3.connect(DB_PATH) as conn:
            for code, fm in conn.execute(
                "SELECT code, float_mcap FROM extra_info WHERE date=?", (today,)
            ):
                if fm:
                    float_mcap[code] = fm

    records = {}

    # 1) zjlx逐股历史 → 5日/20日累计
    # 字段兼容两种格式: preset中文键名 或 原始编码(rq/N001/N003)
    #   主力净额金额(元)=N001, 超大单净买入金额(元)=N003
    def _get_main(r):
        return _f(r.get("主力净额金额(元)", r.get("N001")))

    def _get_jumbo(r):
        return _f(r.get("超大单净买入金额(元)", r.get("N003")))

    def _get_date(r):
        return r.get("日期", r.get("rq"))

    for code, rows in zjlx.items():
        rows = [r for r in rows if _get_date(r) and _d(_get_date(r))]
        if not rows:
            continue
        rows.sort(key=lambda r: str(_get_date(r)))
        main_series = [_get_main(r) for r in rows]
        latest = rows[-1]
        main_today = main_series[-1]
        main_5d = sum(main_series[-5:])
        main_20d = sum(main_series[-20:])
        jumbo = _get_jumbo(latest)
        fm = float_mcap.get(code, 0)
        inflow_rate = (main_today / fm * 100) if fm > 0 else 0.0
        records[code] = {
            "main_net_5d": main_5d,
            "main_net_20d": main_20d,
            "inflow_rate": inflow_rate,
            "jumbo_net": jumbo,
            "main_net_today": main_today,
            "_date": str(_get_date(latest))[:10],
        }

    # 2) screener当日快照 → 仅补充zjlx未覆盖的股票的main_net_today
    for item in screener:
        code = str(item.get("code", ""))
        mn = _f(item.get("main_net_today"))
        if not code or mn == 0:
            continue
        if code in records:
            continue  # zjlx历史数据优先
        records[code] = {
            "main_net_5d": 0,
            "main_net_20d": 0,
            "inflow_rate": 0,
            "jumbo_net": 0,
            "main_net_today": mn,
            "_date": today,
        }

    # 3) 写DB (zjlx数据用其最新交易日日期, screener快照用今日)
    if not records:
        print("无有效数据可入库")
        return 0

    n_full = sum(1 for r in records.values() if r["main_net_20d"] != 0 or r["main_net_5d"] != 0)
    n_snap = len(records) - n_full
    with sqlite3.connect(DB_PATH) as conn:
        for code, r in records.items():
            conn.execute(
                "INSERT OR REPLACE INTO fund_flows(code,date,main_net_5d,main_net_20d,inflow_rate,jumbo_net,main_net_today,fetched_at) "
                "VALUES(?,?,?,?,?,?,?,datetime('now'))",
                (code, r["_date"], r["main_net_5d"], r["main_net_20d"],
                 r["inflow_rate"], r["jumbo_net"], r["main_net_today"]),
            )
            conn.execute(
                "INSERT OR REPLACE INTO fetch_log(code,date,data_type,status,updated_at) "
                "VALUES(?,?,?,'ok',datetime('now'))",
                (code, r["_date"], "fund_flow_tdx"),
            )

    print(f"✅ 资金流入库完成: 共{len(records)}只 (完整5/20日数据{n_full}只 + 当日快照{n_snap}只)")
    for code, r in list(records.items())[:8]:
        print(f"  {code}: 5日{r['main_net_5d']/1e8:+.2f}亿 20日{r['main_net_20d']/1e8:+.2f}亿 "
              f"今日{r['main_net_today']/1e8:+.2f}亿 ({r['_date']})")
    if len(records) > 8:
        print(f"  ... 其余{len(records)-8}只")
    return len(records)


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_RAW
    if not os.path.exists(path):
        print(f"输入文件不存在: {path}")
        sys.exit(1)
    ingest(path)
