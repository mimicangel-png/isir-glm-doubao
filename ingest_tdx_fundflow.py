#!/usr/bin/env python3
"""
通达信(tdx)资金流入库脚本 — 资金流优先通道 (2026-09-18建立)
用途: AI会话通过 tdx-connector MCP 工具 tdx_api_data 逐只拉取
      entry="TdxSharePCCW.tdxf10_gg_jyds" fixedTag="zjlx" 的20日资金流表,
      转录为紧凑行格式存 output/fundflow_tdx.txt, 本脚本解析写入库。
      tdx口径特点: 逐日完整收盘数据(主力净额/超大单/占比/收盘价),
      盘中调用返回截至最近收盘; 收盘后调用即含当日完整数据。

输入文件: output/fundflow_tdx.txt (默认)
格式T1(紧凑行, 每行一只):
  code|首行日期|最新日超大单净买入|d1,d2,...,dN
  - code: 纯数字或带sz/sh前缀(取后6位)
  - 首行日期: tdx返回capital_flow表第一行的日期(YYYY-MM-DD), 即d1对应的交易日
  - 最新日超大单净买入: d1当日的超大单净买入金额(元), 无数据写0
  - d1..dN: 逐日主力净额(元), 最新在前, 逗号分隔(尽量20个)
  - #开头行与空行忽略
  例: 301511|2026-09-17|-174220864|-400808256,-69647488,201317760,...

脚本计算(与fund_flows表口径一致):
  【纯tdx】tdx首行日期=今日(收盘后) 或 未提供westock实时:
    main_net_today = d1;  main_net_5d = sum(d1..d5);  main_net_20d = sum(d1..d20)
    jumbo_net = 最新日超大单;  inflow_rate = 今日净额/流通市值×100(腾讯行情换算)
  【混合口径】--merge-westock 且 tdx首行日期<今日(盘中常态):
    main_net_today = westock当日实时MainNetFlow
    main_net_5d    = sum(d1..d4) + 当日实时   (tdx前4收盘日 + 今日实时)
    main_net_20d   = sum(d1..d19) + 当日实时  (tdx前19收盘日 + 今日实时)
    jumbo_net / inflow_rate = westock口径
    → 当日实时用westock、5D/20D用tdx逐日收盘拼合，两者互补。
    tdx首行日期=今日的标的自动走纯tdx(收盘完整口径, 不与westock重复计入)。
  date = 合并时为今日; 纯tdx时为首行日期(昨日记录REPLACE与tdx定时任务同日覆盖无冲突)

用法: python3 ingest_tdx_fundflow.py [tdx.txt路径] [--merge-westock [westock.txt路径]]
      --merge-westock 不带路径时默认 output/fundflow_westock.txt (紧凑行格式)
"""

import os, sys, sqlite3, urllib.request

SELF_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(SELF_DIR, "output", "stock_cache.db")
DEFAULT_RAW = os.path.join(SELF_DIR, "output", "fundflow_tdx.txt")


def _f(v, default=0.0):
    try:
        return float(v) if v not in (None, "", "-", "--") else default
    except (ValueError, TypeError):
        return default


def _to_symbol(code):
    if code.startswith(("6", "9", "58")):
        return "sh" + code
    if code.startswith(("8", "4", "92")):
        return "bj" + code
    return "sz" + code


def _circ_mcap_yuan(code):
    """qt.gtimg.cn 实时行情取流通市值(亿)→元, 失败返回0"""
    try:
        sym = _to_symbol(code)
        req = urllib.request.Request(f"https://qt.gtimg.cn/q={sym}",
                                     headers={"User-Agent": "Mozilla/5.0"})
        data = urllib.request.urlopen(req, timeout=5).read().decode("gbk", errors="replace")
        vals = data.split('"')[1].split("~") if '"' in data else []
        if len(vals) > 44 and vals[44]:
            return float(vals[44]) * 1e8
    except Exception:
        pass
    return 0.0


def _load_westock_today(path):
    """解析westock紧凑行(当日实时): code|MainNetFlow|5D|20D|JumboNetFlow|MainInflowCircRate"""
    out = {}
    if not path or not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) < 6:
                continue
            out[parts[0][-6:]] = {
                "today": _f(parts[1]),
                "jumbo": _f(parts[4]),
                "rate": _f(parts[5]),
            }
    return out


def ingest(raw_path=DEFAULT_RAW, westock_path=None):
    from datetime import datetime
    today_str = datetime.now().strftime("%Y-%m-%d")
    records = {}

    with open(raw_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) < 4:
                continue
            code = parts[0][-6:]
            date = parts[1][:10]
            jumbo = _f(parts[2])
            days = [_f(x) for x in parts[3].split(",") if x.strip() != ""]
            if not code or not date or not days:
                continue
            records[(code, date)] = {"days": days, "jumbo": jumbo}

    if not records:
        print("无有效数据可入库")
        return 0

    # westock当日实时(可选合并源)
    ws_today = _load_westock_today(westock_path)

    out = {}
    merged = 0
    pure_tdx_keys = set()
    for (code, date), r in records.items():
        days = r["days"]
        w = ws_today.get(code)
        if w is not None and date != today_str:
            # 混合口径: tdx截至昨收, 当日实时来自westock
            t = w["today"]
            out[(code, today_str)] = {
                "main_net_today": t,
                "main_net_5d": sum(days[:4]) + t,
                "main_net_20d": sum(days[:19]) + t,
                "jumbo_net": w["jumbo"],
                "inflow_rate": w["rate"],
            }
            merged += 1
        else:
            # 纯tdx: 已含今日(收盘后) 或 无westock实时
            out[(code, date)] = {
                "main_net_today": days[0],
                "main_net_5d": sum(days[:5]),
                "main_net_20d": sum(days[:20]),
                "jumbo_net": r["jumbo"],
                "inflow_rate": 0.0,  # 稍后统一换算
            }
            pure_tdx_keys.add((code, date))

    # 纯tdx记录的inflow_rate: 主力净额/流通市值×100 (对齐westock口径)
    for key in pure_tdx_keys:
        code = key[0]
        mcap = _circ_mcap_yuan(code)
        if mcap > 0:
            out[key]["inflow_rate"] = out[key]["main_net_today"] / mcap * 100.0

    with sqlite3.connect(DB_PATH) as conn:
        for (code, date), r in out.items():
            conn.execute(
                "INSERT OR REPLACE INTO fund_flows(code,date,main_net_5d,main_net_20d,inflow_rate,jumbo_net,main_net_today,fetched_at) "
                "VALUES(?,?,?,?,?,?,?,datetime('now'))",
                (code, date, r["main_net_5d"], r["main_net_20d"],
                 r["inflow_rate"], r["jumbo_net"], r["main_net_today"]),
            )
            conn.execute(
                "INSERT OR REPLACE INTO fetch_log(code,date,data_type,status,updated_at) "
                "VALUES(?,?,?,'ok',datetime('now'))",
                (code, date, "fund_flow_tdx"),
            )

    n = len(out)
    dates = sorted({d for _, d in out})
    mix = f", 混合实时{merged}只(tdx历史+westock当日)" if merged else ""
    lag = "" if dates[-1] == today_str else f" (⚠️ 数据截至{dates[-1]}, 非今日实时)"
    print(f"✅ tdx资金流入库完成: {n}只, 日期{','.join(dates)}{mix}{lag}")
    rows = sorted(out.items(), key=lambda kv: -kv[1]["main_net_5d"])
    for (c, d), r in rows[:3]:
        print(f"  5日净流入TOP {c}: 5日{r['main_net_5d']/1e8:+.2f}亿 20日{r['main_net_20d']/1e8:+.2f}亿")
    for (c, d), r in rows[-3:]:
        print(f"  5日净流出TOP {c}: 5日{r['main_net_5d']/1e8:+.2f}亿 20日{r['main_net_20d']/1e8:+.2f}亿")
    return n


if __name__ == "__main__":
    args = sys.argv[1:]
    westock_path = None
    if "--merge-westock" in args:
        i = args.index("--merge-westock")
        rest = args[i + 1:]
        if rest and not rest[0].startswith("-"):
            westock_path = rest[0]
            del args[i:i + 2]
        else:
            westock_path = os.path.join(SELF_DIR, "output", "fundflow_westock.txt")
            del args[i:i + 1]
    candidates = args[:1] if args else [DEFAULT_RAW]
    path = next((p for p in candidates if os.path.exists(p)), None)
    if path is None:
        print(f"输入文件不存在: {candidates[0]}")
        sys.exit(1)
    ingest(path, westock_path)
