#!/usr/bin/env python3
"""
东财push2实时资金流全池拉取脚本 (2026-09-18建立) — 资金流终极通道
数据源: 东方财富 push2 资金流K线接口, 免鉴权, 盘中分钟级实时更新。
实测验证(2026-09-18): 1.push2.eastmoney.com / push2delay 节点可用,
push2主节点当日502(节点轮换故障, 脚本自动多节点容错)。

一次调用同时拿到:
  - 当日实时净额(盘中末行=当日累计, 收盘后=当日最终)
  - 20日逐日历史(求和得5D/20D)
  - 五档分层: 主力/小单/中单/大单/超大单

口径(与fund_flows表对齐):
  main_net_today = klines末行(今日)主力净额
  main_net_5d    = 末5行(含今日)主力求和
  main_net_20d   = 末20行(含今日)主力求和
  jumbo_net      = 末行超大单净额
  inflow_rate    = today / 流通市值×100 (qt.gtimg.cn换算, 对齐westock口径)

用法: python3 fetch_eastmoney_fundflow.py [代码列表文件] (默认 stock_codes.txt 全池)
      代码文件每行一个纯数字代码。输出: 全池入库 + 摘要打印。
"""

import os, sys, json, time, sqlite3, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

SELF_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(SELF_DIR, "output", "stock_cache.db")
CODES_FILE = os.path.join(SELF_DIR, "stock_codes.txt")

# 节点按优先级轮换(2026-09-18实测: 主节点502时1.push2正常)
EM_HOSTS = [
    "https://1.push2.eastmoney.com",
    "https://push2delay.eastmoney.com",
    "https://2.push2.eastmoney.com",
    "https://push2.eastmoney.com",
    "https://push2his.eastmoney.com",  # 历史-only兜底(不含今日)
]
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Referer": "https://quote.eastmoney.com/",
}


def _secid(code):
    """6/9开头→1.(沪) 其余→0.(深/北)"""
    return ("1." if code.startswith(("6", "9")) else "0.") + code


def _get_json(url, timeout=5, retries=1):
    for _ in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            return json.loads(urllib.request.urlopen(req, timeout=timeout).read().decode("utf-8"))
        except Exception:
            time.sleep(0.3)
    return None


def fetch_one(code):
    """单只: push2拿今日盘中实时(盘中该接口仅返回当日1行) + push2his拿20日历史(不含今日), 合并。
    返回(code, rows) rows=[[date,main,small,mid,large,super],...] 日期升序"""
    secid = _secid(code)
    # ① 今日实时: push2 klt=101 (盘中仅返回当日行, 收盘后=当日最终)
    today_url = (f"/api/qt/stock/fflow/kline/get?secid={secid}"
                 f"&klt=101&lmt=1&fields1=f1,f2,f3,f7&fields2=f51,f52,f53,f54,f55,f56")
    today_row = None
    for host in EM_HOSTS[:-1]:  # 排除push2his(不含今日)
        d = _get_json(host + today_url)
        if d and d.get("data") and d["data"].get("klines"):
            p = d["data"]["klines"][-1].split(",")
            if len(p) >= 6:
                today_row = (p[0], float(p[1]), float(p[2]), float(p[3]), float(p[4]), float(p[5]))
            if today_row:
                break
    # ② 20日历史: push2his daykline (截至昨收, 不含今日)
    hist_url = (f"/api/qt/stock/fflow/daykline/get?secid={secid}"
                f"&fields1=f1,f2,f3,f7&fields2=f51,f52,f53,f54,f55,f56&klt=101&lmt=20")
    rows = []
    d = _get_json("https://push2his.eastmoney.com" + hist_url, timeout=8, retries=3)
    if d and d.get("data") and d["data"].get("klines"):
        for line in d["data"]["klines"]:
            p = line.split(",")
            if len(p) >= 6:
                rows.append((p[0], float(p[1]), float(p[2]), float(p[3]), float(p[4]), float(p[5])))
    # ③ 合并(历史在前, 今日在后; push2his若已含今日则不重复)
    if today_row:
        if not rows or rows[-1][0] != today_row[0]:
            rows.append(today_row)
        else:
            rows[-1] = today_row  # 同日以push2实时覆盖
    if not rows:
        return code, None
    return code, rows


def fetch_all_today_hist(codes):
    """全池拉取(线程池), 返回 {code: rows}"""
    out, done = {}, 0
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(fetch_one, c): c for c in codes}
        for fu in as_completed(futs):
            code, rows = fu.result()
            done += 1
            if done % 80 == 0:
                print(f"  ... 已拉取 {done}/{len(codes)}", flush=True)
            if rows:
                out[code] = rows
    return out


def _to_symbol(code):
    if code.startswith(("6", "9", "58")):
        return "sh" + code
    if code.startswith(("8", "4", "92")):
        return "bj" + code
    return "sz" + code


def _circ_mcap_batch(codes):
    """qt.gtimg.cn批量流通市值(元), 返回{code: mcap}"""
    mcaps = {}
    for i in range(0, len(codes), 60):
        batch = codes[i:i + 60]
        syms = ",".join(_to_symbol(c) for c in batch)
        try:
            req = urllib.request.Request(f"https://qt.gtimg.cn/q={syms}", headers=HEADERS)
            data = urllib.request.urlopen(req, timeout=6).read().decode("gbk", errors="replace")
            for line in data.split(";"):
                if '"' not in line:
                    continue
                vals = line.split('"')[1].split("~")
                if len(vals) > 44 and vals[2]:
                    code, v44 = vals[2][-6:], vals[44]
                    if v44:
                        mcaps[code] = float(v44) * 1e8
        except Exception:
            pass
    return mcaps


def main():
    from datetime import datetime
    today_str = datetime.now().strftime("%Y-%m-%d")
    path = sys.argv[1] if len(sys.argv) > 1 else CODES_FILE
    codes = [l.strip() for l in open(path) if l.strip() and not l.startswith("#")]
    print(f"东财push2资金流: 全池{len(codes)}只, 多节点容错+8线程...")

    t0 = time.time()
    data = fetch_all_today_hist(codes)
    print(f"首轮拉取完成: {len(data)}/{len(codes)}只, 耗时{time.time()-t0:.0f}s")

    # 补拉轮: 历史缺失(只有今日1行)的股票低并发重拉 (push2his并发限流会静默丢部分)
    bad = [c for c, rows in data.items() if len(rows) < 2]
    if bad:
        print(f"补拉历史: {len(bad)}只(3线程+重试)...")
        with ThreadPoolExecutor(max_workers=3) as ex:
            for code, rows in ex.map(fetch_one, bad):
                if rows and len(rows) >= 2:
                    data[code] = rows
        still = [c for c in bad if len(data.get(c, [])) < 2]
        print(f"补拉后仍缺历史: {len(still)}只" + (f" {','.join(still[:10])}" if still else ""))

    # 流通市值批量换算inflow_rate
    mcaps = _circ_mcap_batch(list(data.keys()))

    records = {}
    realtime_n = 0
    for code, rows in data.items():
        rows = sorted(rows, key=lambda r: r[0])  # 日期升序
        last = rows[-1]
        main_hist = [r[1] for r in rows]
        rec = {
            "main_net_today": last[1],
            "main_net_5d": sum(main_hist[-5:]),
            "main_net_20d": sum(main_hist[-20:]),
            "jumbo_net": last[5],
            "inflow_rate": 0.0,
            # 修复(2026-09-22): 记录数据自身日期。原实现无论末行是否为今日, 都按 today_str 写库,
            # 会把昨日数据冒充当日数据(5D/20D 也随之错位)。现在按真实日期落库。
            "data_date": last[0],
        }
        if last[0] == today_str:
            realtime_n += 1
        mcap = mcaps.get(code, 0)
        if mcap > 0:
            rec["inflow_rate"] = rec["main_net_today"] / mcap * 100.0
        records[code] = rec

    with sqlite3.connect(DB_PATH) as conn:
        for code, r in records.items():
            # 修复(2026-09-22): 落库日期 = 数据自身日期(非抓取日), fetch_log 仍记抓取日
            data_date = r.get("data_date") or today_str
            conn.execute(
                "INSERT OR REPLACE INTO fund_flows(code,date,main_net_5d,main_net_20d,inflow_rate,jumbo_net,main_net_today,fetched_at) "
                "VALUES(?,?,?,?,?,?,?,datetime('now'))",
                (code, data_date, r["main_net_5d"], r["main_net_20d"],
                 r["inflow_rate"], r["jumbo_net"], r["main_net_today"]),
            )
            conn.execute(
                "INSERT OR REPLACE INTO fetch_log(code,date,data_type,status,updated_at) "
                "VALUES(?,?,?,'ok',datetime('now'))",
                (code, today_str, "fund_flow_eastmoney"),
            )

    print(f"✅ 东财push2资金流入库完成: {len(records)}只 | 含今日实时{realtime_n}只" + (f" | ⚠️{len(records)-realtime_n}只按数据实际日期回填(非今日)" if len(records) > realtime_n else ""))
    rows = sorted(records.items(), key=lambda kv: -kv[1]["main_net_5d"])
    for c, r in rows[:3]:
        print(f"  5日净流入TOP {c}: 今日{r['main_net_today']/1e8:+.2f}亿 5日{r['main_net_5d']/1e8:+.2f}亿 20日{r['main_net_20d']/1e8:+.2f}亿")
    for c, r in rows[-3:]:
        print(f"  5日净流出TOP {c}: 今日{r['main_net_today']/1e8:+.2f}亿 5日{r['main_net_5d']/1e8:+.2f}亿 20日{r['main_net_20d']/1e8:+.2f}亿")
    return len(records)


if __name__ == "__main__":
    main()
