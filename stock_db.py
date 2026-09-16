#!/usr/bin/env python3
"""
股票数据本地 SQLite 缓存模块。
统一数据层：K线、实时行情、资金流、公告事件
"""

import os, json, sqlite3, urllib.request, subprocess, re
from datetime import datetime, timedelta
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

DB_PATH = None

def _get_db_path():
    global DB_PATH
    if DB_PATH is None:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
        os.makedirs(output_dir, exist_ok=True)
        DB_PATH = os.path.join(output_dir, "stock_cache.db")
    return DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS klines (
    code TEXT NOT NULL, date TEXT NOT NULL,
    open REAL, high REAL, low REAL, close REAL, volume REAL,
    fetched_at TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (code, date)
);
CREATE TABLE IF NOT EXISTS extra_info (
    code TEXT NOT NULL, date TEXT NOT NULL,
    name TEXT, price REAL, change_pct REAL,
    pe_ttm REAL, pb REAL, mcap REAL, turnover REAL, vol_ratio REAL,
    float_mcap REAL, zt_price REAL, dt_price REAL,
    fetched_at TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (code, date)
);
CREATE TABLE IF NOT EXISTS fund_flows (
    code TEXT NOT NULL, date TEXT NOT NULL,
    main_net_5d REAL, main_net_20d REAL,
    inflow_rate REAL, jumbo_net REAL, main_net_today REAL,
    fetched_at TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (code, date)
);
CREATE TABLE IF NOT EXISTS events (
    code TEXT NOT NULL, date TEXT NOT NULL,
    title TEXT NOT NULL, event_type TEXT, base_score REAL,
    fetched_at TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (code, date, title)
);
CREATE TABLE IF NOT EXISTS fetch_log (
    code TEXT NOT NULL, date TEXT NOT NULL,
    data_type TEXT NOT NULL,
    status TEXT DEFAULT 'failed', retry_count INTEGER DEFAULT 0,
    error_msg TEXT,
    updated_at TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (code, date, data_type)
);
CREATE INDEX IF NOT EXISTS idx_klines_code ON klines(code);
CREATE INDEX IF NOT EXISTS idx_klines_date ON klines(date);
CREATE INDEX IF NOT EXISTS idx_events_code ON events(code);
CREATE INDEX IF NOT EXISTS idx_events_date ON events(date);
CREATE INDEX IF NOT EXISTS idx_fetch_log_status ON fetch_log(status);
"""

class StockDB:
    def __init__(self, db_path=None):
        self.db_path = db_path or _get_db_path()
        self._init_db()
        self._node_script = None  # 已改用纯Python实现，不再依赖Node.js
        self._workers = 20

    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _init_db(self):
        with self._connect() as conn:
            conn.executescript(SCHEMA)
            # 迁移: 旧表补充新列(幂等)
            cols = {r[1] for r in conn.execute("PRAGMA table_info(extra_info)")}
            for col, ddl in [("float_mcap","ALTER TABLE extra_info ADD COLUMN float_mcap REAL"),
                             ("zt_price","ALTER TABLE extra_info ADD COLUMN zt_price REAL"),
                             ("dt_price","ALTER TABLE extra_info ADD COLUMN dt_price REAL")]:
                if col not in cols:
                    conn.execute(ddl)

    @staticmethod
    def _to_symbol(code):
        if code.startswith(("8", "4", "920")):
            return f"bj{code}"
        return f"sh{code}" if code.startswith(("6", "9", "58")) else f"sz{code}"

    def get_klines(self, codes, days=130, refresh_today=True):
        """refresh_today=True: 当日K线即使已缓存也强制重抓(修复盘中快照滞留问题,
        确保收盘后运行能拿到最终收盘价; 抓取失败时回退用缓存值)
        (2026-09-16 本地合并: 从本地v3.3移植到远端v3.5基线, 适配盘中三时段自动化)"""
        today = datetime.now().strftime("%Y-%m-%d")
        all_klines = {}
        missing = []
        now = datetime.now()
        wd = now.weekday()
        if wd == 5: latest_td = (now - timedelta(days=1)).strftime("%Y-%m-%d")
        elif wd == 6: latest_td = (now - timedelta(days=2)).strftime("%Y-%m-%d")
        else: latest_td = today

        with self._connect() as conn:
            for code in codes:
                rows = conn.execute(
                    "SELECT date, open, high, low, close, volume FROM klines WHERE code=? ORDER BY date",
                    (code,)
                ).fetchall()
                if len(rows) >= days:
                    last_date = rows[-1][0]
                    parsed = [{"date":r[0],"open":r[1],"high":r[2],"low":r[3],"close":r[4],"volume":r[5]} for r in rows]
                    if last_date >= latest_td:
                        all_klines[code] = parsed[-days:]
                        if refresh_today:
                            # 当日K线可能为盘中快照, 强制重抓以获取最新/收盘价
                            missing.append(code)
                    else:
                        all_klines[code] = parsed
                        missing.append(code)
                elif len(rows) > 0:
                    all_klines[code] = [{"date":r[0],"open":r[1],"high":r[2],"low":r[3],"close":r[4],"volume":r[5]} for r in rows]
                    missing.append(code)
                else:
                    missing.append(code)

        if missing:
            self._fetch_klines_batch(missing, days, today, all_klines)

        for code in list(all_klines.keys()):
            if len(all_klines[code]) > days:
                all_klines[code] = all_klines[code][-days:]
        return all_klines

    def _fetch_klines_batch(self, codes, days, today, all_klines):
        print(f"  [DB] 增量抓取 {len(codes)} 只K线...")
        def fetch_one(code):
            sym = self._to_symbol(code)
            url = f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={sym},day,,,{days},qfq"
            req = urllib.request.Request(url, headers={"User-Agent":"Mozilla/5.0"})
            try:
                resp = urllib.request.urlopen(req, timeout=10)
                data = json.loads(resp.read().decode("utf-8"))
                klines = data.get("data",{}).get(sym,{}).get("qfqday",[]) or data.get("data",{}).get(sym,{}).get("day",[])
                return code, klines, None
            except Exception as e:
                return code, None, str(e)[:200]

        completed = 0
        with ThreadPoolExecutor(max_workers=self._workers) as ex:
            futures = {ex.submit(fetch_one, c): c for c in codes}
            for f in as_completed(futures):
                code, kls, err = f.result()
                completed += 1
                if completed % 30 == 0: print(f"  K线: {completed}/{len(codes)}")
                if kls:
                    parsed = []
                    for k in kls:
                        try:
                            parsed.append({"date":k[0],"open":float(k[1]),"close":float(k[2]),"high":float(k[3]),"low":float(k[4]),"volume":float(k[5]) if len(k)>5 else 0})
                        except (IndexError, ValueError, TypeError):
                            continue
                    if parsed:
                        existing = {r["date"]:r for r in all_klines.get(code,[])}
                        for r in parsed: existing[r["date"]] = r
                        all_klines[code] = sorted(existing.values(), key=lambda x:x["date"])
                        self._save_klines(code, parsed)
                        self._log_fetch(code, today, "klines", "ok")
                else:
                    self._log_fetch(code, today, "klines", "failed", err)

    def _save_klines(self, code, klines):
        with self._connect() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO klines(code,date,open,high,low,close,volume) VALUES(?,?,?,?,?,?,?)",
                [(code,k["date"],k["open"],k["high"],k["low"],k["close"],k["volume"]) for k in klines])

    def get_extra_info(self, codes, force_refresh=False):
        today = datetime.now().strftime("%Y-%m-%d")
        result = {}
        need_fetch = []
        if force_refresh:
            need_fetch = list(codes)
        else:
            with self._connect() as conn:
                for code in codes:
                    row = conn.execute(
                        "SELECT name,price,change_pct,pe_ttm,pb,mcap,turnover,vol_ratio,float_mcap,zt_price,dt_price FROM extra_info WHERE code=? AND date=?",
                        (code, today)).fetchone()
                    if row:
                        result[code] = {"name":row[0],"price":row[1],"change_pct":row[2],"pe_ttm":row[3],"pb":row[4],"mcap":row[5],"turnover":row[6],"vol_ratio":row[7],"float_mcap":row[8] or 0,"zt_price":row[9] or 0,"dt_price":row[10] or 0}
                    else:
                        need_fetch.append(code)
        if need_fetch:
            fetched = self._fetch_extra_info_batch(need_fetch, today)
            result.update(fetched)
        return result

    def _fetch_extra_info_batch(self, codes, today):
        result = {}
        prefixed = [self._to_symbol(c) for c in codes]
        for i in range(0, len(prefixed), 60):
            batch = prefixed[i:i+60]
            url = "https://qt.gtimg.cn/q=" + ",".join(batch)
            req = urllib.request.Request(url, headers={"User-Agent":"Mozilla/5.0"})
            try:
                resp = urllib.request.urlopen(req, timeout=10)
                for line in resp.read().decode("gbk").strip().split(";"):
                    if "=" not in line or '"' not in line: continue
                    vals = line.split('"')[1].split("~")
                    if len(vals) < 55: continue
                    code = line.split("=")[0].split("_")[-1][2:]
                    # 字段口径(2026-09-14验证): vals[44]=流通市值(亿) vals[45]=总市值(亿) vals[47]=涨停价 vals[48]=跌停价
                    info = {"name":vals[1],"price":float(vals[3]) if vals[3] else 0,
                            "change_pct":float(vals[32]) if vals[32] else 0,
                            "pe_ttm":float(vals[39]) if vals[39] else 0,
                            "pb":float(vals[46]) if vals[46] else 0,
                            "mcap":float(vals[45])*1e8 if vals[45] else 0,        # 总市值(元) — 修复: 原误取字段44流通市值
                            "float_mcap":float(vals[44])*1e8 if vals[44] else 0,   # 流通市值(元)
                            "zt_price":float(vals[47]) if vals[47] else 0,         # 涨停价(精确)
                            "dt_price":float(vals[48]) if vals[48] else 0,         # 跌停价(精确)
                            "turnover":float(vals[38]) if vals[38] else 0,
                            "vol_ratio":float(vals[49]) if vals[49] else 0}
                    result[code] = info
                    self._save_extra(code, today, info)
                    self._log_fetch(code, today, "extra", "ok")
            except Exception as e:
                for c in batch:
                    # 修复: strip("shsz")会误剥bj前缀, 统一去市场前缀取后6位数字代码
                    self._log_fetch(re.sub(r"^(sh|sz|bj)", "", c), today, "extra", "failed", str(e)[:200])
        return result

    def _save_extra(self, code, date, info):
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO extra_info(code,date,name,price,change_pct,pe_ttm,pb,mcap,turnover,vol_ratio,float_mcap,zt_price,dt_price) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (code,date,info["name"],info["price"],info["change_pct"],info["pe_ttm"],info["pb"],info["mcap"],info["turnover"],info["vol_ratio"],info.get("float_mcap",0),info.get("zt_price",0),info.get("dt_price",0)))

    def get_fund_flows(self, codes):
        today = datetime.now().strftime("%Y-%m-%d")
        result = {}
        need_fetch = []
        with self._connect() as conn:
            for code in codes:
                # 修复(2026-09-14): 资金流数据源改为通达信MCP定时任务写入(收盘后16:35),
                # 数据可能滞后当日(盘前读到的是昨日完整数据)。取≤today的最近记录而非严格等于today,
                # 避免tdx完整5/20日数据(写在上一个交易日)被错过。
                row = conn.execute(
                    "SELECT main_net_5d,main_net_20d,inflow_rate,jumbo_net,main_net_today FROM fund_flows "
                    "WHERE code=? AND date<=? ORDER BY date DESC LIMIT 1",
                    (code, today)).fetchone()
                if row: result[code] = dict(zip(["main_net_5d","main_net_20d","inflow_rate","jumbo_net","main_net_today"], row))
                else: need_fetch.append(code)
        if need_fetch:
            fetched = self._fetch_fund_flows_batch(need_fetch, today)
            result.update(fetched)
        return result

    def _fetch_fund_flows_batch(self, codes, today):
        """纯Python实现资金流数据获取，不依赖Node.js"""
        result = {}
        for i in range(0, len(codes), 30):
            batch = codes[i:i+30]
            for code in batch:
                try:
                    sym = self._to_symbol(code)
                    # 东方财富资金流API
                    # 0=主力净流入, 1=小单, 2=中单, 3=大单
                    mkt = f"{sym[:2]}"  # sh/sz/bj
                    secid = self._get_eastmoney_secid(code)
                    if not secid: continue
                    url = f"https://push2.eastmoney.com/api/qt/stock/fflow/kline/get?secid={secid}&lmt=0&klt=101&fields1=f1,f2,f3,f7&fields2=f51,f52,f53,f54,f55,f56,f57&ut=b2884a393a59ad64002292a3e90d46a5"
                    req = urllib.request.Request(url, headers={"User-Agent":"Mozilla/5.0","Referer":"https://data.eastmoney.com"})
                    resp = urllib.request.urlopen(req, timeout=10)
                    data = json.loads(resp.read().decode("utf-8"))
                    klines = data.get("data",{}).get("klines",[])
                    if not klines: continue
                    # 最近5天和20天主力净流入
                    recent = klines[-20:] if len(klines) >= 20 else klines
                    main_flows = []
                    for line in recent:
                        parts = line.split(",")
                        if len(parts) >= 6:
                            main_flows.append(float(parts[1]))  # 主力净流入(元)
                    main_net_5d = sum(main_flows[-5:]) if len(main_flows) >= 5 else sum(main_flows)
                    main_net_20d = sum(main_flows)
                    main_net_today = main_flows[-1] if main_flows else 0
                    # 估算主力流入占比 = 今日主力净流入 / 流通市值
                    today_info = {}
                    try:
                        ex_url = f"https://qt.gtimg.cn/q={self._to_symbol(code)}"
                        ex_req = urllib.request.Request(ex_url, headers={"User-Agent":"Mozilla/5.0"})
                        ex_resp = urllib.request.urlopen(ex_req, timeout=5)
                        ex_data = ex_resp.read().decode("gbk", errors="replace")
                        ex_vals = ex_data.split('"')[1].split("~") if '"' in ex_data else []
                        if len(ex_vals) > 44:
                            circ_mcap = float(ex_vals[44]) * 1e8 if ex_vals[44] else 0  # 流通市值(亿→元)
                            inflow_rate = (main_net_today / circ_mcap * 100) if circ_mcap > 0 else 0
                        else:
                            inflow_rate = 0
                    except Exception:
                        inflow_rate = 0
                    info = {
                        "main_net_5d": main_net_5d,
                        "main_net_20d": main_net_20d,
                        "inflow_rate": inflow_rate,
                        "jumbo_net": 0,
                        "main_net_today": main_net_today,
                    }
                    result[code] = info
                    self._save_fund(code, today, info)
                    self._log_fetch(code, today, "fund_flow", "ok")
                except Exception as e:
                    self._log_fetch(code, today, "fund_flow", "failed", str(e)[:200])
        return result

    @staticmethod
    def _get_eastmoney_secid(code):
        """获取东方财富secid格式: 1.600000 / 0.000001 / 0.300001"""
        if code.startswith(("6", "9", "58")):
            return f"1.{code}"
        elif code.startswith(("8", "4", "920")):
            return f"0.{code}"  # 北交所
        else:
            return f"0.{code}"

    def _save_fund(self, code, date, info):
        with self._connect() as conn:
            conn.execute("INSERT OR REPLACE INTO fund_flows(code,date,main_net_5d,main_net_20d,inflow_rate,jumbo_net,main_net_today) VALUES(?,?,?,?,?,?,?)",
                (code,date,info["main_net_5d"],info["main_net_20d"],info["inflow_rate"],info["jumbo_net"],info["main_net_today"]))

    def _log_fetch(self, code, date, data_type, status, error_msg=None):
        with self._connect() as conn:
            conn.execute("INSERT OR REPLACE INTO fetch_log(code,date,data_type,status,error_msg,updated_at) VALUES(?,?,?,?,?,datetime('now'))",
                (code,date,data_type,status,error_msg))

    def check_data_freshness(self, codes):
        now = datetime.now()
        wd = now.weekday()
        if wd == 5: expected_td = (now - timedelta(days=1)).strftime("%Y-%m-%d")
        elif wd == 6: expected_td = (now - timedelta(days=2)).strftime("%Y-%m-%d")
        else: expected_td = now.strftime("%Y-%m-%d")
        with self._connect() as conn:
            kline_latest = conn.execute("SELECT MAX(date) FROM klines").fetchone()[0] or "-"
            extra_latest = conn.execute("SELECT MAX(date) FROM extra_info").fetchone()[0] or "-"
            rows = conn.execute("SELECT date, COUNT(DISTINCT code) FROM klines GROUP BY date ORDER BY date DESC LIMIT 5").fetchall()
            kline_counts = {r[0]:r[1] for r in rows}
            missing_klines = [c for c in codes if conn.execute("SELECT 1 FROM klines WHERE code=? AND date=? LIMIT 1",(c,kline_latest)).fetchone() is None]
            missing_extra = [c for c in codes if conn.execute("SELECT 1 FROM extra_info WHERE code=? AND date=? LIMIT 1",(c,extra_latest)).fetchone() is None]
        fresh = (kline_latest >= expected_td) and len(missing_klines) == 0 and len(missing_extra) == 0
        return {"kline_latest":kline_latest,"extra_latest":extra_latest,"expected_td":expected_td,
                "missing_klines":missing_klines,"missing_extra":missing_extra,
                "kline_counts":kline_counts,"total_codes":len(codes),"fresh":fresh}

    def stats(self):
        with self._connect() as conn:
            kc = conn.execute("SELECT COUNT(*) FROM klines").fetchone()[0]
            ec = conn.execute("SELECT COUNT(*) FROM extra_info").fetchone()[0]
            stocks = conn.execute("SELECT COUNT(DISTINCT code) FROM klines").fetchone()[0]
            last_kline = conn.execute("SELECT MAX(date) FROM klines").fetchone()[0] or "-"
        db_size = os.path.getsize(self.db_path)/(1024*1024)
        print(f"\n  StockDB: K线 {kc}条({stocks}只, 最新{last_kline}) | DB {db_size:.1f}MB")
