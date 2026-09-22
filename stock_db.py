#!/usr/bin/env python3
"""
股票数据本地 SQLite 缓存模块。
统一数据层：K线、实时行情、资金流、公告事件
"""

import os, json, sqlite3, urllib.request, subprocess, re, time
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

# ===== 增量K线抓取参数 (2026-09-22新增) =====
INC_BARS_MAX = 45        # 增量模式单次最多抓取根数(≈1.5个月交易日)
INC_MIN_BARS = 5         # 增量模式最少抓取根数(覆盖盘中当日快照刷新)
INC_MAX_GAP_DAYS = 30    # 日历天缺口上限(≈21个交易日), 超过则走全量重抓
INC_PRICE_TOL = 0.005    # 重叠日收盘价容差(0.5%): 超差视为前复权因子变化 → 全量重抓

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
        """refresh_today=True: 当日K线即使已缓存也强制刷新(修复盘中快照滞留问题,
        确保收盘后运行能拿到最终收盘价; 抓取失败时回退用缓存值)
        (2026-09-16 本地合并: 从本地v3.3移植到远端v3.5基线, 适配盘中三时段自动化)

        2026-09-22 增量改造: 缓存命中时只回补最近 INC_MIN_BARS~INC_BARS_MAX 根
        (按缺口日历天数动态计算), 不再每轮全量重抓 days 根(512只×300根 → 约1/10流量)。
        增量数据合并前做"重叠历史日价格一致性校验"(见 _overlap_ok): 不一致说明前复权
        因子已变化(除权除息)或数据源口径不同, 该只自动降级为全量重抓, 保证序列连续。"""
        today = datetime.now().strftime("%Y-%m-%d")
        all_klines = {}
        inc_codes = []    # [(code, pull_bars)] 只回补最近若干根
        full_codes = []   # 历史不足 / 缺口过大 → 全量抓取
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
                    all_klines[code] = parsed[-days:] if last_date >= latest_td else parsed
                    if last_date >= latest_td and not refresh_today:
                        continue          # 不要求刷新当日快照 → 缓存即最终版, 无需抓取
                    gap_days = 0
                    if last_date < latest_td:
                        try:
                            gap_days = (datetime.strptime(latest_td, "%Y-%m-%d")
                                        - datetime.strptime(last_date, "%Y-%m-%d")).days
                        except ValueError:
                            gap_days = INC_MAX_GAP_DAYS + 1
                    if gap_days <= INC_MAX_GAP_DAYS:
                        # 增量回补: 日历天 → 交易日约 ×5/7, 再留 INC_MIN_BARS 缓冲, 夹在范围内
                        need = int(gap_days * 5 / 7) + INC_MIN_BARS
                        inc_codes.append((code, min(INC_BARS_MAX, max(INC_MIN_BARS, need))))
                    else:
                        full_codes.append(code)
                elif len(rows) > 0:
                    all_klines[code] = [{"date":r[0],"open":r[1],"high":r[2],"low":r[3],"close":r[4],"volume":r[5]} for r in rows]
                    full_codes.append(code)
                else:
                    full_codes.append(code)

        if full_codes:
            self._fetch_klines_batch(full_codes, days, today, all_klines, incremental=False)
        if inc_codes:
            self._fetch_klines_batch([c for c, _ in inc_codes], days, today, all_klines,
                                     incremental=True, pull_map=dict(inc_codes))

        # 2026-09-22: 兜底——仍有标的缺当日bar(新浪盘中无当日数据)时用qt实时快照合成
        if latest_td == today:
            stale = [c for c in codes
                     if all_klines.get(c) and all_klines[c][-1]["date"] < today]
            if stale:
                self._patch_today_from_qt(stale, today, all_klines)

        for code in list(all_klines.keys()):
            if len(all_klines[code]) > days:
                all_klines[code] = all_klines[code][-days:]
        return all_klines

    def _fetch_klines_batch(self, codes, days, today, all_klines, incremental=False, pull_map=None):
        """incremental=True: 只抓最近 pull 根(默认 INC_BARS_MAX), 合并前做重叠一致性校验;
        incremental=False: 全量抓取 days 根(冷启动 / 缺口过大 / 增量校验失败时的兜底)。
        2026-09-22 增量改造: 原 refresh_today 场景下 512 只全部重抓 300 根, 现按需回补。"""
        pull_default = INC_BARS_MAX if incremental else days
        pull_map = pull_map or {}
        print(f"  [DB] {'增量' if incremental else '全量'}抓取 {len(codes)} 只K线...")
        def fetch_one(code):
            pull = int(pull_map.get(code) or pull_default)
            sym = self._to_symbol(code)
            url = f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={sym},day,,,{pull},qfq"
            req = urllib.request.Request(url, headers={"User-Agent":"Mozilla/5.0"})
            try:
                resp = urllib.request.urlopen(req, timeout=10)
                data = json.loads(resp.read().decode("utf-8"))
                klines = data.get("data",{}).get(sym,{}).get("qfqday",[]) or data.get("data",{}).get(sym,{}).get("day",[])
                return code, klines, None, "tencent"
            except Exception as e:
                # 腾讯失败(如WAF限流501) → 东财前复权通道(含当日bar) → 新浪备用
                em = self._fetch_em_kline(code, pull)
                if em:
                    return code, em, None, "eastmoney"
                fb = self._fetch_sina_kline(code, sym, pull)
                if fb is not None:
                    return code, fb, None, "sina"
                return code, None, str(e)[:200], "tencent"

        completed = 0
        with ThreadPoolExecutor(max_workers=self._workers) as ex:
            futures = {ex.submit(fetch_one, c): c for c in codes}
            for f in as_completed(futures):
                code, kls, err, src = f.result()
                completed += 1
                if completed % 30 == 0: print(f"  K线: {completed}/{len(codes)}")
                if kls:
                    if src == "sina":
                        # 新浪返回 [{day,open,high,low,close,volume(股)}] → 腾讯兼容口径(手)
                        parsed = []
                        for k in kls:
                            try:
                                parsed.append({"date": k["day"][:10], "open": float(k["open"]), "close": float(k["close"]),
                                               "high": float(k["high"]), "low": float(k["low"]), "volume": float(k.get("volume", 0)) / 100.0})
                            except (KeyError, ValueError, TypeError):
                                continue
                    else:
                        parsed = []
                        for k in kls:
                            try:
                                parsed.append({"date":k[0],"open":float(k[1]),"close":float(k[2]),"high":float(k[3]),"low":float(k[4]),"volume":float(k[5]) if len(k)>5 else 0})
                            except (IndexError, ValueError, TypeError):
                                continue
                    if parsed:
                        # 2026-09-22: 增量合并前校验重叠历史日价格一致性。不一致说明前复权
                        # 因子已变化(除权除息)或数据源口径不同, 增量拼接会造成历史断层
                        # → 立即降级为全量重抓该只
                        if incremental and not self._overlap_ok(all_klines.get(code, []), parsed, today):
                            self._log_fetch(code, today, "klines", "inc_mismatch")
                            self._fetch_klines_batch([code], days, today, all_klines, incremental=False)
                            continue
                        existing = {r["date"]:r for r in all_klines.get(code,[])}
                        # 2026-09-22: 新浪为不复权口径, 只允许补缺口(不覆盖腾讯/东财前复权数据), 防口径混用
                        if src == "sina":
                            new_rows = [r for r in parsed if r["date"] not in existing]
                            for r in parsed:
                                existing.setdefault(r["date"], r)
                        else:
                            new_rows = parsed
                            for r in parsed:
                                existing[r["date"]] = r
                        if not new_rows:
                            self._log_fetch(code, today, "klines", f"skip_{src}")
                            continue
                        all_klines[code] = sorted(existing.values(), key=lambda x:x["date"])
                        self._save_klines(code, new_rows)
                        self._log_fetch(code, today, "klines", f"ok_{src}")
                else:
                    self._log_fetch(code, today, "klines", "failed", err)

    def _overlap_ok(self, local, fetched, today=None):
        """重叠历史日收盘价一致性校验(2026-09-22新增, 增量抓取专用)。
        返回 True  → 新旧序列同口径连续, 可安全合并;
        返回 False → 前复权因子已变化(除权除息)或数据源口径不同, 增量拼接会造成历史断层,
                     调用方必须降级为全量重抓。
        注意: 当日bar只可能是盘中快照(随行情变动), 不参与校验——否则每次盘中运行都会
        因价格变动被误判为"口径不一致", 反而退化成全量抓取。"""
        if not local or not fetched:
            return False
        lb = {r["date"]: r for r in local}
        fb = {r["date"]: r for r in fetched}
        overlap = sorted(d for d in (set(lb) & set(fb)) if d != today)
        if not overlap:
            return False
        for d in overlap[-3:]:          # 校验最近3个重叠历史日
            try:
                lc = float(lb[d].get("close") or 0)
                fc = float(fb[d].get("close") or 0)
            except (TypeError, ValueError):
                return False
            if lc <= 0 or fc <= 0:
                return False
            if abs(lc - fc) / lc > INC_PRICE_TOL:
                return False
        return True

    def _fetch_em_kline(self, code, days):
        """东财前复权日K备用通道(2026-09-22新增, 优先级高于新浪):
        腾讯fqkline被WAF限流时的首选降级。返回腾讯兼容口径
        [[date, open, close, high, low, volume(手)], ...] 或 None。
        优势: fqt=1真前复权(与腾讯qfq同口径)且**含当日盘中bar**; 新浪则不复权且盘中无当日bar。"""
        secid = ("1." if code.startswith(("6", "5", "9")) else "0.") + code
        url = ("https://push2his.eastmoney.com/api/qt/stock/kline/get?secid=" + secid +
               "&fields1=f1,f2,f3,f4,f5,f6&fields2=f51,f52,f53,f54,f55,f56&klt=101&fqt=1&end=20500101&lmt=" + str(days + 10))
        try:
            resp = urllib.request.urlopen(urllib.request.Request(
                url, headers={"User-Agent": "Mozilla/5.0"}), timeout=10)
            data = json.loads(resp.read().decode("utf-8"))
            rows = (data.get("data") or {}).get("klines") or []
            out = []
            for r in rows:
                p = r.split(",")
                if len(p) < 6:
                    continue
                # 东财: date,open,close,high,low,volume(手)
                out.append([p[0], p[1], p[2], p[3], p[4], p[5]])
            return out or None
        except Exception:
            return None

    def _patch_today_from_qt(self, codes, today, all_klines):
        """qt.gtimg.cn 实时快照合成当日K线(2026-09-22新增):
        腾讯+东财均失败时, 新浪通道盘中不含当日bar → 用实时快照补当日bar,
        保证盘中运行的因子计算不落在昨日收盘上。仅当快照时间戳为今日才写入(防非交易日/停牌造伪bar)。"""
        ymd = today.replace("-", "")
        patched = 0
        for i in range(0, len(codes), 20):
            chunk = codes[i:i+20]
            syms = ",".join(self._to_symbol(c) for c in chunk)
            try:
                raw = urllib.request.urlopen(urllib.request.Request(
                    "https://qt.gtimg.cn/q=" + syms, headers={"User-Agent": "Mozilla/5.0"}), timeout=10).read().decode("gbk", "ignore")
            except Exception:
                break
            for line in raw.strip().split(";"):
                p = line.strip().split("~")
                if len(p) < 35:
                    continue
                c, ts = p[2], p[30]
                if not c or not ts.startswith(ymd):
                    continue
                try:
                    bar = {"date": today, "open": float(p[5]), "high": float(p[33]),
                           "low": float(p[34]), "close": float(p[3]), "volume": float(p[6])}
                except (ValueError, IndexError):
                    continue
                if bar["close"] <= 0 or bar["open"] <= 0:
                    continue
                existing = {r["date"]: r for r in all_klines.get(c, [])}
                if len(existing) < 20:      # 历史太短, 补一根也凑不出因子
                    continue
                existing[today] = bar
                all_klines[c] = sorted(existing.values(), key=lambda x: x["date"])
                self._save_klines(c, [bar])
                self._log_fetch(c, today, "klines", "ok_qt")
                patched += 1
            time.sleep(0.2)
        if patched:
            print(f"  [DB] qt实时快照补当日K线: {patched}只")
        return patched

    def _fetch_sina_kline(self, code, sym, days):
        """新浪日K备用通道(2026-09-16新增): 腾讯fqkline被WAF限流时的自动降级。
        返回新浪原始列表[{day,open,high,low,close,volume(股)}] 或 None。
        注意: ① 不复权价(该接口无复权参数, 适合近期无除权的标的; 有除权时会引入口径污染,
        好在下次腾讯通道恢复后会以qfq覆盖) ② 盘中当日bar可能缺失, 收盘后才有。"""
        url = (f"https://quotes.sina.cn/cn/api/jsonp_v2.php/var%20_foo=/"
               f"CN_MarketDataService.getKLineData?symbol={sym}&scale=240&ma=no&datalen={days}")
        req = urllib.request.Request(url, headers={"User-Agent":"Mozilla/5.0", "Referer":"https://finance.sina.com.cn"})
        try:
            resp = urllib.request.urlopen(req, timeout=10)
            text = resp.read().decode("utf-8")
            # jsonp_v2包装: var _foo=(...) → 提取JSON数组部分
            start = text.find("[")
            end = text.rfind("]")
            if start < 0 or end <= start:
                return None
            return json.loads(text[start:end+1])
        except Exception:
            return None

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
                # 修复(2026-09-22): 回调带出记录自身的 date 并标记 stale, 使调用方能够区分
                # "今日实时口径"与"昨日口径", 避免旧数据静默进入当日评分。
                row = conn.execute(
                    "SELECT main_net_5d,main_net_20d,inflow_rate,jumbo_net,main_net_today,date FROM fund_flows "
                    "WHERE code=? AND date<=? ORDER BY date DESC LIMIT 1",
                    (code, today)).fetchone()
                if row:
                    rec = dict(zip(["main_net_5d","main_net_20d","inflow_rate","jumbo_net","main_net_today","date"], row))
                    rec["stale"] = (rec.get("date") != today)
                    result[code] = rec
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
