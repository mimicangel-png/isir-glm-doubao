#!/usr/bin/env python3
"""
refresh_fundflow_realtime.py — 盘中实时资金流刷新 (2026-09-21 建立)
============================================================
彻底解决: SS分资金面(55%权重)盘中读昨收口径导致排名滞后的底层问题。

原理: 用 westock CLI(独立二进制, 脚本可直连)拉取全股票池当日实时资金流,
      写入 stock_cache.db 的 fund_flows 表【当日】记录。
      统一评分引擎 get_fund_flows 用 `date<=today ORDER BY date DESC LIMIT 1`
      读最近记录 → 盘中即读到当日实时(而非昨收), SS分/ISIR/GLM资金流因子全部实时化。

westock CLI 返回字段(实测 sz300497):
  EndDate=当日  MainNetFlow=当日主力净额(实时)  MainNetFlow5D/20D=实时累计
  JumboNetFlow=超大单净额  MainInflowCircRate=主力流入占流通比(%)

与 tdx 关系: tdx 逐只查历史收盘(5D/20D精确拼合)、westock 给当日实时, 互补。
本脚本只负责"当日实时"这一路, 入库 date=今日, 不覆盖 tdx 历史。

用法:
  python3 refresh_fundflow_realtime.py              # 拉全股票池(stock_codes.txt)
  python3 refresh_fundflow_realtime.py --pool-only  # 只拉持仓+focus+共识关键池(快)
  python3 refresh_fundflow_realtime.py --dry-run    # 只拉不写库, 打印结果

失败回退: 全池失败→重试1次; 仍失败→打印警告不写库(引擎回退读昨收, 不中断流程)。
"""
import os, sys, subprocess, sqlite3, json, time
from datetime import datetime

SELF_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(SELF_DIR, "output", "stock_cache.db")
WESTOCK = os.path.expanduser("~/.local/bin/westock")

FIELDS = ["SecuCode", "EndDate", "MainNetFlow", "MainNetFlow5D", "MainNetFlow20D",
          "JumboNetFlow", "MainInflowCircRate"]

# westock CLI 每批上限(实测10只稳)
BATCH_SIZE = 10


def _to_symbol(code):
    if code.startswith(("6", "9", "58")):
        return "sh" + code
    if code.startswith(("8", "4", "92")):
        return "bj" + code
    return "sz" + code


def load_codes(pool_only=False):
    """全池 stock_codes.txt; --pool-only 用持仓+focus+共识关键池"""
    if not pool_only:
        f = os.path.join(SELF_DIR, "stock_codes.txt")
        if os.path.exists(f):
            codes = [l.strip() for l in open(f, encoding="utf-8") if l.strip()]
            if codes:
                return codes
    # 关键池: gen_fundflow_pool.py 输出的核心池(持仓+focus+共识)
    try:
        sys.path.insert(0, SELF_DIR)
        import gen_fundflow_pool as g
        core, full = g._load_pool()
        return sorted(full)
    except Exception:
        pass
    # 兜底: 用评分JSON里的全部code
    try:
        today = datetime.now().strftime("%Y-%m-%d")
        p = os.path.join(SELF_DIR, "output", f"unified_{today}.json")
        if os.path.exists(p):
            return [x["code"] for x in json.load(open(p))]
    except Exception:
        pass
    return []


def parse_table(text):
    rows = []
    lines = [l for l in text.splitlines() if l.strip().startswith("|")]
    if len(lines) < 3:
        return rows
    header = [c.strip() for c in lines[0].strip("|").split("|")]
    idx = {f: header.index(f) for f in FIELDS if f in header}
    for line in lines[2:]:
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < len(header):
            continue
        rows.append({f: cells[i] for f, i in idx.items()})
    return rows


def _f(v, default=0.0):
    try:
        return float(v) if v not in (None, "", "-", "--") else default
    except (ValueError, TypeError):
        return default


def fetch_batch(codes):
    """westock CLI 拉一批, 失败重试1次"""
    for attempt in (1, 2):
        try:
            r = subprocess.run([WESTOCK, "fund", "flow", ",".join(codes)],
                               capture_output=True, text=True, timeout=90)
            rows = parse_table(r.stdout)
            if rows:
                return rows
        except Exception:
            pass
        time.sleep(3)
    return []


def main():
    pool_only = "--pool-only" in sys.argv
    dry_run = "--dry-run" in sys.argv
    codes = load_codes(pool_only)
    if not codes:
        print("❌ 无股票池, 退出")
        return 1

    print(f"[实时资金流刷新] 池{len(codes)}只 | westock CLI | 每批{BATCH_SIZE}只")
    got, missing = {}, []
    for i in range(0, len(codes), BATCH_SIZE):
        batch = codes[i:i + BATCH_SIZE]
        syms = [_to_symbol(c) for c in batch]
        for r in fetch_batch(syms):
            sc = str(r.get("SecuCode", ""))[-6:]
            if sc:
                got[sc] = r
        if (i // BATCH_SIZE + 1) % 5 == 0:
            print(f"  进度 {i + BATCH_SIZE}/{len(codes)} | 累计{len(got)}只", file=sys.stderr)

    # 缺失的单独补一次
    for code in codes:
        if code not in got:
            rows = fetch_batch([_to_symbol(code)])
            for r in rows:
                if str(r.get("SecuCode", ""))[-6:] == code:
                    got[code] = r
                    break
            if code not in got:
                missing.append(code)

    if not got:
        print("❌ westock CLI 全池拉取失败, 不写库(引擎将回退读昨收)")
        return 1

    # 校验 EndDate 是否为今日(实时性判断)
    dates = sorted({str(r.get("EndDate", ""))[:10] for r in got.values()})
    today = datetime.now().strftime("%Y-%m-%d")
    is_realtime = dates[-1] == today if dates else False

    if dry_run:
        print(f"  [dry-run] 拉取{len(got)}只 EndDate={dates} 实时={is_realtime}")
        rows = sorted(got.items(), key=lambda kv: -_f(kv[1].get("MainNetFlow")))
        for c, r in rows[:5]:
            print(f"    净流入TOP {c}: 今日{_f(r.get('MainNetFlow'))/1e8:+.2f}亿 5日{_f(r.get('MainNetFlow5D'))/1e8:+.2f}亿")
        for c, r in rows[-5:]:
            print(f"    净流出TOP {c}: 今日{_f(r.get('MainNetFlow'))/1e8:+.2f}亿 5日{_f(r.get('MainNetFlow5D'))/1e8:+.2f}亿")
        return 0

    # 写库: date=今日, INSERT OR REPLACE
    n = 0
    with sqlite3.connect(DB_PATH) as conn:
        for code, r in got.items():
            date = str(r.get("EndDate", ""))[:10] or today
            conn.execute(
                "INSERT OR REPLACE INTO fund_flows(code,date,main_net_5d,main_net_20d,inflow_rate,jumbo_net,main_net_today,fetched_at) "
                "VALUES(?,?,?,?,?,?,?,datetime('now'))",
                (code, date, _f(r.get("MainNetFlow5D")), _f(r.get("MainNetFlow20D")),
                 _f(r.get("MainInflowCircRate")), _f(r.get("JumboNetFlow")), _f(r.get("MainNetFlow"))),
            )
            conn.execute(
                "INSERT OR REPLACE INTO fetch_log(code,date,data_type,status,updated_at) "
                "VALUES(?,?,?,'ok',datetime('now'))",
                (code, date, "fund_flow_westock_realtime"),
            )
            n += 1

    flag = "✅实时" if is_realtime else "⚠️非当日"
    print(f"✅ 实时资金流入库: {n}只 | EndDate={dates} {flag} | 缺失{len(missing)}只")
    if missing:
        print(f"  缺失: {','.join(missing[:20])}{'...' if len(missing) > 20 else ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
