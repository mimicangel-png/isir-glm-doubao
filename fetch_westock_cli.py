#!/usr/bin/env python3
"""用westock CLI分批拉全池当日资金流, 生成紧凑行文件 output/fundflow_westock.txt
格式: code|MainNetFlow|MainNetFlow5D|MainNetFlow20D|JumboNetFlow|MainInflowCircRate
CLI单次最多10只; 失败批次重试1次, 仍失败记录到stderr并继续。
"""
import subprocess, sys, time, os

SELF_DIR = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(SELF_DIR, "output", "fundflow_westock.txt")
WESTOCK = os.path.expanduser("~/.local/bin/westock")

POOL = """sh588160 sh588170 sh589180 sh600021 sh600089 sh600226 sh600584 sh600605 sh600633 sh600941
sh601208 sh603083 sh603389 sh603618 sh603626 sh603650 sh688146 sh688183 sh688352 sh688381
sh688521 sh688545 sh688766 sh688795 sz000034 sz000628 sz000636 sz000938 sz001399 sz002149
sz002185 sz002222 sz002285 sz002338 sz002396 sz002536 sz002851 sz002916 sz002927 sz300093
sz300124 sz300223 sz300308 sz300346 sz300418 sz300497 sz300499 sz300570 sz300642 sz300666
sz301080 sz301122 sz301183 sz301218 sz301358 sz301486 sz301511 sz301526""".split()

FIELDS = ["SecuCode", "MainNetFlow", "MainNetFlow5D", "MainNetFlow20D",
          "JumboNetFlow", "MainInflowCircRate", "EndDate"]


def parse_table(text):
    """解析CLI markdown表格 -> [dict]"""
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


def fetch_batch(codes):
    for attempt in (1, 2):
        try:
            r = subprocess.run([WESTOCK, "fund", "flow", ",".join(codes)],
                               capture_output=True, text=True, timeout=60)
            rows = parse_table(r.stdout)
            if rows:
                return rows
        except Exception:
            pass
        time.sleep(3)
    return []


def main():
    got, missing = {}, []
    for i in range(0, len(POOL), 10):
        batch = POOL[i:i + 10]
        for r in fetch_batch(batch):
            code = r.get("SecuCode", "")
            if code:
                got[code] = r
        print(f"批次{i//10+1}: 累计{len(got)}只", file=sys.stderr)
    # 缺失的单独补一次
    for code in POOL:
        if code not in got:
            rows = fetch_batch([code])
            for r in rows:
                if r.get("SecuCode") == code:
                    got[code] = r
    for code in POOL:
        if code not in got:
            missing.append(code)

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("# westock资金流当日快照(2026-09-18收盘后, CLI通道)\n")
        for code in POOL:
            r = got.get(code)
            if not r:
                continue
            f.write("|".join([code, r.get("MainNetFlow", "0"), r.get("MainNetFlow5D", "0"),
                             r.get("MainNetFlow20D", "0"), r.get("JumboNetFlow", "0"),
                             r.get("MainInflowCircRate", "0")]) + "\n")
    dates = sorted({r.get("EndDate", "") for r in got.values()})
    print(f"✅ 完成{len(got)}/{len(POOL)}只, EndDate={dates}, 缺失:{missing or '无'}")


if __name__ == "__main__":
    main()
