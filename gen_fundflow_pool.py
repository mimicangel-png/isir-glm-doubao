#!/usr/bin/env python3
"""
生成实时资金流拉取的关键池代码串。
用法:
  python3 gen_fundflow_pool.py          # westock批量模式: 输出sh/sz前缀代码逗号串(≤60只/行)
  python3 gen_fundflow_pool.py --tdx    # tdx优先模式(2026-09-18): 输出两行纯数字代码(空格分隔):
                                        #   第1行=核心池(持仓+focus+共识, tdx逐只查)
                                        #   第2行=扩展池(TOP30补充, westock批量查)

关键池构成: 用户真实持仓 + focus追踪列表 + 最新一期共识 + ISIR/GLM各TOP30 (去重)。
持仓清单维护在本文件 HOLDINGS 常量(用户清仓/新买后手动更新)。
"""

import os, sys, json, glob

SELF_DIR = os.path.dirname(os.path.abspath(__file__))

# 用户真实持仓(2026-09-17口径, 东田微已清仓; 9/17新增彤程新材成本74) — 变动后手动更新
HOLDINGS = [
    "600089",  # 特变电工
    "300497",  # 富祥股份
    "300093",  # 金刚光伏
    "301358",  # 湖南裕能
    "002338",  # 奥普光电
    "300124",  # 汇川技术
    "002149",  # 西部材料
    "002222",  # 福晶科技
    "603389",  # 亚振家居
    "600021",  # 上海电力
    "601208",  # 东材科技
    "600633",  # 浙数文化
    "300418",  # 昆仑万维
    "000938",  # 紫光股份
    "300346",  # 南大光电
    "002927",  # 泰永长征
    "000034",  # 神州数码
    "600941",  # 中国移动
    "600226",  # 亨通股份
    "603650",  # 彤程新材(2026-09-17 14时许买入, 成本74)
    "588170",  # 科创半导体ETF
    "588160",  # 科创新材料ETF
    "589180",  # 科创材基ETF
]


def _pref(code):
    if code.startswith(("6", "9", "58")):
        return "sh" + code
    if code.startswith(("8", "4", "92")):
        return "bj" + code
    return "sz" + code


def _load_pool():
    """返回 (core_set, full_set): core=持仓+focus+共识, full=core+TOP30扩展"""
    core = set(HOLDINGS)

    # focus列表
    try:
        sys.path.insert(0, SELF_DIR)
        from focus_summary import FOCUS
        core |= set(FOCUS.keys())
    except Exception:
        pass

    full = set(core)

    # 最新一期评分: 共识 + ISIR/GLM TOP30 (排除带时段后缀与历史文件)
    try:
        today = None
        import datetime
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        cands = [p for p in glob.glob(os.path.join(SELF_DIR, "output", "unified_*.json"))
                 if not any(x in p for x in ("rank_history", "signal_history", "trade_ledger"))
                 and "backtest" not in p and "_" not in os.path.basename(p)[len("unified_"):-5] or
                 os.path.basename(p) == f"unified_{today}.json"]
        target = os.path.join(SELF_DIR, "output", f"unified_{today}.json")
        if not os.path.exists(target):
            target = max(cands, key=os.path.getmtime) if cands else None
        if target and os.path.exists(target):
            d = json.load(open(target))
            for s in sorted(d, key=lambda x: x["isir_rank"])[:30]:
                full.add(s["code"])
            for s in sorted(d, key=lambda x: x["glm_rank"])[:30]:
                full.add(s["code"])
            for s in d:
                if s.get("consensus"):
                    core.add(s["code"])
                    full.add(s["code"])
    except Exception as e:
        print(f"# 评分JSON读取失败({e}), 仅用持仓+focus", file=sys.stderr)

    return core, full


def main():
    core, full = _load_pool()

    if "--tdx" in sys.argv:
        # tdx优先模式: 第1行核心池(逐只tdx), 第2行扩展池(westock批量或并入tdx)
        print(" ".join(sorted(core)))
        print(" ".join(sorted(full - core)))
        return

    # westock批量模式: 全池sh/sz前缀逗号串
    syms = sorted(_pref(c) for c in full)
    # 分批输出, 每批≤60只
    for i in range(0, len(syms), 60):
        print(",".join(syms[i:i + 60]))


if __name__ == "__main__":
    main()
