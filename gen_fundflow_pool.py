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

# 用户真实持仓(2026-09-21 15:21同花顺截图口径, 两账户合并) — 变动后手动更新
# 格式: code  # 名称 股数(平安+国金) 成本区间
# 平安证券**2528市值约186.7万(盈亏约-46.1万) + 国金证券**7341市值约348.1万(盈亏约-69.6万, 含港股)
# 两账户合计市值约534.8万, 总浮亏约-115.7万(约-17.8%)
# 已清仓: 东田微301183(9/16@284)/生益电子688183(9/15@126.25)/国际复材(9/15@32.05)/沃森生物(9/11止损)/蓝色光标/三变科技/奥飞数据/龙蟠科技
# 港股: 资玩6400股(HK$9.68, 引擎不支持不入池)
# 注: 9/21四张截图两两拼合(平安2张+国金2张,中段重叠)已确认为完整持仓, 无缺失个股
HOLDINGS = [
    "600089",  # 特变电工 18200股(国金) 成本12.277 +49.06%
    "300497",  # 富祥股份 32100股(平安19900@19.112+国金12200@18.581) +8.99%/+12.10%
    "300093",  # 金刚光伏 11900股(国金) 成本23.917 -6.55%
    "301358",  # 湖南裕能 1000股(平安) 成本84.808 -37.41%
    "002338",  # 奥普光电 100股(平安,已大幅减仓) 成本58.211 -26.82%
    "300124",  # 汇川技术 2800股(国金) 成本77.878 -32.15%
    "002149",  # 西部材料 2700股(平安) 成本73.594 -53.80%
    "002222",  # 福晶科技 7800股(平安1300@182.277+国金6500@85.995) -61.73%/-18.88%
    "603389",  # 亚振家居 32100股(平安2800@89.431+国金29300@60.086) -48.46%/-23.29%
    "600021",  # 上海电力 16100股(国金) 成本20.067 -33.08%
    "601208",  # 东材科技 4100股(平安,补仓摊低成本70.1→63.917) -17.47%
    "600633",  # 浙数文化 10300股(国金) 成本14.350 -28.02%
    "300418",  # 昆仑万维 1900股(国金) 成本56.500 -22.81%
    "000938",  # 紫光股份 1000股(国金) 成本40.326 -16.08%
    "300346",  # 南大光电 1500股(平安) 成本57.507 -3.28%
    "002927",  # 泰永长征 2000股(平安) 成本23.039 -10.98%
    "000034",  # 神州数码 500股(平安) 成本27.939 -16.10%
    "600941",  # 中国移动 200股(国金) 成本108.057 -11.15%
    "600226",  # 亨通股份 10000股(平安) 成本6.971 -13.06%
    "603650",  # 彤程新材 600股(平安,9/17买) 成本73.867 -2.54%
    "002536",  # 飞龙股份 1200股(国金,重新买入) 成本55.751 -3.46%
    "600605",  # 汇通能源 9900股(平安,新持仓) 成本26.090 +4.25%
    "300570",  # 太辰光 300股(平安,新持仓) 成本225.772 -0.92%
    "002463",  # 沪电股份 400股(平安,新持仓) 成本125.371 +0.50%
    "601869",  # 长飞光纤 100股(平安,新持仓) 成本458.615 +0.08%
    "300666",  # 江丰电子 400股(平安,新持仓) 成本258.738 -0.65%
    "588170",  # 科创半导体ETF 52700份 成本1.398 -27.62%
    "588160",  # 科创新材料ETF 35300份 成本1.416 -22.61%
    "589180",  # 科创材基ETF 57200份 成本2.397 -27.16%
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
