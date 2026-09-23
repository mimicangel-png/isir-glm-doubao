# -*- coding: utf-8 -*-
"""
focus_summary.py - 重点科技股走势追踪摘要
监控: 福晶科技(002222) 东田微(301183) 飞龙股份(002536)
      金刚光伏(300093) 亚振家居(603389) 特变电工(600089)
      生益电子(688183) 中际旭创(300308) 麦格米特(002851) 采纳股份(301122)
      南大光电(300346) 深南电路(002916) 紫光股份(000938)
      华是科技(301218) 富祥股份(300497)
数据: SQLite(output/stock_cache.db) 行情 + K线 + 当日评分JSON
输出: 文本摘要(含今日行情、评分排名、近5日走势、技术判断)
"""
import json
import sqlite3
import os
import sys
import glob

PROJ = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(PROJ, "output", "stock_cache.db")

# code -> (名称, 标签)
# 2026-09-21更新: 按用户要求"所有持仓+清仓+重点关注每次分析都详细分析"全覆盖
# 持仓股标[持仓], 已清仓但持续追踪标[清仓追踪], 纯关注标原行业标签
# 2026-09-23更新: 新增【机动池】标记 —— 最近新入且市值2万~20万的小仓, 用于高频波段操作(用户指定)
FOCUS = {
    # ===== 持仓股 (2026-09-21 15:21截图口径) =====
    "300497": ("富祥股份", "持仓(国金12200@18.581)/平安仓9-23@20.15清仓"),
    "601208": ("东材科技", "持仓/新材料(补仓摊成本63.9,市值超20万不入机动池)"),
    "603389": ("亚振家居", "持仓/家居(深套-48%/-23%)"),
    "300666": ("江丰电子", "机动池/半导体靶材(9/21新买)"),
    "002149": ("西部材料", "持仓/钛材(深套-54%)"),
    "002222": ("福晶科技", "持仓/激光晶体(深套-62%/-19%,回补需证据)"),
    "300346": ("南大光电", "机动池/光刻胶(近期加仓)"),
    "300570": ("太辰光", "机动池/光模块连接(新买)"),
    "600226": ("亨通股份", "持仓/光纤"),
    "301358": ("湖南裕能", "持仓/锂电正极(深套-37%)"),
    "002463": ("沪电股份", "机动池/AI-PCB(新买)"),
    "601869": ("长飞光纤", "机动池/光纤(新买)"),
    "603650": ("彤程新材", "机动池/光刻胶(9/17买)"),
    "002927": ("泰永长征", "持仓/低压电器"),
    "000034": ("神州数码", "持仓/IT分销/信创"),
    "002338": ("奥普光电", "持仓/光电(已减至100股)"),
    "600089": ("特变电工", "持仓/输变电(+49%最大盈利)"),
    "300093": ("金刚光伏", "持仓/光伏(危险仓)"),
    "002536": ("飞龙股份", "持仓/热管理(重新买入)"),
    "600021": ("上海电力", "持仓/电力(深套-33%)"),
    "300124": ("汇川技术", "持仓/工控(深套-32%)"),
    "600633": ("浙数文化", "持仓/数字文化(深套-28%)"),
    "300418": ("昆仑万维", "持仓/AI应用(深套-23%)"),
    "000938": ("紫光股份", "持仓/ICT设备"),
    "600941": ("中国移动", "持仓/运营商"),
    "588170": ("科创半导体ETF", "持仓ETF(52700份)"),
    "588160": ("科创新材料ETF", "持仓ETF(35300份)"),
    "589180": ("科创材基ETF", "持仓ETF(57200份)"),
    # ===== 清仓追踪 =====
    "301183": ("东田微", "清仓追踪(9/16@284卖出)"),
    "688183": ("生益电子", "清仓追踪(9/15@126.25卖出)"),
    "600605": ("汇通能源", "清仓追踪(9/23@31卖出, 成本26.090 +18.8%)"),
    # ===== 重点关注(非持仓) =====
    "300308": ("中际旭创", "光模块"),
    "002851": ("麦格米特", "电力电子"),
    "301122": ("采纳股份", "医疗器械"),
    "002916": ("深南电路", "PCB/封装基板"),
    "301218": ("华是科技", "智慧城市/AI视觉"),
}


def latest_date(conn):
    return conn.execute("SELECT MAX(date) FROM extra_info").fetchone()[0]


def load_ranks(today):
    """从最新评分 JSON 加载排名信息"""
    # 只匹配日期文件 unified_YYYY-MM-DD.json，排除 rank/signal/trade_ledger 等
    json_files = sorted(
        glob.glob(os.path.join(PROJ, "output", "unified_20*.json"))
    )
    if not json_files:
        return {}
    with open(json_files[-1], encoding="utf-8") as f:
        data = json.load(f)
    ranks = {}
    for r in data:
        ranks[r.get("code", "")] = {
            "isir": r.get("isir_rank"),
            "glm": r.get("glm_rank"),
            "ss": r.get("doubao_rank"),
            "consensus": r.get("consensus", False),
            "vcp": r.get("vcp_status", ""),
            "oversold": r.get("rebound_stage") or r.get("oversold_score") or r.get("rebound_status", ""),
        }
    return ranks


def main():
    conn = sqlite3.connect(DB)
    date = latest_date(conn)
    ranks = load_ranks(date)

    print(f"📅 数据日期: {date}")
    print("=" * 64)
    for code, (name, tag) in FOCUS.items():
        row = conn.execute(
            "SELECT price, change_pct, turnover, vol_ratio FROM extra_info WHERE code=? AND date=?",
            (code, date),
        ).fetchone()
        # 近5日K线
        kls = conn.execute(
            "SELECT date, open, high, low, close, volume FROM klines WHERE code=? ORDER BY date DESC LIMIT 6",
            (code,),
        ).fetchall()

        print(f"\n◆ {code} {name}（{tag}）")
        if row:
            price, chg, turn, vr = row[0], row[1] or 0, row[2], row[3]
            chg_sym = "+" if chg >= 0 else ""
            print(f"  今日: {price} ({chg_sym}{chg:.2f}%) | 换手{turn}% 量比{vr}")
        else:
            print("  今日: 无行情数据")
            chg = 0

        # 近5日走势
        if kls:
            trend = []
            prev = None
            for d, o, h, l, c, v in reversed(kls):
                cchg = ((c - prev) / prev * 100) if prev else 0
                trend.append(f"{d[5:]}:{c:.1f}({cchg:+.1f}%)")
                prev = c
            print(f"  近6日: {'  '.join(trend)}")
            # 5日累计
            if len(kls) >= 2:
                c5 = (kls[0][4] / kls[-1][4] - 1) * 100
                print(f"  5日累计: {c5:+.2f}%")

        # 评分排名
        rk = ranks.get(code)
        if rk:
            cons = "✅共识" if rk["consensus"] else "—"
            vcp = f" VCP:{rk['vcp']}" if rk.get("vcp") else ""
            print(
                f"  排名: ISIR#{rk['isir']} GLM#{rk['glm']} SS#{rk['ss']} {cons}{vcp}"
            )
        else:
            print("  排名: 无(未进JSON)")

    conn.close()
    print("\n" + "=" * 64)


if __name__ == "__main__":
    main()
