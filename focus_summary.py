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
# 2026-09-16更新: +3只持仓ETF(9/15入池, 远端v3.5股票池458)
FOCUS = {
    "002222": ("福晶科技", "激光晶体"),
    "301183": ("东田微", "光学滤光片"),
    "002536": ("飞龙股份", "热管理/泵"),
    "300093": ("金刚光伏", "光伏"),
    "603389": ("亚振家居", "家居"),
    "600089": ("特变电工", "输变电"),
    "688183": ("生益电子", "AI-PCB"),
    "300308": ("中际旭创", "光模块"),
    "002851": ("麦格米特", "电力电子"),
    "301122": ("采纳股份", "医疗器械"),
    "300497": ("富祥股份", "医药中间体/CDMO"),
    "300346": ("南大光电", "光刻胶/前驱体"),
    "002916": ("深南电路", "PCB/封装基板"),
    "000938": ("紫光股份", "ICT设备/云计算"),
    "301218": ("华是科技", "智慧城市/AI视觉"),
    "588170": ("科创半导体ETF", "持仓ETF"),
    "588160": ("科创新材料ETF", "持仓ETF"),
    "589180": ("科创材基ETF", "持仓ETF"),
    "600605": ("汇通能源", "风电运营/综合能源"),
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
