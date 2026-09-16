# -*- coding: utf-8 -*-
"""遍历股票池科技股 2026 半年报，计算同比增速与加速度，筛大超预期"""
import baostock as bs
import csv, time, sys, json

TECH_SECTORS = ['半导体/芯片', 'AI/算力/通信', '电子/消费电子', '智能制造']

def load_tech_codes():
    import sector_map
    d = sector_map.STOCK_SECTOR
    return {c: s for c, s in d.items() if s in TECH_SECTORS}

def bs_code(code):
    return ('sh.' if code.startswith('6') else 'sz.') + code

def query_np(code, year, quarter, tries=3):
    for i in range(tries):
        try:
            rs = bs.query_profit_data(code=bs_code(code), year=year, quarter=quarter)
            rows = []
            while rs.error_code == '0' and rs.next():
                rows.append(rs.get_row_data())
            if rows:
                r = rows[0]
                return {
                    'pubDate': r[1], 'netProfit': float(r[6] or 0),
                    'roeAvg': float(r[3] or 0), 'npMargin': float(r[4] or 0),
                    'gpMargin': float(r[5] or 0), 'revenue': float(r[8] or 0)
                }
            return None
        except Exception:
            time.sleep(0.5)
    return None

def yoy(cur, prev):
    """同比增速，处理负基数"""
    if prev is None or cur is None:
        return None
    if prev > 0:
        return (cur / prev - 1) * 100
    if prev == 0:
        return None
    # prev<0: 扭亏为盈按正增长口径
    return (cur / abs(prev)) * 100 if cur > 0 else (cur / prev - 1) * 100

def main():
    codes = load_tech_codes()
    print(f'科技板块标的: {len(codes)}只', flush=True)
    lg = bs.login()
    results = []
    i = 0
    for code, sector in codes.items():
        i += 1
        h1_26 = query_np(code, 2026, 2)   # 2026 H1
        h1_25 = query_np(code, 2025, 2)   # 2025 H1
        q1_26 = query_np(code, 2026, 1)   # 2026 Q1
        q1_25 = query_np(code, 2025, 1)   # 2025 Q1
        if i % 30 == 0:
            print(f'  ...{i}/{len(codes)}', flush=True)
        results.append({
            'code': code, 'sector': sector,
            'h1_26': h1_26, 'h1_25': h1_25, 'q1_26': q1_26, 'q1_25': q1_25,
        })
    bs.logout()

    # 计算指标
    out = []
    for r in results:
        h1_26, h1_25, q1_26, q1_25 = r['h1_26'], r['h1_25'], r['q1_26'], r['q1_25']
        np_h1_26 = h1_26['netProfit'] if h1_26 else None
        np_h1_25 = h1_25['netProfit'] if h1_25 else None
        np_q1_26 = q1_26['netProfit'] if q1_26 else None
        np_q1_25 = q1_25['netProfit'] if q1_25 else None
        h1_yoy = yoy(np_h1_26, np_h1_25)
        q1_yoy = yoy(np_q1_26, np_q1_25)
        # 单Q2净利 = H1 - Q1
        q2_26 = (np_h1_26 - np_q1_26) if (np_h1_26 is not None and np_q1_26 is not None) else None
        q2_25 = (np_h1_25 - np_q1_25) if (np_h1_25 is not None and np_q1_25 is not None) else None
        q2_yoy = yoy(q2_26, q2_25)
        accel = (h1_yoy - q1_yoy) if (h1_yoy is not None and q1_yoy is not None) else None
        out.append({
            'code': r['code'], 'sector': r['sector'],
            'reported': h1_26 is not None,
            'pubDate': h1_26['pubDate'] if h1_26 else '',
            'np_h1_26': np_h1_26, 'np_h1_25': np_h1_25,
            'np_q1_26': np_q1_26, 'np_q1_25': np_q1_25,
            'h1_yoy': round(h1_yoy, 1) if h1_yoy is not None else None,
            'q1_yoy': round(q1_yoy, 1) if q1_yoy is not None else None,
            'q2_yoy': round(q2_yoy, 1) if q2_yoy is not None else None,
            'accel': round(accel, 1) if accel is not None else None,
            'roe': h1_26['roeAvg'] if h1_26 else None,
            'npm': h1_26['npMargin'] if h1_26 else None,
            'gpm': h1_26['gpMargin'] if h1_26 else None,
            'rev': h1_26['revenue'] if h1_26 else None,
        })

    with open('output/h1_2026_scan.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    json.dump(out, open('output/h1_2026_scan.json', 'w'), ensure_ascii=False, indent=1)
    print(f'完成，写入 output/h1_2026_scan.csv/json，共{len(out)}只', flush=True)

if __name__ == '__main__':
    main()
