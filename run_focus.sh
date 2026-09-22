#!/bin/bash
# ============================================================
# run_focus.sh - 每日评分 + 重点科技股走势追踪
# 对齐线上 GitHub Actions 机制：跑统一评分引擎 + 输出重点标的摘要
# 重点监控: 福晶科技(002222) 东田微(301183) 飞龙股份(002536)
#           金刚光伏(300093) 亚振家居(603389) 特变电工(600089)
#           生益电子(688183) 中际旭创(300308) 麦格米特(002851) 采纳股份(301122)
#           南大光电(300346) 深南电路(002916) 紫光股份(000938) 华是科技(301218)
# 用法: bash run_focus.sh [--no-engine]  (--no-engine 跳过引擎只出摘要)
# 每次运行后自动按"日期_时段"独立存档，不覆盖历史:
#   output/unified_YYYY-MM-DD.html          <- 最新一份(canonical)
#   output/unified_YYYY-MM-DD_HHMM.html     <- 本次时段独立存档
# ============================================================
set -e
cd "$(dirname "$0")"
PY=/Users/bytedance/.workbuddy/binaries/python/envs/default/bin/python

TODAY=$(date +%F)
SLOT=$(date +%H%M)
HOUR=$(date +%H)

# ===== 实时资金流刷新 (2026-09-21新增) =====
# 盘中(HOUR>=9:30)自动拉 westock 当日实时资金流入库, 解决SS分资金面盘中读昨收滞后问题。
# 盘前(9:00前, 未开盘无实时资金流)跳过; 午盘/收盘/手动盘中均刷实时。
# --no-fundflow 跳过(手动降级); 拉取失败不中断引擎(引擎回退读昨收)。
if [ "$1" != "--no-fundflow" ] && [ "${HOUR#0}" -ge 9 ]; then
  echo "========== [0/3] 刷新盘中实时资金流 (westock) =========="
  $PY refresh_fundflow_realtime.py 2>&1 | tail -8
else
  echo "========== [0/3] 跳过实时资金流 (盘前/手动降级) =========="
fi

if [ "$1" != "--no-engine" ]; then
  echo "========== [1/3] 运行统一评分引擎 =========="
  # 完整输出落盘(供自动化/复盘直接读取大盘门控/共识/信号/回测摘要, 免解析HTML), 控制台仅留尾部
  mkdir -p output/logs
  LOG="output/logs/engine_${TODAY}_${SLOT}.log"
  $PY unified_scoring_engine.py 2>&1 | tee "$LOG" | tail -30
  echo "  📄 引擎完整输出已存: ${LOG}"
else
  echo "========== [1/3] 跳过引擎(使用已有数据) =========="
fi

echo ""
echo "========== [2/3] 时段独立存档 (${TODAY}_${SLOT}) =========="
ARCHIVED=""
for ext in html json; do
  SRC="output/unified_${TODAY}.${ext}"
  DST="output/unified_${TODAY}_${SLOT}.${ext}"
  if [ -f "$SRC" ]; then
    cp "$SRC" "$DST"
    echo "  ✅ ${DST}  (from ${SRC})"
    ARCHIVED="$ARCHIVED $DST"
  else
    echo "  ⚠ 未找到 ${SRC}，跳过存档"
  fi
done
if [ -z "$ARCHIVED" ]; then
  echo "  ⚠ 本次没有可存档的报告文件——引擎可能失败，检查上方输出"
fi

echo ""
echo "========== [3/3] 重点科技股走势追踪 =========="
mkdir -p output/logs
$PY focus_summary.py 2>&1 | tee "output/logs/focus_${TODAY}_${SLOT}.log"

echo ""
echo "✅ 完成 | 存档: ${TODAY}_${SLOT}"
