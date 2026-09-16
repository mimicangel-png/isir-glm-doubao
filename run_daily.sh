#!/bin/bash
# 本地每日评分运行脚本 — 对齐 GitHub Actions daily-report.yml
# 用法: ./run_daily.sh
# 逻辑: 加载 .env → 跑评分引擎 → 发邮件(3次重试)
set -u
cd "$(dirname "$0")"

PY=/Users/bytedance/.workbuddy/binaries/python/envs/default/bin/python

# 加载 .env
if [ -f .env ]; then
  set -a; source .env; set +a
fi

echo "============================================================"
echo "  本地每日评分  $(date '+%Y-%m-%d %H:%M:%S %A')"
echo "============================================================"

# 1. 运行评分引擎（允许非零退出，仍尝试发邮件）
$PY unified_scoring_engine.py
engine_rc=$?

# 2. 检查报告是否生成
date_str=$(date '+%Y-%m-%d')
if ls output/unified_${date_str}.html 1>/dev/null 2>&1; then
  echo "✅ 报告已生成: output/unified_${date_str}.html"
else
  echo "❌ 报告未生成 (引擎退出码 $engine_rc)"
  exit 1
fi

# 3. 发送邮件
echo "--- 发送邮件 ---"
$PY send_report.py
echo "============================================================"
echo "  完成 $(date '+%H:%M:%S')"
echo "============================================================"
