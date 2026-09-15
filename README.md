# 统一评分引擎 v3.1 (Unified Stock Scoring Engine)

A股量化评分系统，整合 ISIR / GLM / 豆包 三套评分体系 + VCP形态确认 + 市场趋势门控，每日生成综合决策报告。

## v3.1 更新 (2026-08-04)

### ICIR 权重重标定
- 用 85 交易日 5 日前向收益计算 Spearman rank IC
- sector_rsi 0.015→0.0956, dev_ma20 0.024→0.0897, vwap_premium 0.022→0.0889 等大幅提升
- amplitude_z / vol_ratio_5d 方向反转（IC为负，A股高振幅/高量比不利）
- 无回测数据的因子保留原始手调权重

### 市场趋势门控
- 获取上证综指 K 线，计算指数 vs 50 日均线
- 多头（指数 > MA50）：仓位上限 30 只，正常操作
- 空头（指数 < MA50）：仓位上限减半至 15 只 + 浮亏持仓强制减仓

### VCP 波动收缩形态标注
- Minervini VCP 检测：Stage 2 趋势模板 + Swing 收缩形态 + 量能枯竭 + 突破确认
- 在 ISIR/GLM/豆包 TOP30 标的上标注 "VCP突破" 或 "VCP预突破"
- 回测验证：VCP 突破 5 日胜率 53.7%（+3.9pp vs 全市场），10 日 52.9%（+4.7pp）

## 核心功能

### 三套独立评分
- **ISIR** — 31因子 ICIR 重标定加权排名
- **GLM** — mfi/pct_52w 方向反转排名（基于 GLM 方向矛盾诊断）
- **豆包** — SS 传统技术评分（技术面35% + 资金面55% + 信息面10%）

### 共识标记 + VCP 确认
ISIR ∩ GLM ∩ 豆包 三者 TOP30 交集，VCP 突破/预突破标注。

### 实战买卖策略 + 市场门控
- **买入**：进入 TOP30，T+1 开盘价买入（空头市场不开新仓）
- **仓位上限**：多头 30 只 / 空头 15 只
- **退出条件**（任一触发）：
  - 时间到期：持仓 10 个交易日
  - 止损：浮亏 ≤ -8%
  - 止盈：浮盈 ≥ +15%
  - 排名崩溃：跌到后 50%
  - 空头减仓：空头市场浮亏持仓强制平仓

### 报告功能
- **纵览面板**：市场宽度（MA20占比）、涨跌统计、RSI水位、板块强弱
- **4 个独立评分面板**：SS分 / ISIR / GLM / 豆包，各自独立排序筛选
- **按板块视图**：10 个板块分组纵览
- **操作记录**：按股票视角展示全部交易历史，点击展开明细
- **点击展开详情**：可读技术面解读（均线/RSI/MACD/资金面/信息面）
- **5日/10日/20日涨跌**：每行显示多周期涨跌
- **180天回测对比**：自动选择最优策略作为主评分 👑

## 快速开始

```bash
# 安装依赖
pip install numpy

# 运行引擎（生成当日报告）
python3 unified_scoring_engine.py

# 回溯14天交易信号（首次使用）
python3 backfill_signals.py

# 完整策略回测（85天）
python3 strategy_backtest.py
```

## 文件结构

```
├── unified_scoring_engine.py   # 主引擎（评分+报告+信号追踪）
├── stock_db.py                 # 数据层（SQLite缓存+腾讯财经API）
├── sector_map.py               # 板块映射表（446只A股，10个板块）
├── stock_codes.txt             # 股票池代码列表
├── backfill_signals.py         # 14天信号回溯脚本
├── strategy_backtest.py        # 策略回测脚本（85天）
├── .gitignore
├── README.md
└── output/                     # 输出目录（自动生成）
    ├── unified_YYYY-MM-DD.html # 每日报告
    ├── unified_YYYY-MM-DD.json # 排名数据
    ├── unified_trade_ledger.json   # 交易账本
    ├── unified_signal_history.json # 信号历史
    ├── unified_rank_history.json   # 排名历史
    ├── strategy_backtest.json      # 回测结果
    └── stock_cache.db              # SQLite数据缓存
```

## 评分算法

### ISIR / GLM 权重（来自 ICIR 回测标定）

| 因子 | 权重 | 说明 |
|------|------|------|
| turnover_z | 0.451 | 换手率异常 |
| log_mcap | 0.162 | 市值规模 |
| mfi | ±0.153 | MFI资金流（GLM取反） |
| pct_52w | ±0.091 | 52周位置（GLM取反） |
| pe_percentile | 0.078 | PE分位 |
| ... | ... | 共31个因子 |

### SS 评分（豆包）

```
技术面 = clamp(50 + MA排列± + MACD + RSI调整, 5, 95)
资金面 = clamp(50 + CMF加减分, 5, 95)
信息面 = 50（固定）
综合SS = 技术面×0.35 + 资金面×0.55 + 信息面×0.10
```

## 回测结果（85天，71个有效交易日）

### 实战策略

| 策略 | 交易笔数 | 胜率 | 均收益 | 累积收益 | 均持仓 |
|------|:---:|:---:|:---:|:---:|:---:|
| ISIR 👑 | 380 | 47.1% | +2.0% | +762% | 5.5天 |
| GLM | 503 | 47.1% | +1.8% | +920% | 4.2天 |
| 豆包 | 368 | 44.8% | +1.3% | +462% | 5.7天 |

### 固定持仓基准（选股能力上限）

| 策略 | 5日胜率 | 10日胜率 | 20日胜率 |
|------|:---:|:---:|:---:|
| ISIR | 53.3% | 56.6% | 60.2% |
| GLM | 51.6% | 54.2% | 57.9% |
| 豆包 | 51.1% | 53.5% | 55.5% |

## 数据来源

- **K线数据**：腾讯财经 API (`web.ifzq.gtimg.cn`)
- **实时行情**：腾讯行情 API (`qt.gtimg.cn`)
- **资金流/事件**：westock-data Node.js 脚本

## 技术栈

- Python 3.10+
- numpy（截面标准化计算）
- SQLite（本地数据缓存）
- 纯标准库（urllib, sqlite3, json, subprocess）

## 换设备迁移指南 (2026-09-15)

代码与关键数据均在 GitHub 仓库（isir-glm-doubau 的 main 分支），更换 PC 后按以下步骤恢复迭代：

```bash
# 1. 克隆仓库
git clone https://github.com/mimicangel-png/isir-glm-doubau.git
cd isir-glm-doubau

# 2. 生成 output 目录（git 不跟踪目录本身）
mkdir -p output

# 3. 运行引擎 —— K线缓存(output/stock_cache.db)会自动增量抓取重建,
#    首次约 3-5 分钟拉取 458 只 x 300 日
python3 unified_scoring_engine.py
```

**Git 跟踪的关键数据（换设备自动随 clone 恢复）**：
- `output/unified_signal_history.json` — 信号追踪历史（三体系持仓/已结算交易/累积收益）
- `output/quality_rebound_backtest.json` — 第五视图质量反弹回测数据

**仅存在于本地、需手动迁移（可选）**：
- `output/stock_cache.db` (~35MB) — K线缓存。不迁移也行，引擎会自动重拉；迁移可省首跑时间（U盘/网盘/AirDrop 拷贝到 output/ 下即可）

**本地独有的配置（不在 Git，需手动迁移）**：
- WorkBuddy 自动化任务（盘前9点/午盘12点/收盘16点/资金流16:35）在新设备 WorkBuddy 里重新创建
- `.workbuddy/memory/` 项目记忆（持仓清单、分析约定）——建议迁移以保持分析连续性

**股票池调整注意事项**（重要踩坑记录）：
1. 新代码加入 `stock_codes.txt` 后，需删除 `output/stock_cache.db` 中该代码的 klines/extra_info/fund_flows 记录（或直接删整个db），引擎才会增量抓取新标的
2. 沪市 ETF（588/589 开头）前缀已由 stock_db.py 的 `_to_symbol` 处理（2026-09-15 修复），深市 ETF（159 开头）走默认 sz 分支
3. 新标的需同步在 `sector_map.py` 的 STOCK_SECTOR 补板块映射，否则行业相对因子（sector_rsi/sector_momentum）失真

## 免责声明

本项目仅供学习和研究使用，不构成投资建议。投资有风险，入市需谨慎。

## License

MIT
