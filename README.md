# Quant Platform

基于 Qlib + Wind + RD-Agent 的AI量化交易平台，面向A股和ETF市场。

## 快速开始

### 离线体验（无需Wind终端）

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 生成模拟数据（25只股票 + 20只ETF）
python -m src mock-data --stocks 25 --etfs 20

# 3. 运行端到端Demo（数据生成 → 因子评估 → 策略回测 → 报告生成）
python scripts/demo.py

# 4. 启动交互式Dashboard
python -m src serve --port 8000
# 浏览器访问 http://localhost:8000
```

Demo会在 `data/reports/` 下生成HTML回测报告，包含净值曲线和绩效指标。

### 生产环境（需要Wind终端）

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置
#    编辑 config/settings.yaml — MySQL/Redis等连接信息
#    编辑 config/wind_config.yaml — Wind终端参数
#    编辑 config/rd_agent_config.yaml — LLM API Key

# 3. 初始化数据（首次全量，需要Wind终端运行中）
python -m src init-data --start 2015-01-01 --end 2026-08-01

# 4. 每日增量更新
python -m src update-data --date 2026-08-18

# 5. 验证数据
python -m src verify-data

# 6. 运行回测
python -m src backtest --strategy etf_trend --start 2021-01-01 --end 2026-08-01

# 7. 启动Dashboard
python -m src serve --port 8000
# 访问 http://localhost:8000/docs
```

## CLI命令一览

```bash
python -m src --help              # 查看所有命令
python -m src mock-data           # 生成模拟数据
python -m src demo                # 端到端Demo
python -m src init-data            # 从Wind初始化数据
python -m src update-data         # 增量更新
python -m src verify-data         # 数据验证
python -m src backtest            # 运行回测
python -m src list-factors        # 列出可用因子
python -m src serve               # 启动Dashboard
python -m src mine-factors        # RD-Agent因子挖掘
```

## 项目结构

```
quant-platform/
├── config/                         # 配置文件
│   ├── settings.yaml               # 全局配置(MySQL/Redis/Kafka/ES/MinIO)
│   ├── wind_config.yaml            # Wind数据源配置
│   ├── qlib_config.yaml            # Qlib回测/模型配置
│   └── rd_agent_config.yaml        # RD-Agent LLM配置
├── data/                           # 数据目录
│   ├── qlib_bin/                   # Qlib bin格式数据
│   │   ├── calendars/              # 交易日历
│   │   ├── instruments/             # instrument文件
│   │   └── features/                # OHLCV bin文件
│   ├── raw/                        # Wind原始数据缓存
│   ├── cache/                      # 中间处理缓存
│   ├── reports/                    # 回测报告
│   └── rd_agent_workspace/         # RD-Agent工作目录
├── docker/docker-compose.yml      # 补充服务(Jupyter/Grafana)
├── docs/design.md                  # 技术设计文档
├── scripts/
│   ├── init_data.py                # 数据初始化(Wind/Mock)
│   ├── update_data.py              # 每日增量更新
│   ├── demo.py                     # 端到端Demo
│   ├── init_db.sql                 # 数据库建表
│   └── setup_env.ps1              # 环境搭建
├── src/
│   ├── cli.py                      # CLI入口
│   ├── data/                       # 数据管道
│   │   ├── wind/                   # Wind数据采集
│   │   │   ├── wind_client.py      # WindPy封装
│   │   │   ├── stock_data.py       # A股行情/财务/指数成分
│   │   │   ├── etf_data.py         # ETF行情/成分/申赎
│   │   │   └── realtime.py         # 实时行情订阅
│   │   ├── qlib_adapter/           # Qlib格式适配
│   │   │   ├── converter.py        # Wind→Qlib bin(含日历对齐)
│   │   │   ├── calendar.py         # 交易日历生成
│   │   │   ├── instrument.py       # instrument文件生成
│   │   │   └── feature.py         # 特征表达式定义
│   │   ├── mock_data.py            # 模拟数据生成(离线开发)
│   │   └── pipeline.py             # 数据管道编排
│   ├── strategy/                   # 策略
│   │   ├── stock/                  # 股票因子(7个量价因子)
│   │   └── etf/                    # ETF策略(趋势/均值回归/轮动)
│   ├── backtest/                   # 回测引擎
│   │   ├── backtest_engine.py      # 双模式(Qlib原生/策略)
│   │   ├── metrics.py              # 绩效指标
│   │   ├── portfolio.py            # 组合构建
│   │   └── report.py               # HTML报告
│   ├── rd_agent/                   # RD-Agent集成
│   ├── execution/                  # 实盘执行(QMT/风控)
│   ├── dashboard/api/              # FastAPI后端
│   └── utils/                      # 工具函数
├── tests/                          # 25个单元测试
├── requirements.txt
└── README.md
```

## 技术栈

| 组件 | 技术 | 用途 |
|------|------|------|
| 数据源 | Wind Py | A股/ETF专业行情数据 |
| 量化引擎 | Qlib | AI建模/特征工程/回测 |
| AI研发 | RD-Agent | 自动因子挖掘/模型优化 |
| 后端 | FastAPI | REST API + WebSocket |
| 数据库 | MySQL | 历史数据/回测结果/因子元数据 |
| 缓存 | Redis | 实时行情/任务队列 |
| 前端 | HTML + ECharts | 净值曲线/回测报告 |

## 开发路线

详见 [docs/design.md](docs/design.md)

| 阶段 | 内容 | 状态 |
|------|------|------|
| 一 | 数据管道(Wind→Qlib, Mock, 特征) | ✅ 完成 |
| 二 | 回测引擎(双模式, 报告, Demo) | ✅ 完成 |
| 三 | RD-Agent集成 | 🔧 代码完成, 待环境验证 |
| 四 | ETF策略(趋势/回归/轮动/优化) | ✅ 完成 |
| 五 | Dashboard(5页面, ECharts) | ✅ 完成 |
| 六 | 实盘对接(QMT/风控) | 🔧 代码完成, 待对接 |






 
