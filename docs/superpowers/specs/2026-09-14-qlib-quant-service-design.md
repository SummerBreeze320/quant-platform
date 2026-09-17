# 基于 Qlib 与 WindPy 的纯 Python 量化服务系统设计规范

## 1. 项目概述与设计目标

### 1.1 项目背景
在量化投研与交易实践中，微软开源的 **Qlib** 提供了卓越的 AI 导向投研基础设施，包括高性能列式二进制特征存储、灵活的 Alpha 表达式引擎、丰富的机器学习/深度学习模型支持以及组合投资模拟回测。

本项目旨在构建一个**完全纯 Python 架构的量化服务系统**：
1. **投研底座**：以 **Microsoft Qlib** 为核心计算与策略引擎。
2. **数据源**：以 **WindPy（万得金融数据终端 Python API）** 作为权威中国 A 股日频/分钟频行情、指数成分、交易日历与财务数据源。
3. **基础设施**：利用 **Docker Compose** 容器化编排数据库（PostgreSQL）与缓存/任务队列中间件（Redis），降低运维复杂度并与 Python 宿主应用环境解耦。
4. **AI 研发协同**：集成微软 **RD-Agent** 研发智能体，实现基于 LLM 的假设生成、因子代码编写、Qlib 沙箱回测验证与高质量因子自动入库沉淀。
5. **服务形态**：基于 **FastAPI** 与内置调度器（APScheduler / Celery），提供全流程自动化运维与对外 RESTful API 接口。

---

## 2. 总体系统拓扑与基础设施编排

### 2.1 部署与通信拓扑
```
┌─────────────────────────────────────────────────────────────┐
│                       Docker 基础设施容器                   │
│                                                             │
│   ┌───────────────────────────┐ ┌─────────────────────────┐ │
│   │   PostgreSQL 16           │ │   Redis 7-alpine        │ │
│   │   - 资产与成分股元数据     │ │   - Celery / 任务 Broker │ │
│   │   - 因子库元数据与指标     │ │   - 任务进度与状态缓存  │ │
│   │   - 模型与回测表现记录     │ │   - 分布式并发防重锁    │ │
│   └─────────────▲─────────────┘ └────────────▲────────────┘ │
└─────────────────┼────────────────────────────┼──────────────┘
                  │ 5432                       │ 6379
                  │                            │
┌─────────────────┴────────────────────────────┴──────────────┐
│                  纯 Python 应用服务宿主 (Windows / 本地)     │
│                                                             │
│  ┌───────────────────────┐       ┌───────────────────────┐  │
│  │   FastAPI 服务与路由   │       │   定时调度引擎        │  │
│  │   (/api/v1/...)       │       │   (APScheduler/Celery)│  │
│  └───────────┬───────────┘       └───────────┬───────────┘  │
│              ▼                               ▼              │
│  ┌───────────────────────────────────────────────────────┐  │
│  │                     核心业务模块                      │  │
│  │                                                       │  │
│  │  [WindPy Collector]  ──► 原始清洗 ──► [Qlib Dump Bin] │  │
│  │                                              │        │  │
│  │  [Qlib Engine]       ◄───────────────────────┘        │  │
│  │     - 表达式引擎 (Alpha158/自定义因子)                │  │
│  │     - 模型训练与日度推理 (LightGBM/NN)                │  │
│  │     - 组合投资模拟回测 (Top-K/滑点/成本)              │  │
│  │                                                       │  │
│  │  [RD-Agent Subsystem] ──► LLM 假设生成与因子演进      │  │
│  │     - 因子代码自动生成 ──► Qlib 沙箱验证 ──► 因子入库 │  │
│  └───────────────────────────────────────────────────────┘  │
│              │                                              │
│              ▼                                              │
│   ┌─────────────────────┐       ┌────────────────────────┐  │
│   │   本地文件存储      │       │   万得金融终端 (COM)   │  │
│   │   data/qlib_data/   │       │   WindPy 进程通信      │  │
│   └─────────────────────┘       └────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Docker 编排定义 (`docker/docker-compose.yml`)
* **PostgreSQL (v16)**:
  * 端口暴露：`5432:5432`
  * 数据卷持久化：`./data/postgres_data:/var/lib/postgresql/data`
* **Redis (v7-alpine)**:
  * 端口暴露：`6379:6379`
  * 数据卷持久化：`./data/redis_data:/data`

---

## 3. 工程目录结构设计

```text
QuantCopliot/
├── docker/
│   └── docker-compose.yml           # PostgreSQL 与 Redis 编排
├── config/
│   ├── config.yaml                  # 系统基础配置 (标的池、字段映射、回测参数)
│   ├── .env.example                 # 环境变量模版 (DB连接、Redis连接、LLM API Key)
│   └── .env                         # 实际环境变量 (不纳入版本控制)
├── src/
│   ├── common/                      # 基础支撑库
│   │   ├── config.py                # Pydantic 强类型配置解析
│   │   ├── db.py                    # SQLAlchemy 异步/同步连接与 Session
│   │   ├── redis_client.py          # Redis 连接与分布式锁封装
│   │   └── logger.py                # Loguru 日志统一管理
│   ├── models/                      # 数据库 ORM 实体
│   │   ├── base.py                  # DeclarativeBase 与公共审计字段
│   │   ├── factor.py                # 因子元数据表 (公式、IC、IR、作者、分类)
│   │   ├── model_registry.py        # 模型版本与指标记录表
│   │   ├── backtest_record.py       # 回测任务与详细统计表
│   │   └── sync_log.py              # 数据同步批次日志表
│   ├── data_pipeline/               # 数据采集与清洗模块
│   │   ├── wind_client.py           # WindPy 接口生命周期管理、连接池与重试
│   │   ├── collector.py             # 行情、日历、成分股、财务数据抽取器
│   │   ├── transformer.py           # 原始数据清洗、复权计算与格式对齐
│   │   └── qlib_dumper.py           # 转码生成 Qlib 二进制 (.bin) 与增量追加
│   ├── qlib_engine/                 # Qlib 核心封装
│   │   ├── initializer.py           # Qlib 启动配置 (provider_uri, 缓存控制)
│   │   ├── factor_handler.py        # 表达式特征计算与 Alpha 因子工程
│   │   ├── model_trainer.py         # LightGBM / CatBoost / NN 训练与推理封装
│   │   └── backtest.py              # 组合模拟回测引擎与绩效指标评估
│   ├── agent_research/              # RD-Agent 自动化研发
│   │   ├── prompt_templates.py      # 金融假设与因子代码生成 Prompt
│   │   ├── hypothesis_agent.py      # 假设提出与逻辑推理 Agent
│   │   ├── factor_coder_agent.py    # 因子代码/表达式编写 Agent
│   │   ├── sandbox_evaluator.py     # Qlib 回测沙箱与 IC/IR 门槛过滤器
│   │   └── evolution_loop.py        # 多轮迭代反思与自我演进总线
│   ├── service/                     # FastAPI 服务层
│   │   ├── schemas/                 # Pydantic 请求与响应模型
│   │   ├── routers/                 # 模块化路由
│   │   │   ├── data_router.py       # 数据管理接口
│   │   │   ├── factor_router.py     # 因子库接口
│   │   │   ├── model_router.py      # 模型与预测接口
│   │   │   ├── backtest_router.py   # 回测接口
│   │   │   └── agent_router.py      # RD-Agent 任务接口
│   │   └── app.py                   # FastAPI 应用入口与中间件
│   └── tasks/                       # 任务与调度层
│       ├── scheduler.py             # APScheduler 定时作业总控
│       └── jobs.py                  # 盘后同步、每日打分、Agent 离线挖掘任务
├── data/
│   ├── qlib_data/                   # Qlib 二进制特征库
│   │   ├── calendars/               # day.txt
│   │   ├── instruments/             # all.txt, csi300.txt, csi500.txt 等
│   │   └── features/                # 股票特征独立 bin 文件
│   └── temp/                        # 临时中间 Parquet 文件
├── tests/                           # 单元测试与集成测试
├── docs/                            # 项目规格与设计文档
├── requirements.txt                 # 生产依赖列表
├── main.py                          # 服务一键启动脚本
└── README.md                        # 项目指南
```

---

## 4. 核心模块详细设计

### 4.1 WindPy 数据采集与 Qlib 二进制格式转换流水线

#### 4.1.1 WindPy 接口封装 (`wind_client.py`)
* **连接与容错**：
  * 使用上下文管理器管理 `w.start()` 与 `w.stop()`，并在初始化时通过 `w.isconnected()` 验证连接状态。
  * 对 Wind 常见的限流代码（如 `-40522017` 并发达到上限）内置重试机制（指数退避算法）。
* **核心抽取接口**：
  * `w.tdays(start_date, end_date)`：获取标准交易日历；
  * `w.wset("sectorconstituent", "date=...;sectorid=...")`：获取成分股历史名单；
  * `w.wsd(symbols, fields, start_date, end_date, "adj=None")`：按股票批次拉取原始开高低收、成交量、成交额、前收价、复权因子（`adjfactor`）、换手率、股本。

#### 4.1.2 数据清洗与 Qlib 转换 (`qlib_dumper.py`)
* **字段映射规范**：
  * Wind 字段映射至 Qlib 标准基础特征：
    * `OPEN` -> `$open`
    * `HIGH` -> `$high`
    * `LOW` -> `$low`
    * `CLOSE` -> `$close`
    * `VOLUME` -> `$volume`
    * `AMT` -> `$money`
    * `ADJFACTOR` -> `$factor`
    * `VWAP` -> `$vwap`
* **Qlib 数据落盘格式**：
  * `calendars/day.txt`：包含格式为 `YYYY-MM-DD` 的交易日。
  * `instruments/{pool}.txt`：按行记录 `symbol \t start_date \t end_date`。
  * `features/{symbol}/{field}.day.bin`：float32 连续二进制格式存储。
* **增量更新机制**：
  * 数据库维护 `sync_log`，记录各标的最新更新日期。
  * 每日收盘后运行增量同步，仅获取缺失日期的数据追加至二进制文件尾部，避免大规模重排。

---

### 4.2 Qlib 核心投研与回测引擎

#### 4.2.1 因子处理与特征工程 (`factor_handler.py`)
* 基于 Qlib 的表达式解析器，构建统一的数据集 Handler：
  * 支持公式语法直接表达量价因子，如：
    * 收益率反转：`Ref($close, -1) / $close - 1`
    * 量比放大：`$volume / Mean($volume, 20)`
    * 波动率因子：`Std($close / Ref($close, 1) - 1, 20)`
  * 支持自定义 Python 复合因子（如财务与量价交叉因子）。
  * 预处理器默认启用：`DropnaLabel`, `CSZScoreNorm`（截面 Z-Score 标准化）。

#### 4.2.2 模型训练与推理 (`model_trainer.py`)
* **模型适配**：
  * 默认集成 `LightGBM`（梯度提升树）和 `DoubleEnsemble`，兼顾训练速度与非线性拟合能力。
  * 提供统一接口：`fit(train_dataset, valid_dataset)`、`predict(test_dataset)`、`save(path)`、`load(path)`。
* **滚动训练 (Walk-Forward Validation)**：
  * 配置滚动时间窗口（例如训练集 24 个月，验证集 3 个月，测试推理 1 个月），防止未来函数（Lookahead Bias）。

#### 4.2.3 组合回测与绩效评估 (`backtest.py`)
* **交易约束与成本模型**：
  * 基准指数：沪深300 (`000300.SH`) 或中证500 (`000905.SH`)。
  * 持仓调仓：`TopkDropoutStrategy`（每期选出预测得分最高的 Top 30 只股票，落出 Top 50 时卖出，减少无效换手）。
  * 交易费用：印花税 0.05%（仅卖出收取），佣金万分之二，单边滑点万分之五。
* **输出指标**：
  * 收益指标：年化收益率 (ARR)、累计超额收益 (Alpha ARR)。
  * 风险指标：最大回撤 (Max Drawdown)、年化波动率。
  * 综合指标：夏普比率 (Sharpe Ratio)、信息比率 (IR)、Rank IC 均值、ICIR、月度胜率。

---

### 4.3 RD-Agent 自动化研发集成

#### 4.3.1 R&D 智能体工作流
```
┌──────────────────────────────────────────────────────────┐
│                   RD-Agent 演进工作流                    │
│                                                          │
│   ┌──────────────────────────────────────────────────┐   │
│   │ 1. 假设生成 (Hypothesis Generator)               │   │
│   │    - 基于当前市场风格提出因子构建逻辑            │   │
│   └────────────────────────┬─────────────────────────┘   │
│                            ▼                             │
│   ┌──────────────────────────────────────────────────┐   │
│   │ 2. 因子代码编写 (Factor Coder)                   │   │
│   │    - 生成合法的 Qlib 表达式或 Python 计算代码    │   │
│   └────────────────────────┬─────────────────────────┘   │
│                            ▼                             │
│   ┌──────────────────────────────────────────────────┐   │
│   │ 3. Qlib 沙箱验证与指标计算 (Sandbox Evaluator)   │   │
│   │    - 计算全截面 Rank IC, ICIR, 单调性得分        │   │
│   └────────────────────────┬─────────────────────────┘   │
│                            ▼                             │
│   ┌──────────────────────────────────────────────────┐   │
│   │ 4. 质量门槛过滤 (Quality Gate)                   │   │
│   │    - |Rank IC| > 0.035 且 ICIR > 0.5             │   │
│   │    - 与既有因子相关系数 < 0.6                    │   │
│   └────────────┬─────────────────────────┬───────────┘   │
│                │ 达标                    │ 未达标        │
│                ▼                         ▼               │
│   ┌───────────────────────┐  ┌───────────────────────┐   │
│   │ 自动注册入库          │  │ 反思原因并调整假设    │   │
│   │ (PostgreSQL 因子表)   │  │ (Self-Reflection 循环)│   │
│   └───────────────────────┘  └───────────────────────┘   │
└──────────────────────────────────────────────────────────┘
```

#### 4.3.2 因子入库与资产化
通过质量门槛的因子自动持久化至 PostgreSQL，记录如下字段：
* 因子名称与唯一英文标识符；
* 经济学/金融学假设描述；
* Qlib 表达式公式或 Python 源码；
* 验证期历史表现（Rank IC 均值、ICIR、t 统计量、分层收益曲线）；
* 生成该因子的 LLM Prompt 与思考路径记录。

---

### 4.4 定时任务与自动化调度设计 (`tasks/`)

| 触发时间 | 任务名称 | 职责说明 | 容错机制 |
| :--- | :--- | :--- | :--- |
| **交易日 16:30** | `daily_data_sync` | 同步 WindPy 当日最新行情与成分股，更新 Qlib 二进制库 | 自动重试 3 次，失败发送警告日志 |
| **交易日 17:15** | `daily_model_predict`| 加载最佳模型，计算当日特征，输出下一个交易日选股打分清单 | 依赖 `daily_data_sync` 完成状态检查 |
| **周末 10:00** | `weekend_model_retrain` | 使用滑窗数据对生产模型进行增量再训练，评估并保存新权重 | 历史权重版本化备份，防止退化 |
| **周日 14:00** | `rd_agent_auto_mining`| 启动 RD-Agent 离线迭代多轮，挖掘新因子并自动丰富因子库 | 设置运行轮次与超时熔断保护 |

---

## 5. 接口设计 (FastAPI RESTful API)

### 5.1 数据管理模块 (`/api/v1/data`)
* `POST /sync/daily`: 手动触发日度增量同步任务。
* `POST /sync/full`: 触发指定起止时间的全量历史数据重拉与转码。
* `GET /status`: 查看数据更新最新日期、覆盖的标的数及交易日历范围。

### 5.2 因子中心模块 (`/api/v1/factor`)
* `GET /list`: 分页查询当前系统的有效因子列表（支持按类型、IC 排序）。
* `GET /{factor_id}/metrics`: 查询指定因子的详细绩效（IC 时序、分层月度收益）。
* `POST /register`: 手动注册自定义因子公式。
* `GET /correlation`: 计算输入因子集合在指定时间段内的相关性矩阵。

### 5.3 模型与预测模块 (`/api/v1/model`)
* `POST /train`: 提交模型训练任务（传入训练集起止日期、因子组合、模型类型）。
* `GET /predict/latest`: 查询最新交易日生成的预测持仓打分清单（Top-K）。
* `GET /models`: 查看所有已训练模型的历史表现与当前生效的模型版本。

### 5.4 组合回测模块 (`/api/v1/backtest`)
* `POST /run`: 提交回测任务（传入模型ID、回测区间、基准指数、手续费率、持仓数量）。
* `GET /{task_id}/result`: 获取回测绩效综合指标（ARR, Sharpe, MDD, 净值曲线时序）。

### 5.5 研发智能体模块 (`/api/v1/rd-agent`)
* `POST /tasks`: 提交因子挖掘探索任务（指定探索方向、股票池、迭代次数）。
* `GET /tasks/{task_id}`: 查询当前 Agent 探索进度、当前轮次假设与待审核因子。

---

## 6. 验证与测试策略

1. **WindPy 连通性与采集测试**：验证 `WindClient` 的启动、心跳及数据结构规范化。
2. **Qlib 二进制生成与读取测试**：验证 `dump_bin` 生成的 `features` 和 `calendars` 能被 `qlib.data.D.features` 毫秒级正常加载与计算。
3. **因子计算与回测全链路测试**：测试预设 Alpha 表达式在 Qlib 引擎下的计算正确性及轻量 LightGBM 模型的全流程跑通。
4. **RD-Agent 模拟闭环测试**：验证单轮假设提出 -> 表达式编写 -> 沙箱计算 IC -> 自动落库的完整交互。
5. **Docker 基础设施协同测试**：验证 FastAPI 与 PostgreSQL / Redis 的连接池生命周期及分布式锁功能。
