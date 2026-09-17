# 基于 Qlib 与 WindPy 的纯 Python 量化服务实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建一个完全纯 Python 的量化服务系统，以 Microsoft Qlib 为核心底座、WindPy 为权威数据源、Docker Compose 编排基础设施（PostgreSQL + Redis）、深度集成微软 RD-Agent 自动化因子研发闭环，并通过 FastAPI 与定时调度器实现服务化运维。

**Architecture:** 采用模块化任务驱动分层架构。基础设施由 Docker 提供 Postgres（持久化存储因子库、回测记录和同步日志）与 Redis（队列与缓存）；Python 服务层由数据管道（WindPy 抽取清洗 -> Qlib dump_bin 二进制生成）、Qlib 投研引擎（Alpha 表达式、模型训练、组合回测）、RD-Agent 研发子系统（假设生成 -> 因子编写 -> 沙箱检验 -> 入库）、以及 FastAPI 与 APScheduler 服务层组成。

**Tech Stack:** Python 3.10+, pyqlib, WindPy, RD-Agent, FastAPI, Pydantic v2, SQLAlchemy 2.0, APScheduler, Redis-py, LightGBM, Loguru, Docker & Docker Compose, Pytest.

**Spec:** `docs/superpowers/specs/2026-09-14-qlib-quant-service-design.md`

## Global Constraints

- 纯 Python 开发与兼容 Windows 宿主机运行环境（保障 WindPy 与 Qlib 本地高性能读写）。
- 代码中必须严格处理 WindPy 异常与无 Wind 客户端环境下的 Mock 降级，保障自动化单测通过。
- 遵循 DRY、YAGNI 与 TDD 原则，每个模块均具备单元测试。
- 配置统一使用 Pydantic BaseSettings / Settings，禁止硬编码密码或端口。

---

### Task 1: 基础设施编排、配置与公共工具层

**Files:**
- Create: `docker/docker-compose.yml`
- Create: `config/config.yaml`
- Create: `config/.env.example`
- Create: `src/common/config.py`
- Create: `src/common/logger.py`
- Create: `requirements.txt`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces:
  - `src.common.config.Settings`: 强类型配置单例 `get_settings()`
  - `src.common.logger.logger`: 预先配置好的 Loguru 日志输出对象

- [ ] **Step 1: 编写测试用例 `tests/test_config.py`**

```python
import os
import pytest
from src.common.config import Settings, get_settings

def test_settings_load_defaults(monkeypatch):
    monkeypatch.setenv("POSTGRES_DB", "quant_test")
    monkeypatch.setenv("REDIS_HOST", "127.0.0.1")
    settings = Settings()
    assert settings.POSTGRES_DB == "quant_test"
    assert settings.REDIS_HOST == "127.0.0.1"
    assert settings.DATABASE_URL.startswith("postgresql+asyncpg://") or settings.DATABASE_URL.startswith("postgresql://")
```

- [ ] **Step 2: 运行测试并验证失败**

Run: `pytest tests/test_config.py -v`
Expected: FAIL with ModuleNotFoundError: No module named 'src'

- [ ] **Step 3: 编写依赖清单 `requirements.txt` 与 Docker 编排文件**

创建 `docker/docker-compose.yml`：
```yaml
version: '3.8'

services:
  postgres:
    image: postgres:16-alpine
    container_name: quant_postgres
    restart: always
    environment:
      POSTGRES_USER: ${POSTGRES_USER:-quant_user}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:-quant_pass_2026}
      POSTGRES_DB: ${POSTGRES_DB:-quant_db}
    ports:
      - "${POSTGRES_PORT:-5432}:5432"
    volumes:
      - ../data/postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER:-quant_user}"]
      interval: 5s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    container_name: quant_redis
    restart: always
    ports:
      - "${REDIS_PORT:-6379}:6379"
    volumes:
      - ../data/redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 5s
      timeout: 5s
      retries: 5
```

- [ ] **Step 4: 实现 `src/common/config.py` 与 `src/common/logger.py`**

```python
from pydantic_settings import BaseSettings
from functools import lru_cache
from pathlib import Path

class Settings(BaseSettings):
    PROJECT_NAME: str = "QuantCopliot"
    ENV: str = "development"
    
    # Database
    POSTGRES_USER: str = "quant_user"
    POSTGRES_PASSWORD: str = "quant_pass_2026"
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "quant_db"
    
    # Redis
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    
    # Qlib Storage
    QLIB_DATA_DIR: str = str(Path(__file__).resolve().parents[2] / "data" / "qlib_data")
    
    # LLM Settings for RD-Agent
    LLM_API_KEY: str = ""
    LLM_BASE_URL: str = "https://api.openai.com/v1"
    LLM_MODEL: str = "gpt-4o"
    
    @property
    def DATABASE_URL(self) -> str:
        return f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
    
    class Config:
        env_file = ".env"
        extra = "ignore"

@lru_cache()
def get_settings() -> Settings:
    return Settings()
```

- [ ] **Step 5: 运行测试并验证通过**

Run: `pytest tests/test_config.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add docker/ config/ src/common/ requirements.txt tests/test_config.py
git commit -m "feat: setup infrastructure compose, config and logging modules"
```

---

### Task 2: 数据库 ORM 实体与 Redis 客户端封装

**Files:**
- Create: `src/common/db.py`
- Create: `src/common/redis_client.py`
- Create: `src/models/base.py`
- Create: `src/models/factor.py`
- Create: `src/models/sync_log.py`
- Create: `src/models/model_registry.py`
- Create: `src/models/backtest_record.py`
- Test: `tests/test_models.py`

**Interfaces:**
- Consumes: `src.common.config.Settings`
- Produces:
  - `src.common.db.get_db()`: 数据库 Session 工厂
  - `src.models.factor.FactorMetadata`: 因子定义 ORM
  - `src.common.redis_client.RedisClient`: 包含分布式锁与 KV 操作客户端

- [ ] **Step 1: 编写 ORM 模型测试用例 `tests/test_models.py`**

```python
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.models.base import Base
from src.models.factor import FactorMetadata
from src.models.sync_log import SyncLog

@pytest.fixture
def session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()

def test_create_factor(session):
    factor = FactorMetadata(
        name="rev_5d",
        expression="Ref($close, -5) / $close - 1",
        category="reversal",
        ic_mean=0.042,
        icir=0.65,
        created_by="RD-Agent"
    )
    session.add(factor)
    session.commit()
    assert factor.id is not None
    assert factor.name == "rev_5d"
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_models.py -v`
Expected: FAIL with ModuleNotFoundError

- [ ] **Step 3: 编写 SQLAlchemy ORM 实体与 DB/Redis 工具**

实现 `src/models/base.py`, `src/models/factor.py`, `src/models/sync_log.py`, `src/common/db.py`, `src/common/redis_client.py`。
- [ ] **Step 4: 运行测试并验证通过**

Run: `pytest tests/test_models.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/models/ src/common/db.py src/common/redis_client.py tests/test_models.py
git commit -m "feat: implement database ORM models and redis client"
```

---

### Task 3: WindPy 接口封装与数据采集清洗管道

**Files:**
- Create: `src/data_pipeline/wind_client.py`
- Create: `src/data_pipeline/collector.py`
- Create: `src/data_pipeline/transformer.py`
- Test: `tests/test_collector.py`

**Interfaces:**
- Consumes: `src.common.config.Settings`, `src.common.logger`
- Produces:
  - `src.data_pipeline.wind_client.WindClient`: Wind 连接管理器
  - `src.data_pipeline.collector.WindDataCollector`: 行情/日历/成分股抓取
  - `src.data_pipeline.transformer.DataTransformer`: 格式清洗与复权换算

- [ ] **Step 1: 编写数据采集与转换测试用例 `tests/test_collector.py` (含 Mock 机制)**

```python
import pandas as pd
import pytest
from unittest.mock import MagicMock
from src.data_pipeline.transformer import DataTransformer

def test_transform_raw_market_data():
    raw_df = pd.DataFrame({
        "SEC_CODE": ["000001.SZ", "000001.SZ"],
        "DATETIME": ["2026-09-11", "2026-09-12"],
        "OPEN": [10.0, 10.2],
        "HIGH": [10.5, 10.6],
        "LOW": [9.9, 10.1],
        "CLOSE": [10.3, 10.4],
        "VOLUME": [100000, 120000],
        "AMT": [1030000, 1248000],
        "ADJFACTOR": [1.0, 1.0]
    })
    transformed = DataTransformer.standardize_quotes(raw_df)
    assert "$close" in transformed.columns
    assert "$factor" in transformed.columns
    assert len(transformed) == 2
```

- [ ] **Step 2: 运行测试并验证失败**

Run: `pytest tests/test_collector.py -v`
Expected: FAIL

- [ ] **Step 3: 实现 `wind_client.py` 与 `transformer.py`**

在 `wind_client.py` 中处理 Wind 初始化与 Mock 降级支持；在 `transformer.py` 中完成字段对齐与前/后复权转换。
- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_collector.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/data_pipeline/ tests/test_collector.py
git commit -m "feat: implement windpy client and data transformer"
```

---

### Task 4: Qlib 原生二进制转码器与增量更新引擎

**Files:**
- Create: `src/data_pipeline/qlib_dumper.py`
- Create: `src/qlib_engine/initializer.py`
- Test: `tests/test_qlib_dumper.py`

**Interfaces:**
- Consumes: `src.data_pipeline.transformer`
- Produces:
  - `src.data_pipeline.qlib_dumper.QlibDumper.dump_all(...)`
  - `src.data_pipeline.qlib_dumper.QlibDumper.append_daily(...)`
  - `src.qlib_engine.initializer.init_qlib()`

- [ ] **Step 1: 编写 Qlib 转码与读取验证测试 `tests/test_qlib_dumper.py`**

```python
import tmp_path
import pandas as pd
from src.data_pipeline.qlib_dumper import QlibDumper

def test_qlib_dumper_writes_valid_bins(tmp_path):
    target_dir = tmp_path / "qlib_data"
    dumper = QlibDumper(str(target_dir))
    df = pd.DataFrame({
        "symbol": ["000001.SZ"] * 3,
        "date": pd.to_datetime(["2026-09-01", "2026-09-02", "2026-09-03"]),
        "$close": [10.0, 10.5, 10.3],
        "$volume": [1000.0, 1200.0, 1100.0]
    })
    dumper.dump_from_df(df, calendar=["2026-09-01", "2026-09-02", "2026-09-03"])
    assert (target_dir / "calendars" / "day.txt").exists()
    assert (target_dir / "instruments" / "all.txt").exists()
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_qlib_dumper.py -v`
Expected: FAIL

- [ ] **Step 3: 实现 `qlib_dumper.py` 二进制生成逻辑与 `initializer.py`**
- [ ] **Step 4: 运行测试验证通过**

Run: `pytest tests/test_qlib_dumper.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/data_pipeline/qlib_dumper.py src/qlib_engine/initializer.py tests/test_qlib_dumper.py
git commit -m "feat: implement qlib binary dump and dataset initialization"
```

---

### Task 5: Qlib 因子工程、模型训练与回测评估模块

**Files:**
- Create: `src/qlib_engine/factor_handler.py`
- Create: `src/qlib_engine/model_trainer.py`
- Create: `src/qlib_engine/backtest.py`
- Test: `tests/test_qlib_engine.py`

**Interfaces:**
- Consumes: `src.qlib_engine.initializer`
- Produces:
  - `FactorHandler.compute_factors(expressions)`
  - `ModelTrainer.train_lightgbm(...)`
  - `BacktestEngine.run_portfolio_backtest(...)` -> 返回绩效报告与净值曲线

- [ ] **Step 1: 编写 Qlib 因子计算与回测评估测试 `tests/test_qlib_engine.py`**
- [ ] **Step 2: 运行测试确认失败**
- [ ] **Step 3: 实现因子处理器、LightGBM 训练包装器与 Top-K 组合回测模拟器**
- [ ] **Step 4: 运行测试验证通过**
- [ ] **Step 5: Commit**

```bash
git add src/qlib_engine/ tests/test_qlib_engine.py
git commit -m "feat: implement qlib factor handler, model trainer and backtest engine"
```

---

### Task 6: RD-Agent 自动化因子研发子系统

**Files:**
- Create: `src/agent_research/prompt_templates.py`
- Create: `src/agent_research/hypothesis_agent.py`
- Create: `src/agent_research/factor_coder_agent.py`
- Create: `src/agent_research/sandbox_evaluator.py`
- Create: `src/agent_research/evolution_loop.py`
- Test: `tests/test_agent_research.py`

**Interfaces:**
- Consumes: `src.qlib_engine`, `src.models.factor`
- Produces:
  - `RDAgentLoop.run_exploration_cycle(goal, rounds)`: 执行研发循环并输出通过准入门槛的因子

- [ ] **Step 1: 编写 RD-Agent 闭环流程单测 `tests/test_agent_research.py` (Mock LLM)**
- [ ] **Step 2: 运行测试确认失败**
- [ ] **Step 3: 实现 Prompt、假设提出、因子代码生成、沙箱 IC 检验与自动注册逻辑**
- [ ] **Step 4: 运行测试验证通过**
- [ ] **Step 5: Commit**

```bash
git add src/agent_research/ tests/test_agent_research.py
git commit -m "feat: implement RD-Agent automated factor research subsystem"
```

---

### Task 7: 定时自动化调度总控

**Files:**
- Create: `src/tasks/jobs.py`
- Create: `src/tasks/scheduler.py`
- Test: `tests/test_tasks.py`

**Interfaces:**
- Consumes: `src.data_pipeline`, `src.qlib_engine`, `src.agent_research`
- Produces:
  - `start_scheduler()`: 启动 APScheduler，挂载收盘同步、日度打分与周末挖掘作业

- [ ] **Step 1: 编写调度任务触发测试 `tests/test_tasks.py`**
- [ ] **Step 2: 实现日常作业逻辑并集成 Redis 分布式防重锁**
- [ ] **Step 3: 运行测试验证通过**
- [ ] **Step 4: Commit**

```bash
git add src/tasks/ tests/test_tasks.py
git commit -m "feat: implement automated scheduled jobs with redis locks"
```

---

### Task 8: FastAPI 服务层、路由与系统启动入口

**Files:**
- Create: `src/service/schemas/data_schema.py`
- Create: `src/service/schemas/factor_schema.py`
- Create: `src/service/schemas/model_schema.py`
- Create: `src/service/routers/data_router.py`
- Create: `src/service/routers/factor_router.py`
- Create: `src/service/routers/model_router.py`
- Create: `src/service/routers/agent_router.py`
- Create: `src/service/app.py`
- Create: `main.py`
- Test: `tests/test_api.py`

**Interfaces:**
- Produces:
  - FastAPI App (`/api/v1/...`)
  - `python main.py` 一键启动命令

- [x] **Step 1: 编写 FastAPI 路由接口端到端测试 `tests/test_api.py` (TestClient)**
- [x] **Step 2: 运行测试确认失败**
- [x] **Step 3: 实现所有 RESTful 路由、Pydantic 请求体验证与统一异常拦截**
- [x] **Step 4: 编写 `main.py` 整合 FastAPI 与 Background Scheduler 启动**
- [x] **Step 5: 运行全量测试验证**

Run: `pytest tests/ -v`
Expected: 全部测试 PASS

- [x] **Step 6: Commit**

```bash
git add src/service/ main.py tests/test_api.py docs/superpowers/plans/2026-09-14-qlib-quant-service.md
git commit -m "feat: implement fastapi service layer and main entrypoint"
```
