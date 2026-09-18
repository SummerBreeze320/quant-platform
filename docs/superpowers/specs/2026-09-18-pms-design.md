# 机构级多策略与多账户资产管理系统（PMS: Portfolio Management System）设计规范

## 1. 概述与设计目标

### 1.1 背景与业务定位
在机构量化投资实践中，资产管理往往不是单一策略或单一账本的运作，而是**多策略、多账户协同运作的母子架构（Master-Sub Architecture）**：
- **母账户（Master Account）**：统一统筹总资金池，设定母级流动性预备金，向各个量化子策略进行资金配额与动态风险调配；
- **子策略账户（Strategy Account）**：各个策略独立运行（如：沪深300指增、中证500多因子、微盘Alpha、行业轮动），拥有独立的资金预算、交易持仓、净值核算与独立风控；
- **跨策略穿透式风险聚合（Look-Through Aggregation）**：不同策略可能会买入相同的个股或行业，单一策略在微观上均未超限，但合并计算后母账户层面的个股集中度或总杠杆可能严重突破红线。

本项目旨在为 QuantCopilot 打造一个生产级、机构化的多策略资产管理系统（PMS）：
1. **母子账户树状层级管理**：支持子策略动态注册、启停、资金划拨（Deposit / Withdrawal）与资产归集；
2. **多算法策略动态配资引擎**：支持等权（Equal Weight）、固定基准预算（Fixed Budget）、风险平价（Risk Parity / 波动率倒数加权）与夏普动态动量加权（Sharpe Ratio Weighting），并自动生成资金划转计划；
3. **全局穿透式持仓合并与宏观风控红线**：实时抵消合并跨策略重合标的，监控全平台单股真实集中度（如 $\le 15\%$）与母级总杠杆率，联动日内熔断状态机进行全平台总资产熔断保护；
4. **横向策略绩效对比与 RESTful 服务**：提供跨策略收益贡献度、夏普对比以及标准化 FastAPI 接口。

---

## 2. 系统拓扑与模块组织

### 2.1 目录结构
```text
src/pms/
├── __init__.py                # 模块统一导出声明
├── models.py                  # 核心数据模型：MasterAccount, StrategyAccount, CapitalAllocationPlan, ConsolidatedPortfolio
├── manager.py                 # 资产组合管理器 PortfolioManager (母子账户生命周期与资金划转)
├── allocator.py               # 策略配资引擎 CapitalAllocator (等权/固定/风险平价/夏普)
└── aggregator.py              # 全局穿透聚合器 PortfolioAggregator (合并持仓与母级宏观风控)

src/service/routers/
└── pms_router.py              # RESTful API 路由 (/api/v1/pms/...)

tests/
└── test_pms.py                # 独立全量自动化集成测试
```

### 2.2 核心业务流程图
```text
┌────────────────────────────────────────────────────────────────────────┐
│                        投资总监 / 资管投资决策委员会                     │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                        PortfolioManager                                │
│  - 管理 MasterAccount (总资产、预备金 reserve_cash)                     │
│  - 管理 StrategyAccount (多策略独立账本: CSI300指增 / CSI500多因子...)   │
└───────────────┬───────────────────────────────────────┬────────────────┘
                │                                       │
                │ 定期 / 调资触发                        │ 实时持仓数据归集
                ▼                                       ▼
┌────────────────────────────────┐    ┌──────────────────────────────────┐
│        CapitalAllocator        │    │       PortfolioAggregator        │
│  (多策略动态配资引擎)           │    │  (跨策略全局穿透式持仓聚合器)      │
├────────────────────────────────┤    ├──────────────────────────────────┤
│ - 等权配资 (Equal Weight)      │    │ - 穿透合并跨策略重叠标的持仓      │
│ - 固定预算 (Fixed Budget)      │    │ - 计算母账户总杠杆率              │
│ - 风险平价 (Risk Parity/1/σ)   │    │ - 监控全平台单股集中度 (<=15%)    │
│ - 夏普加权 (Sharpe Momentum)   │    │ - 联动 CircuitBreakerManager     │
│                                │    │   (触发全平台母级日内回撤熔断)     │
│ ──► 输出: CapitalAllocationPlan│    └──────────────────────────────────┘
└────────────────────────────────┘
```

---

## 3. 核心数据模型规范 (`models.py`)

### 3.1 枚举定义
```python
from enum import Enum

class StrategyType(str, Enum):
    ALPHA = "ALPHA"
    INDEX_ENHANCEMENT = "INDEX_ENHANCEMENT"
    SECTOR_ROTATION = "SECTOR_ROTATION"
    MOMENTUM = "MOMENTUM"

class AllocationMethod(str, Enum):
    EQUAL_WEIGHT = "EQUAL_WEIGHT"
    FIXED_BUDGET = "FIXED_BUDGET"
    RISK_PARITY = "RISK_PARITY"
    SHARPE_WEIGHTED = "SHARPE_WEIGHTED"
```

### 3.2 实体模型定义
1. **`StrategyAccount`（子策略账户）**：
   - `strategy_id: str`（唯一标识，如 `csi300_alpha`）
   - `name: str`（策略名称）
   - `strategy_type: StrategyType`
   - `allocated_budget: float`（当前分配的基础资金预算）
   - `current_cash: float`（当前可用现金）
   - `total_equity: float`（当前策略总权益 = 现金 + 各持仓市值）
   - `positions: Dict[str, Position]`（当前持仓字典，标的 -> Position）
   - `is_active: bool = True`

2. **`MasterAccount`（母账户）**：
   - `master_id: str = "master_default"`
   - `reserve_cash: float`（母账户未分配的预备金现金）
   - `total_equity: float`（母账户全局总资产 = reserve_cash + $\sum$ active_sub_equities）
   - `strategies: Dict[str, StrategyAccount]`

3. **`CashTransfer` 与 `CapitalAllocationPlan`**：
   - `CashTransfer`：
     - `strategy_id: str`
     - `transfer_amount: float`（正数表示母账户向子策略注资，负数表示子策略向母账户抽资）
     - `reason: str`
   - `CapitalAllocationPlan`：
     - `method: AllocationMethod`
     - `target_ratios: Dict[str, float]`（各子策略目标资金权重，和为 1.0）
     - `transfers: List[CashTransfer]`

4. **`ConsolidatedPosition` 与 `ConsolidatedPortfolio`**：
   - `ConsolidatedPosition`：
     - `symbol: str`
     - `total_volume: int`（全平台合并持股总量）
     - `total_market_value: float`（全平台合并总市值）
     - `weight_in_master: float`（占母账户总资产比例）
     - `contributing_strategies: List[str]`（参与持有的策略 ID 列表）
   - `ConsolidatedPortfolio`：
     - `master_id: str`
     - `total_equity: float`
     - `total_cash: float`
     - `total_market_value: float`
     - `leverage_ratio: float`
     - `positions: Dict[str, ConsolidatedPosition]`
     - `max_stock_concentration: float`
     - `top5_concentration: float`
     - `circuit_breaker_level: int`
     - `macro_alerts: List[str]`

5. **`StrategyPerformance`（策略绩效评估对比）**：
   - `strategy_id: str`
   - `name: str`
   - `allocated_budget: float`
   - `current_equity: float`
   - `cumulative_return: float`
   - `contribution_to_master: float`

---

## 4. 详细模块设计与算法规范

### 4.1 资产组合管理器 (`manager.py`)

`PortfolioManager` 负责管理母子账户的生命周期与资金划转：
1. **策略注册与初始化 `register_strategy(strategy_id, name, strategy_type, initial_budget)`**：
   - 检查 `master.reserve_cash >= initial_budget`；
   - 若足够，从 `master.reserve_cash` 扣除 `initial_budget`，创建并激活 `StrategyAccount`，其 `allocated_budget` 和 `current_cash` 初始化为 `initial_budget`；
2. **资金划拨与抽调 `transfer_cash(strategy_id, amount)`**：
   - 若 `amount > 0`（注资）：`master.reserve_cash -= amount`，`strategy.current_cash += amount`；
   - 若 `amount < 0`（抽资）：需保证 `strategy.current_cash >= |amount|`，`strategy.current_cash += amount`，`master.reserve_cash += |amount|`；
3. **与交易执行中台联动**：
   - 当策略需要执行调仓时，调用现有 `ExecutionCoordinator.execute_rebalance(account_id=strategy_id, ...)`；
   - 交易完成后，通过 `PaperBroker.get_account(strategy_id)` 自动同步最新持仓与可用现金至 `StrategyAccount`。

---

### 4.2 策略配资引擎 (`allocator.py`)

`CapitalAllocator` 包含 4 大资产分配算法模型：
1. **等权分配 (`EQUAL_WEIGHT`)**：
   $$w_i = \frac{1}{N}$$
2. **固定基准预算 (`FIXED_BUDGET`)**：
   依照配置的预设基准比例归一化分配；
3. **风险平价 / 波动率倒数加权 (`RISK_PARITY`)**：
   - 输入：各子策略的历史波动率字典 `strategy_volatilities: Dict[str, float]`；
   - 计算：
     $$w_i = \frac{1 / \sigma_i}{\sum_{j=1}^N (1 / \sigma_j)}$$
   - 波动率低、收益回撤表现稳健的策略获得更高的资本分配，使得各策略对母组合的方差风险贡献大致对等；
4. **夏普动态动量加权 (`SHARPE_WEIGHTED`)**：
   - 输入：各子策略的历史夏普比率字典 `strategy_sharpes: Dict[str, float]`；
   - 负夏普值保底截断为 0.01：
     $$w_i = \frac{\max(Sharpe_i, 0.01)}{\sum_{j=1}^N \max(Sharpe_j, 0.01)}$$
   - 业绩强劲、风险调整收益高的 Alpha 策略将获得更高的资金增量倾斜；
5. **资金调配划转计划生成 `generate_plan(...)`**：
   - 设定可投资金基数 $C_{invest} = MasterEquity \times investable\_ratio$（默认 95%）；
   - 各策略目标金额 $TargetCap_i = C_{invest} \times w_i$；
   - 划转差额 $\Delta C_i = TargetCap_i - StrategyEquity_i$；
   - 生成 `CapitalAllocationPlan`。

---

### 4.3 全局穿透聚合器与宏观风控红线 (`aggregator.py`)

`PortfolioAggregator` 负责全局穿透式合并与风控防御：
1. **跨策略标的合并计算**：
   - 对所有激活状态子策略的持仓进行字典合并；
   - 对于相同标的，累加总持股数 `total_volume` 与市值 `total_market_value`；
   - 计算该标的占母账户全局总资产比例：
     $$Weight\_in\_Master_k = \frac{\sum_{i} MarketValue_{i, k}}{MasterTotalEquity}$$
2. **宏观合规红线校验**：
   - **全平台单股集中度限制（默认 $\le 15\%$）**：
     - 若任意标的的 $Weight\_in\_Master > 0.15$，产生宏观风控报警 `CRITICAL` 告警；
   - **全平台总杠杆率限制（默认 $\le 1.0$）**：
     - 若 $\frac{TotalMarketValue}{MasterTotalEquity} > 1.0$，产生杠杆超标告警；
3. **母级日内回撤全局熔断联动**：
   - 计算母账户最新总权益 $MasterEquity = TotalCash + TotalMarketValue$；
   - 调用 `CircuitBreakerManager.update_equity(master_id, MasterEquity)`；
   - 当整个平台总资产日内回撤触发 `ORANGE` 或 `RED` 熔断时，产生全局熔断通知，并在宏观层面阻断所有子策略的新增开仓。

---

## 5. FastAPI 服务路由规范 (`src/service/routers/pms_router.py`)

在主应用中挂载：`app.include_router(pms_router, prefix="/api/v1/pms", tags=["Portfolio Management System"])`。

| 路径 | 方法 | 描述 | 请求体 / 参数 |
| :--- | :--- | :--- | :--- |
| `/api/v1/pms/strategies` | POST | 注册新的量化子策略账户 | `StrategyCreateRequest` (id, name, type, budget) |
| `/api/v1/pms/strategies` | GET | 查询所有子策略列表与绩效对比 | 无 |
| `/api/v1/pms/master` | GET | 查询母账户全局资产负债表 | Query param: `master_id` (默认 default) |
| `/api/v1/pms/allocate` | POST | 运行配资算法并执行资金划转 | `AllocationRunRequest` (method, metrics, apply) |
| `/api/v1/pms/lookthrough` | GET | 查询跨策略全局穿透式持仓与风控指标 | Query param: `master_id` |

---

## 6. 验证与测试规范 (`tests/test_pms.py`)

构建独立自动化集成测试集，确保以下核心场景 100% 验证通过：
1. **`test_pms_models_and_enums`**：验证数据模型约束与枚举合法性；
2. **`test_strategy_registration_and_cash_flow`**：验证母账户预备金充足时的成功注册、从母账户向子策略的资金划转与抽资逻辑；
3. **`test_capital_allocator_algorithms`**：
   - 验证 `EQUAL_WEIGHT` 等权分配精确性；
   - 验证 `RISK_PARITY` 波动率倒数权重计算；
   - 验证 `SHARPE_WEIGHTED` 夏普权重动态倾斜；
   - 验证生成的 `CapitalAllocationPlan` 划转差额准确无误；
4. **`test_portfolio_lookthrough_aggregation`**：
   - 构建策略 A（持有茅台 500 股）与策略 B（持有茅台 1000 股）的重叠持仓场景；
   - 验证穿透合并后茅台总股数为 1500 股，且 `contributing_strategies` 包含两个策略；
   - 验证母级单股集中度与总杠杆率计算；
5. **`test_master_macro_risk_overlay`**：
   - 验证合并持仓突破 15% 时的宏观风控拦截与报警；
   - 验证母账户总资产联动 `CircuitBreakerManager` 的熔断触发；
6. **`test_pms_api_endpoints`**：使用 `starlette.testclient.TestClient` 验证 `/api/v1/pms/*` 全部接口的 HTTP 交互；
7. **全量回归验证**：确保原有的 66 项测试集全部通过，系统全量测试总量达到约 72+ 项且 100% 通过。
