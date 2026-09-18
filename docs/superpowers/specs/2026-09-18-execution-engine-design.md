# 智能算法交易执行中台与 A 股仿真撮合系统（Execution Engine & Paper Broker）设计规范

## 1. 概述与设计目标

### 1.1 背景与业务定位
在量化投研与资产管理闭环中，组合优化器（如 `ConvexOptimizer` 或 `allocate_industry_neutral`）负责输出标的组合的“目标持仓权重（Target Weights）”，风控中台（`Risk Engine`）负责合规底线防御。然而，从“宏观目标权重”到“微观真实成交”之间存在巨大的执行鸿沟：
1. **调仓差额转换问题**：当前组合持仓与目标权重之间的差额必须转换为符合 A 股现货规则（如 100 股整手、先卖后买回笼资金、T+1 可用持仓限制）的具体委托订单；
2. **大单市场冲击问题**：若直接以大单全额市价入场，极易造成严重的盘口冲击成本与滑点亏损，必须依赖算法拆单（TWAP 时间加权、VWAP 成交量加权）；
3. **仿真记账与风控联动闭环**：需要具备一个无需外部券商实盘柜台、纯 Python 运行的高保真 A 股 T+1 撮合记账器（Paper Broker），并在委托前后与已建成的 `PreTradeRiskChecker` 和 `CircuitBreakerManager` 实现无缝闭环联动。

### 1.2 核心目标与原则
1. **分层解耦架构**：拆单算法（Algo）、订单生成（Rebalance）、撮合网关（Gateway）与风控协调（Coordinator）严格分层，支持未来零成本切换实盘柜台（QMT/PTrade）；
2. **严格 A 股 T+1 现货交易机制**：支持买入当日冻结不可卖、次日自动解冻、卖出单向征收千分之0.5印花税、动态滑点与双向佣金；
3. **时序安全与资金保护**：调仓生成器严格遵循“先卖后买（Sell First, Then Buy）”原则，优先释放流动性现金；
4. **全链路实时风控闭环**：每笔拆单子单在进柜台前必须经过 `PreTradeRiskChecker` 事前校验，成交记账后自动刷新账户净值并触发 `CircuitBreakerManager` 日内回撤监控；
5. **100% 易测性与纯 Python 兼容**：无硬性第三方柜台依赖，跨平台全量单测秒级运行。

---

## 2. 系统拓扑与模块组织

### 2.1 目录结构
```text
src/execution_engine/
├── __init__.py                # 模块统一导出声明
├── models.py                  # 核心数据实体：Order, Trade, Position, AccountState, RebalancePlan
├── rebalance.py               # 再平衡调仓订单生成器 RebalanceOrderGenerator
├── algos/                     # 拆单算法池
│   ├── __init__.py
│   ├── base.py                # BaseExecutionAlgo 算法抽象基类
│   ├── direct.py              # DirectAlgo 小单直达算法
│   ├── twap.py                # TwapAlgo 时间加权均匀拆单算法
│   └── vwap.py                # VwapAlgo A股日内U型成交量加权拆单算法
├── gateway/                   # 柜台网关抽象与实现
│   ├── __init__.py
│   ├── base.py                # BaseBrokerGateway 抽象网关接口
│   ├── paper_broker.py        # PaperBroker A股高保真T+1仿真撮合器与账本
│   └── qmt_gateway.py         # QmtBrokerGateway 实盘柜台接口扩展骨架
└── coordinator.py             # 交易执行总调度协调器 ExecutionCoordinator

src/service/routers/
└── execution_router.py        # RESTful API 路由 (/api/v1/execution/...)

tests/
└── test_execution_engine.py   # 独立全量自动化集成测试
```

### 2.2 核心数据流闭环
```text
┌────────────────────────────────────────────────────────────────────────┐
│                        投资经理 / 策略优化器 / API                      │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ 传入 Target Weights (如 {"600519.SH": 0.08, ...})
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                      RebalanceOrderGenerator                           │
│  - 计算各标的目标市值与 100 股整手股数                                     │
│  - 对比当前 Account 持仓，生成差额订单                                     │
│  - 严格编排：先卖后买 (Sell Orders -> Buy Orders)                        │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                       ExecutionCoordinator                             │
│                                                                        │
│   ┌────────────────────────────────────────────────────────────────┐   │
│   │ 算法拆单池 (Algo Pool): Direct / TWAP / VWAP                   │   │
│   │ - 将大额母单精准切分为 N 笔整手子单                             │   │
│   └───────────────────────────────┬────────────────────────────────┘   │
│                                   │ 遍历每个子单                        │
│                                   ▼                                    │
│   ┌────────────────────────────────────────────────────────────────┐   │
│   │ PreTradeRiskChecker (事前风控拦截)                             │   │
│   │ - 校验：黑名单 / 单笔限额 / ADV流动性 / 集中度 / 现金 / 熔断状态 │   │
│   └───────────────────────────────┬────────────────────────────────┘   │
│                                   │ [放行 PASS]                        │
│                                   ▼                                    │
│   ┌────────────────────────────────────────────────────────────────┐   │
│   │ BaseBrokerGateway (当前为 PaperBroker 仿真撮合器)              │   │
│   │ - 计算动态滑点 & 撮合成交                                       │   │
│   │ - 扣除双向佣金 & 卖出单向印花税 (0.05%)                        │   │
│   │ - 记账：买入增加 frozen_volume (T+1), 卖出扣减 available_volume │   │
│   └───────────────────────────────┬────────────────────────────────┘   │
│                                   │ 最新成交与净值变动                  │
│                                   ▼                                    │
│   ┌────────────────────────────────────────────────────────────────┐   │
│   │ CircuitBreakerManager (事后日内熔断监控)                        │   │
│   │ - 刷新日内动态回撤，若触发 ORANGE/RED 立即熔断后续挂单           │   │
│   └────────────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 3. 核心数据模型契约 (`models.py`)

### 3.1 枚举定义
```python
from enum import Enum

class OrderDirection(str, Enum):
    BUY = "BUY"
    SELL = "SELL"

class OrderType(str, Enum):
    LIMIT = "LIMIT"
    MARKET = "MARKET"

class OrderStatus(str, Enum):
    PENDING = "PENDING"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    REJECTED = "REJECTED"
    CANCELLED = "CANCELLED"

class AlgoType(str, Enum):
    DIRECT = "DIRECT"
    TWAP = "TWAP"
    VWAP = "VWAP"
```

### 3.2 实体模型定义
1. **`Position`（持仓明细）**：
   - `symbol: str`
   - `total_volume: int`（总持股数）
   - `available_volume: int`（当日可用卖出股数）
   - `frozen_volume: int`（当日买入冻结股数，T+1 日方可解冻）
   - `avg_cost: float`（持仓均价）
   - `market_value: float`（最新市值）

2. **`AccountState`（账户资产状态）**：
   - `account_id: str = "default"`
   - `strategy_id: Optional[str] = None`
   - `total_equity: float`（总资产净值 = 现金 + 各持仓市值）
   - `available_cash: float`（可用可用买入资金）
   - `positions: Dict[str, Position]`
   - `updated_at: str`

3. **`Order`（委托订单）**：
   - `order_id: str`
   - `account_id: str`
   - `symbol: str`
   - `direction: OrderDirection`
   - `order_type: OrderType = OrderType.LIMIT`
   - `price: float`
   - `volume: int`（必须为正数）
   - `filled_volume: int = 0`
   - `filled_amount: float = 0.0`
   - `status: OrderStatus = OrderStatus.PENDING`
   - `algo_type: AlgoType = AlgoType.DIRECT`
   - `parent_order_id: Optional[str] = None`
   - `reject_reason: Optional[str] = None`
   - `created_at: str`

4. **`Trade`（成交明细回报）**：
   - `trade_id: str`
   - `order_id: str`
   - `account_id: str`
   - `symbol: str`
   - `direction: OrderDirection`
   - `price: float`（包含滑点后的实际成交价）
   - `volume: int`
   - `amount: float`（$Price \times Volume$）
   - `commission: float`（佣金费用）
   - `stamp_tax: float`（卖方印花税）
   - `executed_at: str`

5. **`RebalancePlan`（调仓计划）**：
   - `account_id: str`
   - `target_weights: Dict[str, float]`
   - `sell_orders: List[Order]`（先卖队列）
   - `buy_orders: List[Order]`（后买队列）
   - `estimated_cash_change: float`

---

## 4. 详细模块设计与算法规范

### 4.1 调仓再平衡订单生成器 (`rebalance.py`)

`RebalanceOrderGenerator` 负责桥接策略输出与实盘订单：
```python
class RebalanceOrderGenerator:
    @staticmethod
    def generate_plan(
        account: AccountState,
        target_weights: Dict[str, float],
        current_prices: Dict[str, float],
        algo_type: AlgoType = AlgoType.DIRECT
    ) -> RebalancePlan:
        ...
```
**生成规则**：
1. **总资产基准**：以 `account.total_equity` 作为计算基准；
2. **标的遍历与股数计算**：
   - 合并 `target_weights.keys()` 与 `account.positions.keys()` 的并集；
   - 目标市值 $TV_i = target\_weights.get(i, 0.0) \times total\_equity$；
   - 目标整手股数 $TS_i = \lfloor \frac{TV_i}{price_i \times 100} \rfloor \times 100$；
   - 当前股数 $CS_i = account.positions[i].total\_volume$（若无持仓则为 0）；
   - 差额股数 $\Delta S_i = TS_i - CS_i$；
3. **分流至买卖队列**：
   - 若 $\Delta S_i < 0$（需要减仓或清仓）：
     - 卖出股数 $SellVol = \min(|\Delta S_i|, account.positions[i].available\_volume)$；
     - 若 $SellVol > 0$，创建 `OrderDirection.SELL` 订单并推入 `sell_orders`；
   - 若 $\Delta S_i > 0$（需要买入加仓或建仓）：
     - 创建 `OrderDirection.BUY` 订单（股数为 $\Delta S_i$），推入 `buy_orders`；
4. **时序安全保证**：
   - `sell_orders` 必须优先于 `buy_orders` 执行，通过卖出回流可用现金。

---

### 4.2 拆单算法池 (`src/execution_engine/algos/`)

#### 4.2.1 算法基类 `BaseExecutionAlgo`
```python
class BaseExecutionAlgo(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def slice_order(self, parent_order: Order, ctx: Optional[Dict[str, Any]] = None) -> List[Order]:
        """将父单切分为若干子单列表"""
        pass
```

#### 4.2.2 具体算法实现
1. **`DirectAlgo`（直达算法）**：
   - 直接返回包含单个父单的列表（`[parent_order]`）；
2. **`TwapAlgo`（时间加权平均拆单）**：
   - 参数：`num_slices: int = 5`（默认拆分为 5 笔子单）；
   - 切分逻辑：
     - 单笔基础股数 $base = \lfloor \frac{total\_volume}{num\_slices \times 100} \rfloor \times 100$；
     - 前 $num\_slices - 1$ 笔子单均分配 $base$ 股；
     - 最后一笔子单分配剩余所有股数 $total\_volume - (num\_slices - 1) \times base$；
     - 每笔子单标记 `parent_order_id = parent_order.order_id`，`algo_type = AlgoType.TWAP`；
3. **`VwapAlgo`（成交量加权平均拆单）**：
   - 依据 A 股经典的 5 阶段日内 U 型成交量权重：
     $$\text{Profile} = [0.30, 0.15, 0.10, 0.15, 0.30]$$
     （代表早盘 9:30~10:00 活跃 30%，上午 10:00~11:30 15%，午后 13:00~14:00 10%，下午 14:00~14:30 15%，尾盘 14:30~15:00 抢筹 30%）；
   - 前 4 笔子单按 $Profile[i]$ 分配并向下圆整为 100 股整手，最后一笔承接余量，确保总股数完全精确匹配。

---

### 4.3 A 股高保真仿真撮合器 (`PaperBroker`)

继承自 `BaseBrokerGateway`，完整实现真实 A 股柜台的交易结算机制：
1. **参数配置**：
   - `commission_rate: float = 0.0002`（万分之二佣金，买卖均收）；
   - `stamp_tax_rate: float = 0.0005`（万分之五印花税，**仅在卖出方向单向征收**）；
   - `slippage_rate: float = 0.0005`（万分之五滑点）；
2. **撮合成交逻辑 `submit_order(order: Order) -> OrderExecutionResult`**：
   - **滑点价格校准**：
     - 买入成交价：$FillPrice = order.price \times (1.0 + slippage\_rate)$；
     - 卖出成交价：$FillPrice = order.price \times (1.0 - slippage\_rate)$；
   - **资金与可用量硬约束**：
     - 买单：需满足 $FillPrice \times volume \times (1 + commission\_rate) \le available\_cash$，不足则状态变为 `REJECTED`；
     - 卖单：需满足 $volume \le position.available\_volume$，不足则状态变为 `REJECTED`（严禁超卖，严禁卖出今日冻结股份）；
   - **资产划转与 T+1 记账**：
     - 买入成交：
       - `available_cash` 扣减 $成交额 + 佣金$；
       - 持仓 `total_volume` 增加，`frozen_volume` 增加（**T+1 冻结**，`available_volume` 不增加）；
     - 卖出成交：
       - `available_cash` 增加 $成交额 - 佣金 - 印花税$；
       - 持仓 `total_volume` 减少，`available_volume` 减少；
3. **跨日清算与解冻接口 `settle_overnight(account_id: Optional[str] = None)`**：
   - 模拟每日闭市后清算：
   - 遍历各标的持仓，将 `frozen_volume` 全部合并加至 `available_volume`，并将 `frozen_volume` 清零；
   - 模拟进入新交易日，昨日买入的股票全额变为可用持仓。

---

### 4.4 交易执行总协调器与风控全自动闭环 (`ExecutionCoordinator`)

`ExecutionCoordinator` 将所有部件组装为具备风控强免疫力的自动化流水线：
```python
class ExecutionCoordinator:
    def __init__(
        self,
        gateway: Optional[BaseBrokerGateway] = None,
        risk_checker: Optional[PreTradeRiskChecker] = None,
        circuit_breaker: Optional[CircuitBreakerManager] = None
    ):
        ...
```
**执行流水线**：
1. **再平衡生成**：调用 `RebalanceOrderGenerator.generate_plan(...)`；
2. **优先执行卖单队列**：
   - 对每笔卖单调用对应拆单算法（如 TWAP/VWAP）；
   - 每笔子单提交前调用 `risk_checker.check_order(sub_order)` 进行事前合规检查；
   - 若风控放行，提交 `gateway.submit_order(sub_order)` 成交，回笼资金；
3. **顺序执行买单队列**：
   - 同步调用拆单算法与 `risk_checker.check_order(sub_order)`；
   - 提交 `gateway.submit_order(sub_order)` 成交；
4. **实时熔断联动**：
   - 每次有订单成交后，获取账户最新动态权益 `total_equity`；
   - 调用 `circuit_breaker.update_equity(account_id, total_equity)`；
   - 若检测到触发了 `ORANGE_RESTRICT_BUY`（禁买限制）或 `RED_HALT`（全局熔断），协调器立即**自动终止并撤销所有后续待买订单**，并产生告警通知。

---

## 5. FastAPI 服务路由规范 (`src/service/routers/execution_router.py`)

在主应用中挂载：`app.include_router(execution_router, prefix="/api/v1/execution", tags=["Execution Engine"])`。

| 路径 | 方法 | 描述 | 请求体 / 参数 |
| :--- | :--- | :--- | :--- |
| `/api/v1/execution/rebalance` | POST | 提交目标权重执行调仓闭环 | `RebalanceRequest` (含 account_id, target_weights, algo_type, prices) |
| `/api/v1/execution/order` | POST | 提交单笔委托执行 | `OrderRequest` (含 symbol, direction, price, volume, algo_type) |
| `/api/v1/execution/account` | GET | 查询账户资金、持仓与 T+1 状态 | Query param: `account_id: str = "default"` |
| `/api/v1/execution/settle` | POST | 触发跨日清算解冻 T+1 持仓 | `{"account_id": "default"}` |
| `/api/v1/execution/trades` | GET | 查询账户历史成交回报记录 | Query param: `account_id: str = "default"` |

---

## 6. 验证与测试规范 (`tests/test_execution_engine.py`)

构建独立自动化集成测试集，确保以下核心场景 100% 验证通过：
1. **`test_models_and_enums`**：验证数据实体模型校验与状态枚举；
2. **`test_rebalance_order_generator`**：
   - 验证目标权重转换为整手股数；
   - 验证“先卖后买”时序编排；
   - 验证根据 T+1 可用持仓限制卖出上限；
3. **`test_algo_slicers`**：
   - 验证 `DirectAlgo` 单笔直达；
   - 验证 `TwapAlgo` 均匀切分与余数保全；
   - 验证 `VwapAlgo` 基于 U 型权重切分与余数保全；
4. **`test_paper_broker_spot_t1`**：
   - 验证买入当天计入 `frozen_volume`，不可在当日卖出；
   - 验证买入扣除佣金，卖出扣除佣金与单向千0.5印花税；
   - 验证滑点对买卖价格的实际调整；
   - 验证调用 `settle_overnight()` 后冻结持仓成功解冻为可用；
5. **`test_execution_coordinator_risk_closed_loop`**：
   - 验证子单被 `PreTradeRiskChecker`（如黑名单或超额）精准拦截；
   - 验证成交后最新资产正确刷新 `CircuitBreakerManager`；
   - 验证当熔断状态为 `ORANGE/RED` 时，买单全额被熔断拒绝；
6. **`test_execution_api_endpoints`**：使用 `starlette.testclient.TestClient` 验证 `/api/v1/execution/*` 全部端点的 HTTP 交互；
7. **全量回归验证**：确保原有的 60 项测试集全部通过，系统测试总量达到约 67 项且 100% 通过。
