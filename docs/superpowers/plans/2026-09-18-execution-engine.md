# Execution Engine & Paper Broker Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建生产级量化算法交易执行中台与高保真 A 股 T+1 仿真撮合系统（再平衡差额订单生成、TWAP/VWAP 算法拆单、A 股 T+1 仿真记账、与事前风控和日内熔断全自动闭环联动，以及 FastAPI RESTful 接口）。

**Architecture:** 采用分层解耦执行管道架构。优化器输出的目标权重由 `RebalanceOrderGenerator` 转化为整手买卖订单（先卖后买）；大单经由 `TwapAlgo`/`VwapAlgo` 切分；每笔子单进柜台前强制调用 `PreTradeRiskChecker`；在 `PaperBroker` 中高保真撮合，计提滑点、佣金与卖方印花税，并严格记录 T+1 冻结股份；成交后实时回调 `CircuitBreakerManager` 监控回撤与触发熔断。

**Tech Stack:** Python 3.11, Pydantic, FastAPI, Starlette TestClient, pytest, NumPy, Pandas, PreTradeRiskChecker, CircuitBreakerManager.

**Spec:** [docs/superpowers/specs/2026-09-18-execution-engine-design.md](file:///d:/BaiduSyncdisk/PycharmProjects/QuantCopliot/docs/superpowers/specs/2026-09-18-execution-engine-design.md)

## Global Constraints
- 纯 Python 开发，严格保持 Windows 与 Linux 跨平台兼容。
- 严格遵循 A 股现货规则：买入 100 股整手圆整、当日买入次日解冻（T+1）、卖方单向 0.05% 印花税。
- 测试运行环境固定使用：`& 'D:\Program Files\anaconda3\envs\python311\python.exe' -m pytest`。
- 保证既有 60 项全量自动化测试 100% 通过无回归破坏。

---

### Task 1: 核心数据模型与枚举 (`src/execution_engine/models.py`)

**Files:**
- Create: `src/execution_engine/models.py`
- Test: `tests/test_execution_models.py`

**Interfaces:**
- Consumes: Standard library `enum`, `typing`, `pydantic.BaseModel`
- Produces: `OrderDirection`, `OrderType`, `OrderStatus`, `AlgoType`, `Position`, `AccountState`, `Order`, `Trade`, `RebalancePlan`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_execution_models.py
from src.execution_engine.models import (
    OrderDirection, OrderType, OrderStatus, AlgoType,
    Position, AccountState, Order, Trade, RebalancePlan
)

def test_execution_models_instantiation():
    pos = Position(
        symbol="600519.SH",
        total_volume=500,
        available_volume=300,
        frozen_volume=200,
        avg_cost=1800.0,
        market_value=900000.0
    )
    assert pos.total_volume == 500
    assert pos.available_volume == 300
    assert pos.frozen_volume == 200

    acc = AccountState(
        account_id="acc_01",
        total_equity=1000000.0,
        available_cash=100000.0,
        positions={"600519.SH": pos},
        updated_at="2026-09-18T10:00:00"
    )
    assert acc.total_equity == 1000000.0

    order = Order(
        order_id="ord_01",
        account_id="acc_01",
        symbol="600519.SH",
        direction=OrderDirection.BUY,
        price=1800.0,
        volume=100,
        created_at="2026-09-18T10:00:00"
    )
    assert order.status == OrderStatus.PENDING
    assert order.algo_type == AlgoType.DIRECT
```

- [ ] **Step 2: Run test to verify it fails**

Run: `& 'D:\Program Files\anaconda3\envs\python311\python.exe' -m pytest tests/test_execution_models.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.execution_engine'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/execution_engine/models.py
from enum import Enum
from datetime import datetime
from typing import Optional, Dict, List
from pydantic import BaseModel, Field

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

class Position(BaseModel):
    symbol: str
    total_volume: int = 0
    available_volume: int = 0
    frozen_volume: int = 0
    avg_cost: float = 0.0
    market_value: float = 0.0

class AccountState(BaseModel):
    account_id: str = "default"
    strategy_id: Optional[str] = None
    total_equity: float = 0.0
    available_cash: float = 0.0
    positions: Dict[str, Position] = Field(default_factory=dict)
    updated_at: str = Field(default_factory=lambda: datetime.now().isoformat())

class Order(BaseModel):
    order_id: str
    account_id: str = "default"
    strategy_id: Optional[str] = None
    symbol: str
    direction: OrderDirection
    order_type: OrderType = OrderType.LIMIT
    price: float
    volume: int
    filled_volume: int = 0
    filled_amount: float = 0.0
    status: OrderStatus = OrderStatus.PENDING
    algo_type: AlgoType = AlgoType.DIRECT
    parent_order_id: Optional[str] = None
    reject_reason: Optional[str] = None
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())

class Trade(BaseModel):
    trade_id: str
    order_id: str
    account_id: str = "default"
    symbol: str
    direction: OrderDirection
    price: float
    volume: int
    amount: float
    commission: float = 0.0
    stamp_tax: float = 0.0
    executed_at: str = Field(default_factory=lambda: datetime.now().isoformat())

class RebalancePlan(BaseModel):
    account_id: str
    target_weights: Dict[str, float]
    sell_orders: List[Order] = Field(default_factory=list)
    buy_orders: List[Order] = Field(default_factory=list)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `& 'D:\Program Files\anaconda3\envs\python311\python.exe' -m pytest tests/test_execution_models.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/execution_engine/models.py tests/test_execution_models.py
git commit -m "feat(execution): 建立交易执行数据模型与状态枚举"
```

---

### Task 2: 调仓再平衡订单生成器 (`src/execution_engine/rebalance.py`)

**Files:**
- Create: `src/execution_engine/rebalance.py`
- Test: `tests/test_rebalance_generator.py`

**Interfaces:**
- Consumes: `src.execution_engine.models` (`AccountState`, `Position`, `Order`, `OrderDirection`, `RebalancePlan`)
- Produces: `RebalanceOrderGenerator`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_rebalance_generator.py
from src.execution_engine.models import AccountState, Position, OrderDirection
from src.execution_engine.rebalance import RebalanceOrderGenerator

def test_rebalance_order_generator():
    pos_600519 = Position(symbol="600519.SH", total_volume=300, available_volume=300, avg_cost=1000.0, market_value=300000.0)
    pos_000858 = Position(symbol="000858.SZ", total_volume=500, available_volume=500, avg_cost=100.0, market_value=50000.0)
    
    acc = AccountState(
        account_id="acc_test",
        total_equity=1000000.0,
        available_cash=650000.0,
        positions={"600519.SH": pos_600519, "000858.SZ": pos_000858}
    )

    # Target: 600519.SH -> 10% (100k, price 1000 => 100 shares, sell 200)
    # Target: 000858.SZ -> 0% (liquidate, sell 500)
    # Target: 601318.SH -> 20% (200k, price 50 => 4000 shares, buy 4000)
    target_weights = {
        "600519.SH": 0.10,
        "601318.SH": 0.20
    }
    prices = {
        "600519.SH": 1000.0,
        "000858.SZ": 100.0,
        "601318.SH": 50.0
    }

    plan = RebalanceOrderGenerator.generate_plan(acc, target_weights, prices)
    assert len(plan.sell_orders) == 2  # Sells: 600519 (200), 000858 (500)
    assert len(plan.buy_orders) == 1   # Buys: 601318 (4000)

    # Verify sell order details
    sell_symbols = {o.symbol: o.volume for o in plan.sell_orders}
    assert sell_symbols["600519.SH"] == 200
    assert sell_symbols["000858.SZ"] == 500

    # Verify buy order details
    assert plan.buy_orders[0].symbol == "601318.SH"
    assert plan.buy_orders[0].volume == 4000
    assert plan.buy_orders[0].direction == OrderDirection.BUY
```

- [ ] **Step 2: Run test to verify it fails**

Run: `& 'D:\Program Files\anaconda3\envs\python311\python.exe' -m pytest tests/test_rebalance_generator.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.execution_engine.rebalance'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/execution_engine/rebalance.py
import uuid
from typing import Dict, Optional
from src.execution_engine.models import (
    AccountState, Order, OrderDirection, OrderType, AlgoType, RebalancePlan
)

class RebalanceOrderGenerator:
    @staticmethod
    def generate_plan(
        account: AccountState,
        target_weights: Dict[str, float],
        current_prices: Dict[str, float],
        algo_type: AlgoType = AlgoType.DIRECT
    ) -> RebalancePlan:
        equity = account.total_equity
        sell_orders = []
        buy_orders = []

        all_symbols = set(target_weights.keys()).union(account.positions.keys())

        for sym in sorted(list(all_symbols)):
            price = current_prices.get(sym, 0.0)
            if price <= 0:
                continue

            target_weight = target_weights.get(sym, 0.0)
            target_mv = target_weight * equity
            target_shares = int((target_mv / price) // 100) * 100

            current_pos = account.positions.get(sym)
            current_shares = current_pos.total_volume if current_pos else 0
            available_shares = current_pos.available_volume if current_pos else 0

            delta = target_shares - current_shares

            if delta < 0:
                # Sell order (capped at available_volume for T+1 safety)
                sell_vol = min(abs(delta), available_shares)
                if sell_vol > 0:
                    sell_orders.append(Order(
                        order_id=f"ord_{uuid.uuid4().hex[:8]}",
                        account_id=account.account_id,
                        symbol=sym,
                        direction=OrderDirection.SELL,
                        order_type=OrderType.LIMIT,
                        price=price,
                        volume=sell_vol,
                        algo_type=algo_type
                    ))
            elif delta > 0:
                # Buy order (must be multiples of 100)
                buy_vol = int(delta // 100) * 100
                if buy_vol > 0:
                    buy_orders.append(Order(
                        order_id=f"ord_{uuid.uuid4().hex[:8]}",
                        account_id=account.account_id,
                        symbol=sym,
                        direction=OrderDirection.BUY,
                        order_type=OrderType.LIMIT,
                        price=price,
                        volume=buy_vol,
                        algo_type=algo_type
                    ))

        return RebalancePlan(
            account_id=account.account_id,
            target_weights=target_weights,
            sell_orders=sell_orders,
            buy_orders=buy_orders
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `& 'D:\Program Files\anaconda3\envs\python311\python.exe' -m pytest tests/test_rebalance_generator.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/execution_engine/rebalance.py tests/test_rebalance_generator.py
git commit -m "feat(execution): 实现再平衡调仓订单生成器 RebalanceOrderGenerator"
```

---

### Task 3: 算法拆单池与时间/量价切分 (`src/execution_engine/algos/`)

**Files:**
- Create: `src/execution_engine/algos/base.py`
- Create: `src/execution_engine/algos/direct.py`
- Create: `src/execution_engine/algos/twap.py`
- Create: `src/execution_engine/algos/vwap.py`
- Create: `src/execution_engine/algos/__init__.py`
- Test: `tests/test_execution_algos.py`

**Interfaces:**
- Consumes: `src.execution_engine.models` (`Order`, `AlgoType`)
- Produces: `BaseExecutionAlgo`, `DirectAlgo`, `TwapAlgo`, `VwapAlgo`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_execution_algos.py
from src.execution_engine.models import Order, OrderDirection, AlgoType
from src.execution_engine.algos import DirectAlgo, TwapAlgo, VwapAlgo

def test_algos_slicing():
    parent = Order(
        order_id="p_01",
        symbol="600519.SH",
        direction=OrderDirection.BUY,
        price=100.0,
        volume=1000
    )

    # 1. DirectAlgo
    direct = DirectAlgo()
    slices_direct = direct.slice_order(parent)
    assert len(slices_direct) == 1
    assert slices_direct[0].volume == 1000

    # 2. TwapAlgo (5 slices of 200)
    twap = TwapAlgo(num_slices=5)
    slices_twap = twap.slice_order(parent)
    assert len(slices_twap) == 5
    assert sum(s.volume for s in slices_twap) == 1000
    assert all(s.volume % 100 == 0 for s in slices_twap)
    assert all(s.algo_type == AlgoType.TWAP for s in slices_twap)

    # 3. VwapAlgo (U-shape profile)
    vwap = VwapAlgo(volume_profile=[0.30, 0.15, 0.10, 0.15, 0.30])
    slices_vwap = vwap.slice_order(parent)
    assert len(slices_vwap) == 5
    assert sum(s.volume for s in slices_vwap) == 1000
    assert all(s.volume % 100 == 0 for s in slices_vwap)
    assert slices_vwap[0].volume == 300
    assert slices_vwap[1].volume == 100 or slices_vwap[1].volume == 200
```

- [ ] **Step 2: Run test to verify it fails**

Run: `& 'D:\Program Files\anaconda3\envs\python311\python.exe' -m pytest tests/test_execution_algos.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.execution_engine.algos'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/execution_engine/algos/base.py
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from src.execution_engine.models import Order

class BaseExecutionAlgo(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def slice_order(self, parent_order: Order, ctx: Optional[Dict[str, Any]] = None) -> List[Order]:
        pass
```

```python
# src/execution_engine/algos/direct.py
from typing import List, Optional, Dict, Any
from src.execution_engine.models import Order, AlgoType
from src.execution_engine.algos.base import BaseExecutionAlgo

class DirectAlgo(BaseExecutionAlgo):
    def __init__(self):
        super().__init__(name="DIRECT")

    def slice_order(self, parent_order: Order, ctx: Optional[Dict[str, Any]] = None) -> List[Order]:
        sub = parent_order.model_copy()
        sub.algo_type = AlgoType.DIRECT
        return [sub]
```

```python
# src/execution_engine/algos/twap.py
import uuid
from typing import List, Optional, Dict, Any
from src.execution_engine.models import Order, AlgoType
from src.execution_engine.algos.base import BaseExecutionAlgo

class TwapAlgo(BaseExecutionAlgo):
    def __init__(self, num_slices: int = 5):
        super().__init__(name="TWAP")
        self.num_slices = max(1, num_slices)

    def slice_order(self, parent_order: Order, ctx: Optional[Dict[str, Any]] = None) -> List[Order]:
        total_vol = parent_order.volume
        if self.num_slices <= 1 or total_vol <= 100:
            sub = parent_order.model_copy()
            sub.algo_type = AlgoType.TWAP
            return [sub]

        base_vol = int((total_vol / self.num_slices) // 100) * 100
        slices = []
        allocated = 0

        for i in range(self.num_slices - 1):
            if allocated + base_vol <= total_vol:
                vol = base_vol
            else:
                vol = 0
            allocated += vol
            slices.append(Order(
                order_id=f"sub_twap_{uuid.uuid4().hex[:8]}",
                account_id=parent_order.account_id,
                symbol=parent_order.symbol,
                direction=parent_order.direction,
                order_type=parent_order.order_type,
                price=parent_order.price,
                volume=vol,
                algo_type=AlgoType.TWAP,
                parent_order_id=parent_order.order_id
            ))

        # Remainder in last slice
        last_vol = total_vol - allocated
        slices.append(Order(
            order_id=f"sub_twap_{uuid.uuid4().hex[:8]}",
            account_id=parent_order.account_id,
            symbol=parent_order.symbol,
            direction=parent_order.direction,
            order_type=parent_order.order_type,
            price=parent_order.price,
            volume=last_vol,
            algo_type=AlgoType.TWAP,
            parent_order_id=parent_order.order_id
        ))

        return [s for s in slices if s.volume > 0]
```

```python
# src/execution_engine/algos/vwap.py
import uuid
from typing import List, Optional, Dict, Any
from src.execution_engine.models import Order, AlgoType
from src.execution_engine.algos.base import BaseExecutionAlgo

class VwapAlgo(BaseExecutionAlgo):
    DEFAULT_PROFILE = [0.30, 0.15, 0.10, 0.15, 0.30]

    def __init__(self, volume_profile: Optional[List[float]] = None):
        super().__init__(name="VWAP")
        raw_profile = volume_profile or self.DEFAULT_PROFILE
        total_p = sum(raw_profile)
        self.profile = [p / total_p for p in raw_profile] if total_p > 0 else [1.0]

    def slice_order(self, parent_order: Order, ctx: Optional[Dict[str, Any]] = None) -> List[Order]:
        total_vol = parent_order.volume
        n = len(self.profile)
        if n <= 1 or total_vol <= 100:
            sub = parent_order.model_copy()
            sub.algo_type = AlgoType.VWAP
            return [sub]

        slices = []
        allocated = 0

        for i in range(n - 1):
            ratio = self.profile[i]
            vol = int((total_vol * ratio) // 100) * 100
            allocated += vol
            slices.append(Order(
                order_id=f"sub_vwap_{uuid.uuid4().hex[:8]}",
                account_id=parent_order.account_id,
                symbol=parent_order.symbol,
                direction=parent_order.direction,
                order_type=parent_order.order_type,
                price=parent_order.price,
                volume=vol,
                algo_type=AlgoType.VWAP,
                parent_order_id=parent_order.order_id
            ))

        last_vol = total_vol - allocated
        slices.append(Order(
            order_id=f"sub_vwap_{uuid.uuid4().hex[:8]}",
            account_id=parent_order.account_id,
            symbol=parent_order.symbol,
            direction=parent_order.direction,
            order_type=parent_order.order_type,
            price=parent_order.price,
            volume=last_vol,
            algo_type=AlgoType.VWAP,
            parent_order_id=parent_order.order_id
        ))

        return [s for s in slices if s.volume > 0]
```

```python
# src/execution_engine/algos/__init__.py
from src.execution_engine.algos.base import BaseExecutionAlgo
from src.execution_engine.algos.direct import DirectAlgo
from src.execution_engine.algos.twap import TwapAlgo
from src.execution_engine.algos.vwap import VwapAlgo

__all__ = ["BaseExecutionAlgo", "DirectAlgo", "TwapAlgo", "VwapAlgo"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `& 'D:\Program Files\anaconda3\envs\python311\python.exe' -m pytest tests/test_execution_algos.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/execution_engine/algos/ tests/test_execution_algos.py
git commit -m "feat(execution): 实现拆单算法池 Direct, TWAP 与 VWAP"
```

---

### Task 4: 柜台网关抽象与 A 股 T+1 高保真仿真撮合器 (`src/execution_engine/gateway/`)

**Files:**
- Create: `src/execution_engine/gateway/base.py`
- Create: `src/execution_engine/gateway/paper_broker.py`
- Create: `src/execution_engine/gateway/qmt_gateway.py`
- Create: `src/execution_engine/gateway/__init__.py`
- Test: `tests/test_paper_broker.py`

**Interfaces:**
- Consumes: `src.execution_engine.models` (`Order`, `Trade`, `Position`, `AccountState`, `OrderDirection`, `OrderStatus`)
- Produces: `BaseBrokerGateway`, `PaperBroker`, `QmtBrokerGateway`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_paper_broker.py
from src.execution_engine.gateway.paper_broker import PaperBroker
from src.execution_engine.models import Order, OrderDirection, OrderStatus

def test_paper_broker_spot_t1():
    broker = PaperBroker(commission_rate=0.0002, stamp_tax_rate=0.0005, slippage_rate=0.0005)
    acc_id = "test_acc_t1"
    broker.create_account(account_id=acc_id, initial_cash=1000000.0)

    # 1. Buy order: 100 shares at price 100.0
    # Fill price with slippage: 100.0 * 1.0005 = 100.05
    # Amount = 100.05 * 100 = 10005.0. Commission = 10005 * 0.0002 = 2.001
    buy_order = Order(order_id="o_buy_01", account_id=acc_id, symbol="600519.SH", direction=OrderDirection.BUY, price=100.0, volume=100)
    trade_buy = broker.submit_order(buy_order)
    assert trade_buy is not None
    assert buy_order.status == OrderStatus.FILLED
    assert trade_buy.price == 100.05

    # Check position under T+1
    acc = broker.get_account(acc_id)
    pos = acc.positions["600519.SH"]
    assert pos.total_volume == 100
    assert pos.frozen_volume == 100
    assert pos.available_volume == 0  # CANNOT be sold today!

    # 2. Try to sell immediately today -> MUST be rejected due to 0 available_volume
    sell_order = Order(order_id="o_sell_01", account_id=acc_id, symbol="600519.SH", direction=OrderDirection.SELL, price=105.0, volume=100)
    trade_sell_fail = broker.submit_order(sell_order)
    assert trade_sell_fail is None
    assert sell_order.status == OrderStatus.REJECTED

    # 3. Simulate Overnight Settlement (Next day open)
    broker.settle_overnight(acc_id)
    acc_next_day = broker.get_account(acc_id)
    pos_next = acc_next_day.positions["600519.SH"]
    assert pos_next.frozen_volume == 0
    assert pos_next.available_volume == 100  # UNLOCKED!

    # 4. Sell on next day -> Succeeds!
    sell_order_ok = Order(order_id="o_sell_02", account_id=acc_id, symbol="600519.SH", direction=OrderDirection.SELL, price=105.0, volume=100)
    trade_sell = broker.submit_order(sell_order_ok)
    assert trade_sell is not None
    assert trade_sell.stamp_tax > 0  # Stamp tax charged on sell
    assert broker.get_account(acc_id).positions["600519.SH"].total_volume == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `& 'D:\Program Files\anaconda3\envs\python311\python.exe' -m pytest tests/test_paper_broker.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.execution_engine.gateway'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/execution_engine/gateway/base.py
from abc import ABC, abstractmethod
from typing import Optional, Dict
from src.execution_engine.models import Order, Trade, AccountState, Position

class BaseBrokerGateway(ABC):
    @abstractmethod
    def submit_order(self, order: Order) -> Optional[Trade]:
        pass

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        pass

    @abstractmethod
    def get_account(self, account_id: str) -> AccountState:
        pass
```

```python
# src/execution_engine/gateway/paper_broker.py
import uuid
from typing import Optional, Dict, List
from src.execution_engine.models import (
    Order, Trade, Position, AccountState, OrderDirection, OrderStatus
)
from src.execution_engine.gateway.base import BaseBrokerGateway
from src.common.logger import logger

class PaperBroker(BaseBrokerGateway):
    def __init__(
        self,
        commission_rate: float = 0.0002,
        stamp_tax_rate: float = 0.0005,
        slippage_rate: float = 0.0005
    ):
        self.commission_rate = commission_rate
        self.stamp_tax_rate = stamp_tax_rate
        self.slippage_rate = slippage_rate
        self.accounts: Dict[str, AccountState] = {}
        self.trades: List[Trade] = []

    def create_account(self, account_id: str, initial_cash: float = 1_000_000.0) -> AccountState:
        acc = AccountState(
            account_id=account_id,
            total_equity=initial_cash,
            available_cash=initial_cash,
            positions={}
        )
        self.accounts[account_id] = acc
        return acc

    def get_account(self, account_id: str) -> AccountState:
        if account_id not in self.accounts:
            self.create_account(account_id)
        acc = self.accounts[account_id]
        # Re-calc market value and total equity
        mv_total = sum(p.market_value for p in acc.positions.values())
        acc.total_equity = acc.available_cash + mv_total
        return acc

    def cancel_order(self, order_id: str) -> bool:
        return True

    def submit_order(self, order: Order) -> Optional[Trade]:
        acc = self.get_account(order.account_id)
        sym = order.symbol
        vol = order.volume
        price = order.price

        if order.direction == OrderDirection.BUY:
            fill_price = round(price * (1.0 + self.slippage_rate), 4)
            gross_amount = fill_price * vol
            commission = gross_amount * self.commission_rate
            total_cost = gross_amount + commission

            if total_cost > acc.available_cash:
                order.status = OrderStatus.REJECTED
                order.reject_reason = f"现金不足: 所需 {total_cost:.2f} 大于可用 {acc.available_cash:.2f}"
                return None

            # Execute Buy
            acc.available_cash -= total_cost
            pos = acc.positions.get(sym, Position(symbol=sym))
            new_total = pos.total_volume + vol
            new_cost = (pos.total_volume * pos.avg_cost + gross_amount) / new_total if new_total > 0 else fill_price
            pos.total_volume = new_total
            pos.frozen_volume += vol  # T+1 freeze
            pos.avg_cost = new_cost
            pos.market_value = new_total * fill_price
            acc.positions[sym] = pos

            order.status = OrderStatus.FILLED
            order.filled_volume = vol
            order.filled_amount = gross_amount

            trade = Trade(
                trade_id=f"trd_{uuid.uuid4().hex[:8]}",
                order_id=order.order_id,
                account_id=order.account_id,
                symbol=sym,
                direction=OrderDirection.BUY,
                price=fill_price,
                volume=vol,
                amount=gross_amount,
                commission=commission,
                stamp_tax=0.0
            )
            self.trades.append(trade)
            self.get_account(order.account_id)
            return trade

        elif order.direction == OrderDirection.SELL:
            pos = acc.positions.get(sym)
            if not pos or pos.available_volume < vol:
                avail = pos.available_volume if pos else 0
                order.status = OrderStatus.REJECTED
                order.reject_reason = f"可用持仓不足(T+1受限): 申报 {vol} 可用 {avail}"
                return None

            fill_price = round(price * (1.0 - self.slippage_rate), 4)
            gross_amount = fill_price * vol
            commission = gross_amount * self.commission_rate
            stamp_tax = gross_amount * self.stamp_tax_rate
            net_proceeds = gross_amount - commission - stamp_tax

            # Execute Sell
            acc.available_cash += net_proceeds
            pos.total_volume -= vol
            pos.available_volume -= vol
            pos.market_value = pos.total_volume * fill_price
            if pos.total_volume == 0:
                del acc.positions[sym]

            order.status = OrderStatus.FILLED
            order.filled_volume = vol
            order.filled_amount = gross_amount

            trade = Trade(
                trade_id=f"trd_{uuid.uuid4().hex[:8]}",
                order_id=order.order_id,
                account_id=order.account_id,
                symbol=sym,
                direction=OrderDirection.SELL,
                price=fill_price,
                volume=vol,
                amount=gross_amount,
                commission=commission,
                stamp_tax=stamp_tax
            )
            self.trades.append(trade)
            self.get_account(order.account_id)
            return trade

        return None

    def settle_overnight(self, account_id: Optional[str] = None) -> None:
        target_accounts = [self.get_account(account_id)] if account_id else list(self.accounts.values())
        for acc in target_accounts:
            for pos in acc.positions.values():
                pos.available_volume += pos.frozen_volume
                pos.frozen_volume = 0
            logger.info(f"Overnight settlement completed for account {acc.account_id} (T+1 shares unlocked)")
```

```python
# src/execution_engine/gateway/qmt_gateway.py
from typing import Optional
from src.execution_engine.models import Order, Trade, AccountState
from src.execution_engine.gateway.base import BaseBrokerGateway
from src.common.logger import logger

class QmtBrokerGateway(BaseBrokerGateway):
    """Placeholder gateway adapter for XtQuant / QMT counter API integration."""
    def __init__(self, mini_qmt_path: Optional[str] = None):
        self.mini_qmt_path = mini_qmt_path

    def submit_order(self, order: Order) -> Optional[Trade]:
        logger.info(f"[QmtGateway] Submitting order {order.order_id} to QMT counter.")
        return None

    def cancel_order(self, order_id: str) -> bool:
        return True

    def get_account(self, account_id: str) -> AccountState:
        return AccountState(account_id=account_id)
```

```python
# src/execution_engine/gateway/__init__.py
from src.execution_engine.gateway.base import BaseBrokerGateway
from src.execution_engine.gateway.paper_broker import PaperBroker
from src.execution_engine.gateway.qmt_gateway import QmtBrokerGateway

__all__ = ["BaseBrokerGateway", "PaperBroker", "QmtBrokerGateway"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `& 'D:\Program Files\anaconda3\envs\python311\python.exe' -m pytest tests/test_paper_broker.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/execution_engine/gateway/ tests/test_paper_broker.py
git commit -m "feat(execution): 实现 A 股 T+1 高保真仿真撮合器 PaperBroker"
```

---

### Task 5: 交易执行总协调器与风控全自动闭环 (`src/execution_engine/coordinator.py`)

**Files:**
- Create: `src/execution_engine/coordinator.py`
- Test: `tests/test_execution_coordinator.py`

**Interfaces:**
- Consumes: `src.execution_engine.models`, `src.execution_engine.rebalance`, `src.execution_engine.algos`, `src.execution_engine.gateway`, `src.risk_engine.pre_trade`, `src.risk_engine.circuit_breaker`
- Produces: `ExecutionCoordinator`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_execution_coordinator.py
from src.execution_engine.coordinator import ExecutionCoordinator
from src.execution_engine.gateway.paper_broker import PaperBroker
from src.risk_engine.pre_trade import PreTradeRiskChecker
from src.risk_engine.circuit_breaker import CircuitBreakerManager
from src.execution_engine.models import AlgoType, CircuitBreakerLevel

def test_execution_coordinator_rebalance_and_risk_loop():
    broker = PaperBroker()
    acc_id = "coord_acc"
    broker.create_account(acc_id, initial_cash=1000000.0)

    cb = CircuitBreakerManager()
    risk_checker = PreTradeRiskChecker(cb_manager=cb)
    # Add blacklisted symbol
    risk_checker.add_blacklist("000001.SZ")

    coord = ExecutionCoordinator(
        gateway=broker,
        risk_checker=risk_checker,
        circuit_breaker=cb
    )

    # Target: 600519.SH 5% (50k, price 500 => 100 shares), 000001.SZ 5% (blacklisted!)
    target_weights = {
        "600519.SH": 0.05,
        "000001.SZ": 0.05
    }
    prices = {
        "600519.SH": 500.0,
        "000001.SZ": 10.0
    }

    result = coord.execute_rebalance(
        account_id=acc_id,
        target_weights=target_weights,
        current_prices=prices,
        algo_type=AlgoType.DIRECT
    )

    # 600519 should be FILLED
    assert len(result["executed_trades"]) > 0
    assert result["executed_trades"][0].symbol == "600519.SH"

    # 000001.SZ should be intercepted by pre-trade risk checker!
    assert len(result["rejected_orders"]) > 0
    assert result["rejected_orders"][0].symbol == "000001.SZ"
    assert "黑名单" in result["rejected_orders"][0].reject_reason
```

- [ ] **Step 2: Run test to verify it fails**

Run: `& 'D:\Program Files\anaconda3\envs\python311\python.exe' -m pytest tests/test_execution_coordinator.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.execution_engine.coordinator'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/execution_engine/coordinator.py
from typing import Optional, Dict, Any, List
from src.execution_engine.models import (
    Order, Trade, AccountState, OrderDirection, OrderStatus, AlgoType, RebalancePlan
)
from src.execution_engine.rebalance import RebalanceOrderGenerator
from src.execution_engine.algos import BaseExecutionAlgo, DirectAlgo, TwapAlgo, VwapAlgo
from src.execution_engine.gateway.base import BaseBrokerGateway
from src.execution_engine.gateway.paper_broker import PaperBroker
from src.risk_engine.pre_trade import PreTradeRiskChecker
from src.risk_engine.circuit_breaker import CircuitBreakerManager
from src.risk_engine.models import OrderRiskRequest, CircuitBreakerLevel
from src.common.logger import logger

class ExecutionCoordinator:
    def __init__(
        self,
        gateway: Optional[BaseBrokerGateway] = None,
        risk_checker: Optional[PreTradeRiskChecker] = None,
        circuit_breaker: Optional[CircuitBreakerManager] = None
    ):
        self.gateway = gateway or PaperBroker()
        self.risk_checker = risk_checker or PreTradeRiskChecker()
        self.circuit_breaker = circuit_breaker or CircuitBreakerManager()
        self.algo_registry: Dict[AlgoType, BaseExecutionAlgo] = {
            AlgoType.DIRECT: DirectAlgo(),
            AlgoType.TWAP: TwapAlgo(),
            AlgoType.VWAP: VwapAlgo()
        }

    def execute_rebalance(
        self,
        account_id: str,
        target_weights: Dict[str, float],
        current_prices: Dict[str, float],
        algo_type: AlgoType = AlgoType.DIRECT
    ) -> Dict[str, Any]:
        acc = self.gateway.get_account(account_id)
        plan = RebalanceOrderGenerator.generate_plan(
            account=acc,
            target_weights=target_weights,
            current_prices=current_prices,
            algo_type=algo_type
        )

        algo = self.algo_registry.get(algo_type, DirectAlgo())
        executed_trades: List[Trade] = []
        rejected_orders: List[Order] = []

        # Sequence: SELL orders first, then BUY orders
        order_queue = plan.sell_orders + plan.buy_orders

        for parent_order in order_queue:
            # Check Circuit Breaker before processing
            cb_state = self.circuit_breaker.get_state(account_id)
            if cb_state.level == CircuitBreakerLevel.RED_HALT:
                parent_order.status = OrderStatus.REJECTED
                parent_order.reject_reason = "RED_HALT 熔断激活，全局终止交易"
                rejected_orders.append(parent_order)
                break
            elif cb_state.level == CircuitBreakerLevel.ORANGE_RESTRICT_BUY and parent_order.direction == OrderDirection.BUY:
                parent_order.status = OrderStatus.REJECTED
                parent_order.reject_reason = "ORANGE 熔断激活，禁止开新仓/加仓"
                rejected_orders.append(parent_order)
                continue

            # Slice order
            sub_orders = algo.slice_order(parent_order)

            for sub_order in sub_orders:
                # 1. Pre-trade Risk Check
                acc_now = self.gateway.get_account(account_id)
                curr_pos_vol = acc_now.positions.get(sub_order.symbol).total_volume if acc_now.positions.get(sub_order.symbol) else 0

                risk_req = OrderRiskRequest(
                    account_id=account_id,
                    order_id=sub_order.order_id,
                    symbol=sub_order.symbol,
                    direction=sub_order.direction.value,
                    price=sub_order.price,
                    volume=sub_order.volume,
                    current_position=curr_pos_vol,
                    total_equity=acc_now.total_equity,
                    available_cash=acc_now.available_cash
                )
                risk_res = self.risk_checker.check_order(risk_req)

                if not risk_res.passed:
                    sub_order.status = OrderStatus.REJECTED
                    sub_order.reject_reason = "; ".join(risk_res.violations)
                    rejected_orders.append(sub_order)
                    continue

                # 2. Submit to Gateway
                trade = self.gateway.submit_order(sub_order)
                if trade:
                    executed_trades.append(trade)
                    # 3. Post-trade Equity Update to Circuit Breaker
                    updated_acc = self.gateway.get_account(account_id)
                    self.circuit_breaker.update_equity(account_id, updated_acc.total_equity)
                else:
                    rejected_orders.append(sub_order)

        final_acc = self.gateway.get_account(account_id)
        return {
            "account_id": account_id,
            "executed_trades": executed_trades,
            "rejected_orders": rejected_orders,
            "final_equity": final_acc.total_equity,
            "final_cash": final_acc.available_cash,
            "circuit_breaker_level": self.circuit_breaker.get_state(account_id).level
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `& 'D:\Program Files\anaconda3\envs\python311\python.exe' -m pytest tests/test_execution_coordinator.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/execution_engine/coordinator.py tests/test_execution_coordinator.py
git commit -m "feat(execution): 实现交易执行总调度协调器与风控闭环 ExecutionCoordinator"
```

---

### Task 6: 模块主入口导出 (`src/execution_engine/__init__.py`)

**Files:**
- Create: `src/execution_engine/__init__.py`
- Test: `tests/test_execution_init.py`

**Interfaces:**
- Produces: Package exports for `src.execution_engine`

- [ ] **Step 1: Write test to verify exports**

```python
# tests/test_execution_init.py
def test_execution_engine_exports():
    from src.execution_engine import (
        OrderDirection, OrderType, OrderStatus, AlgoType,
        Position, AccountState, Order, Trade, RebalancePlan,
        RebalanceOrderGenerator,
        BaseExecutionAlgo, DirectAlgo, TwapAlgo, VwapAlgo,
        BaseBrokerGateway, PaperBroker, QmtBrokerGateway,
        ExecutionCoordinator
    )
    assert ExecutionCoordinator is not None
    assert PaperBroker is not None
```

- [ ] **Step 2: Write implementation**

```python
# src/execution_engine/__init__.py
from src.execution_engine.models import (
    OrderDirection,
    OrderType,
    OrderStatus,
    AlgoType,
    Position,
    AccountState,
    Order,
    Trade,
    RebalancePlan,
)
from src.execution_engine.rebalance import RebalanceOrderGenerator
from src.execution_engine.algos import (
    BaseExecutionAlgo,
    DirectAlgo,
    TwapAlgo,
    VwapAlgo,
)
from src.execution_engine.gateway import (
    BaseBrokerGateway,
    PaperBroker,
    QmtBrokerGateway,
)
from src.execution_engine.coordinator import ExecutionCoordinator

__all__ = [
    "OrderDirection",
    "OrderType",
    "OrderStatus",
    "AlgoType",
    "Position",
    "AccountState",
    "Order",
    "Trade",
    "RebalancePlan",
    "RebalanceOrderGenerator",
    "BaseExecutionAlgo",
    "DirectAlgo",
    "TwapAlgo",
    "VwapAlgo",
    "BaseBrokerGateway",
    "PaperBroker",
    "QmtBrokerGateway",
    "ExecutionCoordinator",
]
```

- [ ] **Step 3: Run test to verify it passes**

Run: `& 'D:\Program Files\anaconda3\envs\python311\python.exe' -m pytest tests/test_execution_init.py -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add src/execution_engine/__init__.py tests/test_execution_init.py
git commit -m "feat(execution): 建立执行引擎主包导出声明"
```

---

### Task 7: FastAPI 路由与主应用挂载 (`src/service/routers/execution_router.py` & `src/service/app.py`)

**Files:**
- Create: `src/service/routers/execution_router.py`
- Modify: `src/service/routers/__init__.py`
- Modify: `src/service/app.py`
- Test: `tests/test_execution_api.py`

**Interfaces:**
- Consumes: `src.execution_engine`
- Produces: FastAPI router mounted at `/api/v1/execution`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_execution_api.py
import pytest
from starlette.testclient import TestClient
from src.service.app import app

@pytest.fixture
def client():
    return TestClient(app)

def test_api_rebalance_and_account(client):
    # 1. Check account
    resp_acc = client.get("/api/v1/execution/account?account_id=api_exec_acc")
    assert resp_acc.status_code == 200
    assert resp_acc.json()["total_equity"] >= 0

    # 2. Execute rebalance
    payload = {
        "account_id": "api_exec_acc",
        "target_weights": {"600519.SH": 0.05},
        "current_prices": {"600519.SH": 500.0},
        "algo_type": "DIRECT"
    }
    resp_reb = client.post("/api/v1/execution/rebalance", json=payload)
    assert resp_reb.status_code == 200
    data = resp_reb.json()
    assert len(data["executed_trades"]) > 0

    # 3. Overnight settle
    resp_set = client.post("/api/v1/execution/settle", json={"account_id": "api_exec_acc"})
    assert resp_set.status_code == 200

    # 4. Trades
    resp_trd = client.get("/api/v1/execution/trades?account_id=api_exec_acc")
    assert resp_trd.status_code == 200
    assert len(resp_trd.json()) > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `& 'D:\Program Files\anaconda3\envs\python311\python.exe' -m pytest tests/test_execution_api.py -v`
Expected: FAIL with 404 Not Found

- [ ] **Step 3: Write minimal implementation**

```python
# src/service/routers/execution_router.py
from typing import Optional, Dict, Any, List
from fastapi import APIRouter, Query
from pydantic import BaseModel, Field
from src.execution_engine import (
    ExecutionCoordinator, PaperBroker, AlgoType, AccountState, Trade
)
from src.risk_engine import PreTradeRiskChecker, CircuitBreakerManager

router = APIRouter(prefix="/execution", tags=["Execution Engine"])

global_broker = PaperBroker()
global_coordinator = ExecutionCoordinator(gateway=global_broker)

class RebalanceRequest(BaseModel):
    account_id: str = "default"
    target_weights: Dict[str, float]
    current_prices: Dict[str, float]
    algo_type: str = "DIRECT"

class SettleRequest(BaseModel):
    account_id: str = "default"

@router.post("/rebalance")
def execute_rebalance(req: RebalanceRequest):
    algo = AlgoType(req.algo_type.upper()) if req.algo_type.upper() in AlgoType._value2member_map_ else AlgoType.DIRECT
    return global_coordinator.execute_rebalance(
        account_id=req.account_id,
        target_weights=req.target_weights,
        current_prices=req.current_prices,
        algo_type=algo
    )

@router.get("/account", response_model=AccountState)
def get_account(account_id: str = Query("default")):
    return global_broker.get_account(account_id)

@router.post("/settle")
def settle_overnight(req: SettleRequest):
    global_broker.settle_overnight(req.account_id)
    return {"status": "SUCCESS", "message": f"Overnight settlement completed for {req.account_id}"}

@router.get("/trades", response_model=List[Trade])
def get_trades(account_id: str = Query("default")):
    return [t for t in global_broker.trades if t.account_id == account_id]
```

Modify `src/service/routers/__init__.py` to export `execution_router`:
```python
from src.service.routers.execution_router import router as execution_router
__all__.append("execution_router")
```

Modify `src/service/app.py` to include `execution_router`:
```python
app.include_router(execution_router, prefix="/api/v1")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `& 'D:\Program Files\anaconda3\envs\python311\python.exe' -m pytest tests/test_execution_api.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/service/routers/execution_router.py src/service/routers/__init__.py src/service/app.py tests/test_execution_api.py
git commit -m "feat(service): 挂载算法交易与执行中台 RESTful 接口"
```

---

### Task 8: 全量自动化集成测试与基线回归验证 (`tests/test_execution_engine.py`)

**Files:**
- Create: `tests/test_execution_engine.py`
- Delete: Temporary modular test files (`tests/test_execution_models.py`, `tests/test_rebalance_generator.py`, `tests/test_execution_algos.py`, `tests/test_paper_broker.py`, `tests/test_execution_coordinator.py`, `tests/test_execution_init.py`, `tests/test_execution_api.py`) consolidated into `tests/test_execution_engine.py`.

- [ ] **Step 1: Run comprehensive execution tests**

Run: `& 'D:\Program Files\anaconda3\envs\python311\python.exe' -m pytest tests/test_execution_engine.py -v`
Expected: All execution tests PASS

- [ ] **Step 2: Run full regression suite across whole repo**

Run: `& 'D:\Program Files\anaconda3\envs\python311\python.exe' -m pytest tests -q`
Expected: All 60 original platform tests + all new execution tests (total ~67+) 100% PASS with 0 errors.

- [ ] **Step 3: Commit**

```bash
git add tests/
git commit -m "test(execution): 整理全量交易执行自动化测试套件并通过全量回归"
```
