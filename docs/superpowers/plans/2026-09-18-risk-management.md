# Risk Management & Circuit Breaker Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建生产级/机构级量化风控中台系统（事前订单合规拦截、事后组合监控、动态三级梯级熔断机、告警分发中心与 FastAPI RESTful API）。

**Architecture:** 采用模块化可插拔规则管道架构。以纯 Python 内存微秒级检查为核心，无缝融合现有的 `BarraRiskModel` 因子风险模型与 `redis_client.py` 优雅降级机制；提供单向熔断不可降级保护、环形告警队列与 RESTful 端点。

**Tech Stack:** Python 3.11, Pydantic, FastAPI, Starlette TestClient, pytest, NumPy, Pandas, Qlib BarraRiskModel.

**Spec:** [docs/superpowers/specs/2026-09-18-risk-management-design.md](file:///d:/BaiduSyncdisk/PycharmProjects/QuantCopliot/docs/superpowers/specs/2026-09-18-risk-management-design.md)

## Global Constraints
- 纯 Python 开发，严格保证 Windows 与 Linux 跨平台兼容。
- 绝不引入硬外部强依赖（Redis 不可用时无条件降级回退纯内存安全运行）。
- 测试运行环境固定使用：`& 'D:\Program Files\anaconda3\envs\python311\python.exe' -m pytest`。
- 保证既有 53 项自动化单元测试 100% 通过无回归破坏。

---

### Task 1: 核心数据模型与枚举 (`src/risk_engine/models.py`)

**Files:**
- Create: `src/risk_engine/models.py`
- Test: `tests/test_risk_models.py`

**Interfaces:**
- Consumes: None (Standard library `enum`, `dataclasses`, `typing`, `pydantic.BaseModel`)
- Produces: `RiskAction`, `CircuitBreakerLevel`, `AlertLevel`, `OrderRiskRequest`, `OrderRiskResult`, `PortfolioRiskRequest`, `PortfolioRiskResult`, `CircuitBreakerState`, `RiskAlertEvent`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_risk_models.py
import pytest
from pydantic import ValidationError

def test_risk_models_instantiation():
    from src.risk_engine.models import (
        RiskAction, CircuitBreakerLevel, AlertLevel,
        OrderRiskRequest, OrderRiskResult,
        PortfolioRiskRequest, PortfolioRiskResult,
        CircuitBreakerState, RiskAlertEvent
    )

    # Test Enums
    assert RiskAction.PASS == "PASS"
    assert CircuitBreakerLevel.RED_HALT == 3
    assert AlertLevel.CRITICAL == "CRITICAL"

    # Test OrderRiskRequest
    req = OrderRiskRequest(
        account_id="acc_01",
        symbol="600519.SH",
        direction="BUY",
        price=1800.0,
        volume=100,
        total_equity=1000000.0,
        available_cash=500000.0
    )
    assert req.price == 1800.0
    assert req.direction == "BUY"

    # Test OrderRiskResult
    res = OrderRiskResult(
        passed=True,
        action=RiskAction.PASS,
        violations=[],
        warnings=[]
    )
    assert res.passed is True

    # Test CircuitBreakerState
    cb = CircuitBreakerState(
        account_id="acc_01",
        level=CircuitBreakerLevel.NORMAL,
        high_watermark=1000000.0,
        current_equity=1000000.0,
        max_drawdown=0.0
    )
    assert cb.level == CircuitBreakerLevel.NORMAL
```

- [ ] **Step 2: Run test to verify it fails**

Run: `& 'D:\Program Files\anaconda3\envs\python311\python.exe' -m pytest tests/test_risk_models.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.risk_engine'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/risk_engine/models.py
from enum import Enum, IntEnum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

class RiskAction(str, Enum):
    PASS = "PASS"
    WARN = "WARN"
    REJECT = "REJECT"

class CircuitBreakerLevel(IntEnum):
    NORMAL = 0
    YELLOW_WARN = 1
    ORANGE_RESTRICT_BUY = 2
    RED_HALT = 3

class AlertLevel(str, Enum):
    INFO = "INFO"
    WARN = "WARN"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class OrderRiskRequest(BaseModel):
    account_id: str = "default"
    order_id: Optional[str] = None
    symbol: str
    direction: str
    price: float
    volume: int
    current_position: int = 0
    total_equity: float
    available_cash: float
    adv_20: Optional[float] = None
    timestamp: Optional[str] = None

class OrderRiskResult(BaseModel):
    passed: bool
    action: RiskAction
    violations: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    metrics: Dict[str, Any] = Field(default_factory=dict)

class PortfolioRiskRequest(BaseModel):
    account_id: str = "default"
    total_equity: float
    positions: Dict[str, float] = Field(default_factory=dict)
    high_watermark: Optional[float] = None
    benchmark_weights: Optional[Dict[str, float]] = None
    factor_exposures: Optional[Dict[str, Dict[str, float]]] = None

class PortfolioRiskResult(BaseModel):
    account_id: str
    current_equity: float
    high_watermark: float
    drawdown: float
    circuit_breaker_level: CircuitBreakerLevel
    max_stock_weight: float
    top5_weight: float
    leverage_ratio: float
    active_risk: Optional[Dict[str, float]] = None
    alerts: List[str] = Field(default_factory=list)

class CircuitBreakerState(BaseModel):
    account_id: str
    level: CircuitBreakerLevel
    high_watermark: float
    current_equity: float
    max_drawdown: float
    last_triggered_at: Optional[str] = None
    trigger_reason: Optional[str] = None

class RiskAlertEvent(BaseModel):
    event_id: str
    timestamp: str
    level: AlertLevel
    account_id: str
    rule_name: str
    message: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `& 'D:\Program Files\anaconda3\envs\python311\python.exe' -m pytest tests/test_risk_models.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/risk_engine/models.py tests/test_risk_models.py
git commit -m "feat(risk): 建立风控模型与枚举定义"
```

---

### Task 2: 抽象风控规则基类与开箱即用规则集 (`src/risk_engine/rules.py`)

**Files:**
- Create: `src/risk_engine/rules.py`
- Test: `tests/test_risk_rules.py`

**Interfaces:**
- Consumes: `src.risk_engine.models` (`RiskAction`, `CircuitBreakerLevel`, `OrderRiskRequest`)
- Produces: `BaseRiskRule`, `BlacklistRule`, `MaxOrderAmountRule`, `MaxStockWeightRule`, `AdvLiquidityRule`, `CashSufficiencyRule`, `CircuitBreakerRule`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_risk_rules.py
from src.risk_engine.models import OrderRiskRequest, RiskAction, CircuitBreakerLevel
from src.risk_engine.rules import (
    BlacklistRule, MaxOrderAmountRule, MaxStockWeightRule,
    AdvLiquidityRule, CashSufficiencyRule, CircuitBreakerRule
)

def test_blacklist_rule():
    rule = BlacklistRule(blacklist={"000001.SZ", "600000.SH"})
    req_pass = OrderRiskRequest(symbol="600519.SH", direction="BUY", price=100.0, volume=100, total_equity=1e6, available_cash=1e5)
    action, msg = rule.check(req_pass, {})
    assert action == RiskAction.PASS
    assert msg is None

    req_reject = OrderRiskRequest(symbol="000001.SZ", direction="BUY", price=100.0, volume=100, total_equity=1e6, available_cash=1e5)
    action, msg = rule.check(req_reject, {})
    assert action == RiskAction.REJECT
    assert "黑名单" in msg

def test_max_order_amount_rule():
    rule = MaxOrderAmountRule(max_order_amount=500000.0)
    req_pass = OrderRiskRequest(symbol="600519.SH", direction="BUY", price=1000.0, volume=100, total_equity=1e6, available_cash=1e5) # 100k
    action, _ = rule.check(req_pass, {})
    assert action == RiskAction.PASS

    req_reject = OrderRiskRequest(symbol="600519.SH", direction="BUY", price=1000.0, volume=600, total_equity=1e6, available_cash=1e6) # 600k
    action, msg = rule.check(req_reject, {})
    assert action == RiskAction.REJECT
    assert "单笔限额" in msg

def test_max_stock_weight_rule():
    rule = MaxStockWeightRule(max_stock_weight=0.10) # max 10%
    # total equity 1,000,000. current pos 500 shares * 100 = 50k. buy 600 shares * 100 = 60k. post = 110k (11% > 10%)
    req_reject = OrderRiskRequest(symbol="600519.SH", direction="BUY", price=100.0, volume=600, current_position=500, total_equity=1e6, available_cash=1e5)
    action, msg = rule.check(req_reject, {})
    assert action == RiskAction.REJECT
    assert "集中度限制" in msg

def test_adv_liquidity_rule():
    rule = AdvLiquidityRule(max_adv_ratio=0.10)
    req = OrderRiskRequest(symbol="600519.SH", direction="BUY", price=100.0, volume=1500, total_equity=1e6, available_cash=1e5, adv_20=10000.0)
    action, msg = rule.check(req, {})
    assert action == RiskAction.REJECT
    assert "冲击成本上限" in msg

def test_cash_sufficiency_rule():
    rule = CashSufficiencyRule(fee_buffer_rate=0.002)
    # price 100 * volume 1000 = 100,000. available cash 90,000
    req = OrderRiskRequest(symbol="600519.SH", direction="BUY", price=100.0, volume=1000, total_equity=1e6, available_cash=90000.0)
    action, msg = rule.check(req, {})
    assert action == RiskAction.REJECT
    assert "可用现金不足" in msg
```

- [ ] **Step 2: Run test to verify it fails**

Run: `& 'D:\Program Files\anaconda3\envs\python311\python.exe' -m pytest tests/test_risk_rules.py -v`
Expected: FAIL with `ImportError: cannot import name 'BlacklistRule' from 'src.risk_engine.rules'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/risk_engine/rules.py
from abc import ABC, abstractmethod
from typing import Tuple, Optional, Dict, Any, Set
from src.risk_engine.models import RiskAction, CircuitBreakerLevel, OrderRiskRequest

class BaseRiskRule(ABC):
    def __init__(self, name: str, enabled: bool = True):
        self.name = name
        self.enabled = enabled

    @abstractmethod
    def check(self, req: OrderRiskRequest, ctx: Dict[str, Any]) -> Tuple[RiskAction, Optional[str]]:
        pass

class BlacklistRule(BaseRiskRule):
    def __init__(self, blacklist: Optional[Set[str]] = None, enabled: bool = True):
        super().__init__(name="BlacklistRule", enabled=enabled)
        self.blacklist: Set[str] = set(blacklist) if blacklist else set()

    def check(self, req: OrderRiskRequest, ctx: Dict[str, Any]) -> Tuple[RiskAction, Optional[str]]:
        if not self.enabled:
            return RiskAction.PASS, None
        if req.symbol in self.blacklist:
            return RiskAction.REJECT, f"标的 {req.symbol} 命中合规禁投黑名单"
        return RiskAction.PASS, None

class MaxOrderAmountRule(BaseRiskRule):
    def __init__(self, max_order_amount: float = 1_000_000.0, enabled: bool = True):
        super().__init__(name="MaxOrderAmountRule", enabled=enabled)
        self.max_order_amount = max_order_amount

    def check(self, req: OrderRiskRequest, ctx: Dict[str, Any]) -> Tuple[RiskAction, Optional[str]]:
        if not self.enabled:
            return RiskAction.PASS, None
        amount = req.price * req.volume
        if amount > self.max_order_amount:
            return RiskAction.REJECT, f"委托金额 {amount:.2f} 超过单笔限额 {self.max_order_amount:.2f}"
        return RiskAction.PASS, None

class MaxStockWeightRule(BaseRiskRule):
    def __init__(self, max_stock_weight: float = 0.10, enabled: bool = True):
        super().__init__(name="MaxStockWeightRule", enabled=enabled)
        self.max_stock_weight = max_stock_weight

    def check(self, req: OrderRiskRequest, ctx: Dict[str, Any]) -> Tuple[RiskAction, Optional[str]]:
        if not self.enabled:
            return RiskAction.PASS, None
        if req.direction.upper() == "BUY":
            if req.total_equity <= 0:
                return RiskAction.REJECT, "总权益为0或非正，禁止买入操作"
            post_mv = (req.current_position + req.volume) * req.price
            weight = post_mv / req.total_equity
            if weight > self.max_stock_weight:
                return RiskAction.REJECT, f"买入后持仓权重 {weight:.2%} 超过个股集中度限制 {self.max_stock_weight:.2%}"
        return RiskAction.PASS, None

class AdvLiquidityRule(BaseRiskRule):
    def __init__(self, max_adv_ratio: float = 0.10, enabled: bool = True):
        super().__init__(name="AdvLiquidityRule", enabled=enabled)
        self.max_adv_ratio = max_adv_ratio

    def check(self, req: OrderRiskRequest, ctx: Dict[str, Any]) -> Tuple[RiskAction, Optional[str]]:
        if not self.enabled:
            return RiskAction.PASS, None
        if req.adv_20 is not None and req.adv_20 > 0:
            ratio = req.volume / req.adv_20
            if ratio > self.max_adv_ratio:
                return RiskAction.REJECT, f"委托数量占过去20日日均成交量比例 {ratio:.2%} 超过冲击成本上限 {self.max_adv_ratio:.2%}"
        return RiskAction.PASS, None

class CashSufficiencyRule(BaseRiskRule):
    def __init__(self, fee_buffer_rate: float = 0.002, enabled: bool = True):
        super().__init__(name="CashSufficiencyRule", enabled=enabled)
        self.fee_buffer_rate = fee_buffer_rate

    def check(self, req: OrderRiskRequest, ctx: Dict[str, Any]) -> Tuple[RiskAction, Optional[str]]:
        if not self.enabled:
            return RiskAction.PASS, None
        if req.direction.upper() == "BUY":
            required_cash = req.price * req.volume * (1.0 + self.fee_buffer_rate)
            if required_cash > req.available_cash:
                return RiskAction.REJECT, f"可用现金 {req.available_cash:.2f} 不足支付预估费用 {required_cash:.2f}"
        return RiskAction.PASS, None

class CircuitBreakerRule(BaseRiskRule):
    def __init__(self, cb_manager: Optional[Any] = None, enabled: bool = True):
        super().__init__(name="CircuitBreakerRule", enabled=enabled)
        self.cb_manager = cb_manager

    def check(self, req: OrderRiskRequest, ctx: Dict[str, Any]) -> Tuple[RiskAction, Optional[str]]:
        if not self.enabled or not self.cb_manager:
            return RiskAction.PASS, None
        state = self.cb_manager.get_state(req.account_id)
        if state.level == CircuitBreakerLevel.RED_HALT:
            return RiskAction.REJECT, "账户处于 RED_HALT 紧急熔断状态，禁止一切交易"
        elif state.level == CircuitBreakerLevel.ORANGE_RESTRICT_BUY and req.direction.upper() == "BUY":
            return RiskAction.REJECT, "账户处于 ORANGE 限制买入熔断状态，禁止新开仓/加仓"
        return RiskAction.PASS, None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `& 'D:\Program Files\anaconda3\envs\python311\python.exe' -m pytest tests/test_risk_rules.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/risk_engine/rules.py tests/test_risk_rules.py
git commit -m "feat(risk): 实现可插拔风控规则抽象与预置规则池"
```

---

### Task 3: 动态阶梯熔断管理器 (`src/risk_engine/circuit_breaker.py`)

**Files:**
- Create: `src/risk_engine/circuit_breaker.py`
- Test: `tests/test_circuit_breaker.py`

**Interfaces:**
- Consumes: `src.risk_engine.models` (`CircuitBreakerLevel`, `CircuitBreakerState`)
- Produces: `CircuitBreakerManager`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_circuit_breaker.py
from src.risk_engine.circuit_breaker import CircuitBreakerManager
from src.risk_engine.models import CircuitBreakerLevel

def test_circuit_breaker_transitions():
    cb = CircuitBreakerManager(warn_drawdown=0.015, restrict_buy_drawdown=0.025, halt_drawdown=0.035)
    acc = "acc_test"

    # Initial equity 1,000,000 (drawdown = 0)
    state = cb.update_equity(acc, 1000000.0)
    assert state.level == CircuitBreakerLevel.NORMAL
    assert state.max_drawdown == 0.0

    # Drawdown 1.8% -> Yellow Warn
    state = cb.update_equity(acc, 982000.0)
    assert state.level == CircuitBreakerLevel.YELLOW_WARN

    # Drawdown 2.8% -> Orange Restrict Buy
    state = cb.update_equity(acc, 972000.0)
    assert state.level == CircuitBreakerLevel.ORANGE_RESTRICT_BUY

    # Drawdown 3.8% -> Red Halt
    state = cb.update_equity(acc, 962000.0)
    assert state.level == CircuitBreakerLevel.RED_HALT

    # Intraday bounce (equity recovers to 980,000, dd 2.0%)
    # One-way protection: MUST remain RED_HALT until manual reset
    state = cb.update_equity(acc, 980000.0)
    assert state.level == CircuitBreakerLevel.RED_HALT

    # Manual Reset
    reset_state = cb.reset(acc, reset_watermark=True)
    assert reset_state.level == CircuitBreakerLevel.NORMAL
    assert reset_state.high_watermark == 980000.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `& 'D:\Program Files\anaconda3\envs\python311\python.exe' -m pytest tests/test_circuit_breaker.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.risk_engine.circuit_breaker'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/risk_engine/circuit_breaker.py
import json
from datetime import datetime
from typing import Dict, Optional, Any
from src.risk_engine.models import CircuitBreakerLevel, CircuitBreakerState
from src.common.logger import logger

class CircuitBreakerManager:
    def __init__(
        self,
        warn_drawdown: float = 0.015,
        restrict_buy_drawdown: float = 0.025,
        halt_drawdown: float = 0.035,
        redis_client: Optional[Any] = None
    ):
        self.warn_drawdown = warn_drawdown
        self.restrict_buy_drawdown = restrict_buy_drawdown
        self.halt_drawdown = halt_drawdown
        self.redis_client = redis_client
        self._states: Dict[str, CircuitBreakerState] = {}

    def get_state(self, account_id: str) -> CircuitBreakerState:
        if account_id not in self._states:
            self._states[account_id] = CircuitBreakerState(
                account_id=account_id,
                level=CircuitBreakerLevel.NORMAL,
                high_watermark=0.0,
                current_equity=0.0,
                max_drawdown=0.0
            )
        return self._states[account_id]

    def update_equity(
        self,
        account_id: str,
        equity: float,
        high_watermark: Optional[float] = None
    ) -> CircuitBreakerState:
        state = self.get_state(account_id)
        if high_watermark is not None and high_watermark > 0:
            hw = max(high_watermark, equity)
        else:
            hw = max(state.high_watermark, equity)

        drawdown = (hw - equity) / hw if hw > 0 else 0.0
        max_dd = max(state.max_drawdown, drawdown)

        # Calculate tier
        if drawdown >= self.halt_drawdown:
            target_level = CircuitBreakerLevel.RED_HALT
            reason = f"日内回撤达到 {drawdown:.2%} 触碰全局硬熔断线 {self.halt_drawdown:.2%}"
        elif drawdown >= self.restrict_buy_drawdown:
            target_level = CircuitBreakerLevel.ORANGE_RESTRICT_BUY
            reason = f"日内回撤达到 {drawdown:.2%} 触碰禁买限制线 {self.restrict_buy_drawdown:.2%}"
        elif drawdown >= self.warn_drawdown:
            target_level = CircuitBreakerLevel.YELLOW_WARN
            reason = f"日内回撤达到 {drawdown:.2%} 触碰预警线 {self.warn_drawdown:.2%}"
        else:
            target_level = CircuitBreakerLevel.NORMAL
            reason = None

        # One-way ratchet: cannot automatically downgrade without manual reset
        new_level = max(state.level, target_level)
        triggered_at = datetime.now().isoformat() if new_level > state.level else state.last_triggered_at
        active_reason = reason if new_level > state.level else state.trigger_reason

        updated_state = CircuitBreakerState(
            account_id=account_id,
            level=new_level,
            high_watermark=hw,
            current_equity=equity,
            max_drawdown=max_dd,
            last_triggered_at=triggered_at,
            trigger_reason=active_reason
        )
        self._states[account_id] = updated_state
        self._sync_redis(account_id, updated_state)
        return updated_state

    def reset(self, account_id: str, reset_watermark: bool = True) -> CircuitBreakerState:
        state = self.get_state(account_id)
        hw = state.current_equity if (reset_watermark and state.current_equity > 0) else state.high_watermark
        new_state = CircuitBreakerState(
            account_id=account_id,
            level=CircuitBreakerLevel.NORMAL,
            high_watermark=hw,
            current_equity=state.current_equity,
            max_drawdown=0.0,
            last_triggered_at=None,
            trigger_reason="Manual reset"
        )
        self._states[account_id] = new_state
        self._sync_redis(account_id, new_state)
        logger.info(f"Circuit breaker reset for account {account_id}")
        return new_state

    def daily_reset(self) -> None:
        for acc in list(self._states.keys()):
            self.reset(acc, reset_watermark=True)

    def _sync_redis(self, account_id: str, state: CircuitBreakerState) -> None:
        if not self.redis_client:
            return
        try:
            client = self.redis_client.get_client() if hasattr(self.redis_client, "get_client") else self.redis_client
            if client:
                client.set(f"quant:risk:cb:{account_id}", state.model_dump_json(), ex=86400)
        except Exception as e:
            logger.debug(f"Redis cb sync skipped: {e}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `& 'D:\Program Files\anaconda3\envs\python311\python.exe' -m pytest tests/test_circuit_breaker.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/risk_engine/circuit_breaker.py tests/test_circuit_breaker.py
git commit -m "feat(risk): 建立三级动态熔断状态机与单向保护机制"
```

---

### Task 4: 告警中心与 Webhook 分发 (`src/risk_engine/alert.py`)

**Files:**
- Create: `src/risk_engine/alert.py`
- Test: `tests/test_risk_alert.py`

**Interfaces:**
- Consumes: `src.risk_engine.models` (`AlertLevel`, `RiskAlertEvent`)
- Produces: `RiskAlertManager`, `BaseWebhookAdapter`, `FeishuWebhookAdapter`, `DingTalkWebhookAdapter`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_risk_alert.py
from src.risk_engine.alert import RiskAlertManager, FeishuWebhookAdapter, DingTalkWebhookAdapter
from src.risk_engine.models import AlertLevel, RiskAlertEvent

def test_alert_manager_ring_buffer():
    mgr = RiskAlertManager(maxlen=5)
    for i in range(10):
        mgr.record_alert(
            event_id=f"evt_{i}",
            level=AlertLevel.WARN if i % 2 == 0 else AlertLevel.CRITICAL,
            account_id="acc_01",
            rule_name="TestRule",
            message=f"Alert message {i}"
        )

    alerts = mgr.get_recent_alerts(limit=10)
    assert len(alerts) == 5  # Ring buffer capped at 5
    assert alerts[-1].event_id == "evt_9"

    # Test filtering
    critical_alerts = mgr.get_recent_alerts(level=AlertLevel.CRITICAL)
    assert all(a.level == AlertLevel.CRITICAL for a in critical_alerts)

def test_webhook_adapters():
    evt = RiskAlertEvent(
        event_id="evt_01",
        timestamp="2026-09-18T10:00:00",
        level=AlertLevel.CRITICAL,
        account_id="acc_01",
        rule_name="CircuitBreakerRule",
        message="RED_HALT triggered"
    )
    feishu_payload = FeishuWebhookAdapter.format(evt)
    assert feishu_payload["msg_type"] == "interactive"

    ding_payload = DingTalkWebhookAdapter.format(evt)
    assert ding_payload["msgtype"] == "markdown"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `& 'D:\Program Files\anaconda3\envs\python311\python.exe' -m pytest tests/test_risk_alert.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.risk_engine.alert'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/risk_engine/alert.py
import uuid
from collections import deque
from datetime import datetime
from typing import Optional, List, Dict, Any
from src.risk_engine.models import AlertLevel, RiskAlertEvent
from src.common.logger import logger

class BaseWebhookAdapter:
    @staticmethod
    def format(event: RiskAlertEvent) -> Dict[str, Any]:
        return event.model_dump()

class FeishuWebhookAdapter(BaseWebhookAdapter):
    @staticmethod
    def format(event: RiskAlertEvent) -> Dict[str, Any]:
        color = "red" if event.level == AlertLevel.CRITICAL else "orange" if event.level == AlertLevel.WARN else "blue"
        return {
            "msg_type": "interactive",
            "card": {
                "header": {
                    "title": {"tag": "plain_text", "content": f"【风控告警】{event.rule_name}"},
                    "template": color
                },
                "elements": [
                    {
                        "tag": "div",
                        "text": {"tag": "lark_md", "content": f"**级别**: {event.level.value}\n**账户**: {event.account_id}\n**时间**: {event.timestamp}\n**详情**: {event.message}"}
                    }
                ]
            }
        }

class DingTalkWebhookAdapter(BaseWebhookAdapter):
    @staticmethod
    def format(event: RiskAlertEvent) -> Dict[str, Any]:
        return {
            "msgtype": "markdown",
            "markdown": {
                "title": f"风控告警: {event.rule_name}",
                "text": f"### 风控告警 - {event.level.value}\n- **账户**: {event.account_id}\n- **规则**: {event.rule_name}\n- **详情**: {event.message}\n- **时间**: {event.timestamp}"
            }
        }

class RiskAlertManager:
    def __init__(self, maxlen: int = 200, webhook_url: Optional[str] = None):
        self._history: deque = deque(maxlen=maxlen)
        self.webhook_url = webhook_url

    def record_alert(
        self,
        event_id: Optional[str] = None,
        level: AlertLevel = AlertLevel.WARN,
        account_id: str = "default",
        rule_name: str = "GeneralRisk",
        message: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ) -> RiskAlertEvent:
        evt = RiskAlertEvent(
            event_id=event_id or f"evt_{uuid.uuid4().hex[:8]}",
            timestamp=datetime.now().isoformat(),
            level=level,
            account_id=account_id,
            rule_name=rule_name,
            message=message,
            metadata=metadata or {}
        )
        self._history.append(evt)
        if level in (AlertLevel.CRITICAL, AlertLevel.ERROR):
            logger.error(f"[RISK-{level.value}] {rule_name}: {message} (Account: {account_id})")
        else:
            logger.warning(f"[RISK-{level.value}] {rule_name}: {message} (Account: {account_id})")

        self._dispatch_webhook(evt)
        return evt

    def get_recent_alerts(
        self,
        limit: int = 50,
        level: Optional[AlertLevel] = None
    ) -> List[RiskAlertEvent]:
        events = list(self._history)
        if level:
            events = [e for e in events if e.level == level]
        return events[-limit:]

    def _dispatch_webhook(self, event: RiskAlertEvent) -> None:
        if not self.webhook_url:
            return
        try:
            import urllib.request
            data = json.dumps(BaseWebhookAdapter.format(event)).encode("utf-8")
            req = urllib.request.Request(self.webhook_url, data=data, headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=2)
        except Exception as e:
            logger.debug(f"Webhook push failed (silent ignore): {e}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `& 'D:\Program Files\anaconda3\envs\python311\python.exe' -m pytest tests/test_risk_alert.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/risk_engine/alert.py tests/test_risk_alert.py
git commit -m "feat(risk): 实现环形历史告警队列与多平台 Webhook 适配"
```

---

### Task 5: 事前风控协调器 (`src/risk_engine/pre_trade.py`)

**Files:**
- Create: `src/risk_engine/pre_trade.py`
- Test: `tests/test_pre_trade.py`

**Interfaces:**
- Consumes: `src.risk_engine.models`, `src.risk_engine.rules`, `src.risk_engine.circuit_breaker`, `src.risk_engine.alert`
- Produces: `PreTradeRiskChecker`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pre_trade.py
from src.risk_engine.pre_trade import PreTradeRiskChecker
from src.risk_engine.circuit_breaker import CircuitBreakerManager
from src.risk_engine.models import OrderRiskRequest, RiskAction

def test_pre_trade_checker_pipeline():
    cb = CircuitBreakerManager()
    checker = PreTradeRiskChecker(cb_manager=cb)
    checker.add_blacklist("000001.SZ")

    # Order on blacklisted stock
    req = OrderRiskRequest(
        account_id="acc_01",
        symbol="000001.SZ",
        direction="BUY",
        price=10.0,
        volume=1000,
        total_equity=1e6,
        available_cash=5e5
    )
    result = checker.check_order(req)
    assert result.passed is False
    assert result.action == RiskAction.REJECT
    assert any("黑名单" in v for v in result.violations)

    # Clean order
    req_clean = OrderRiskRequest(
        account_id="acc_01",
        symbol="600519.SH",
        direction="BUY",
        price=100.0,
        volume=100,
        total_equity=1e6,
        available_cash=5e5
    )
    result_clean = checker.check_order(req_clean)
    assert result_clean.passed is True
    assert result_clean.action == RiskAction.PASS
```

- [ ] **Step 2: Run test to verify it fails**

Run: `& 'D:\Program Files\anaconda3\envs\python311\python.exe' -m pytest tests/test_pre_trade.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.risk_engine.pre_trade'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/risk_engine/pre_trade.py
from typing import List, Optional, Set, Dict, Any
from src.risk_engine.models import (
    RiskAction, AlertLevel, OrderRiskRequest, OrderRiskResult
)
from src.risk_engine.rules import (
    BaseRiskRule, BlacklistRule, MaxOrderAmountRule,
    MaxStockWeightRule, AdvLiquidityRule, CashSufficiencyRule,
    CircuitBreakerRule
)
from src.risk_engine.circuit_breaker import CircuitBreakerManager
from src.risk_engine.alert import RiskAlertManager

class PreTradeRiskChecker:
    def __init__(
        self,
        cb_manager: Optional[CircuitBreakerManager] = None,
        alert_manager: Optional[RiskAlertManager] = None,
        max_order_amount: float = 1_000_000.0,
        max_stock_weight: float = 0.10,
        max_adv_ratio: float = 0.10,
        fee_buffer_rate: float = 0.002
    ):
        self.cb_manager = cb_manager or CircuitBreakerManager()
        self.alert_manager = alert_manager or RiskAlertManager()
        self.blacklist_rule = BlacklistRule()
        self.rules: List[BaseRiskRule] = [
            self.blacklist_rule,
            CircuitBreakerRule(cb_manager=self.cb_manager),
            MaxOrderAmountRule(max_order_amount=max_order_amount),
            MaxStockWeightRule(max_stock_weight=max_stock_weight),
            AdvLiquidityRule(max_adv_ratio=max_adv_ratio),
            CashSufficiencyRule(fee_buffer_rate=fee_buffer_rate)
        ]

    def add_blacklist(self, symbol: str) -> None:
        self.blacklist_rule.blacklist.add(symbol)

    def remove_blacklist(self, symbol: str) -> None:
        self.blacklist_rule.blacklist.discard(symbol)

    def get_blacklist(self) -> Set[str]:
        return set(self.blacklist_rule.blacklist)

    def check_order(
        self,
        request: OrderRiskRequest,
        fail_fast: bool = False
    ) -> OrderRiskResult:
        violations: List[str] = []
        warnings: List[str] = []
        ctx: Dict[str, Any] = {}

        for rule in self.rules:
            if not rule.enabled:
                continue
            action, msg = rule.check(request, ctx)
            if action == RiskAction.REJECT:
                violations.append(f"[{rule.name}] {msg}")
                self.alert_manager.record_alert(
                    level=AlertLevel.CRITICAL,
                    account_id=request.account_id,
                    rule_name=rule.name,
                    message=msg or "Rejected",
                    metadata={"symbol": request.symbol, "volume": request.volume}
                )
                if fail_fast:
                    break
            elif action == RiskAction.WARN:
                warnings.append(f"[{rule.name}] {msg}")
                self.alert_manager.record_alert(
                    level=AlertLevel.WARN,
                    account_id=request.account_id,
                    rule_name=rule.name,
                    message=msg or "Warning",
                    metadata={"symbol": request.symbol}
                )

        if violations:
            final_action = RiskAction.REJECT
            passed = False
        elif warnings:
            final_action = RiskAction.WARN
            passed = True
        else:
            final_action = RiskAction.PASS
            passed = True

        metrics = {
            "order_amount": request.price * request.volume,
            "post_weight": ((request.current_position + request.volume) * request.price / request.total_equity) if request.total_equity > 0 else 0.0,
            "adv_ratio": (request.volume / request.adv_20) if (request.adv_20 and request.adv_20 > 0) else None
        }

        return OrderRiskResult(
            passed=passed,
            action=final_action,
            violations=violations,
            warnings=warnings,
            metrics=metrics
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `& 'D:\Program Files\anaconda3\envs\python311\python.exe' -m pytest tests/test_pre_trade.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/risk_engine/pre_trade.py tests/test_pre_trade.py
git commit -m "feat(risk): 实现事前风控协调器 PreTradeRiskChecker"
```

---

### Task 6: 事后组合监控与 Barra 因子模型集成 (`src/risk_engine/post_trade.py`)

**Files:**
- Create: `src/risk_engine/post_trade.py`
- Test: `tests/test_post_trade.py`

**Interfaces:**
- Consumes: `src.risk_engine.models`, `src.risk_engine.circuit_breaker`, `src.qlib_engine.risk_model.BarraRiskModel`
- Produces: `PostTradeRiskMonitor`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_post_trade.py
from src.risk_engine.post_trade import PostTradeRiskMonitor
from src.risk_engine.circuit_breaker import CircuitBreakerManager
from src.risk_engine.models import PortfolioRiskRequest, CircuitBreakerLevel

def test_post_trade_monitor():
    cb = CircuitBreakerManager(warn_drawdown=0.015, restrict_buy_drawdown=0.025, halt_drawdown=0.035)
    monitor = PostTradeRiskMonitor(cb_manager=cb)

    # 1. Total equity 1,000,000, positions sum to 800,000
    req = PortfolioRiskRequest(
        account_id="acc_01",
        total_equity=1000000.0,
        high_watermark=1000000.0,
        positions={"600519.SH": 200000.0, "000858.SZ": 150000.0, "601318.SH": 100000.0}
    )
    res = monitor.analyze_portfolio(req)
    assert res.drawdown == 0.0
    assert res.circuit_breaker_level == CircuitBreakerLevel.NORMAL
    assert res.max_stock_weight == 0.20 # 200k / 1M
    assert res.leverage_ratio == 0.45   # 450k / 1M

    # 2. Equity drops to 960,000 (dd = 4.0% > 3.5% -> RED_HALT)
    req_drawdown = PortfolioRiskRequest(
        account_id="acc_01",
        total_equity=960000.0,
        high_watermark=1000000.0,
        positions={"600519.SH": 180000.0}
    )
    res_dd = monitor.analyze_portfolio(req_drawdown)
    assert res_dd.drawdown == 0.04
    assert res_dd.circuit_breaker_level == CircuitBreakerLevel.RED_HALT
    assert len(res_dd.alerts) > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `& 'D:\Program Files\anaconda3\envs\python311\python.exe' -m pytest tests/test_post_trade.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.risk_engine.post_trade'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/risk_engine/post_trade.py
from typing import Optional, Dict, Any, List
import pandas as pd
import numpy as np
from src.risk_engine.models import (
    PortfolioRiskRequest, PortfolioRiskResult, CircuitBreakerLevel, AlertLevel
)
from src.risk_engine.circuit_breaker import CircuitBreakerManager
from src.risk_engine.alert import RiskAlertManager
from src.qlib_engine.risk_model import BarraRiskModel

class PostTradeRiskMonitor:
    def __init__(
        self,
        cb_manager: Optional[CircuitBreakerManager] = None,
        alert_manager: Optional[RiskAlertManager] = None,
        max_concentration_threshold: float = 0.20,
        max_top5_threshold: float = 0.60
    ):
        self.cb_manager = cb_manager or CircuitBreakerManager()
        self.alert_manager = alert_manager or RiskAlertManager()
        self.max_concentration_threshold = max_concentration_threshold
        self.max_top5_threshold = max_top5_threshold
        self.barra_model = BarraRiskModel()

    def analyze_portfolio(self, request: PortfolioRiskRequest) -> PortfolioRiskResult:
        equity = request.total_equity
        positions = request.positions
        alerts: List[str] = []

        # 1. Update circuit breaker state
        cb_state = self.cb_manager.update_equity(
            account_id=request.account_id,
            equity=equity,
            high_watermark=request.high_watermark
        )
        if cb_state.level > CircuitBreakerLevel.NORMAL:
            msg = f"账户熔断级别更新为 {cb_state.level.name}: {cb_state.trigger_reason}"
            alerts.append(msg)
            self.alert_manager.record_alert(
                level=AlertLevel.CRITICAL if cb_state.level == CircuitBreakerLevel.RED_HALT else AlertLevel.WARN,
                account_id=request.account_id,
                rule_name="CircuitBreaker",
                message=msg
            )

        # 2. Portfolio concentration metrics
        total_mv = sum(positions.values()) if positions else 0.0
        leverage_ratio = total_mv / equity if equity > 0 else 0.0

        weights = [mv / equity for mv in positions.values()] if equity > 0 else []
        max_stock_weight = max(weights) if weights else 0.0
        sorted_weights = sorted(weights, reverse=True)
        top5_weight = sum(sorted_weights[:5])

        if max_stock_weight > self.max_concentration_threshold:
            msg = f"单一持仓集中度 {max_stock_weight:.2%} 超过警示阈值 {self.max_concentration_threshold:.2%}"
            alerts.append(msg)
            self.alert_manager.record_alert(
                level=AlertLevel.WARN,
                account_id=request.account_id,
                rule_name="ConcentrationRisk",
                message=msg
            )

        if top5_weight > self.max_top5_threshold:
            msg = f"Top-5 持仓集中度 {top5_weight:.2%} 超过警示阈值 {self.max_top5_threshold:.2%}"
            alerts.append(msg)

        # 3. Optional Barra active risk calculation
        active_risk_metrics = None
        if request.benchmark_weights and request.factor_exposures and positions:
            active_risk_metrics = self._calc_barra_risk(request)

        return PortfolioRiskResult(
            account_id=request.account_id,
            current_equity=equity,
            high_watermark=cb_state.high_watermark,
            drawdown=(cb_state.high_watermark - equity) / cb_state.high_watermark if cb_state.high_watermark > 0 else 0.0,
            circuit_breaker_level=cb_state.level,
            max_stock_weight=max_stock_weight,
            top5_weight=top5_weight,
            leverage_ratio=leverage_ratio,
            active_risk=active_risk_metrics,
            alerts=alerts
        )

    def _calc_barra_risk(self, request: PortfolioRiskRequest) -> Optional[Dict[str, float]]:
        try:
            symbols = list(request.positions.keys())
            if not symbols:
                return None
            w = np.array([request.positions[s] / request.total_equity for s in symbols])
            bw = np.array([request.benchmark_weights.get(s, 0.0) for s in symbols])
            # Build mock/passed factor matrix
            factor_names = list(next(iter(request.factor_exposures.values())).keys())
            X = np.array([[request.factor_exposures[s].get(f, 0.0) for f in factor_names] for s in symbols])
            F = np.eye(len(factor_names)) * 0.0001
            delta = np.ones(len(symbols)) * 0.0004
            return self.barra_model.predict_portfolio_risk(
                weights=w,
                exposures=X,
                factor_cov=F,
                specific_var=delta,
                benchmark_weights=bw
            )
        except Exception:
            return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `& 'D:\Program Files\anaconda3\envs\python311\python.exe' -m pytest tests/test_post_trade.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/risk_engine/post_trade.py tests/test_post_trade.py
git commit -m "feat(risk): 实现事后组合监控与 Barra 因子风险联动"
```

---

### Task 7: 模块总入口导出 (`src/risk_engine/__init__.py`)

**Files:**
- Create: `src/risk_engine/__init__.py`
- Test: `tests/test_risk_engine_init.py`

**Interfaces:**
- Produces: Package exports for `src.risk_engine`

- [ ] **Step 1: Write test to verify clean exports**

```python
# tests/test_risk_engine_init.py
def test_risk_engine_exports():
    from src.risk_engine import (
        PreTradeRiskChecker, PostTradeRiskMonitor,
        CircuitBreakerManager, RiskAlertManager,
        OrderRiskRequest, OrderRiskResult,
        PortfolioRiskRequest, PortfolioRiskResult,
        CircuitBreakerLevel, RiskAction, AlertLevel
    )
    assert PreTradeRiskChecker is not None
    assert PostTradeRiskMonitor is not None
```

- [ ] **Step 2: Write implementation**

```python
# src/risk_engine/__init__.py
from src.risk_engine.models import (
    RiskAction,
    CircuitBreakerLevel,
    AlertLevel,
    OrderRiskRequest,
    OrderRiskResult,
    PortfolioRiskRequest,
    PortfolioRiskResult,
    CircuitBreakerState,
    RiskAlertEvent,
)
from src.risk_engine.rules import (
    BaseRiskRule,
    BlacklistRule,
    MaxOrderAmountRule,
    MaxStockWeightRule,
    AdvLiquidityRule,
    CashSufficiencyRule,
    CircuitBreakerRule,
)
from src.risk_engine.circuit_breaker import CircuitBreakerManager
from src.risk_engine.alert import RiskAlertManager
from src.risk_engine.pre_trade import PreTradeRiskChecker
from src.risk_engine.post_trade import PostTradeRiskMonitor

__all__ = [
    "RiskAction",
    "CircuitBreakerLevel",
    "AlertLevel",
    "OrderRiskRequest",
    "OrderRiskResult",
    "PortfolioRiskRequest",
    "PortfolioRiskResult",
    "CircuitBreakerState",
    "RiskAlertEvent",
    "BaseRiskRule",
    "BlacklistRule",
    "MaxOrderAmountRule",
    "MaxStockWeightRule",
    "AdvLiquidityRule",
    "CashSufficiencyRule",
    "CircuitBreakerRule",
    "CircuitBreakerManager",
    "RiskAlertManager",
    "PreTradeRiskChecker",
    "PostTradeRiskMonitor",
]
```

- [ ] **Step 3: Run test to verify it passes**

Run: `& 'D:\Program Files\anaconda3\envs\python311\python.exe' -m pytest tests/test_risk_engine_init.py -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add src/risk_engine/__init__.py tests/test_risk_engine_init.py
git commit -m "feat(risk): 建立风控模块主包导出声明"
```

---

### Task 8: FastAPI 风控路由与主应用挂载 (`src/service/routers/risk_router.py` & `src/service/app.py`)

**Files:**
- Create: `src/service/routers/risk_router.py`
- Modify: `src/service/app.py`
- Test: `tests/test_risk_api.py`

**Interfaces:**
- Consumes: `src.risk_engine`
- Produces: FastAPI router mounted at `/api/v1/risk`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_risk_api.py
import pytest
from starlette.testclient import TestClient
from src.service.app import app

@pytest.fixture
def client():
    return TestClient(app)

def test_api_check_order(client):
    payload = {
        "account_id": "api_acc_01",
        "symbol": "600519.SH",
        "direction": "BUY",
        "price": 1800.0,
        "volume": 100,
        "total_equity": 1000000.0,
        "available_cash": 500000.0
    }
    resp = client.post("/api/v1/risk/check_order", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["passed"] is True
    assert data["action"] == "PASS"

def test_api_circuit_breaker_status_and_reset(client):
    # Query status
    resp = client.get("/api/v1/risk/circuit_breaker/status?account_id=api_acc_01")
    assert resp.status_code == 200
    assert resp.json()["account_id"] == "api_acc_01"

    # Reset
    resp_reset = client.post("/api/v1/risk/circuit_breaker/reset", json={"account_id": "api_acc_01"})
    assert resp_reset.status_code == 200
    assert resp_reset.json()["level"] == 0

def test_api_blacklist(client):
    # Add
    resp_add = client.post("/api/v1/risk/blacklist", json={"action": "add", "symbols": ["300750.SZ"]})
    assert resp_add.status_code == 200
    assert "300750.SZ" in resp_add.json()["blacklist"]

    # Check order rejected
    payload = {
        "account_id": "api_acc_01",
        "symbol": "300750.SZ",
        "direction": "BUY",
        "price": 200.0,
        "volume": 100,
        "total_equity": 1000000.0,
        "available_cash": 500000.0
    }
    resp_check = client.post("/api/v1/risk/check_order", json=payload)
    assert resp_check.json()["passed"] is False

    # Remove
    resp_del = client.post("/api/v1/risk/blacklist", json={"action": "remove", "symbols": ["300750.SZ"]})
    assert "300750.SZ" not in resp_del.json()["blacklist"]

def test_api_alerts(client):
    resp = client.get("/api/v1/risk/alerts?limit=10")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `& 'D:\Program Files\anaconda3\envs\python311\python.exe' -m pytest tests/test_risk_api.py -v`
Expected: FAIL with 404 Not Found

- [ ] **Step 3: Write minimal implementation**

```python
# src/service/routers/risk_router.py
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, Query, Body
from pydantic import BaseModel
from src.risk_engine.models import (
    OrderRiskRequest, OrderRiskResult,
    PortfolioRiskRequest, PortfolioRiskResult,
    CircuitBreakerState, RiskAlertEvent, AlertLevel
)
from src.risk_engine.circuit_breaker import CircuitBreakerManager
from src.risk_engine.alert import RiskAlertManager
from src.risk_engine.pre_trade import PreTradeRiskChecker
from src.risk_engine.post_trade import PostTradeRiskMonitor

router = APIRouter()

# Singleton instances for API workers
global_cb = CircuitBreakerManager()
global_alert = RiskAlertManager()
global_checker = PreTradeRiskChecker(cb_manager=global_cb, alert_manager=global_alert)
global_monitor = PostTradeRiskMonitor(cb_manager=global_cb, alert_manager=global_alert)

class BlacklistRequest(BaseModel):
    action: str  # "add", "remove", "list"
    symbols: List[str] = []

class ResetRequest(BaseModel):
    account_id: str = "default"
    reset_watermark: bool = True

@router.post("/check_order", response_model=OrderRiskResult)
def check_order(req: OrderRiskRequest, fail_fast: bool = False):
    return global_checker.check_order(req, fail_fast=fail_fast)

@router.post("/check_portfolio", response_model=PortfolioRiskResult)
def check_portfolio(req: PortfolioRiskRequest):
    return global_monitor.analyze_portfolio(req)

@router.get("/circuit_breaker/status", response_model=CircuitBreakerState)
def get_circuit_breaker_status(account_id: str = Query("default")):
    return global_cb.get_state(account_id)

@router.post("/circuit_breaker/reset", response_model=CircuitBreakerState)
def reset_circuit_breaker(req: ResetRequest):
    return global_cb.reset(account_id=req.account_id, reset_watermark=req.reset_watermark)

@router.get("/alerts", response_model=List[RiskAlertEvent])
def get_alerts(limit: int = Query(50, ge=1, le=200), level: Optional[str] = Query(None)):
    alert_lvl = AlertLevel(level) if level in AlertLevel._value2member_map_ else None
    return global_alert.get_recent_alerts(limit=limit, level=alert_lvl)

@router.post("/blacklist")
def manage_blacklist(req: BlacklistRequest):
    if req.action == "add":
        for s in req.symbols:
            global_checker.add_blacklist(s)
    elif req.action == "remove":
        for s in req.symbols:
            global_checker.remove_blacklist(s)
    return {"blacklist": sorted(list(global_checker.get_blacklist()))}
```

挂载至 `src/service/app.py`:
```python
from src.service.routers.risk_router import router as risk_router
app.include_router(risk_router, prefix="/api/v1/risk", tags=["Risk Management"])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `& 'D:\Program Files\anaconda3\envs\python311\python.exe' -m pytest tests/test_risk_api.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/service/routers/risk_router.py src/service/app.py tests/test_risk_api.py
git commit -m "feat(service): 挂载量化风控与熔断中台 RESTful 接口"
```

---

### Task 9: 综合集成测试与全量回归验证 (`tests/test_risk_management.py`)

**Files:**
- Create: `tests/test_risk_management.py`
- Delete: Temporary modular test files (`tests/test_risk_models.py`, `tests/test_risk_rules.py`, `tests/test_circuit_breaker.py`, `tests/test_risk_alert.py`, `tests/test_pre_trade.py`, `tests/test_post_trade.py`, `tests/test_risk_engine_init.py`, `tests/test_risk_api.py`) -> Consolidated into `tests/test_risk_management.py`.

- [ ] **Step 1: Run comprehensive risk tests and baseline tests**

Run: `& 'D:\Program Files\anaconda3\envs\python311\python.exe' -m pytest tests/test_risk_management.py -v`
Expected: All risk tests PASS

- [ ] **Step 2: Run full regression suite**

Run: `& 'D:\Program Files\anaconda3\envs\python311\python.exe' -m pytest tests -q`
Expected: 53 original tests + all new risk tests (approx 65+ total) 100% PASS with 0 errors.

- [ ] **Step 3: Commit**

```bash
git add tests/
git commit -m "test(risk): 整理全量风控自动化测试套件并通过全量回归"
```
