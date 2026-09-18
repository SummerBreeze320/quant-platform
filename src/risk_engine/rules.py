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
                return RiskAction.REJECT, f"可用现金不足: 可用现金 {req.available_cash:.2f} 小于预估所需费用 {required_cash:.2f}"
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
