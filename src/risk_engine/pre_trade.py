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
