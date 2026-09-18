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
