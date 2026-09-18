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
