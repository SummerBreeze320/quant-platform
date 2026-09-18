from src.execution_engine.coordinator import ExecutionCoordinator
from src.execution_engine.gateway.paper_broker import PaperBroker
from src.risk_engine.pre_trade import PreTradeRiskChecker
from src.risk_engine.circuit_breaker import CircuitBreakerManager
from src.execution_engine.models import AlgoType
from src.risk_engine.models import CircuitBreakerLevel

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
