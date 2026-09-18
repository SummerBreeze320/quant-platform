import pytest
from src.market_feed.models import SignalEvent, SignalDirection
from src.market_feed.router import SignalRouter
from src.risk_engine.pre_trade import PreTradeRiskChecker
from src.execution_engine.gateway.paper_broker import PaperBroker

def test_signal_router_auto_execution_success():
    broker = PaperBroker()
    acc = broker.create_account("strat_hft", initial_cash=5_000_000.0)
    risk_checker = PreTradeRiskChecker(max_order_amount=500_000.0, max_stock_weight=0.20)
    router = SignalRouter(broker=broker, risk_checker=risk_checker)

    router.register_strategy(strategy_id="strat_hft", auto_execute=True, default_volume=100)

    sig = SignalEvent(
        signal_id="sig_test_01",
        strategy_id="strat_hft",
        symbol="600519.SH",
        direction=SignalDirection.BUY,
        price=1800.0,
        volume=100,
        reason="测试突破信号"
    )
    result = router.route_signal(sig)
    assert result["status"] == "EXECUTED"
    assert result["trade_id"] is not None

    # Account position should be updated in broker
    acc_updated = broker.get_account("strat_hft")
    assert "600519.SH" in acc_updated.positions
    assert acc_updated.positions["600519.SH"].total_volume == 100

def test_signal_router_auto_execution_risk_rejected():
    broker = PaperBroker()
    broker.create_account("strat_hft", initial_cash=5_000_000.0)
    # Set tight single-order limit to 50,000 (100 shares @ 1800 = 180,000 will breach)
    risk_checker = PreTradeRiskChecker(max_order_amount=50_000.0, max_stock_weight=0.20)
    router = SignalRouter(broker=broker, risk_checker=risk_checker)
    router.register_strategy(strategy_id="strat_hft", auto_execute=True)

    sig = SignalEvent(
        signal_id="sig_test_02",
        strategy_id="strat_hft",
        symbol="600519.SH",
        direction=SignalDirection.BUY,
        price=1800.0,
        volume=100,
        reason="测试超限买入"
    )
    result = router.route_signal(sig)
    assert result["status"] == "REJECTED"
    assert "MaxOrderAmountRule" in result["detail"]

def test_signal_router_manual_mode():
    broker = PaperBroker()
    router = SignalRouter(broker=broker)
    router.register_strategy(strategy_id="strat_manual", auto_execute=False)

    sig = SignalEvent(
        signal_id="sig_test_03",
        strategy_id="strat_manual",
        symbol="600519.SH",
        direction=SignalDirection.BUY,
        price=1800.0,
        volume=100,
        reason="观察模式信号"
    )
    result = router.route_signal(sig)
    assert result["status"] == "BROADCASTED"
