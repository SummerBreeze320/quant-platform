import time
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from fastapi.testclient import TestClient

from src.execution_engine.models import (
    Order, OrderDirection, OrderType, Position, AccountState
)
from src.execution_engine.gateway.paper_broker import PaperBroker
from src.execution_engine.gateway.qmt_gateway import QmtBrokerGateway
from src.execution_engine.persistence import TradingStorage
from src.market_feed.models import MarketTick
from src.models.trading import Base
from src.risk_engine.circuit_breaker import CircuitBreakerManager
from src.risk_engine.models import CircuitBreakerLevel
from src.service.app import create_app
from src.service.runtime import ServiceRuntime


@pytest.fixture
def sqlite_storage():
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)
    storage = TradingStorage(session_factory=session_factory)
    storage._enabled = True
    return storage


def test_paper_broker_mark_to_market():
    broker = PaperBroker()
    acc = broker.create_account("acc_mtm", initial_cash=1_000_000.0)

    # Buy 1000 shares of 600519.SH at 100.0
    order = Order(
        order_id="ord_mtm_1",
        account_id="acc_mtm",
        symbol="600519.SH",
        direction=OrderDirection.BUY,
        order_type=OrderType.LIMIT,
        price=100.0,
        volume=1000,
    )
    trade = broker.submit_order(order)
    assert trade is not None

    pos = broker.get_account("acc_mtm").positions["600519.SH"]
    cost = pos.avg_cost

    # 1. Price increases to 120.0
    updated_accs = broker.update_market_price("600519.SH", 120.0)
    assert len(updated_accs) == 1
    assert updated_accs[0].account_id == "acc_mtm"

    pos = broker.get_account("acc_mtm").positions["600519.SH"]
    assert pos.last_price == 120.0
    assert pos.market_value == 120_000.0
    assert pos.unrealized_pnl == pytest.approx((120.0 - cost) * 1000, rel=1e-3)
    assert pos.unrealized_pnl_ratio == pytest.approx((120.0 - cost) / cost, rel=1e-3)
    assert pos.unrealized_pnl > 0

    # 2. Price drops to 80.0
    broker.update_market_price("600519.SH", 80.0)
    pos = broker.get_account("acc_mtm").positions["600519.SH"]
    assert pos.last_price == 80.0
    assert pos.market_value == 80_000.0
    assert pos.unrealized_pnl < 0
    assert pos.unrealized_pnl_ratio < 0


def test_qmt_gateway_mark_to_market():
    gw = QmtBrokerGateway(account_id="qmt_acc", mock_mode=True)
    gw._mock_broker.create_account("qmt_acc", initial_cash=500_000.0)

    # Buy 500 shares of 000001.SZ at 20.0
    order = Order(
        order_id="qmt_ord_mtm",
        account_id="qmt_acc",
        symbol="SZ000001",
        direction=OrderDirection.BUY,
        order_type=OrderType.LIMIT,
        price=20.0,
        volume=500,
    )
    gw.submit_order(order)

    # Mark to market via QmtGateway
    updated = gw.update_market_price("000001.SZ", 25.0)
    assert len(updated) == 1
    acc = gw.get_account("qmt_acc")
    pos = acc.positions["000001.SZ"]
    assert pos.last_price == 25.0
    assert pos.market_value == 12500.0
    assert pos.unrealized_pnl > 0


def test_storage_persistence_of_mtm(sqlite_storage):
    broker = PaperBroker(storage=sqlite_storage)
    acc = broker.create_account("persist_mtm_acc", initial_cash=200_000.0)

    order = Order(
        order_id="p_ord_1",
        account_id="persist_mtm_acc",
        symbol="600036.SH",
        direction=OrderDirection.BUY,
        order_type=OrderType.LIMIT,
        price=30.0,
        volume=1000,
    )
    broker.submit_order(order)

    # Revalue
    broker.update_market_price("600036.SH", 36.0)

    # Restore in fresh broker instance
    fresh_broker = PaperBroker(storage=sqlite_storage)
    restored_count = fresh_broker.restore_from_storage()
    assert restored_count >= 1

    restored_acc = fresh_broker.get_account("persist_mtm_acc")
    assert "600036.SH" in restored_acc.positions
    pos = restored_acc.positions["600036.SH"]
    assert pos.last_price == 36.0
    assert pos.unrealized_pnl > 0


def test_market_runtime_tick_triggers_circuit_breaker():
    runtime = ServiceRuntime()
    # Allocate position to hft_stream_01 account
    acc = runtime.broker.get_account("hft_stream_01")
    # Simulate an established high watermark: initial equity 5,000,000
    runtime.circuit_breaker.update_equity("hft_stream_01", 5_000_000.0)

    # Buy a heavy position
    order = Order(
        order_id="heavy_buy",
        account_id="hft_stream_01",
        symbol="600519.SH",
        direction=OrderDirection.BUY,
        order_type=OrderType.LIMIT,
        price=2000.0,
        volume=2000, # Value ~ 4,000,000
    )
    runtime.broker.submit_order(order)
    runtime.circuit_breaker.update_equity("hft_stream_01", runtime.broker.get_account("hft_stream_01").total_equity)

    # Initial tick at 2000.0: state NORMAL
    tick1 = MarketTick(
        symbol="600519.SH",
        timestamp="2026-09-19T09:30:00",
        last_price=2000.0,
        volume=10000,
        turnover=20000000.0,
        bid_prices=[1999.0],
        bid_volumes=[10],
        ask_prices=[2001.0],
        ask_volumes=[10],
    )
    runtime.market.on_tick(tick1)
    cb_state = runtime.circuit_breaker.get_state("hft_stream_01")
    assert cb_state.level == CircuitBreakerLevel.NORMAL

    # Tick price drops from 2000.0 to 1800.0 (a loss of 200 * 2000 = 400,000 on 5M capital = 8% drawdown)
    # This exceeds halt_drawdown (3.5%), which must trigger RED_HALT
    tick2 = MarketTick(
        symbol="600519.SH",
        timestamp="2026-09-19T09:35:00",
        last_price=1800.0,
        volume=20000,
        turnover=38000000.0,
        bid_prices=[1799.0],
        bid_volumes=[10],
        ask_prices=[1801.0],
        ask_volumes=[10],
    )
    runtime.market.on_tick(tick2)

    cb_state_after = runtime.circuit_breaker.get_state("hft_stream_01")
    assert cb_state_after.level == CircuitBreakerLevel.RED_HALT
    assert "日内回撤达到" in (cb_state_after.trigger_reason or "")

    # Alert should have been generated
    alerts = runtime.alert_manager.get_recent_alerts(account_id="hft_stream_01")
    assert len(alerts) >= 1
    assert alerts[0].rule_name == "DynamicDrawdownCircuitBreaker"


def test_mark_to_market_api():
    app = create_app()
    client = TestClient(app)

    # 1. Place a rebalance trade to buy 601318.SH (weight <= 10% risk threshold)
    res = client.post("/api/v1/execution/rebalance", json={
        "account_id": "api_mtm_acc",
        "target_weights": {"601318.SH": 0.08},
        "current_prices": {"601318.SH": 100.0},
        "algo_type": "DIRECT",
        "execution_mode": "SYNC"
    })
    assert res.status_code == 200

    # 2. Call mark_to_market endpoint
    mtm_res = client.post("/api/v1/execution/mark_to_market", json={
        "symbol": "601318.SH",
        "price": 115.0
    })
    assert mtm_res.status_code == 200
    data = mtm_res.json()
    assert data["status"] == "SUCCESS"
    assert data["symbol"] == "601318.SH"
    assert data["price"] == 115.0
    assert len(data["updated_accounts"]) >= 1

    # 3. Query positions endpoint
    pos_res = client.get("/api/v1/execution/positions?account_id=api_mtm_acc")
    assert pos_res.status_code == 200
    positions = pos_res.json()
    assert len(positions) >= 1
    p = [pos for pos in positions if pos["symbol"] == "601318.SH"][0]
    assert p["last_price"] == 115.0
    assert p["market_value"] > 0
    assert p["unrealized_pnl"] > 0
    assert p["unrealized_pnl_ratio"] > 0
