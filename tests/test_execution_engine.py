import pytest
from starlette.testclient import TestClient
from src.service.app import app
from src.execution_engine import (
    OrderDirection,
    OrderType,
    OrderStatus,
    AlgoType,
    Position,
    AccountState,
    Order,
    Trade,
    RebalancePlan,
    RebalanceOrderGenerator,
    DirectAlgo,
    TwapAlgo,
    VwapAlgo,
    PaperBroker,
    ExecutionCoordinator,
)
from src.risk_engine import PreTradeRiskChecker, CircuitBreakerManager, CircuitBreakerLevel

@pytest.fixture
def client():
    return TestClient(app)

def test_models_and_enums():
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
        positions={"600519.SH": pos}
    )
    assert acc.total_equity == 1000000.0

def test_rebalance_order_generator():
    pos_600519 = Position(symbol="600519.SH", total_volume=300, available_volume=300, avg_cost=1000.0, market_value=300000.0)
    pos_000858 = Position(symbol="000858.SZ", total_volume=500, available_volume=500, avg_cost=100.0, market_value=50000.0)
    
    acc = AccountState(
        account_id="acc_test",
        total_equity=1000000.0,
        available_cash=650000.0,
        positions={"600519.SH": pos_600519, "000858.SZ": pos_000858}
    )

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

    sell_symbols = {o.symbol: o.volume for o in plan.sell_orders}
    assert sell_symbols["600519.SH"] == 200
    assert sell_symbols["000858.SZ"] == 500

    assert plan.buy_orders[0].symbol == "601318.SH"
    assert plan.buy_orders[0].volume == 4000

def test_algos_slicing():
    parent = Order(
        order_id="p_01",
        symbol="600519.SH",
        direction=OrderDirection.BUY,
        price=100.0,
        volume=1000
    )

    # 1. Direct
    direct = DirectAlgo()
    assert len(direct.slice_order(parent)) == 1

    # 2. Twap
    twap = TwapAlgo(num_slices=5)
    slices_twap = twap.slice_order(parent)
    assert len(slices_twap) == 5
    assert sum(s.volume for s in slices_twap) == 1000
    assert all(s.volume % 100 == 0 for s in slices_twap)

    # 3. Vwap
    vwap = VwapAlgo(volume_profile=[0.30, 0.15, 0.10, 0.15, 0.30])
    slices_vwap = vwap.slice_order(parent)
    assert len(slices_vwap) == 5
    assert sum(s.volume for s in slices_vwap) == 1000
    assert all(s.volume % 100 == 0 for s in slices_vwap)

def test_paper_broker_spot_t1():
    broker = PaperBroker(commission_rate=0.0002, stamp_tax_rate=0.0005, slippage_rate=0.0005)
    acc_id = "test_acc_t1"
    broker.create_account(account_id=acc_id, initial_cash=1000000.0)

    # 1. Buy order
    buy_order = Order(order_id="o_buy_01", account_id=acc_id, symbol="600519.SH", direction=OrderDirection.BUY, price=100.0, volume=100)
    trade_buy = broker.submit_order(buy_order)
    assert trade_buy is not None
    assert buy_order.status == OrderStatus.FILLED
    assert trade_buy.price == 100.05

    # Position frozen under T+1
    acc = broker.get_account(acc_id)
    pos = acc.positions["600519.SH"]
    assert pos.total_volume == 100
    assert pos.frozen_volume == 100
    assert pos.available_volume == 0

    # 2. Try to sell immediately today -> rejected
    sell_order = Order(order_id="o_sell_01", account_id=acc_id, symbol="600519.SH", direction=OrderDirection.SELL, price=105.0, volume=100)
    trade_sell_fail = broker.submit_order(sell_order)
    assert trade_sell_fail is None
    assert sell_order.status == OrderStatus.REJECTED

    # 3. Overnight Settlement
    broker.settle_overnight(acc_id)
    acc_next_day = broker.get_account(acc_id)
    pos_next = acc_next_day.positions["600519.SH"]
    assert pos_next.frozen_volume == 0
    assert pos_next.available_volume == 100

    # 4. Sell next day -> success with stamp tax
    sell_order_ok = Order(order_id="o_sell_02", account_id=acc_id, symbol="600519.SH", direction=OrderDirection.SELL, price=105.0, volume=100)
    trade_sell = broker.submit_order(sell_order_ok)
    assert trade_sell is not None
    assert trade_sell.stamp_tax > 0

def test_execution_coordinator_risk_closed_loop():
    broker = PaperBroker()
    acc_id = "coord_acc"
    broker.create_account(acc_id, initial_cash=1000000.0)

    cb = CircuitBreakerManager()
    risk_checker = PreTradeRiskChecker(cb_manager=cb)
    risk_checker.add_blacklist("000001.SZ")

    coord = ExecutionCoordinator(
        gateway=broker,
        risk_checker=risk_checker,
        circuit_breaker=cb
    )

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
        algo_type=AlgoType.TWAP
    )

    assert len(result["executed_trades"]) > 0
    assert len(result["rejected_orders"]) > 0
    assert result["rejected_orders"][0].symbol == "000001.SZ"

def test_execution_api(client):
    resp_acc = client.get("/api/v1/execution/account?account_id=api_test_acc")
    assert resp_acc.status_code == 200

    payload = {
        "account_id": "api_test_acc",
        "target_weights": {"600519.SH": 0.05},
        "current_prices": {"600519.SH": 500.0},
        "algo_type": "VWAP"
    }
    resp_reb = client.post("/api/v1/execution/rebalance", json=payload)
    assert resp_reb.status_code == 200
    assert len(resp_reb.json()["executed_trades"]) > 0

    resp_set = client.post("/api/v1/execution/settle", json={"account_id": "api_test_acc"})
    assert resp_set.status_code == 200

    resp_trd = client.get("/api/v1/execution/trades?account_id=api_test_acc")
    assert resp_trd.status_code == 200
    assert len(resp_trd.json()) > 0
