import pytest
from fastapi.testclient import TestClient

from src.execution_engine.models import (
    Order, OrderDirection, OrderType, AlgoType
)
from src.execution_engine.algos import IcebergAlgo, ImplementationShortfallAlgo
from src.execution_engine.coordinator import ExecutionCoordinator
from src.execution_engine.gateway.paper_broker import PaperBroker
from src.service.app import create_app


def test_iceberg_algo_basic_slicing():
    algo = IcebergAlgo(display_ratio=0.15, randomize=False)
    parent = Order(
        order_id="iceberg_p_1",
        account_id="acc_iceberg",
        symbol="600000.SH",
        direction=OrderDirection.BUY,
        order_type=OrderType.LIMIT,
        price=10.0,
        volume=10000,
    )

    slices = algo.slice_order(parent)
    assert len(slices) > 1
    # Check volume conservation
    assert sum(s.volume for s in slices) == 10000

    # Check 100 shares lot size
    for s in slices:
        assert s.volume % 100 == 0
        assert s.volume > 0
        assert s.algo_type == AlgoType.ICEBERG
        assert s.parent_order_id == "iceberg_p_1"


def test_iceberg_algo_with_fixed_volume_and_randomize():
    algo = IcebergAlgo(display_volume=2000, randomize=True, variance_pct=0.2)
    parent = Order(
        order_id="iceberg_p_2",
        account_id="acc_iceberg",
        symbol="000001.SZ",
        direction=OrderDirection.SELL,
        order_type=OrderType.LIMIT,
        price=15.0,
        volume=12500,
    )

    slices = algo.slice_order(parent)
    assert len(slices) >= 5
    assert sum(s.volume for s in slices) == 12500
    for s in slices:
        assert s.volume % 100 == 0
        assert s.volume >= 100

    # Small order test
    small_order = Order(
        order_id="iceberg_p_small",
        account_id="acc_iceberg",
        symbol="000001.SZ",
        direction=OrderDirection.BUY,
        order_type=OrderType.LIMIT,
        price=15.0,
        volume=100,
    )
    s_slices = algo.slice_order(small_order)
    assert len(s_slices) == 1
    assert s_slices[0].volume == 100


def test_is_algo_urgency_decay():
    # Low urgency vs High urgency
    algo_low = ImplementationShortfallAlgo(num_slices=5, default_urgency=0.0)
    algo_high = ImplementationShortfallAlgo(num_slices=5, default_urgency=1.5)

    parent = Order(
        order_id="is_p_1",
        account_id="acc_is",
        symbol="600519.SH",
        direction=OrderDirection.BUY,
        order_type=OrderType.LIMIT,
        price=1800.0,
        volume=5000,
    )

    slices_low = algo_low.slice_order(parent)
    slices_high = algo_high.slice_order(parent)

    assert sum(s.volume for s in slices_low) == 5000
    assert sum(s.volume for s in slices_high) == 5000

    # Under high urgency, the first slice should be noticeably larger than under low urgency
    assert slices_high[0].volume > slices_low[0].volume
    for s in slices_high:
        assert s.volume % 100 == 0
        assert s.algo_type == AlgoType.IS


def test_is_algo_price_drift_adaptive():
    algo = ImplementationShortfallAlgo(num_slices=5, default_urgency=0.5)

    # 1. Buy order with adverse price drift (+5% price increase)
    parent_buy = Order(
        order_id="is_p_buy",
        account_id="acc_is",
        symbol="601318.SH",
        direction=OrderDirection.BUY,
        order_type=OrderType.LIMIT,
        price=50.0,
        volume=10000,
    )
    slices_neutral = algo.slice_order(parent_buy, ctx={"arrival_price": 50.0, "current_price": 50.0})
    slices_adverse = algo.slice_order(parent_buy, ctx={"arrival_price": 50.0, "current_price": 52.5})

    assert sum(s.volume for s in slices_neutral) == 10000
    assert sum(s.volume for s in slices_adverse) == 10000
    # Adverse drift accelerates execution in earlier slices
    assert slices_adverse[0].volume >= slices_neutral[0].volume

    # 2. Sell order with adverse price drift (-5% price drop)
    parent_sell = Order(
        order_id="is_p_sell",
        account_id="acc_is",
        symbol="601318.SH",
        direction=OrderDirection.SELL,
        order_type=OrderType.LIMIT,
        price=50.0,
        volume=10000,
    )
    slices_sell_neutral = algo.slice_order(parent_sell, ctx={"arrival_price": 50.0, "current_price": 50.0})
    slices_sell_adverse = algo.slice_order(parent_sell, ctx={"arrival_price": 50.0, "current_price": 47.5})
    assert slices_sell_adverse[0].volume >= slices_sell_neutral[0].volume


def test_coordinator_rebalance_with_iceberg_and_is():
    broker = PaperBroker()
    broker.create_account("coord_test_acc", initial_cash=2_000_000.0)
    coordinator = ExecutionCoordinator(gateway=broker)

    # 1. Sync rebalance with ICEBERG (weight <= 10% risk threshold)
    res_iceberg = coordinator.execute_rebalance(
        account_id="coord_test_acc",
        target_weights={"600000.SH": 0.08},
        current_prices={"600000.SH": 10.0},
        algo_type=AlgoType.ICEBERG,
        execution_mode="SYNC",
    )
    assert len(res_iceberg["executed_trades"]) > 0
    assert len(res_iceberg["rejected_orders"]) == 0

    # 2. Async scheduled rebalance with IS
    res_is = coordinator.execute_rebalance(
        account_id="coord_test_acc",
        target_weights={"600000.SH": 0.05, "000001.SZ": 0.05},
        current_prices={"600000.SH": 10.0, "000001.SZ": 15.0},
        algo_type=AlgoType.IS,
        execution_mode="ASYNC_SCHEDULED",
        interval_seconds=0.01,
    )
    assert res_is["status"] == "SCHEDULED"
    assert len(res_is["tasks"]) > 0


def test_execution_api_endpoints_with_advanced_algos():
    app = create_app()
    client = TestClient(app)

    # Rebalance with ICEBERG
    payload_iceberg = {
        "account_id": "api_adv_algo_acc",
        "target_weights": {"600000.SH": 0.06},
        "current_prices": {"600000.SH": 10.0},
        "algo_type": "ICEBERG",
        "execution_mode": "SYNC",
    }
    resp1 = client.post("/api/v1/execution/rebalance", json=payload_iceberg)
    assert resp1.status_code == 200
    data1 = resp1.json()
    assert len(data1["executed_trades"]) > 0

    # Rebalance with IS
    payload_is = {
        "account_id": "api_adv_algo_acc",
        "target_weights": {"600000.SH": 0.04, "000858.SZ": 0.05},
        "current_prices": {"600000.SH": 10.0, "000858.SZ": 150.0},
        "algo_type": "IS",
        "execution_mode": "ASYNC_SCHEDULED",
        "interval_seconds": 0.01,
    }
    resp2 = client.post("/api/v1/execution/rebalance", json=payload_is)
    assert resp2.status_code == 200
    data2 = resp2.json()
    assert data2["status"] == "SCHEDULED"
    assert len(data2["tasks"]) > 0
