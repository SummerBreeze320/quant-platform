import time
import pytest
from fastapi.testclient import TestClient

from src.execution_engine.models import (
    Order, Trade, Position, AccountState, OrderDirection, OrderType, AlgoType
)
from src.execution_engine.gateway.paper_broker import PaperBroker
from src.execution_engine.gateway.qmt_gateway import QmtBrokerGateway
from src.execution_engine.scheduler import (
    TimeSlicedScheduler, SlicedExecutionTask, ExecutionTaskStatus
)
from src.execution_engine.coordinator import ExecutionCoordinator
from src.risk_engine.circuit_breaker import CircuitBreakerManager
from src.service.app import create_app


def test_qmt_gateway_symbol_normalization():
    assert QmtBrokerGateway.normalize_symbol("600000.SH") == "600000.SH"
    assert QmtBrokerGateway.normalize_symbol("SH600000") == "600000.SH"
    assert QmtBrokerGateway.normalize_symbol("SZ000001") == "000001.SZ"
    assert QmtBrokerGateway.normalize_symbol("000001.SZ") == "000001.SZ"


def test_qmt_gateway_simulation_mode():
    gw = QmtBrokerGateway(account_id="qmt_test_acc", mock_mode=True)
    assert gw.is_connected
    assert gw.mock_mode

    # Initialize account in gateway's broker
    gw.get_account("qmt_test_acc")
    gw._mock_broker.create_account("qmt_test_acc", initial_cash=500_000.0)

    buy_order = Order(
        order_id="qmt_ord_1",
        account_id="qmt_test_acc",
        symbol="SH600000",
        direction=OrderDirection.BUY,
        order_type=OrderType.LIMIT,
        price=10.0,
        volume=1000,
    )
    trade = gw.submit_order(buy_order)
    assert trade is not None
    assert trade.volume == 1000
    assert trade.symbol == "600000.SH"

    acc = gw.get_account("qmt_test_acc")
    assert "600000.SH" in acc.positions
    assert acc.positions["600000.SH"].total_volume == 1000
    assert gw.cancel_order("any_order_id") is True


def test_time_sliced_scheduler_twap_execution():
    broker = PaperBroker()
    broker.create_account("twap_acc", initial_cash=1_000_000.0)
    coordinator = ExecutionCoordinator(gateway=broker)
    scheduler = TimeSlicedScheduler()

    parent = Order(
        order_id="parent_twap_1",
        account_id="twap_acc",
        symbol="600000.SH",
        direction=OrderDirection.BUY,
        order_type=OrderType.LIMIT,
        price=10.0,
        volume=1000,
        algo_type=AlgoType.TWAP,
    )

    twap_algo = coordinator.algo_registry[AlgoType.TWAP]
    slices = twap_algo.slice_order(parent)
    assert len(slices) == 5

    # Run synchronously with tiny interval for unit test speed
    task = scheduler.schedule_execution(
        parent_order=parent,
        slices=slices,
        interval_seconds=0.001,
        coordinator=coordinator,
        run_async=False,
    )

    assert task.status == ExecutionTaskStatus.COMPLETED
    assert task.total_slices == 5
    assert task.completed_slices == 5
    assert task.filled_volume == 1000
    assert len(task.trades) == 5

    # Account position should reflect all slices
    acc = broker.get_account("twap_acc")
    assert acc.positions["600000.SH"].total_volume == 1000


def test_time_sliced_scheduler_cancellation():
    broker = PaperBroker()
    broker.create_account("cancel_acc", initial_cash=1_000_000.0)
    coordinator = ExecutionCoordinator(gateway=broker)
    scheduler = TimeSlicedScheduler()

    parent = Order(
        order_id="parent_twap_cancel",
        account_id="cancel_acc",
        symbol="600000.SH",
        direction=OrderDirection.BUY,
        order_type=OrderType.LIMIT,
        price=10.0,
        volume=2000,
        algo_type=AlgoType.TWAP,
    )

    twap_algo = coordinator.algo_registry[AlgoType.TWAP]
    slices = twap_algo.slice_order(parent)

    # Schedule with 0.1s interval per slice
    task = scheduler.schedule_execution(
        parent_order=parent,
        slices=slices,
        interval_seconds=0.1,
        coordinator=coordinator,
        run_async=True,
    )

    time.sleep(0.05)
    # Cancel task before all slices complete
    cancelled = scheduler.cancel_task(task.task_id)
    assert cancelled is True

    # Wait for worker thread to exit
    time.sleep(0.15)
    updated_task = scheduler.get_task(task.task_id)
    assert updated_task.status == ExecutionTaskStatus.CANCELLED
    assert updated_task.completed_slices < updated_task.total_slices


def test_time_sliced_scheduler_circuit_breaker_interception():
    broker = PaperBroker()
    broker.create_account("cb_acc", initial_cash=1_000_000.0)
    cb = CircuitBreakerManager()
    coordinator = ExecutionCoordinator(gateway=broker, circuit_breaker=cb)
    scheduler = TimeSlicedScheduler()

    # Trip circuit breaker
    cb.update_equity("cb_acc", 1_000_000.0)
    cb.update_equity("cb_acc", 850_000.0)  # > 10% drop -> triggers RED_HALT

    parent = Order(
        order_id="parent_twap_cb",
        account_id="cb_acc",
        symbol="600000.SH",
        direction=OrderDirection.BUY,
        order_type=OrderType.LIMIT,
        price=10.0,
        volume=1000,
        algo_type=AlgoType.TWAP,
    )

    twap_algo = coordinator.algo_registry[AlgoType.TWAP]
    slices = twap_algo.slice_order(parent)

    task = scheduler.schedule_execution(
        parent_order=parent,
        slices=slices,
        interval_seconds=0.001,
        coordinator=coordinator,
        run_async=False,
    )

    assert task.status == ExecutionTaskStatus.CANCELLED
    assert any("RED_HALT" in err for err in task.errors)
    assert task.filled_volume == 0


def test_api_gateway_status_and_tasks_flow():
    app = create_app()
    client = TestClient(app)

    # 1. Query Gateway status
    gw_resp = client.get("/api/v1/execution/gateway/status")
    assert gw_resp.status_code == 200
    gw_info = gw_resp.json()
    assert "gateway_type" in gw_info
    assert "is_connected" in gw_info

    # 2. Register strategy via PMS to have budget
    client.post("/api/v1/pms/strategies", json={
        "strategy_id": "scheduled_strat",
        "name": "Scheduled Strategy",
        "initial_budget": 1_000_000.0,
    })

    # 3. Request ASYNC_SCHEDULED rebalance with TWAP
    rebalance_resp = client.post("/api/v1/execution/rebalance", json={
        "account_id": "scheduled_strat",
        "target_weights": {"600000.SH": 0.10},
        "current_prices": {"600000.SH": 10.0},
        "algo_type": "TWAP",
        "execution_mode": "ASYNC_SCHEDULED",
        "interval_seconds": 0.01,
    })
    assert rebalance_resp.status_code == 200
    res = rebalance_resp.json()
    assert res["status"] == "SCHEDULED"
    assert len(res["tasks"]) >= 1
    task_id = res["tasks"][0]["task_id"]

    # 4. Query tasks list
    tasks_resp = client.get("/api/v1/execution/tasks", params={"account_id": "scheduled_strat"})
    assert tasks_resp.status_code == 200
    tasks = tasks_resp.json()
    assert any(t["task_id"] == task_id for t in tasks)

    # 5. Query specific task detail
    detail_resp = client.get(f"/api/v1/execution/tasks/{task_id}")
    assert detail_resp.status_code == 200
    detail = detail_resp.json()
    assert detail["task_id"] == task_id
    assert detail["total_slices"] > 0
