from src.execution_engine.models import (
    OrderDirection, OrderType, OrderStatus, AlgoType,
    Position, AccountState, Order, Trade, RebalancePlan
)

def test_execution_models_instantiation():
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
        positions={"600519.SH": pos},
        updated_at="2026-09-18T10:00:00"
    )
    assert acc.total_equity == 1000000.0

    order = Order(
        order_id="ord_01",
        account_id="acc_01",
        symbol="600519.SH",
        direction=OrderDirection.BUY,
        price=1800.0,
        volume=100,
        created_at="2026-09-18T10:00:00"
    )
    assert order.status == OrderStatus.PENDING
    assert order.algo_type == AlgoType.DIRECT
