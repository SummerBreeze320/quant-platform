from src.execution_engine.models import AccountState, Position, OrderDirection
from src.execution_engine.rebalance import RebalanceOrderGenerator

def test_rebalance_order_generator():
    pos_600519 = Position(symbol="600519.SH", total_volume=300, available_volume=300, avg_cost=1000.0, market_value=300000.0)
    pos_000858 = Position(symbol="000858.SZ", total_volume=500, available_volume=500, avg_cost=100.0, market_value=50000.0)
    
    acc = AccountState(
        account_id="acc_test",
        total_equity=1000000.0,
        available_cash=650000.0,
        positions={"600519.SH": pos_600519, "000858.SZ": pos_000858}
    )

    # Target: 600519.SH -> 10% (100k, price 1000 => 100 shares, sell 200)
    # Target: 000858.SZ -> 0% (liquidate, sell 500)
    # Target: 601318.SH -> 20% (200k, price 50 => 4000 shares, buy 4000)
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

    # Verify sell order details
    sell_symbols = {o.symbol: o.volume for o in plan.sell_orders}
    assert sell_symbols["600519.SH"] == 200
    assert sell_symbols["000858.SZ"] == 500

    # Verify buy order details
    assert plan.buy_orders[0].symbol == "601318.SH"
    assert plan.buy_orders[0].volume == 4000
    assert plan.buy_orders[0].direction == OrderDirection.BUY
