from src.execution_engine.gateway.paper_broker import PaperBroker
from src.execution_engine.models import Order, OrderDirection, OrderStatus

def test_paper_broker_spot_t1():
    broker = PaperBroker(commission_rate=0.0002, stamp_tax_rate=0.0005, slippage_rate=0.0005)
    acc_id = "test_acc_t1"
    broker.create_account(account_id=acc_id, initial_cash=1000000.0)

    # 1. Buy order: 100 shares at price 100.0
    # Fill price with slippage: 100.0 * 1.0005 = 100.05
    # Amount = 100.05 * 100 = 10005.0. Commission = 10005 * 0.0002 = 2.001
    buy_order = Order(order_id="o_buy_01", account_id=acc_id, symbol="600519.SH", direction=OrderDirection.BUY, price=100.0, volume=100)
    trade_buy = broker.submit_order(buy_order)
    assert trade_buy is not None
    assert buy_order.status == OrderStatus.FILLED
    assert trade_buy.price == 100.05

    # Check position under T+1
    acc = broker.get_account(acc_id)
    pos = acc.positions["600519.SH"]
    assert pos.total_volume == 100
    assert pos.frozen_volume == 100
    assert pos.available_volume == 0  # CANNOT be sold today!

    # 2. Try to sell immediately today -> MUST be rejected due to 0 available_volume
    sell_order = Order(order_id="o_sell_01", account_id=acc_id, symbol="600519.SH", direction=OrderDirection.SELL, price=105.0, volume=100)
    trade_sell_fail = broker.submit_order(sell_order)
    assert trade_sell_fail is None
    assert sell_order.status == OrderStatus.REJECTED

    # 3. Simulate Overnight Settlement (Next day open)
    broker.settle_overnight(acc_id)
    acc_next_day = broker.get_account(acc_id)
    pos_next = acc_next_day.positions["600519.SH"]
    assert pos_next.frozen_volume == 0
    assert pos_next.available_volume == 100  # UNLOCKED!

    # 4. Sell on next day -> Succeeds!
    sell_order_ok = Order(order_id="o_sell_02", account_id=acc_id, symbol="600519.SH", direction=OrderDirection.SELL, price=105.0, volume=100)
    trade_sell = broker.submit_order(sell_order_ok)
    assert trade_sell is not None
    assert trade_sell.stamp_tax > 0  # Stamp tax charged on sell
    assert broker.get_account(acc_id).positions.get("600519.SH") is None
