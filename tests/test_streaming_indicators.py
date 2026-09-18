import pytest
from src.market_feed.models import MarketTick
from src.market_feed.indicators import StreamingVWAP, StreamingBollingerBands, OrderFlowImbalance

def test_streaming_vwap():
    vwap_ind = StreamingVWAP()
    # Tick 1: 1000 shares @ 10.0 -> turnover 10,000
    t1 = MarketTick(symbol="600519.SH", last_price=10.0, volume=1000, turnover=10000.0)
    res1 = vwap_ind.update(t1)
    assert res1.value == 10.0
    assert res1.details["vwap"] == 10.0

    # Tick 2: cumulative volume 3000 (+2000), turnover 34,000 (+24,000 @ 12.0)
    # Total turnover = 34,000, Total volume = 3000 -> VWAP = 34000 / 3000 = 11.3333
    t2 = MarketTick(symbol="600519.SH", last_price=12.0, volume=3000, turnover=34000.0)
    res2 = vwap_ind.update(t2)
    assert pytest.approx(res2.value, 0.001) == 11.3333
    assert pytest.approx(res2.details["bias"], 0.001) == (12.0 - 11.3333) / 11.3333

def test_streaming_bollinger_bands():
    bb = StreamingBollingerBands(window=20, num_std=2.0)
    prices = [10.0] * 20
    for p in prices:
        t = MarketTick(symbol="600519.SH", last_price=p, volume=100, turnover=1000.0)
        res = bb.update(t)
    assert res.value == 10.0
    assert res.details["std"] == 0.0
    assert res.details["upper"] == 10.0
    assert res.details["lower"] == 10.0

    # Price surge to 20.0 -> upper band expands and pct_b > 1.0
    t_surge = MarketTick(symbol="600519.SH", last_price=20.0, volume=200, turnover=4000.0)
    res_surge = bb.update(t_surge)
    assert res_surge.details["pct_b"] > 1.0
    assert res_surge.details["upper"] > 10.0

def test_order_flow_imbalance():
    ofi = OrderFlowImbalance()
    # Snapshot 1: Bid1 100 @ 10.0, Ask1 100 @ 10.1
    t1 = MarketTick(
        symbol="600519.SH", last_price=10.0, volume=1000, turnover=10000.0,
        bid_prices=[10.0], bid_volumes=[100],
        ask_prices=[10.1], ask_volumes=[100]
    )
    res1 = ofi.update(t1)
    assert res1.value == 0.0

    # Snapshot 2: Price same, Bid1 volume increases to 300 (+200 buyer pressure), Ask1 same
    t2 = MarketTick(
        symbol="600519.SH", last_price=10.0, volume=1000, turnover=10000.0,
        bid_prices=[10.0], bid_volumes=[300],
        ask_prices=[10.1], ask_volumes=[100]
    )
    res2 = ofi.update(t2)
    assert res2.value > 0.0 # Positive buyer flow imbalance
    assert res2.details["ofi_raw"] == 200.0
