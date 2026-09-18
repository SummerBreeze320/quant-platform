from src.market_feed.models import SignalDirection, MarketTick, IndicatorValue, SignalEvent

def test_market_feed_enums():
    assert SignalDirection.BUY == "BUY"
    assert SignalDirection.SELL == "SELL"
    assert SignalDirection.FLAT == "FLAT"

def test_market_tick_model():
    tick = MarketTick(
        symbol="600519.SH",
        timestamp="2026-09-18T09:30:00",
        last_price=1800.0,
        volume=10000,
        turnover=18000000.0,
        bid_prices=[1799.5, 1799.0, 1798.5],
        bid_volumes=[100, 200, 300],
        ask_prices=[1800.5, 1801.0, 1801.5],
        ask_volumes=[150, 250, 350],
    )
    assert tick.symbol == "600519.SH"
    assert tick.last_price == 1800.0
    assert len(tick.bid_prices) == 3

def test_indicator_value_model():
    ind = IndicatorValue(
        symbol="600519.SH",
        indicator_name="BOLLINGER",
        value=1.05,
        details={"upper": 1820.0, "lower": 1780.0, "pct_b": 1.05},
        timestamp="2026-09-18T09:30:00"
    )
    assert ind.indicator_name == "BOLLINGER"
    assert ind.details["upper"] == 1820.0

def test_signal_event_model():
    sig = SignalEvent(
        signal_id="sig_001",
        strategy_id="strat_alpha",
        symbol="600519.SH",
        direction=SignalDirection.BUY,
        price=1800.0,
        volume=100,
        strength=0.95,
        reason="OFI买压激增 + 布林带上轨突破",
        timestamp="2026-09-18T09:30:00"
    )
    assert sig.direction == SignalDirection.BUY
    assert sig.volume == 100
