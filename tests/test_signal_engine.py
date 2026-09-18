import pytest
from src.market_feed.models import MarketTick, SignalDirection
from src.market_feed.signal_engine import SignalEngine

def test_signal_engine_bollinger_breakout():
    engine = SignalEngine(cooldown_seconds=1.0)

    # Feed steady ticks around 100.0
    for _ in range(15):
        engine.process_tick(MarketTick(
            symbol="600519.SH", last_price=100.0, volume=1000, turnover=100000.0,
            bid_prices=[100.0], bid_volumes=[100],
            ask_prices=[100.1], ask_volumes=[100]
        ))

    # Sudden surge to 125.0 breaking upper Bollinger Band and above VWAP
    sig = engine.process_tick(MarketTick(
        symbol="600519.SH", last_price=125.0, volume=2000, turnover=220000.0,
        bid_prices=[125.0], bid_volumes=[500],
        ask_prices=[125.1], ask_volumes=[100]
    ))
    assert sig is not None
    assert sig.direction == SignalDirection.BUY
    assert "布林带" in sig.reason or "OFI" in sig.reason

def test_signal_engine_cooldown():
    engine = SignalEngine(cooldown_seconds=5.0)
    for _ in range(15):
        engine.process_tick(MarketTick(symbol="600519.SH", last_price=100.0, volume=1000, turnover=100000.0))

    # First trigger
    sig1 = engine.process_tick(MarketTick(symbol="600519.SH", last_price=125.0, volume=2000, turnover=220000.0))
    assert sig1 is not None

    # Immediate second tick within 5s cooldown should be suppressed
    sig2 = engine.process_tick(MarketTick(symbol="600519.SH", last_price=126.0, volume=2500, turnover=280000.0))
    assert sig2 is None
