import pytest
from src.market_feed.models import MarketTick
from src.market_feed.stream_bus import StreamBus
from src.market_feed.replay import TickReplayEngine
import src.market_feed as mf

def test_tick_replay_engine():
    bus = StreamBus(stream_key="replay_test", force_memory_mode=True)
    engine = TickReplayEngine(stream_bus=bus)

    ticks = [
        MarketTick(symbol="600519.SH", last_price=1800.0, volume=100, turnover=180000.0),
        MarketTick(symbol="600519.SH", last_price=1805.0, volume=200, turnover=361000.0),
        MarketTick(symbol="600519.SH", last_price=1810.0, volume=300, turnover=543000.0),
    ]

    count = engine.replay(ticks, speed=0.0) # Speed 0.0 is instant without delay
    assert count == 3
    assert len(bus.read_ticks(count=10)) == 3

def test_market_feed_exports():
    assert hasattr(mf, "MarketTick")
    assert hasattr(mf, "SignalEvent")
    assert hasattr(mf, "StreamBus")
    assert hasattr(mf, "TickReplayEngine")
    assert hasattr(mf, "StreamingVWAP")
    assert hasattr(mf, "StreamingBollingerBands")
    assert hasattr(mf, "OrderFlowImbalance")
    assert hasattr(mf, "SignalEngine")
    assert hasattr(mf, "SignalRouter")
