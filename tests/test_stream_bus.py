import pytest
from src.market_feed.models import MarketTick
from src.market_feed.stream_bus import StreamBus

def test_stream_bus_memory_mode():
    bus = StreamBus(stream_key="test_stream_mem", force_memory_mode=True)
    assert bus.is_memory_mode is True

    tick1 = MarketTick(symbol="600519.SH", last_price=1800.0, volume=1000, turnover=1800000.0)
    tick2 = MarketTick(symbol="000858.SZ", last_price=150.0, volume=2000, turnover=300000.0)

    msg_id1 = bus.publish_tick(tick1)
    msg_id2 = bus.publish_tick(tick2)
    assert msg_id1 is not None
    assert msg_id2 is not None

    read_ticks = bus.read_ticks(count=10)
    assert len(read_ticks) == 2
    assert read_ticks[0].symbol == "600519.SH"
    assert read_ticks[1].symbol == "000858.SZ"

    # Read again should be empty
    assert len(bus.read_ticks(count=10)) == 0

def test_stream_bus_subscriber_callback():
    bus = StreamBus(stream_key="test_stream_sub", force_memory_mode=True)
    received = []

    def on_tick(tick: MarketTick):
        received.append(tick.symbol)

    bus.subscribe(on_tick)
    bus.publish_tick(MarketTick(symbol="601318.SH", last_price=50.0, volume=500, turnover=25000.0))
    assert len(received) == 1
    assert received[0] == "601318.SH"
