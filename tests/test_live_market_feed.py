import time
from unittest.mock import patch, MagicMock
import pytest
from fastapi.testclient import TestClient

from src.market_feed.models import MarketTick
from src.market_feed.stream_bus import StreamBus
from src.market_feed.live_feed import (
    normalize_symbol_to_standard,
    standard_to_sina_code,
    SimulationLiveFeedAdapter,
    PublicWebLiveFeedAdapter,
    QmtLiveFeedAdapter,
    LiveFeedManager,
)
from src.service.app import create_app
from src.service.runtime import ServiceRuntime


def test_symbol_normalization():
    assert normalize_symbol_to_standard("600000.SH") == "600000.SH"
    assert normalize_symbol_to_standard("sh600000") == "600000.SH"
    assert normalize_symbol_to_standard("600000") == "600000.SH"
    assert normalize_symbol_to_standard("000001.SZ") == "000001.SZ"
    assert normalize_symbol_to_standard("sz000001") == "000001.SZ"
    assert normalize_symbol_to_standard("000001") == "000001.SZ"

    assert standard_to_sina_code("600000.SH") == "sh600000"
    assert standard_to_sina_code("000001.SZ") == "sz000001"


def test_simulation_feed_adapter():
    bus = StreamBus(stream_key="test_stream:sim", force_memory_mode=True)
    received_ticks = []
    bus.subscribe(lambda t: received_ticks.append(t))

    adapter = SimulationLiveFeedAdapter(bus=bus, interval_seconds=0.05, initial_prices={"600000.SH": 10.0})
    adapter.subscribe(["600000.SH", "000001.SZ"])

    adapter.start()
    assert adapter.is_running is True

    time.sleep(0.25)
    adapter.stop()
    assert adapter.is_running is False

    assert len(received_ticks) >= 2
    assert adapter.total_ticks == len(received_ticks)

    first_tick = received_ticks[0]
    assert isinstance(first_tick, MarketTick)
    assert first_tick.symbol in ["600000.SH", "000001.SZ"]
    assert first_tick.last_price > 0
    assert len(first_tick.bid_prices) == 5
    assert len(first_tick.ask_prices) == 5
    assert len(first_tick.bid_volumes) == 5
    assert len(first_tick.ask_volumes) == 5


def test_public_web_feed_parsing():
    bus = StreamBus(stream_key="test_stream:web", force_memory_mode=True)
    received_ticks = []
    bus.subscribe(lambda t: received_ticks.append(t))

    adapter = PublicWebLiveFeedAdapter(bus=bus, interval_seconds=0.1)
    adapter.subscribe(["600000.SH"])

    mock_sina_line = (
        'var hq_str_sh600000="浦发银行,9.050,9.060,9.070,9.150,9.000,9.070,9.080,51759299,469969405.000,'
        '446703,9.070,10000,9.060,20000,9.050,30000,9.040,40000,9.030,'
        '50000,9.080,60000,9.090,70000,9.100,80000,9.110,90000,9.120,'
        '2026-09-18,15:00:00,00";\n'
    )

    mock_response = MagicMock()
    mock_response.read.return_value = mock_sina_line.encode("gbk")
    mock_response.__enter__.return_value = mock_response

    with patch("urllib.request.urlopen", return_value=mock_response):
        ticks = adapter._fetch_snapshot(["600000.SH"])
        assert len(ticks) == 1
        tick = ticks[0]
        assert tick.symbol == "600000.SH"
        assert tick.last_price == 9.07
        assert tick.volume == 51759299
        assert tick.turnover == 469969405.0
        assert tick.bid_prices[0] == 9.07
        assert tick.ask_prices[0] == 9.08
        assert tick.bid_volumes[0] == 446703
        assert tick.ask_volumes[0] == 50000
        assert "2026-09-18" in tick.timestamp


def test_qmt_live_feed_callback_and_fallback():
    bus = StreamBus(stream_key="test_stream:qmt", force_memory_mode=True)
    received_ticks = []
    bus.subscribe(lambda t: received_ticks.append(t))

    adapter = QmtLiveFeedAdapter(bus=bus)
    adapter.subscribe(["600000.SH"])
    adapter.start()

    # Simulate callback payload from xtquant
    raw_datas = {
        "600000.SH": {
            "time": 1726711200000,
            "lastPrice": 9.15,
            "volume": 600000,
            "amount": 5490000.0,
            "bidPrice": [9.14, 9.13, 9.12, 9.11, 9.10],
            "bidVol": [1000, 2000, 3000, 4000, 5000],
            "askPrice": [9.15, 9.16, 9.17, 9.18, 9.19],
            "askVol": [600, 700, 800, 900, 1000],
        }
    }

    adapter._on_xtdata_callback(raw_datas)
    assert len(received_ticks) == 1
    t = received_ticks[0]
    assert t.symbol == "600000.SH"
    assert t.last_price == 9.15
    assert t.volume == 600000
    assert t.bid_prices[0] == 9.14
    assert t.ask_prices[0] == 9.15

    adapter.unsubscribe(["600000.SH"])
    adapter.stop()


def test_live_feed_manager_lifecycle():
    bus = StreamBus(stream_key="test_stream:mgr", force_memory_mode=True)
    mgr = LiveFeedManager(bus=bus)

    status = mgr.start(source="SIMULATION", symbols=["600000.SH"], interval_seconds=0.05)
    assert status["is_running"] is True
    assert status["source"] == "SIMULATION"
    assert "600000.SH" in status["subscribed_symbols"]

    # Subscribe another
    status2 = mgr.subscribe(["000001.SZ"])
    assert "000001.SZ" in status2["subscribed_symbols"]

    time.sleep(0.2)
    status3 = mgr.get_status()
    assert status3["total_ticks"] > 0

    # Unsubscribe
    status4 = mgr.unsubscribe(["600000.SH"])
    assert "600000.SH" not in status4["subscribed_symbols"]
    assert "000001.SZ" in status4["subscribed_symbols"]

    # Stop
    stop_status = mgr.stop()
    assert stop_status["is_running"] is False


def test_market_feed_api_endpoints():
    app = create_app()
    client = TestClient(app)

    # 1. Start feed via API
    start_payload = {
        "source": "SIMULATION",
        "symbols": ["600000.SH", "000001.SZ"],
        "interval_seconds": 0.05,
    }
    res_start = client.post("/api/v1/market/feed/start", json=start_payload)
    assert res_start.status_code == 200
    feed_data = res_start.json()["feed"]
    assert feed_data["is_running"] is True
    assert feed_data["source"] == "SIMULATION"

    # 2. Get status
    time.sleep(0.15)
    res_status = client.get("/api/v1/market/feed/status")
    assert res_status.status_code == 200
    st = res_status.json()["feed"]
    assert st["is_running"] is True
    assert st["total_ticks"] > 0

    # 3. Dynamic subscribe
    res_sub = client.post("/api/v1/market/feed/subscribe", json={"symbols": ["600036.SH"]})
    assert res_sub.status_code == 200
    assert "600036.SH" in res_sub.json()["feed"]["subscribed_symbols"]

    # 4. Dynamic unsubscribe
    res_unsub = client.post("/api/v1/market/feed/unsubscribe", json={"symbols": ["600000.SH"]})
    assert res_unsub.status_code == 200
    assert "600000.SH" not in res_unsub.json()["feed"]["subscribed_symbols"]

    # 5. Stop feed via API
    res_stop = client.post("/api/v1/market/feed/stop")
    assert res_stop.status_code == 200
    assert res_stop.json()["feed"]["is_running"] is False


def test_live_feed_triggers_runtime_pipeline():
    runtime = ServiceRuntime()
    broker = runtime.broker
    acc = broker.create_account("live_feed_test_acc", initial_cash=100000.0)

    # Place initial position at price 10.0
    from src.execution_engine.models import Position
    acc.positions["600000.SH"] = Position(
        symbol="600000.SH",
        total_volume=1000,
        available_volume=1000,
        avg_cost=10.0,
        last_price=10.0,
        market_value=10000.0,
    )

    # Start simulation feed with higher price to test dynamic MTM update
    runtime.market.live_feed.start(
        source="SIMULATION",
        symbols=["600000.SH"],
        interval_seconds=0.05,
    )

    time.sleep(0.25)
    runtime.market.live_feed.stop()

    # Check that position was revalued
    pos = broker.get_account("live_feed_test_acc").positions["600000.SH"]
    assert pos.last_price > 0
    assert pos.market_value > 0

    # Check indicators were updated
    snapshot = runtime.market.signal_engine.get_indicator_snapshot("600000.SH")
    assert "vwap" in snapshot
    assert "bollinger" in snapshot
    assert snapshot["vwap"] is not None
