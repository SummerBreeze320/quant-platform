import pytest
from fastapi.testclient import TestClient

from src.common.redis_client import RedisClient
from src.execution_engine import ExecutionCoordinator
from src.risk_engine.models import OrderRiskRequest
from src.risk_engine import CircuitBreakerManager, PreTradeRiskChecker
from src.service.app import create_app


@pytest.fixture
def client(monkeypatch):
    # Only replace the external Redis probe; all domain services remain real.
    monkeypatch.setattr(RedisClient, "ping", lambda self: False)
    return TestClient(create_app())


def register(client, account="integrated", budget=2_000_000):
    response = client.post("/api/v1/pms/strategies", json={
        "strategy_id": account, "name": account, "initial_budget": budget,
    })
    assert response.status_code == 200, response.text


def rebalance(client, account="integrated", symbol="600000.SH", weight=0.05):
    response = client.post("/api/v1/execution/rebalance", json={
        "account_id": account,
        "target_weights": {symbol: weight},
        "current_prices": {symbol: 10},
    })
    assert response.status_code == 200, response.text
    return response.json()


def test_pms_budget_is_used_by_execution_and_trades_are_visible(client):
    register(client)
    account = client.get("/api/v1/execution/account", params={"account_id": "integrated"}).json()
    assert account["available_cash"] == 2_000_000
    result = rebalance(client)
    assert len(result["executed_trades"]) == 1
    assert result["executed_trades"][0]["volume"] == 10000
    portfolio = client.get("/api/v1/pms/lookthrough").json()
    position = portfolio["positions"]["600000.SH"]
    assert position["total_volume"] == 10000
    assert position["contributing_strategies"] == ["integrated"]


def test_api_blacklist_blocks_execution_and_removal_restores_it(client):
    register(client, "blacklisted")
    client.post("/api/v1/risk/blacklist", json={"action": "add", "symbols": ["600000.SH"]})
    result = rebalance(client, "blacklisted")
    assert result["executed_trades"] == []
    assert len(result["rejected_orders"]) == 1
    assert client.get("/api/v1/risk/alerts").json()
    client.post("/api/v1/risk/blacklist", json={"action": "remove", "symbols": ["600000.SH"]})
    assert len(rebalance(client, "blacklisted")["executed_trades"]) == 1


def test_api_circuit_breaker_blocks_execution_until_reset(client):
    register(client, "halted_api")
    response = client.post("/api/v1/risk/check_portfolio", json={
        "account_id": "halted_api", "total_equity": 1_900_000,
        "high_watermark": 2_000_000, "positions": {},
    })
    assert response.status_code == 200
    assert response.json()["circuit_breaker_level"] == 3
    result = rebalance(client, "halted_api")
    assert result["executed_trades"] == []
    assert result["circuit_breaker_level"] == 3
    client.post("/api/v1/risk/circuit_breaker/reset", json={"account_id": "halted_api"})
    assert len(rebalance(client, "halted_api")["executed_trades"]) == 1


def test_application_instances_do_not_share_blacklists(client):
    client.post("/api/v1/risk/blacklist", json={"action": "add", "symbols": ["ISOLATED.SH"]})
    other = TestClient(create_app())
    result = other.post("/api/v1/risk/blacklist", json={"action": "list"}).json()
    assert "ISOLATED.SH" not in result["blacklist"]


def test_default_coordinator_checker_observes_its_circuit_breaker():
    coordinator = ExecutionCoordinator()
    coordinator.circuit_breaker.update_equity("halted", 100)
    coordinator.circuit_breaker.update_equity("halted", 90)
    result = coordinator.risk_checker.check_order(OrderRiskRequest(
        account_id="halted", symbol="600000.SH", direction="BUY", price=10,
        volume=100, total_equity=1_000_000, available_cash=1_000_000,
    ))
    assert not result.passed


def tick(symbol, price, volume, turnover, bid_volume):
    return {
        "symbol": symbol, "last_price": price, "volume": volume, "turnover": turnover,
        "bid_prices": [9.9], "bid_volumes": [bid_volume],
        "ask_prices": [12.1], "ask_volumes": [100],
    }


def test_market_indicator_api_returns_calculated_values_without_advancing(client):
    client.post("/api/v1/market/tick", json=tick("INDICATOR.SH", 10, 100, 1000, 100))
    client.post("/api/v1/market/tick", json=tick("INDICATOR.SH", 12, 200, 2200, 200))
    snapshot = client.get("/api/v1/market/indicators", params={"symbol": "INDICATOR.SH"}).json()
    assert snapshot["vwap"] == 11
    assert snapshot["bollinger"]["mean"] == 11
    assert snapshot["bollinger"]["pct_b"] == pytest.approx(0.7357)
    assert snapshot["ofi"] == pytest.approx(0.3333)
    assert client.get("/api/v1/market/indicators", params={"symbol": "INDICATOR.SH"}).json() == snapshot


def test_api_blacklist_blocks_market_signal(client):
    client.post("/api/v1/risk/blacklist", json={"action": "add", "symbols": ["MARKET.SH"]})
    client.post("/api/v1/market/tick", json=tick("MARKET.SH", 10, 100, 1000, 100))
    client.post("/api/v1/market/tick", json=tick("MARKET.SH", 10, 200, 2000, 1000))
    signals = client.get("/api/v1/market/signals").json()
    assert signals[-1]["symbol"] == "MARKET.SH"
    assert signals[-1]["status"] == "REJECTED"
    assert client.get("/api/v1/execution/trades", params={"account_id": "hft_stream_01"}).json() == []


def test_market_trade_is_visible_in_execution_and_pms(client):
    client.post("/api/v1/market/tick", json=tick("TRADE.SH", 10, 100, 1000, 100))
    client.post("/api/v1/market/tick", json=tick("TRADE.SH", 10, 200, 2000, 1000))
    trades = client.get("/api/v1/execution/trades", params={"account_id": "hft_stream_01"}).json()
    assert len(trades) == 1
    assert trades[0]["symbol"] == "TRADE.SH"
    portfolio = client.get("/api/v1/pms/lookthrough").json()
    assert portfolio["positions"]["TRADE.SH"]["total_volume"] == 100


def test_replay_honors_requested_interval(client, monkeypatch):
    intervals = []
    monkeypatch.setattr("src.market_feed.replay.time.sleep", intervals.append)
    response = client.post("/api/v1/market/replay", json={
        "ticks": [tick("REPLAY.SH", 10, 100, 1000, 100)], "speed": 0.25,
    })
    assert response.status_code == 200
    assert response.json()["replayed_count"] == 1
    assert intervals == [0.25]


def test_pms_registration_cannot_reset_an_existing_trading_account(client):
    result = rebalance(client, "existing_trader")
    assert result["executed_trades"]
    account_before = client.get("/api/v1/execution/account", params={"account_id": "existing_trader"}).json()
    master_before = client.get("/api/v1/pms/master").json()
    response = client.post("/api/v1/pms/strategies", json={
        "strategy_id": "existing_trader", "name": "collision", "initial_budget": 500_000,
    })
    assert response.status_code == 400
    assert client.get("/api/v1/execution/account", params={"account_id": "existing_trader"}).json() == account_before
    master_after = client.get("/api/v1/pms/master").json()
    assert master_after["reserve_cash"] == master_before["reserve_cash"]
    assert master_after["total_equity"] == master_before["total_equity"]
    assert "existing_trader" not in master_after["strategies"]


def test_market_circuit_breaker_is_shared_with_risk_api(client):
    response = client.post("/api/v1/risk/check_portfolio", json={
        "account_id": "hft_stream_01", "total_equity": 4_750_000,
        "high_watermark": 5_000_000, "positions": {},
    })
    assert response.json()["circuit_breaker_level"] == 3
    client.post("/api/v1/market/tick", json=tick("HALT.SH", 10, 100, 1000, 100))
    client.post("/api/v1/market/tick", json=tick("HALT.SH", 10, 200, 2000, 1000))
    assert client.get("/api/v1/market/signals").json()[-1]["status"] == "REJECTED"
    assert client.get("/api/v1/execution/trades", params={"account_id": "hft_stream_01"}).json() == []
    client.post("/api/v1/risk/circuit_breaker/reset", json={"account_id": "hft_stream_01"})
    # A different symbol avoids the intentional per-symbol signal cooldown.
    client.post("/api/v1/market/tick", json=tick("RESUMED.SH", 10, 100, 1000, 100))
    client.post("/api/v1/market/tick", json=tick("RESUMED.SH", 10, 200, 2000, 1000))
    assert client.get("/api/v1/market/signals").json()[-1]["status"] == "EXECUTED"
    state = client.get("/api/v1/risk/circuit_breaker/status", params={"account_id": "hft_stream_01"}).json()
    assert state["current_equity"] > 4_990_000


def test_market_strategy_budget_is_part_of_master_equity(client):
    master = client.get("/api/v1/pms/master").json()
    assert master["total_equity"] == 10_000_000
    assert master["reserve_cash"] == 5_000_000
    assert master["strategies"]["hft_stream_01"]["current_cash"] == 5_000_000


def test_application_market_indicators_and_websockets_are_isolated(client):
    other = TestClient(create_app())
    with client.websocket_connect("/api/v1/market/ws") as first_ws:
        with other.websocket_connect("/api/v1/market/ws") as second_ws:
            client.post("/api/v1/market/tick", json=tick("FIRST.SH", 10, 100, 1000, 100))
            other.post("/api/v1/market/tick", json=tick("SECOND.SH", 20, 100, 2000, 100))
            assert first_ws.receive_json()["data"]["symbol"] == "FIRST.SH"
            assert second_ws.receive_json()["data"]["symbol"] == "SECOND.SH"
    unknown = other.get("/api/v1/market/indicators", params={"symbol": "FIRST.SH"}).json()
    assert unknown == {"symbol": "FIRST.SH", "vwap": 0, "bollinger": {"mean": 0, "pct_b": 0.5}, "ofi": 0}


def test_negative_replay_interval_is_rejected(client):
    response = client.post("/api/v1/market/replay", json={"ticks": [], "speed": -1})
    assert response.status_code == 422


def test_coordinator_rejects_conflicting_circuit_breakers():
    with pytest.raises(ValueError, match="same circuit breaker"):
        ExecutionCoordinator(risk_checker=PreTradeRiskChecker(), circuit_breaker=CircuitBreakerManager())


def test_pms_withdrawal_does_not_become_a_trading_drawdown(client):
    assert rebalance(client, "hft_stream_01")["executed_trades"]
    allocation = client.post("/api/v1/pms/allocate", json={"investable_ratio": 0.4, "apply": True})
    assert allocation.status_code == 200
    result = rebalance(client, "hft_stream_01", weight=0.08)
    assert result["executed_trades"]
    assert result["circuit_breaker_level"] == 0


def test_capital_flow_preserves_real_drawdown_and_halt(client):
    client.post("/api/v1/risk/check_portfolio", json={
        "account_id": "hft_stream_01", "total_equity": 5_000_000,
        "high_watermark": 6_000_000, "positions": {},
    })
    before = client.get("/api/v1/risk/circuit_breaker/status", params={"account_id": "hft_stream_01"}).json()
    for ratio, expected_equity in [(0.4, 4_000_000), (0.6, 6_000_000)]:
        response = client.post("/api/v1/pms/allocate", json={"investable_ratio": ratio, "apply": True})
        assert response.status_code == 200
        after = client.get("/api/v1/risk/circuit_breaker/status", params={"account_id": "hft_stream_01"}).json()
        assert after["current_equity"] == expected_equity
        assert after["current_equity"] / after["high_watermark"] == pytest.approx(5 / 6)
        assert after["max_drawdown"] == before["max_drawdown"]
        assert after["level"] == before["level"] == 3
        assert after["trigger_reason"] == before["trigger_reason"]
        assert after["last_triggered_at"] == before["last_triggered_at"]
