from fastapi.testclient import TestClient
from src.service.app import app

client = TestClient(app)

def test_market_api_endpoints():
    # 1. Post a tick
    tick_payload = {
        "symbol": "600519.SH",
        "last_price": 1800.0,
        "volume": 1000,
        "turnover": 1800000.0,
        "bid_prices": [1799.0],
        "bid_volumes": [200],
        "ask_prices": [1801.0],
        "ask_volumes": [100]
    }
    r = client.post("/api/v1/market/tick", json=tick_payload)
    assert r.status_code == 200
    res = r.json()
    assert res["status"] == "SUCCESS"
    assert "msg_id" in res

    # 2. Query indicators
    r_ind = client.get("/api/v1/market/indicators?symbol=600519.SH")
    assert r_ind.status_code == 200
    ind_data = r_ind.json()
    assert "vwap" in ind_data
    assert "bollinger" in ind_data
    assert "ofi" in ind_data

    # 3. Query signals
    r_sig = client.get("/api/v1/market/signals")
    assert r_sig.status_code == 200
    assert isinstance(r_sig.json(), list)

    # 4. WebSocket test
    with client.websocket_connect("/api/v1/market/ws") as websocket:
        # Push another tick via REST
        client.post("/api/v1/market/tick", json=tick_payload)
        data = websocket.receive_json()
        assert data["type"] == "TICK"
        assert data["data"]["symbol"] == "600519.SH"
