import pytest
from starlette.testclient import TestClient
from src.service.app import app

@pytest.fixture
def client():
    return TestClient(app)

def test_api_check_order(client):
    payload = {
        "account_id": "api_acc_01",
        "symbol": "600519.SH",
        "direction": "BUY",
        "price": 800.0,
        "volume": 100,
        "total_equity": 1000000.0,
        "available_cash": 500000.0
    }
    resp = client.post("/api/v1/risk/check_order", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["passed"] is True
    assert data["action"] == "PASS"

def test_api_circuit_breaker_status_and_reset(client):
    # Query status
    resp = client.get("/api/v1/risk/circuit_breaker/status?account_id=api_acc_01")
    assert resp.status_code == 200
    assert resp.json()["account_id"] == "api_acc_01"

    # Reset
    resp_reset = client.post("/api/v1/risk/circuit_breaker/reset", json={"account_id": "api_acc_01"})
    assert resp_reset.status_code == 200
    assert resp_reset.json()["level"] == 0

def test_api_blacklist(client):
    # Add
    resp_add = client.post("/api/v1/risk/blacklist", json={"action": "add", "symbols": ["300750.SZ"]})
    assert resp_add.status_code == 200
    assert "300750.SZ" in resp_add.json()["blacklist"]

    # Check order rejected
    payload = {
        "account_id": "api_acc_01",
        "symbol": "300750.SZ",
        "direction": "BUY",
        "price": 200.0,
        "volume": 100,
        "total_equity": 1000000.0,
        "available_cash": 500000.0
    }
    resp_check = client.post("/api/v1/risk/check_order", json=payload)
    assert resp_check.json()["passed"] is False

    # Remove
    resp_del = client.post("/api/v1/risk/blacklist", json={"action": "remove", "symbols": ["300750.SZ"]})
    assert "300750.SZ" not in resp_del.json()["blacklist"]

def test_api_alerts(client):
    resp = client.get("/api/v1/risk/alerts?limit=10")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)
