import pytest
from starlette.testclient import TestClient
from src.service.app import app

@pytest.fixture
def client():
    return TestClient(app)

def test_api_rebalance_and_account(client):
    # 1. Check account
    resp_acc = client.get("/api/v1/execution/account?account_id=api_exec_acc")
    assert resp_acc.status_code == 200
    assert resp_acc.json()["total_equity"] >= 0

    # 2. Execute rebalance
    payload = {
        "account_id": "api_exec_acc",
        "target_weights": {"600519.SH": 0.05},
        "current_prices": {"600519.SH": 500.0},
        "algo_type": "DIRECT"
    }
    resp_reb = client.post("/api/v1/execution/rebalance", json=payload)
    assert resp_reb.status_code == 200
    data = resp_reb.json()
    assert len(data["executed_trades"]) > 0

    # 3. Overnight settle
    resp_set = client.post("/api/v1/execution/settle", json={"account_id": "api_exec_acc"})
    assert resp_set.status_code == 200

    # 4. Trades
    resp_trd = client.get("/api/v1/execution/trades?account_id=api_exec_acc")
    assert resp_trd.status_code == 200
    assert len(resp_trd.json()) > 0
