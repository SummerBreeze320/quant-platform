from fastapi.testclient import TestClient
from src.service.app import app

client = TestClient(app)

def test_pms_api_lifecycle():
    # 1. Query master account
    r_master = client.get("/api/v1/pms/master")
    assert r_master.status_code == 200
    data_m = r_master.json()
    assert data_m["master_id"] == "master_default"
    assert data_m["total_equity"] >= 1_000_000.0

    # 2. Register new sub-strategy
    req_strat = {
        "strategy_id": "api_test_strat_01",
        "name": "API测试沪深300多因子",
        "strategy_type": "INDEX_ENHANCEMENT",
        "initial_budget": 1_000_000.0
    }
    r_reg = client.post("/api/v1/pms/strategies", json=req_strat)
    assert r_reg.status_code == 200
    strat_data = r_reg.json()
    assert strat_data["strategy_id"] == "api_test_strat_01"
    assert strat_data["current_cash"] == 1_000_000.0

    # 3. List strategies & performance
    r_list = client.get("/api/v1/pms/strategies")
    assert r_list.status_code == 200
    strats = r_list.json()
    assert any(s["strategy_id"] == "api_test_strat_01" for s in strats)

    # 4. Request allocation plan
    req_alloc = {
        "method": "EQUAL_WEIGHT",
        "investable_ratio": 0.90,
        "apply": True
    }
    r_alloc = client.post("/api/v1/pms/allocate", json=req_alloc)
    assert r_alloc.status_code == 200
    plan_data = r_alloc.json()
    assert plan_data["method"] == "EQUAL_WEIGHT"
    assert "api_test_strat_01" in plan_data["target_ratios"]

    # 5. Lookthrough aggregation
    r_look = client.get("/api/v1/pms/lookthrough")
    assert r_look.status_code == 200
    look_data = r_look.json()
    assert "positions" in look_data
    assert "max_stock_concentration" in look_data
