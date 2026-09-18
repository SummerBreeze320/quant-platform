import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from src.models.base import Base
from src.models.factor import FactorMetadata
from src.models.sync_log import SyncLog
from src.models.model_registry import ModelRegistry
from src.models.backtest_record import BacktestRecord
from src.common.db import get_db
from src.service.app import create_app

@pytest.fixture
def client_with_db():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    def override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app = create_app()
    app.dependency_overrides[get_db] = override_get_db
    
    with TestClient(app) as client:
        yield client

def test_health_check(client_with_db):
    response = client_with_db.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["service"] == "QuantCopliot"

def test_data_status(client_with_db):
    response = client_with_db.get("/api/v1/data/status")
    assert response.status_code == 200
    assert "qlib_initialized" in response.json()

def test_factor_api_lifecycle(client_with_db):
    # 1. Register factor
    payload = {
        "name": "api_test_factor",
        "expression": "Ref($close, -1) / $close - 1",
        "category": "reversal",
        "description": "Daily return reversal"
    }
    create_resp = client_with_db.post("/api/v1/factor/register", json=payload)
    assert create_resp.status_code == 200
    factor_id = create_resp.json()["id"]
    
    # 2. List factors
    list_resp = client_with_db.get("/api/v1/factor/list")
    assert list_resp.status_code == 200
    assert len(list_resp.json()) >= 1
    assert any(f["name"] == "api_test_factor" for f in list_resp.json())

    # 3. Get metrics
    get_resp = client_with_db.get(f"/api/v1/factor/{factor_id}/metrics")
    assert get_resp.status_code == 200
    assert get_resp.json()["name"] == "api_test_factor"

def test_rd_agent_explore_api(client_with_db):
    payload = {
        "theme": "reversal",
        "rounds": 1
    }
    resp = client_with_db.post("/api/v1/rd-agent/tasks", json=payload)
    # Without LLM API key configured, rdagent is unavailable → 503
    assert resp.status_code == 503
    assert "LLM API key" in resp.json()["detail"]

def test_model_api_lifecycle(client_with_db):
    # 1. LightGBM model
    payload_lgb = {
        "model_name": "test_lgb_model",
        "model_type": "LightGBM",
        "feature_set": ["alpha158", "api_test_factor"],
        "train_start": "2020-01-01",
        "train_end": "2024-12-31"
    }
    resp = client_with_db.post("/api/v1/model/train", json=payload_lgb)
    assert resp.status_code == 200
    assert resp.json()["model_name"] == "test_lgb_model"

    # 2. XGBoost model
    payload_xgb = {
        "model_name": "test_xgb_model",
        "model_type": "XGBoost",
        "feature_set": ["alpha158"],
        "train_start": "2020-01-01",
        "train_end": "2024-12-31",
        "params": {"max_depth": 5}
    }
    resp_xgb = client_with_db.post("/api/v1/model/train", json=payload_xgb)
    assert resp_xgb.status_code == 200
    assert resp_xgb.json()["model_name"] == "test_xgb_model"

    # 3. LSTM model
    payload_lstm = {
        "model_name": "test_lstm_model",
        "model_type": "LSTM",
        "feature_set": ["alpha158"],
        "train_start": "2020-01-01",
        "train_end": "2024-12-31",
        "params": {"hidden_size": 32, "epochs": 5}
    }
    resp_lstm = client_with_db.post("/api/v1/model/train", json=payload_lstm)
    assert resp_lstm.status_code == 200
    assert resp_lstm.json()["model_name"] == "test_lstm_model"

    # 4. Ensemble model
    payload_ensemble = {
        "model_name": "test_ensemble_model",
        "model_type": "Ensemble",
        "feature_set": ["alpha158"],
        "train_start": "2020-01-01",
        "train_end": "2024-12-31",
        "params": {"method": "rank", "base_models": ["LightGBM", "XGBoost"]}
    }
    resp_ens = client_with_db.post("/api/v1/model/train", json=payload_ensemble)
    assert resp_ens.status_code == 200
    assert resp_ens.json()["model_name"] == "test_ensemble_model"

    # 5. List all models
    list_resp = client_with_db.get("/api/v1/model/list")
    assert list_resp.status_code == 200
    names = [m["model_name"] for m in list_resp.json()]
    assert "test_lgb_model" in names
    assert "test_xgb_model" in names
    assert "test_lstm_model" in names
    assert "test_ensemble_model" in names

def test_backtest_api(client_with_db):
    # 1. Standard Top-K backtest
    payload = {
        "strategy_name": "Top30_Test",
        "top_k": 10,
        "benchmark": "000300.SH"
    }
    resp = client_with_db.post("/api/v1/backtest/run", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["strategy_name"] == "Top30_Test"
    assert "annualized_return" in data
    assert "sharpe_ratio" in data
    assert "max_drawdown" in data

    # 2. Industry-neutral backtest with neutralization
    neutral_payload = {
        "strategy_name": "Neutral_Top10",
        "top_k": 10,
        "benchmark": "000300.SH",
        "neutralize": True,
        "industry_neutral_allocation": True
    }
    neutral_resp = client_with_db.post("/api/v1/backtest/run", json=neutral_payload)
    assert neutral_resp.status_code == 200
    neutral_data = neutral_resp.json()
    assert neutral_data["strategy_name"] == "Neutral_Top10"
    assert "annualized_return" in neutral_data

    # 3. HTML Report generation endpoint
    report_payload = {
        "strategy_name": "Report_Alpha_Strategy",
        "top_k": 10,
        "benchmark": "000300.SH",
        "neutralize": True
    }
    report_resp = client_with_db.post("/api/v1/backtest/report/html", json=report_payload)
    assert report_resp.status_code == 200
    assert "text/html" in report_resp.headers["content-type"]
    assert "<!DOCTYPE html>" in report_resp.text
    assert "Report_Alpha_Strategy" in report_resp.text
    assert "Brinson" in report_resp.text

    # 4. Convex optimizer backtest endpoint
    convex_payload = {
        "strategy_name": "QP_Convex_Strategy",
        "benchmark": "000300.SH",
        "use_convex_optimizer": True,
        "risk_aversion": 1.5,
        "max_stock_weight": 0.10,
        "industry_tolerance": 0.02
    }
    convex_resp = client_with_db.post("/api/v1/backtest/run", json=convex_payload)
    assert convex_resp.status_code == 200
    convex_data = convex_resp.json()
    assert convex_data["strategy_name"] == "QP_Convex_Strategy"
    assert "annualized_return" in convex_data
    assert "sharpe_ratio" in convex_data

def test_optimizer_api(client_with_db):
    payload = {
        "alpha_scores": {"SZ000001": 0.05, "SZ000002": 0.02, "SH600000": -0.01},
        "exposures": {
            "SZ000001": {"ind_Bank": 1.0, "style_size": 0.5},
            "SZ000002": {"ind_Bank": 1.0, "style_size": -0.2},
            "SH600000": {"ind_Bank": 1.0, "style_size": 0.1}
        },
        "risk_aversion": 1.0,
        "max_stock_weight": 0.5,
        "industry_tolerance": 0.05
    }
    resp = client_with_db.post("/api/v1/optimizer/optimize", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] in ["optimal", "relaxed_optimal"]
    assert "weights" in data
    assert "predicted_tracking_error" in data
    assert "active_exposures" in data

