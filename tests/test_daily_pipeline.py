import pytest
from fastapi.testclient import TestClient

from src.common.redis_client import get_redis
from src.execution_engine.models import Order, OrderDirection, OrderType
from src.service.app import create_app
from src.service.runtime import ServiceRuntime
from src.tasks.pipeline import PremarketRebalancePipeline, DailySettlementPipeline
from src.tasks.scheduler import get_scheduler


def test_premarket_rebalance_pipeline_with_cached_predictions():
    runtime = ServiceRuntime()
    redis_client = get_redis()

    # Seed mock predictions in Redis
    payload = {
        "date": "2026-09-19",
        "model_name": "lgbm_alpha_test",
        "predictions": [
            {"symbol": "600000.SH", "score": 0.85, "rank": 1},
            {"symbol": "000001.SZ", "score": 0.75, "rank": 2},
            {"symbol": "600036.SH", "score": 0.65, "rank": 3},
            {"symbol": "000858.SZ", "score": 0.55, "rank": 4},
            {"symbol": "601318.SH", "score": 0.45, "rank": 5},
        ]
    }
    try:
        redis_client.set_json("predictions:latest", payload, ex=3600)
    except Exception:
        pass  # Memory mode fallback

    result = PremarketRebalancePipeline.run(
        strategy_id="pipeline_test_strat",
        trade_date="2026-09-19",
        top_k=5,
        max_stock_weight=0.08,
        algo_type="DIRECT",
        execution_mode="SYNC",
        runtime=runtime,
        current_prices_override={
            "600000.SH": 10.0,
            "000001.SZ": 15.0,
            "600036.SH": 30.0,
            "000858.SZ": 150.0,
            "601318.SH": 50.0,
        }
    )

    assert result["status"] == "SUCCESS"
    assert result["strategy_id"] == "pipeline_test_strat"
    assert len(result["target_weights"]) > 0

    # Ensure each weight is bounded by max_stock_weight
    for sym, weight in result["target_weights"].items():
        assert weight <= 0.08 + 1e-4

    # Verify positions in broker
    acc = runtime.broker.get_account("pipeline_test_strat")
    assert len(acc.positions) > 0


def test_premarket_rebalance_pipeline_fallback():
    runtime = ServiceRuntime()

    # Clear predictions to test graceful fallback
    result = PremarketRebalancePipeline.run(
        strategy_id="fallback_test_strat",
        top_k=3,
        max_stock_weight=0.08,
        runtime=runtime,
        current_prices_override={"600000.SH": 10.0, "000001.SZ": 12.0, "600036.SH": 35.0}
    )

    assert result["status"] == "SUCCESS"
    assert len(result["target_weights"]) > 0
    acc = runtime.broker.get_account("fallback_test_strat")
    assert len(acc.positions) > 0


def test_daily_settlement_pipeline():
    runtime = ServiceRuntime()
    broker = runtime.broker
    acc = broker.create_account("settle_test_acc", initial_cash=1_000_000.0)

    # Buy shares to create a position with frozen volume
    order = Order(
        order_id="settle_ord_1",
        account_id="settle_test_acc",
        symbol="000001.SZ",
        direction=OrderDirection.BUY,
        order_type=OrderType.LIMIT,
        price=10.0,
        volume=1000,
    )
    broker.submit_order(order)

    pos = acc.positions["000001.SZ"]
    assert pos.frozen_volume == 1000
    assert pos.available_volume == 0

    # Run daily settlement pipeline
    result = DailySettlementPipeline.run(
        account_id="settle_test_acc",
        trade_date="2026-09-19",
        runtime=runtime,
    )

    assert result["status"] == "SUCCESS"
    assert "settle_test_acc" in result["settled_accounts"]

    # Verify shares unlocked for T+1
    pos_after = broker.get_account("settle_test_acc").positions["000001.SZ"]
    assert pos_after.frozen_volume == 0
    assert pos_after.available_volume == 1000


def test_scheduler_jobs_registration():
    scheduler = get_scheduler()
    scheduler.configure_jobs()
    job_ids = [j.id for j in scheduler.get_jobs()]

    assert "premarket_rebalance" in job_ids
    assert "start_market_feed" in job_ids
    assert "stop_market_feed" in job_ids
    assert "daily_settlement" in job_ids
    assert "daily_data_sync" in job_ids
    assert "daily_model_predict" in job_ids
    assert "weekend_factor_mining" in job_ids


def test_tasks_api_endpoints():
    app = create_app()
    client = TestClient(app)

    # 1. Query scheduler jobs
    resp_jobs = client.get("/api/v1/tasks/jobs")
    assert resp_jobs.status_code == 200
    job_list = resp_jobs.json()
    job_ids = [j["id"] for j in job_list]
    assert "premarket_rebalance" in job_ids
    assert "daily_settlement" in job_ids

    # 2. Trigger rebalance pipeline via API
    reb_payload = {
        "strategy_id": "api_pipeline_strat",
        "top_k": 3,
        "max_stock_weight": 0.08,
        "algo_type": "DIRECT",
        "execution_mode": "SYNC",
        "current_prices_override": {"600000.SH": 10.0, "000001.SZ": 15.0, "600036.SH": 30.0}
    }
    resp_reb = client.post("/api/v1/tasks/rebalance", json=reb_payload)
    assert resp_reb.status_code == 200
    reb_data = resp_reb.json()
    assert reb_data["status"] == "SUCCESS"
    assert reb_data["strategy_id"] == "api_pipeline_strat"
    assert len(reb_data["target_weights"]) > 0

    # 3. Trigger settlement pipeline via API
    settle_payload = {
        "account_id": "api_pipeline_strat",
        "trade_date": "2026-09-19"
    }
    resp_set = client.post("/api/v1/tasks/settle", json=settle_payload)
    assert resp_set.status_code == 200
    set_data = resp_set.json()
    assert set_data["status"] == "SUCCESS"
    assert "api_pipeline_strat" in set_data["settled_accounts"]
