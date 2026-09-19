import math
import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.pms.models import StrategyType, StrategyAccount
from src.pms.manager import PortfolioManager
from src.pms.risk_analytics import RiskAnalyticsEngine, RiskMetricsSummary
from src.pms.attribution_adapter import PmsBrinsonAdapter
from src.execution_engine.models import Position
from src.service.app import create_app
from src.service.runtime import ServiceRuntime


def test_risk_analytics_pure_math():
    # 20 trading days of returns
    rets = [
        0.01, -0.005, 0.015, 0.02, -0.01,
        0.005, -0.008, 0.012, 0.018, -0.015,
        0.025, -0.002, 0.008, -0.012, 0.03,
        -0.005, 0.014, -0.006, 0.02, 0.01
    ]

    metrics = RiskAnalyticsEngine.calculate_metrics(
        returns=rets,
        risk_free_rate=0.02,
        annual_trading_days=252,
    )

    assert isinstance(metrics, RiskMetricsSummary)
    assert metrics.trading_days == 20
    assert metrics.total_return > 0
    assert metrics.annualized_return > 0
    assert metrics.annualized_volatility > 0
    assert 0 < metrics.max_drawdown < 0.1
    assert metrics.sharpe_ratio > 0
    assert metrics.sortino_ratio > 0
    assert metrics.calmar_ratio > 0
    assert 0.5 < metrics.win_rate < 1.0
    assert metrics.profit_loss_ratio > 1.0


def test_risk_analytics_with_benchmark():
    p_rets = [0.01, -0.005, 0.02, 0.015, -0.01, 0.025, -0.008, 0.018, 0.012, -0.005]
    b_rets = [0.008, -0.006, 0.012, 0.01, -0.012, 0.018, -0.005, 0.01, 0.008, -0.004]

    metrics = RiskAnalyticsEngine.calculate_metrics(
        returns=p_rets,
        benchmark_returns=b_rets,
        risk_free_rate=0.02,
    )

    assert metrics.beta is not None
    assert metrics.beta > 0
    assert metrics.alpha is not None
    assert metrics.information_ratio is not None


def test_monthly_matrix_generation():
    records = [
        {"date": "2025-11-03", "return": 0.01},
        {"date": "2025-11-15", "return": 0.02},
        {"date": "2025-12-01", "return": -0.01},
        {"date": "2026-01-05", "return": 0.03},
        {"date": "2026-01-20", "return": 0.015},
        {"date": "2026-02-10", "return": -0.008},
    ]

    res = RiskAnalyticsEngine.generate_monthly_matrix(records)
    assert "years" in res
    assert "2025" in res["years"]
    assert "2026" in res["years"]

    m = res["matrix"]
    assert "11" in m["2025"]
    assert "12" in m["2025"]
    assert "total" in m["2025"]

    assert "01" in m["2026"]
    assert "02" in m["2026"]
    assert "total" in m["2026"]
    assert m["2026"]["01"] > 0


def test_pms_brinson_attribution_with_positions():
    strat = StrategyAccount(
        strategy_id="strat_attr_test",
        name="Attribution Test Strategy",
        strategy_type=StrategyType.ALPHA,
        allocated_budget=1_000_000.0,
        current_cash=500_000.0,
        total_equity=1_000_000.0,
        positions={
            "600000.SH": Position(
                symbol="600000.SH",
                total_volume=20000,
                available_volume=20000,
                avg_cost=10.0,
                last_price=11.0,
                market_value=220_000.0,
                unrealized_pnl=20_000.0,
                unrealized_pnl_ratio=0.10,
            ),
            "000858.SZ": Position(
                symbol="000858.SZ",
                total_volume=1000,
                available_volume=1000,
                avg_cost=140.0,
                last_price=147.0,
                market_value=147_000.0,
                unrealized_pnl=7_000.0,
                unrealized_pnl_ratio=0.05,
            ),
        }
    )

    attr = PmsBrinsonAdapter.attribute_strategy(strat)
    assert "total_allocation" in attr
    assert "total_selection" in attr
    assert "total_interaction" in attr
    assert "total_excess_return" in attr

    # Brinson identity check: alloc + selec + inter == excess
    alloc = attr["total_allocation"]
    selec = attr["total_selection"]
    inter = attr["total_interaction"]
    excess = attr["total_excess_return"]
    assert np.isclose(alloc + selec + inter, excess, atol=1e-5)
    assert len(attr["industry_details"]) > 0


def test_portfolio_manager_nav_and_analytics_flow():
    mgr = PortfolioManager(initial_reserve=5_000_000.0)
    mgr.register_strategy("alpha_01", "Alpha Strat", initial_budget=2_000_000.0)

    # Day 1
    r1 = mgr.record_daily_nav("alpha_01", date="2026-09-15", equity=2_000_000.0, benchmark_return=0.0)
    assert r1.nav == 1.0
    assert r1.daily_return == 0.0

    # Day 2: +2%
    r2 = mgr.record_daily_nav("alpha_01", date="2026-09-16", equity=2_040_000.0, benchmark_return=0.005)
    assert np.isclose(r2.daily_return, 0.02, atol=1e-4)
    assert np.isclose(r2.nav, 1.02, atol=1e-4)

    # Day 3: -1%
    r3 = mgr.record_daily_nav("alpha_01", date="2026-09-17", equity=2_019_600.0, benchmark_return=-0.008)
    assert np.isclose(r3.daily_return, -0.01, atol=1e-4)
    assert r3.nav < 1.02

    # Verify history
    history = mgr.get_nav_history("alpha_01")
    assert len(history) == 3

    # Verify analytics calculation
    metrics = mgr.get_risk_analytics("alpha_01")
    assert metrics.trading_days == 2
    assert metrics.total_return > 0

    # Verify monthly matrix
    m_matrix = mgr.get_monthly_returns_matrix("alpha_01")
    assert "matrix" in m_matrix


def test_pms_analytics_api_endpoints():
    app = create_app()
    client = TestClient(app)

    # 1. Record NAV via API
    nav_payload = {
        "date": "2026-09-18",
        "benchmark_return": 0.008,
    }
    res_rec = client.post("/api/v1/pms/nav/record", json=nav_payload)
    assert res_rec.status_code == 200
    rec_data = res_rec.json()
    assert rec_data["status"] == "SUCCESS"
    assert "master" in rec_data["records"]

    # 2. Query NAV history
    res_hist = client.get("/api/v1/pms/nav/history?account_id=master")
    assert res_hist.status_code == 200
    assert len(res_hist.json()["history"]) >= 1

    # 3. Query Risk Analytics
    res_ana = client.get("/api/v1/pms/analytics?account_id=master")
    assert res_ana.status_code == 200
    metrics = res_ana.json()["metrics"]
    assert "total_return" in metrics
    assert "annualized_return" in metrics
    assert "sharpe_ratio" in metrics
    assert "max_drawdown" in metrics

    # 4. Query Monthly Returns Matrix
    res_mo = client.get("/api/v1/pms/analytics/monthly?account_id=master")
    assert res_mo.status_code == 200
    assert "data" in res_mo.json()

    # 5. Query Brinson Attribution
    res_attr = client.get("/api/v1/pms/analytics/attribution")
    assert res_attr.status_code == 200
    attr_data = res_attr.json()["attribution"]
    assert "total_allocation" in attr_data
    assert "total_selection" in attr_data
    assert "total_interaction" in attr_data
