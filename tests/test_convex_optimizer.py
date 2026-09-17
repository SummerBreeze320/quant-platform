import pytest
import numpy as np
import pandas as pd
from src.qlib_engine.optimizer import ConvexOptimizer
from src.qlib_engine.risk_model import BarraRiskModel

@pytest.fixture
def optimizer_sample_data():
    """Generates synthetic data for convex optimizer tests."""
    np.random.seed(42)
    n_stocks = 20
    symbols = [f"SH{600000+i:06d}" for i in range(n_stocks)]
    
    # Expected alpha score
    alpha = np.random.normal(0.01, 0.02, size=n_stocks)
    
    # 2 Industries
    ind_labels = ["Tech"] * 10 + ["Bank"] * 10
    ind_dummies = pd.get_dummies(ind_labels, dtype=float)
    
    # 1 Style factor
    size = np.random.randn(n_stocks)
    
    exposures = pd.DataFrame({
        "ind_Tech": ind_dummies["Tech"].values,
        "ind_Bank": ind_dummies["Bank"].values,
        "style_size": size
    }, index=symbols)
    
    # Factor covariance (3 x 3)
    F = np.diag([0.0004, 0.0002, 0.0001])
    
    # Specific variance (20,)
    delta = np.full(n_stocks, 0.0005)
    
    # Benchmark weights (50% Tech, 50% Bank, equal in each)
    w_b = np.full(n_stocks, 1.0 / n_stocks)
    
    return {
        "symbols": symbols,
        "alpha": pd.Series(alpha, index=symbols),
        "exposures": exposures,
        "factor_cov": pd.DataFrame(F, index=exposures.columns, columns=exposures.columns),
        "specific_var": pd.Series(delta, index=symbols),
        "benchmark_weights": pd.Series(w_b, index=symbols)
    }

def test_convex_optimizer_basic_long_only(optimizer_sample_data):
    data = optimizer_sample_data
    opt = ConvexOptimizer(
        risk_aversion=1.0,
        max_stock_weight=0.15,
        min_stock_weight=0.0
    )
    
    result = opt.optimize(
        alpha=data["alpha"],
        exposures=data["exposures"],
        factor_cov=data["factor_cov"],
        specific_var=data["specific_var"]
    )
    
    weights = result["weights"]
    assert len(weights) == len(data["symbols"])
    assert np.isclose(weights.sum(), 1.0, atol=1e-4)
    assert np.all(weights >= -1e-6)
    assert np.all(weights <= 0.15 + 1e-4)
    assert result["predicted_tracking_error"] >= 0.0

def test_convex_optimizer_industry_and_style_constraints(optimizer_sample_data):
    data = optimizer_sample_data
    opt = ConvexOptimizer(
        risk_aversion=1.5,
        max_stock_weight=0.15,
        industry_tolerance=0.01,   # Max 1% active industry tilt
        style_tolerance=0.10       # Max 0.1 active style exposure
    )
    
    result = opt.optimize(
        alpha=data["alpha"],
        exposures=data["exposures"],
        factor_cov=data["factor_cov"],
        specific_var=data["specific_var"],
        benchmark_weights=data["benchmark_weights"],
        industry_factors=["ind_Tech", "ind_Bank"],
        style_factors=["style_size"]
    )
    
    weights = result["weights"]
    w_b = data["benchmark_weights"]
    h = weights - w_b
    
    # Check industry tilts: X_ind.T @ h <= tolerance
    ind_X = data["exposures"][["ind_Tech", "ind_Bank"]].values
    ind_tilts = np.abs(ind_X.T @ h.values)
    assert np.all(ind_tilts <= 0.01 + 1e-4)
    
    # Check style tilt
    style_X = data["exposures"][["style_size"]].values
    style_tilt = np.abs(style_X.T @ h.values)
    assert np.all(style_tilt <= 0.10 + 1e-4)

def test_convex_optimizer_turnover_penalty(optimizer_sample_data):
    data = optimizer_sample_data
    
    # Previous holdings concentrated in first 5 stocks
    w0 = pd.Series(0.0, index=data["symbols"])
    w0.iloc[:5] = 0.2
    
    # Optimizer without turnover penalty
    opt_free = ConvexOptimizer(risk_aversion=1.0, max_stock_weight=0.2, turnover_penalty=0.0)
    res_free = opt_free.optimize(
        alpha=data["alpha"],
        exposures=data["exposures"],
        factor_cov=data["factor_cov"],
        specific_var=data["specific_var"],
        prev_weights=w0
    )
    
    # Optimizer with strong turnover penalty
    opt_penalized = ConvexOptimizer(risk_aversion=1.0, max_stock_weight=0.2, turnover_penalty=0.1)
    res_penalized = opt_penalized.optimize(
        alpha=data["alpha"],
        exposures=data["exposures"],
        factor_cov=data["factor_cov"],
        specific_var=data["specific_var"],
        prev_weights=w0
    )
    
    turnover_free = 0.5 * np.sum(np.abs(res_free["weights"] - w0))
    turnover_pen = 0.5 * np.sum(np.abs(res_penalized["weights"] - w0))
    
    assert turnover_pen < turnover_free

def test_convex_optimizer_fallback_on_infeasible(optimizer_sample_data):
    data = optimizer_sample_data
    # 20 stocks, max weight 0.01 -> sum <= 0.20 < 1.0 (impossible to sum to 1.0)
    # The optimizer should relax constraints gracefully and still return a valid sum-to-1 allocation
    opt = ConvexOptimizer(max_stock_weight=0.01)
    res = opt.optimize(
        alpha=data["alpha"],
        exposures=data["exposures"],
        factor_cov=data["factor_cov"],
        specific_var=data["specific_var"]
    )
    
    assert res["status"] in ["optimal", "relaxed_optimal"]
    assert np.isclose(res["weights"].sum(), 1.0, atol=1e-3)

def test_backtest_with_convex_optimizer():
    from src.qlib_engine.backtest import BacktestEngine
    dates = ["2026-01-01", "2026-01-02"]
    symbols = ["S1", "S2", "S3", "S4", "S5"]
    
    records = []
    for d in dates:
        for i, s in enumerate(symbols):
            records.append({
                "date": d,
                "symbol": s,
                "score": float(i * 0.1),
                "ret": float((i - 2) * 0.01),
                "industry": "Tech" if i < 3 else "Bank",
                "market_cap": float(1e9 * (i + 1))
            })
    pred_df = pd.DataFrame(records)
    
    engine = BacktestEngine(
        use_convex_optimizer=True,
        max_stock_weight=0.35,
        risk_aversion=1.0,
        industry_tolerance=0.05
    )
    res = engine.run_backtest(pred_df)
    
    assert "total_return" in res
    assert "sharpe_ratio" in res
    assert res["n_trading_days"] == 2

