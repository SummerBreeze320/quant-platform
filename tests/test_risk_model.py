import pytest
import numpy as np
import pandas as pd
from src.qlib_engine.risk_model import BarraRiskModel

@pytest.fixture
def sample_cross_section_data():
    """Generates synthetic cross-sectional data for Barra risk model tests."""
    np.random.seed(42)
    n_stocks = 50
    symbols = [f"SH{600000+i:06d}" for i in range(n_stocks)]
    
    # 3 Industry indicators (One-Hot)
    industries = ["Bank", "Tech", "Consumer"]
    ind_choices = np.random.choice(industries, size=n_stocks)
    
    # 2 Style factors (Size, Momentum) standard normal
    size = np.random.randn(n_stocks)
    momentum = np.random.randn(n_stocks)
    market_cap = np.exp(size + 10.0)
    
    # True factor returns: Bank=0.01, Tech=0.03, Consumer=0.005, Size=-0.015, Momentum=0.02
    ind_dummy = pd.get_dummies(ind_choices, dtype=float)
    X = pd.DataFrame({
        "industry_Bank": ind_dummy.get("Bank", 0.0),
        "industry_Tech": ind_dummy.get("Tech", 0.0),
        "industry_Consumer": ind_dummy.get("Consumer", 0.0),
        "style_size": size,
        "style_momentum": momentum
    })
    
    true_f = np.array([0.01, 0.03, 0.005, -0.015, 0.02])
    residual = np.random.normal(0, 0.01, size=n_stocks)
    returns = X.values @ true_f + residual
    
    df = pd.DataFrame({
        "symbol": symbols,
        "ret": returns,
        "market_cap": market_cap,
        "industry_Bank": X["industry_Bank"],
        "industry_Tech": X["industry_Tech"],
        "industry_Consumer": X["industry_Consumer"],
        "style_size": size,
        "style_momentum": momentum
    })
    return df, ["industry_Bank", "industry_Tech", "industry_Consumer", "style_size", "style_momentum"]

def test_estimate_factor_returns(sample_cross_section_data):
    df, factor_cols = sample_cross_section_data
    risk_model = BarraRiskModel()
    
    res = risk_model.estimate_factor_returns(
        df=df,
        factor_cols=factor_cols,
        ret_col="ret",
        weight_col="market_cap"
    )
    
    assert "factor_returns" in res
    assert "residuals" in res
    assert "r2" in res
    
    f_returns = res["factor_returns"]
    assert len(f_returns) == len(factor_cols)
    assert res["r2"] > 0.5  # Should capture strong signal
    
    # Residuals should have length equal to stock universe
    residuals = res["residuals"]
    assert len(residuals) == len(df)

def test_factor_covariance_and_psd():
    np.random.seed(42)
    # Generate 60 days of factor returns for 5 factors
    dates = pd.date_range("2026-01-01", periods=60)
    factor_names = ["ind_1", "ind_2", "style_size", "style_value", "style_mom"]
    f_ret_matrix = np.random.randn(60, 5) * 0.02
    factor_returns_df = pd.DataFrame(f_ret_matrix, index=dates, columns=factor_names)
    
    risk_model = BarraRiskModel()
    cov_matrix = risk_model.compute_factor_covariance(factor_returns_df, half_life=21)
    
    assert cov_matrix.shape == (5, 5)
    # Symmetry check
    np.testing.assert_allclose(cov_matrix.values, cov_matrix.values.T, atol=1e-8)
    
    # Positive semi-definite check (eigenvalues >= 0)
    eigenvalues = np.linalg.eigvalsh(cov_matrix.values)
    assert np.all(eigenvalues >= 0.0)

def test_specific_risk_computation():
    np.random.seed(42)
    dates = pd.date_range("2026-01-01", periods=60)
    symbols = ["S1", "S2", "S3", "S4"]
    res_matrix = np.random.normal(0, 0.015, size=(60, 4))
    residuals_df = pd.DataFrame(res_matrix, index=dates, columns=symbols)
    
    risk_model = BarraRiskModel()
    spec_var = risk_model.compute_specific_risk(residuals_df, half_life=21)
    
    assert len(spec_var) == 4
    assert np.all(spec_var.values > 0.0)

def test_predict_risk_decomposition():
    # 3 stocks, 2 factors
    factor_cov = np.array([
        [0.0004, 0.0001],
        [0.0001, 0.0009]
    ])
    exposures = np.array([
        [1.0, 0.5],
        [1.0, -0.5],
        [0.0, 1.2]
    ])
    spec_var = np.array([0.0002, 0.0003, 0.0001])
    
    weights = np.array([0.4, 0.3, 0.3])
    benchmark_weights = np.array([0.33, 0.33, 0.34])
    
    risk_model = BarraRiskModel()
    risk_metrics = risk_model.predict_portfolio_risk(
        weights=weights,
        exposures=exposures,
        factor_cov=factor_cov,
        specific_var=spec_var,
        benchmark_weights=benchmark_weights
    )
    
    assert "total_risk" in risk_metrics
    assert "factor_risk" in risk_metrics
    assert "specific_risk" in risk_metrics
    assert "tracking_error" in risk_metrics
    assert "active_factor_risk" in risk_metrics
    assert "active_specific_risk" in risk_metrics
    
    # Active risk square should decompose into active factor variance + active specific variance
    te_var = risk_metrics["tracking_error"] ** 2
    act_sum = risk_metrics["active_factor_risk"] ** 2 + risk_metrics["active_specific_risk"] ** 2
    np.testing.assert_allclose(te_var, act_sum, rtol=1e-4)
