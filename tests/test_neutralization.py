import pytest
import numpy as np
import pandas as pd
from src.qlib_engine.neutralization import FactorNeutralizer
from src.qlib_engine.optimizer import PortfolioOptimizer
from src.qlib_engine.backtest import BacktestEngine

def test_mad_winsorization():
    data = pd.Series([1.0, 2.0, 2.1, 2.2, 2.0, 1.9, 2.0, 100.0, -100.0])
    winsorized = FactorNeutralizer.winsorize_mad(data, n=3.0)
    assert winsorized.max() < 10.0
    assert winsorized.min() > -10.0
    # Normal in-range points should remain unchanged
    assert winsorized.iloc[1] == 2.0

def test_zscore_standardization():
    data = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])
    standardized = FactorNeutralizer.standardize_zscore(data)
    assert np.isclose(standardized.mean(), 0.0, atol=1e-6)
    assert np.isclose(standardized.std(), 1.0, atol=1e-6)

def test_cross_sectional_neutralization():
    np.random.seed(42)
    n = 200
    industries = np.random.choice(["Bank", "Tech", "Pharma", "Consumer"], size=n)
    market_cap = np.random.uniform(1e9, 1e11, size=n)
    log_cap = np.log(market_cap)

    # Synthetic raw score heavily correlated with Tech industry and Size
    ind_bias = np.array([2.0 if ind == "Tech" else -1.0 if ind == "Bank" else 0.0 for ind in industries])
    cap_bias = log_cap * 0.5
    raw_score = ind_bias + cap_bias + np.random.randn(n) * 0.3

    df = pd.DataFrame({
        "symbol": [f"SZ{i:06d}" for i in range(n)],
        "score": raw_score,
        "industry": industries,
        "market_cap": market_cap
    })

    # Neutralize score against industry and market_cap
    neutralized_df = FactorNeutralizer.neutralize(
        df,
        score_col="score",
        industry_col="industry",
        market_cap_col="market_cap"
    )

    clean_score = neutralized_df["score_neutral"]
    
    # 1. Check correlation with log market cap is effectively zero
    corr_cap = np.corrcoef(clean_score, log_cap)[0, 1]
    assert abs(corr_cap) < 1e-4

    # 2. Check mean score across each industry is zero (no industry bias remaining)
    for ind in ["Bank", "Tech", "Pharma", "Consumer"]:
        mean_ind_score = clean_score[df["industry"] == ind].mean()
        assert abs(mean_ind_score) < 1e-4

def test_portfolio_optimizer_industry_alignment():
    # Benchmark weights
    benchmark_industry_weights = {
        "Bank": 0.40,
        "Tech": 0.30,
        "Pharma": 0.20,
        "Consumer": 0.10
    }

    df = pd.DataFrame({
        "symbol": [f"SZ{i:06d}" for i in range(20)],
        "score": np.random.randn(20),
        "industry": ["Bank"] * 5 + ["Tech"] * 5 + ["Pharma"] * 5 + ["Consumer"] * 5
    })

    weights = PortfolioOptimizer.allocate_industry_neutral(
        df,
        score_col="score",
        industry_col="industry",
        benchmark_industry_weights=benchmark_industry_weights,
        top_n_per_industry=2
    )

    # 1. Total weights should sum to 1.0
    assert np.isclose(weights["weight"].sum(), 1.0, atol=1e-6)

    # 2. Merged industry weights should match benchmark
    ind_weights = weights.groupby("industry")["weight"].sum()
    for ind, w in benchmark_industry_weights.items():
        assert np.isclose(ind_weights[ind], w, atol=1e-5)

def test_backtest_with_neutralization_and_optimizer():
    np.random.seed(42)
    dates = pd.date_range("2026-08-01", periods=10, freq="B").strftime("%Y-%m-%d").tolist()
    symbols = [f"SZ{i:06d}" for i in range(40)]
    industries = ["Bank", "Tech", "Pharma", "Consumer"] * 10
    market_caps = [np.random.uniform(5e9, 5e10) for _ in range(40)]

    records = []
    for d in dates:
        for s, ind, cap in zip(symbols, industries, market_caps):
            records.append({
                "date": d,
                "symbol": s,
                "score": float(np.random.randn()),
                "ret": float(np.random.normal(0.001, 0.02)),
                "industry": ind,
                "market_cap": cap
            })
    pred_df = pd.DataFrame(records)

    engine = BacktestEngine(
        top_k=10,
        neutralize=True,
        industry_neutral_allocation=True
    )
    report = engine.run_backtest(pred_df)

    assert "annualized_return" in report
    assert "sharpe_ratio" in report
    assert "max_drawdown" in report
    assert report["n_trading_days"] == 10
