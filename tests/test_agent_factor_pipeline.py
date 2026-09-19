import numpy as np
import pandas as pd
import pytest

from src.agent_research.evaluator import FactorEvaluator, FactorEvaluationMetrics
from src.agent_research.hypothesis import FactorHypothesis, FactorHypothesisEngine
from src.agent_research.pipeline import AutonomousFactorPipeline


def test_factor_evaluator_positive_alpha():
    np.random.seed(42)
    dates = pd.date_range("2026-01-01", periods=30, freq="B")
    symbols = ["600000.SH", "000001.SZ", "600519.SH", "000858.SZ", "601318.SH"]

    # Generate synthetic true signals and noisy forward returns
    true_signal = np.random.randn(len(dates), len(symbols))
    forward_returns = 0.5 * true_signal + 0.5 * np.random.randn(len(dates), len(symbols)) * 0.1

    factor_df = pd.DataFrame(true_signal, index=dates, columns=symbols)
    returns_df = pd.DataFrame(forward_returns, index=dates, columns=symbols)

    metrics = FactorEvaluator.evaluate(
        factor_df=factor_df,
        forward_returns_df=returns_df,
        ic_threshold=0.02,
        icir_threshold=0.5,
        t_stat_threshold=2.0,
    )

    assert isinstance(metrics, FactorEvaluationMetrics)
    assert metrics.ic_mean > 0.3
    assert metrics.rank_ic_mean > 0.3
    assert metrics.icir > 1.0
    assert metrics.t_stat > 2.0
    assert metrics.long_short_return > 0
    assert metrics.passed_gate is True
    assert len(metrics.gate_violations) == 0


def test_factor_evaluator_poor_factor_gate_rejection():
    np.random.seed(99)
    dates = pd.date_range("2026-01-01", periods=20, freq="B")
    symbols = ["A", "B", "C", "D"]

    # Pure random noise with near-zero IC
    factor_df = pd.DataFrame(np.random.randn(20, 4), index=dates, columns=symbols)
    returns_df = pd.DataFrame(np.random.randn(20, 4) * 0.01, index=dates, columns=symbols)

    metrics = FactorEvaluator.evaluate(
        factor_df=factor_df,
        forward_returns_df=returns_df,
        ic_threshold=0.05,
        icir_threshold=1.0,
    )

    assert metrics.passed_gate is False
    assert len(metrics.gate_violations) > 0


def test_hypothesis_engine_generation_and_computation():
    hyps = FactorHypothesisEngine.generate_hypotheses(category="momentum", count=2, use_llm=False)
    assert len(hyps) >= 1
    assert hyps[0].category == "momentum"
    assert hyps[0].expression != ""

    # Synthetic OHLCV
    dates = pd.date_range("2026-02-01", periods=25, freq="B")
    symbols = ["S1", "S2", "S3"]
    close = pd.DataFrame(np.linspace(10, 20, 25)[:, None] + np.random.randn(25, 3), index=dates, columns=symbols)
    high = close * 1.02
    low = close * 0.98
    volume = pd.DataFrame(np.random.uniform(1000, 5000, (25, 3)), index=dates, columns=symbols)

    price_data = {"close": close, "high": high, "low": low, "volume": volume}

    factor_vals = FactorHypothesisEngine.compute_factor_values(hyps[0], price_data)
    assert factor_vals.shape == (25, 3)
    assert not factor_vals.isna().all().all()


def test_autonomous_pipeline_mining_cycle():
    np.random.seed(42)
    dates = pd.date_range("2026-01-01", periods=30, freq="B")
    symbols = ["600000.SH", "000001.SZ", "600519.SH"]
    close = pd.DataFrame(np.cumprod(1.0 + np.random.randn(30, 3) * 0.01, axis=0) * 10.0, index=dates, columns=symbols)
    high = close * 1.01
    low = close * 0.99
    volume = pd.DataFrame(np.random.uniform(1e5, 5e5, (30, 3)), index=dates, columns=symbols)

    price_data = {"close": close, "high": high, "low": low, "volume": volume}

    pipeline = AutonomousFactorPipeline()
    results = pipeline.run_mining_cycle(price_data, count=2, ic_threshold=0.01, icir_threshold=0.2)

    assert len(results) == 2
    for r in results:
        assert "name" in r
        assert "metrics" in r
        assert "is_active" in r

    # Check listing active factors
    active_factors = pipeline.list_active_factors()
    assert isinstance(active_factors, list)


def test_factor_deprecation_lifecycle():
    pipeline = AutonomousFactorPipeline()

    # Pre-register an active factor in memory
    pipeline._in_memory_registry["factor_alpha_test"] = {
        "name": "factor_alpha_test",
        "category": "momentum",
        "is_active": True,
        "icir": 1.2,
    }

    # Simulate performance degradation
    degraded_metrics = {
        "factor_alpha_test": FactorEvaluationMetrics(
            ic_mean=0.002,
            ic_std=0.05,
            rank_ic_mean=0.001,
            rank_ic_std=0.05,
            icir=0.15,  # severely decayed (< 0.3)
            rank_icir=0.10,
            t_stat=0.4,
            long_short_return=-0.02,
            turnover=0.6,
            monotonicity_score=0.2,
            passed_gate=False,
            gate_violations=["Decayed"],
        )
    }

    retired = pipeline.retire_decayed_factors(degraded_metrics, min_icir=0.3)
    assert "factor_alpha_test" in retired
    assert pipeline._in_memory_registry["factor_alpha_test"]["is_active"] is False
    assert "deprecation_reason" in pipeline._in_memory_registry["factor_alpha_test"]
