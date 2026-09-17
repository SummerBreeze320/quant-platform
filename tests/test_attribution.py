import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from src.qlib_engine.attribution import BrinsonAttribution
from src.qlib_engine.factor_analysis import FactorQuantileAnalyzer
from src.qlib_engine.report_generator import QuantReportGenerator

def test_brinson_attribution_identity():
    # Setup test data with 4 industries
    holdings_df = pd.DataFrame({
        "symbol": ["SZ000001", "SZ000002", "SZ000003", "SZ000004"],
        "industry": ["Bank", "Tech", "Pharma", "Consumer"],
        "weight_p": [0.4, 0.3, 0.2, 0.1],
        "ret_p": [0.05, 0.12, -0.02, 0.08]
    })

    benchmark_weights = {
        "Bank": 0.25,
        "Tech": 0.25,
        "Pharma": 0.25,
        "Consumer": 0.25
    }

    benchmark_returns = {
        "Bank": 0.03,
        "Tech": 0.08,
        "Pharma": 0.01,
        "Consumer": 0.04
    }

    result = BrinsonAttribution.calculate(
        holdings_df=holdings_df,
        benchmark_weights=benchmark_weights,
        benchmark_returns=benchmark_returns
    )

    # 1. Total excess return
    r_p = (holdings_df["weight_p"] * holdings_df["ret_p"]).sum()
    r_b = sum(benchmark_weights[ind] * benchmark_returns[ind] for ind in benchmark_weights)
    excess_expected = r_p - r_b

    # 2. Check Brinson identity: Allocation + Selection + Interaction == Total Excess
    total_alloc = result["total_allocation"]
    total_selec = result["total_selection"]
    total_inter = result["total_interaction"]
    total_excess = result["total_excess_return"]

    assert np.isclose(total_excess, excess_expected, atol=1e-6)
    assert np.isclose(total_alloc + total_selec + total_inter, total_excess, atol=1e-6)
    assert len(result["industry_details"]) == 4

def test_factor_quantile_analysis():
    np.random.seed(42)
    dates = pd.date_range("2026-08-01", periods=15, freq="B").strftime("%Y-%m-%d").tolist()
    symbols = [f"SZ{i:06d}" for i in range(50)]

    records = []
    for d in dates:
        for s in symbols:
            # Score positively correlated with forward return
            sc = float(np.random.randn())
            r = float(sc * 0.02 + np.random.randn() * 0.01)
            records.append({
                "date": d,
                "symbol": s,
                "score": sc,
                "ret": r
            })
    df = pd.DataFrame(records)

    analysis = FactorQuantileAnalyzer.analyze(df, n_quantiles=5)
    assert "quantile_returns" in analysis
    assert "long_short_cumulative" in analysis
    assert "monotonicity_score" in analysis
    assert len(analysis["quantile_returns"]) == 5
    # Q1 should beat Q5 given positive factor-return correlation
    assert analysis["quantile_returns"]["Q1"]["total_return"] > analysis["quantile_returns"]["Q5"]["total_return"]

def test_html_report_generation(tmp_path):
    # Performance summary dict
    perf_summary = {
        "strategy_name": "MultiModel_Ensemble_Alpha",
        "total_return": 0.285,
        "annualized_return": 0.352,
        "benchmark_annualized_return": 0.082,
        "alpha_annualized": 0.270,
        "sharpe_ratio": 2.15,
        "max_drawdown": -0.068,
        "win_rate": 0.62,
        "n_trading_days": 120,
        "cumulative_returns": {f"2026-08-{i:02d}": 1.0 + i * 0.005 for i in range(1, 21)}
    }

    # Brinson attribution sample
    brinson_data = {
        "total_excess_return": 0.052,
        "total_allocation": 0.021,
        "total_selection": 0.025,
        "total_interaction": 0.006,
        "industry_details": [
            {"industry": "Tech", "weight_p": 0.35, "weight_b": 0.20, "ret_p": 0.15, "ret_b": 0.08, "allocation": 0.012, "selection": 0.014, "interaction": 0.005},
            {"industry": "Bank", "weight_p": 0.15, "weight_b": 0.30, "ret_p": 0.04, "ret_b": 0.02, "allocation": 0.009, "selection": 0.006, "interaction": 0.001}
        ]
    }

    # Quantile returns sample
    quantile_data = {
        "Q1": 0.18,
        "Q2": 0.12,
        "Q3": 0.07,
        "Q4": 0.02,
        "Q5": -0.04,
        "long_short_return": 0.22
    }

    report_path = tmp_path / "quant_report.html"
    html_content = QuantReportGenerator.generate_html(
        perf_summary=perf_summary,
        brinson_data=brinson_data,
        quantile_data=quantile_data,
        output_file=str(report_path)
    )

    assert "<!DOCTYPE html>" in html_content
    assert "MultiModel_Ensemble_Alpha" in html_content
    assert "Brinson" in html_content
    assert "<svg" in html_content  # Verifies embedded SVG line chart
    assert report_path.exists()
