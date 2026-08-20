"""P3: Feature Store + Alpha公式 + 数据反馈 — 验证脚本

Tests:
1. FeatureStore save/get/query
2. FeatureStore append_incremental + prune
3. FeatureStore serve API
4. FactorCalculator Feature Store integration
5. compute_alpha_score multi-source fusion
6. QlibSignalGenerator _combine_factors
7. AlternativeDataPipeline feedback interface
"""
import sys
import os
import tempfile
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"


def test_1_feature_store_basic():
    """Test 1: FeatureStore save/get/query"""
    print("\n=== Test 1: FeatureStore save/get ===")
    from src.data.feature_store import FeatureStore
    import pandas as pd
    import numpy as np
    from pathlib import Path

    tmp = Path(tempfile.mkdtemp()) / "fs"
    store = FeatureStore(root=tmp)

    # Create mock data
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    instruments = ["SH600000", "SH600001"]
    idx = pd.MultiIndex.from_product([instruments, dates], names=["instrument", "datetime"])
    data = pd.DataFrame({"kmid": np.random.randn(len(idx))}, index=idx)

    # Save
    version = store.save(
        name="kmid",
        expression="($close - $open) / $open",
        data=data,
        instruments=instruments,
        date_range=("2024-01-01", "2024-01-10"),
        ic_score=0.05,
        icir_score=0.4,
    )
    assert version == "v1"

    # Get
    loaded = store.get("kmid")
    assert loaded is not None
    assert len(loaded) == 20  # 2 instruments * 10 days

    # Get latest
    latest = store.get_latest("kmid")
    assert latest is not None

    # List versions
    versions = store.list_versions("kmid")
    assert len(versions) == 1
    assert versions[0]["ic_score"] == 0.05

    # Query
    results = store.query(min_ic=0.04)
    assert len(results) == 1

    results_low = store.query(min_ic=0.06)
    assert len(results_low) == 0

    # Summary
    summary = store.summary()
    assert summary["total_features"] == 1
    assert summary["best_ic"] == 0.05

    shutil.rmtree(tmp)
    print(f"  Saved v1, loaded {len(loaded)} rows, IC={summary['best_ic']}")
    print(f"{PASS} FeatureStore save/get/query")
    return True


def test_2_feature_store_incremental():
    """Test 2: FeatureStore append_incremental + prune"""
    print("\n=== Test 2: FeatureStore incremental + prune ===")
    from src.data.feature_store import FeatureStore
    import pandas as pd
    import numpy as np
    from pathlib import Path

    tmp = Path(tempfile.mkdtemp()) / "fs"
    store = FeatureStore(root=tmp)

    instruments = ["SH600000"]

    # First batch
    dates1 = pd.date_range("2024-01-01", periods=5, freq="D")
    idx1 = pd.MultiIndex.from_product([instruments, dates1], names=["instrument", "datetime"])
    data1 = pd.DataFrame({"std_20": np.random.randn(len(idx1))}, index=idx1)

    store.save(
        name="std_20",
        expression="Std($close, 20) / $close",
        data=data1,
        instruments=instruments,
        date_range=("2024-01-01", "2024-01-05"),
    )

    # Incremental append
    dates2 = pd.date_range("2024-01-06", periods=5, freq="D")
    idx2 = pd.MultiIndex.from_product([instruments, dates2], names=["instrument", "datetime"])
    data2 = pd.DataFrame({"std_20": np.random.randn(len(idx2))}, index=idx2)

    store.append_incremental(
        name="std_20",
        expression="Std($close, 20) / $close",
        new_data=data2,
        instruments=instruments,
        date_range=("2024-01-01", "2024-01-10"),
    )

    versions = store.list_versions("std_20")
    assert len(versions) == 2

    latest = store.get_latest("std_20")
    assert len(latest) == 10  # 5+5 days

    # Prune to keep only 1 version
    store.prune(keep=1)
    versions_after = store.list_versions("std_20")
    assert len(versions_after) == 1

    shutil.rmtree(tmp)
    print(f"  Appended 5+5→10 rows, pruned to 1 version")
    print(f"{PASS} FeatureStore incremental + prune")
    return True


def test_3_feature_store_serve():
    """Test 3: FeatureStore serve API"""
    print("\n=== Test 3: FeatureStore serve ===")
    from src.data.feature_store import FeatureStore
    import pandas as pd
    import numpy as np
    from pathlib import Path

    tmp = Path(tempfile.mkdtemp()) / "fs"
    store = FeatureStore(root=tmp)

    instruments = ["SH600000", "SH600001"]
    dates = pd.date_range("2024-01-01", periods=10, freq="D")

    for name in ["kmid", "std_20", "rocp_5"]:
        idx = pd.MultiIndex.from_product([instruments, dates], names=["instrument", "datetime"])
        data = pd.DataFrame({name: np.random.randn(len(idx))}, index=idx)
        store.save(
            name=name,
            expression=f"expr_{name}",
            data=data,
            instruments=instruments,
            date_range=("2024-01-01", "2024-01-10"),
        )

    # Serve multiple features
    served = store.serve(["kmid", "std_20", "rocp_5"])
    assert len(served) == 20  # 2 instruments * 10 days
    assert len(served.columns) == 3

    # Serve with date filter
    served_filtered = store.serve(["kmid", "std_20"], start_date="2024-01-05")
    assert len(served_filtered) == 12  # 2 instruments * 6 days (Jan 5-10)

    shutil.rmtree(tmp)
    print(f"  Served 3 features, {len(served)} rows, {len(served.columns)} cols")
    print(f"  Date-filtered: {len(served_filtered)} rows")
    print(f"{PASS} FeatureStore serve API")
    return True


def test_4_factor_calculator_fs():
    """Test 4: FactorCalculator Feature Store integration"""
    print("\n=== Test 4: FactorCalculator FS integration ===")
    from src.data.factor_calculator import FactorCalculator

    fc = FactorCalculator()
    assert fc._feature_store is not None, "FeatureStore not connected"
    assert hasattr(fc, "use_feature_store")
    assert hasattr(fc, "_try_feature_store")
    assert hasattr(fc, "_save_to_feature_store")

    # Check compute() has use_feature_store parameter
    import inspect
    sig = inspect.signature(fc.compute)
    assert "use_feature_store" in sig.parameters

    print(f"  FeatureStore connected: {fc._feature_store is not None}")
    print(f"  compute(use_feature_store={sig.parameters['use_feature_store'].default})")
    print(f"{PASS} FactorCalculator Feature Store integration")
    return True


def test_5_alpha_score():
    """Test 5: compute_alpha_score multi-source fusion"""
    print("\n=== Test 5: compute_alpha_score ===")
    from src.research.factor_library import (
        compute_alpha_score,
        ALPHA_SCORE_WEIGHTS,
    )
    import pandas as pd
    import numpy as np

    codes = ["SH600000", "SH600001", "SH600002", "SH600003", "SH600004"]
    pv = pd.Series(np.random.randn(5), index=codes, name="pv")
    fund = pd.Series(np.random.randn(5), index=codes, name="fund")
    alt = pd.Series(np.random.randn(5), index=codes, name="alt")

    # All sources
    alpha = compute_alpha_score(
        price_volume_score=pv,
        fundamental_score=fund,
        alternative_score=alt,
    )
    assert len(alpha) == 5
    assert alpha.name == "alpha_score"

    # Only one source (weight renormalization)
    alpha_single = compute_alpha_score(price_volume_score=pv)
    assert len(alpha_single) == 5
    # With only one source, alpha should equal the input
    assert np.allclose(alpha_single.values, pv.values)

    # Custom weights
    custom_w = {"price_volume": 0.6, "fundamental": 0.4}
    alpha_custom = compute_alpha_score(
        price_volume_score=pv,
        fundamental_score=fund,
        weights=custom_w,
    )
    assert len(alpha_custom) == 5

    print(f"  Weights: {ALPHA_SCORE_WEIGHTS}")
    print(f"  Alpha (3 sources): {alpha.values[:3]}")
    print(f"  Alpha (1 source, renormalized): matches input")
    print(f"{PASS} compute_alpha_score multi-source fusion")
    return True


def test_6_signal_generator_combine():
    """Test 6: QlibSignalGenerator _combine_factors"""
    print("\n=== Test 6: QlibSignalGenerator _combine_factors ===")
    from src.execution.qlib_signal_generator import QlibSignalGenerator
    import pandas as pd
    import numpy as np

    gen = QlibSignalGenerator()

    codes = ["SH600000", "SH600001", "SH600002"]
    model_scores = pd.Series([0.5, 0.3, 0.1], index=codes)
    alt_scores = pd.Series([0.2, -0.1, 0.3], index=codes)
    fund_scores = pd.Series([0.1, 0.4, -0.2], index=codes)

    alpha = gen._combine_factors(
        model_scores=model_scores,
        alternative_scores=alt_scores,
        fundamental_scores=fund_scores,
    )

    assert len(alpha) == 3
    assert alpha.name == "alpha_score"

    # Check method exists
    assert hasattr(gen, "generate_with_alpha"), "Missing generate_with_alpha"

    print(f"  Model: {model_scores.values}")
    print(f"  Alt:   {alt_scores.values}")
    print(f"  Fund:  {fund_scores.values}")
    print(f"  Alpha: {alpha.values}")
    print(f"{PASS} QlibSignalGenerator _combine_factors")
    return True


def test_7_alternative_feedback():
    """Test 7: AlternativeDataPipeline feedback interface"""
    print("\n=== Test 7: AlternativeDataPipeline feedback ===")
    from src.data.alternative import AlternativeDataPipeline
    import tempfile
    from pathlib import Path

    # Use temp path for feedback
    import src.data.alternative.alternative_pipeline as ap_module
    original_path = ap_module.FEEDBACK_PATH
    tmp = Path(tempfile.mktemp(suffix=".json"))
    ap_module.FEEDBACK_PATH = tmp

    try:
        pipeline = AlternativeDataPipeline()

        # Record feedback
        pipeline.feedback_from_performance(
            strategy_id="strat_test_001",
            pnl=0.05,
            sharpe=1.2,
            max_drawdown=-0.08,
            factor_contributions={
                "news_sentiment": 0.02,
                "consensus_rating": 0.01,
                "vol_ratio_5": -0.005,
            },
        )

        # Check weights adjusted
        weights = pipeline.get_factor_weights()
        assert "news_sentiment" in weights
        assert "consensus_rating" in weights
        assert "vol_ratio_5" in weights

        # Positive contribution → weight > 0.5 (initial)
        assert weights["news_sentiment"] > 0.5
        # Negative contribution → weight < 0.5
        assert weights["vol_ratio_5"] < 0.5

        # Record another feedback
        pipeline.feedback_from_performance(
            strategy_id="strat_test_002",
            pnl=-0.02,
            sharpe=-0.3,
            factor_contributions={"news_sentiment": -0.01},
        )

        summary = pipeline.get_feedback_summary()
        assert summary["total_records"] == 2
        assert "avg_pnl" in summary
        assert "factor_weights" in summary

        print(f"  Feedback records: {summary['total_records']}")
        print(f"  Avg PnL: {summary['avg_pnl']:.4f}")
        print(f"  Factor weights: {weights}")
        print(f"{PASS} AlternativeDataPipeline feedback interface")
        return True

    finally:
        ap_module.FEEDBACK_PATH = original_path
        if tmp.exists():
            tmp.unlink()


def main():
    print("=" * 60)
    print("P3: Feature Store + Alpha公式 + 数据反馈 — 验证")
    print("=" * 60)

    tests = [
        ("FeatureStore save/get", test_1_feature_store_basic),
        ("FeatureStore incremental+prune", test_2_feature_store_incremental),
        ("FeatureStore serve API", test_3_feature_store_serve),
        ("FactorCalculator FS integration", test_4_factor_calculator_fs),
        ("compute_alpha_score", test_5_alpha_score),
        ("SignalGenerator _combine_factors", test_6_signal_generator_combine),
        ("AlternativeDataPipeline feedback", test_7_alternative_feedback),
    ]

    passed = 0
    failed = 0

    for name, fn in tests:
        try:
            if fn():
                passed += 1
            else:
                failed += 1
                print(f"{FAIL} {name}")
        except Exception as e:
            failed += 1
            import traceback
            print(f"{FAIL} {name}")
            print(f"  Error: {e}")
            traceback.print_exc()

    print("\n" + "=" * 60)
    print(f"Results: {passed}/{passed + failed} passed")
    print("=" * 60)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
