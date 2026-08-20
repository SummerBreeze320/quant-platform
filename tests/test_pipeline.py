"""Data pipeline, mock data, features, and strategy tests."""
import pytest
import pandas as pd
import numpy as np
import tempfile
import struct
from pathlib import Path


# === Converter tests ===

def test_converter_normalize():
    from src.data.qlib_adapter.converter import WindToQlibConverter
    assert WindToQlibConverter._normalize_code("600000.SH") == "SH600000"
    assert WindToQlibConverter._normalize_code("000001.SZ") == "SZ000001"
    assert WindToQlibConverter._normalize_code("510300.SH") == "SH510300"


def test_converter_roundtrip():
    from src.data.qlib_adapter.converter import WindToQlibConverter
    with tempfile.TemporaryDirectory() as tmpdir:
        converter = WindToQlibConverter(qlib_dir=tmpdir)
        dates = pd.date_range("2024-01-01", periods=20)
        original = np.random.uniform(10, 20, 20).astype(np.float32)
        df = pd.DataFrame({
            "open": original, "high": original * 1.02, "low": original * 0.98,
            "close": original,
            "volume": np.random.randint(1000, 100000, 20).astype(np.float32),
        }, index=dates)
        converter.convert_stock("600000.SH", df)
        close_path = Path(tmpdir) / "features" / "SH600000" / "close.day.bin"
        assert close_path.exists()
        # Read bin file directly (converter is write-only now)
        with open(close_path, "rb") as f:
            start_idx, end_idx = struct.unpack("ii", f.read(8))
            data = np.frombuffer(f.read(), dtype=np.float32)
        assert start_idx == 0
        assert end_idx == 20
        assert len(data) == 20
        np.testing.assert_array_almost_equal(data, original)


def test_converter_list_instruments():
    from src.data.qlib_adapter.converter import WindToQlibConverter
    with tempfile.TemporaryDirectory() as tmpdir:
        converter = WindToQlibConverter(qlib_dir=tmpdir)
        # No instruments written yet -> features dir empty or missing
        features_dir = Path(tmpdir) / "features"
        assert not features_dir.exists() or len(list(features_dir.iterdir())) == 0
        df = pd.DataFrame({"close": [10.0, 11.0, 12.0]},
                          index=pd.date_range("2024-01-01", periods=3))
        converter.convert_stock("600000.SH", df)
        # Verify by listing the features directory directly
        inst_dirs = [d.name for d in features_dir.iterdir() if d.is_dir()]
        assert "SH600000" in inst_dirs


# === Instrument generator tests ===

def test_instrument_generation():
    from src.data.qlib_adapter.instrument import InstrumentGenerator
    with tempfile.TemporaryDirectory() as tmpdir:
        gen = InstrumentGenerator(qlib_dir=tmpdir)
        gen.generate("test", ["SH600000", "SZ000001"],
                     start_dates=["2015-01-05", "2010-01-04"],
                     end_dates="2024-12-31")
        path = Path(tmpdir) / "instruments" / "test.txt"
        assert path.exists()
        lines = path.read_text().strip().split("\n")
        assert len(lines) == 2
        assert "SH600000" in lines[0]
        assert "2015-01-05" in lines[0]


def test_instrument_end_dates_as_list():
    from src.data.qlib_adapter.instrument import InstrumentGenerator
    with tempfile.TemporaryDirectory() as tmpdir:
        gen = InstrumentGenerator(qlib_dir=tmpdir)
        gen.generate("test", ["A", "B", "C"],
                     end_dates=["2024-01-01", "2024-06-01", "2024-12-31"])
        path = Path(tmpdir) / "instruments" / "test.txt"
        lines = path.read_text().strip().split("\n")
        assert "2024-01-01" in lines[0]
        assert "2024-06-01" in lines[1]


# === Feature definitions tests (now via Qlib native DataProvider) ===

def test_feature_definitions():
    """Verify Qlib native fields are accessible via core module."""
    from src.data.mock_data import MockDataGenerator
    with tempfile.TemporaryDirectory() as tmpdir:
        gen = MockDataGenerator(qlib_dir=tmpdir)
        gen.generate_qlib_data(n_stocks=2, n_etfs=2, seed=42)

        from src.core import init_qlib, get_features
        init_qlib(provider_uri=tmpdir)
        df = get_features(["SH600000"], ["$close", "$open", "$high", "$low", "$volume"])
        assert df is not None
        assert not df.empty
        assert "$close" in df.columns
        assert "$open" in df.columns


def test_factor_templates():
    """Verify Qlib expression-based factors work (e.g. Ref, Mean)."""
    from src.data.mock_data import MockDataGenerator
    with tempfile.TemporaryDirectory() as tmpdir:
        gen = MockDataGenerator(qlib_dir=tmpdir)
        gen.generate_qlib_data(n_stocks=2, n_etfs=2, seed=42)

        from src.core import init_qlib, get_features
        init_qlib(provider_uri=tmpdir)
        # Qlib expression: 5-day mean of close
        df = get_features(["SH600000"], ["Mean($close, 5)"])
        assert df is not None
        assert not df.empty
        assert "Mean($close, 5)" in df.columns


# === Mock data tests ===

def test_mock_data_generation():
    from src.data.mock_data import MockDataGenerator
    with tempfile.TemporaryDirectory() as tmpdir:
        gen = MockDataGenerator(qlib_dir=tmpdir)
        stats = gen.generate_qlib_data(n_stocks=5, n_etfs=5, seed=42)
        assert stats["stocks"] == 5
        assert stats["etfs"] == 5
        assert stats["calendar_days"] > 0

        # Verify data by listing features directory (converter is write-only now)
        features_dir = Path(tmpdir) / "features"
        inst_dirs = sorted([d.name for d in features_dir.iterdir() if d.is_dir()])
        assert len(inst_dirs) == 10
        assert "SH600000" in inst_dirs

        # Read SH600000 close directly from bin file
        close_bin = features_dir / "SH600000" / "close.day.bin"
        assert close_bin.exists()
        with open(close_bin, "rb") as f:
            start_idx, end_idx = struct.unpack("ii", f.read(8))
            data = np.frombuffer(f.read(), dtype=np.float32)
        assert len(data) == end_idx - start_idx
        assert len(data) > 0
        assert (data > 0).all()


# === Factor tests ===

def test_factor_momentum():
    from src.strategy.stock.alpha_factors import MomentumFactor
    df = pd.DataFrame({
        "close": [10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                  20, 19, 18, 17, 16, 15, 14, 13, 12, 11],
    }, index=pd.date_range("2024-01-01", periods=20))
    factor = MomentumFactor(window=5)
    result = factor.calculate(df)
    assert len(result) == 20
    assert result.iloc[5] > 0
    assert result.iloc[-1] < 0


def test_factor_volatility():
    from src.strategy.stock.alpha_factors import VolatilityFactor
    df = pd.DataFrame({
        "close": np.random.uniform(10, 20, 50),
    }, index=pd.date_range("2024-01-01", periods=50))
    factor = VolatilityFactor(window=20)
    result = factor.calculate(df)
    assert len(result) == 50
    assert result.iloc[20:].notna().all()


# === ETF strategy tests ===

def test_etf_ma_trend_signal():
    from src.strategy.etf.trend_following import MATrendStrategy
    prices = list(range(10, 30)) + list(range(30, 10, -1))
    df = pd.DataFrame({
        "close": prices,
        "high": [p * 1.02 for p in prices],
        "low": [p * 0.98 for p in prices],
    }, index=pd.date_range("2024-01-01", periods=len(prices)))
    strategy = MATrendStrategy(short_window=5, long_window=10)
    signal = strategy.generate_signal(df)
    assert len(signal) == len(prices)
    assert signal.max() <= 1
    assert signal.min() >= -1


def test_etf_bollinger_signal():
    from src.strategy.etf.mean_reversion import BollingerStrategy
    np.random.seed(42)
    prices = 10 + np.cumsum(np.random.normal(0, 0.2, 60))
    df = pd.DataFrame({
        "close": prices,
    }, index=pd.date_range("2024-01-01", periods=60))
    strategy = BollingerStrategy(window=20, num_std=2.0)
    signal = strategy.generate_signal(df)
    assert len(signal) == 60
    assert signal.max() <= 1
    assert signal.min() >= -1


def test_etf_breakout_signal():
    from src.strategy.etf.trend_following import BreakoutStrategy
    prices = list(range(10, 50))
    df = pd.DataFrame({
        "close": prices,
        "high": [p + 0.5 for p in prices],
        "low": [p - 0.5 for p in prices],
    }, index=pd.date_range("2024-01-01", periods=len(prices)))
    strategy = BreakoutStrategy(window=20)
    signal = strategy.generate_signal(df)
    assert len(signal) == len(prices)
    assert signal.iloc[-1] == 1  # Should be long in uptrend


# === Backtest tests ===

def test_metrics_calculation():
    from src.backtest.metrics import calc_all_metrics
    returns = pd.Series([0.01, -0.005, 0.02, 0.01, -0.01, 0.015] * 10)
    metrics = calc_all_metrics(returns)
    assert "annual_return" in metrics
    assert "sharpe_ratio" in metrics
    assert "max_drawdown" in metrics
    assert "win_rate" in metrics
    assert isinstance(metrics["annual_return"], float)


def test_portfolio_equal_weight():
    from src.backtest.portfolio import PortfolioBuilder
    scores = pd.Series({"A": 0.5, "B": 0.3, "C": 0.1, "D": 0.2})
    weights = PortfolioBuilder.equal_weight(scores, top_n=2)
    assert weights.sum() == 1.0
    assert weights["A"] == 0.5
    assert weights["B"] == 0.5
    assert weights["C"] == 0.0


def test_strategy_backtest_with_mock_data():
    """End-to-end: mock data -> strategy -> backtest."""
    from src.data.mock_data import MockDataGenerator
    from src.backtest.backtest_engine import BacktestEngine
    from src.strategy.etf.trend_following import MATrendStrategy

    with tempfile.TemporaryDirectory() as tmpdir:
        # Generate mock data
        gen = MockDataGenerator(qlib_dir=tmpdir)
        gen.generate_qlib_data(n_stocks=5, n_etfs=5, seed=42)

        # Run backtest
        engine = BacktestEngine(qlib_dir=tmpdir)
        strategy = MATrendStrategy(short_window=5, long_window=20)
        result = engine.run_strategy_backtest(
            strategy=strategy,
            code="SH510300",
            init_cash=1_000_000,
        )
        assert "error" not in result
        assert "metrics" in result
        assert "equity_curve" in result
        assert result["data_points"] > 0
        assert result["final_value"] > 0


def test_report_generation():
    """Test HTML report generation."""
    from src.backtest.report import BacktestReport
    with tempfile.TemporaryDirectory() as tmpdir:
        report_gen = BacktestReport(output_dir=tmpdir)
        result = {
            "metrics": {
                "annual_return": 0.15,
                "sharpe_ratio": 1.2,
                "max_drawdown": -0.08,
            },
            "cumulative_return": pd.Series(
                [1.0, 1.05, 1.1, 1.08, 1.12, 1.15],
                index=pd.date_range("2024-01-01", periods=6),
            ),
        }
        path = report_gen.generate(result, "test_strategy")
        assert Path(path).exists()
        content = Path(path).read_text(encoding="utf-8")
        assert "test_strategy" in content
        assert "echarts" in content
