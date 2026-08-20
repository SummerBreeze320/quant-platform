"""Qlib converter tests: normalization, conversion, calendar alignment, batch writing."""
import pytest
import pandas as pd
import numpy as np
import tempfile
import struct
from pathlib import Path


def test_normalize_code():
    """Test code normalization."""
    from src.data.qlib_adapter.converter import WindToQlibConverter

    assert WindToQlibConverter._normalize_code("600000.SH") == "SH600000"
    assert WindToQlibConverter._normalize_code("000001.SZ") == "SZ000001"
    assert WindToQlibConverter._normalize_code("510300.SH") == "SH510300"
    assert WindToQlibConverter._normalize_code("SH600000") == "SH600000"


def test_normalize_code_via_core():
    """Test normalize_code from src.core.data_access (new path)."""
    from src.core.data_access import normalize_code

    assert normalize_code("600000.SH") == "SH600000"
    assert normalize_code("000001.SZ") == "SZ000001"
    assert normalize_code("SH600000") == "SH600000"


def test_convert_stock_basic():
    """Test basic single stock conversion."""
    from src.data.qlib_adapter.converter import WindToQlibConverter

    with tempfile.TemporaryDirectory() as tmpdir:
        converter = WindToQlibConverter(qlib_dir=tmpdir)

        dates = pd.date_range("2024-01-01", periods=10)
        df = pd.DataFrame({
            "open": np.random.uniform(10, 20, 10),
            "high": np.random.uniform(15, 25, 10),
            "low": np.random.uniform(8, 15, 10),
            "close": np.random.uniform(10, 20, 10),
            "volume": np.random.randint(1000, 100000, 10).astype(float),
        }, index=dates)

        result = converter.convert_stock("600000.SH", df)
        assert result is not None
        assert result == (0, 10)

        close_bin = Path(tmpdir) / "features" / "SH600000" / "close.day.bin"
        assert close_bin.exists()

        # Verify bin content by reading raw binary
        with open(close_bin, "rb") as f:
            start_idx, end_idx = struct.unpack("ii", f.read(8))
            data = np.frombuffer(f.read(), dtype=np.float32)
        assert start_idx == 0
        assert end_idx == 10
        assert len(data) == 10


def test_convert_with_calendar_alignment():
    """Test that calendar alignment sets correct start_index."""
    from src.data.qlib_adapter.converter import WindToQlibConverter

    with tempfile.TemporaryDirectory() as tmpdir:
        converter = WindToQlibConverter(qlib_dir=tmpdir)

        cal_dates = [d.strftime("%Y-%m-%d") for d in pd.bdate_range("2024-01-01", "2024-01-31")]
        stock_dates = pd.bdate_range("2024-01-15", "2024-01-31")
        df = pd.DataFrame({
            "close": np.linspace(10, 15, len(stock_dates)),
            "open": np.linspace(9, 14, len(stock_dates)),
            "high": np.linspace(11, 16, len(stock_dates)),
            "low": np.linspace(8, 13, len(stock_dates)),
            "volume": np.ones(len(stock_dates)) * 1000,
        }, index=stock_dates)

        result = converter.convert_stock("600000.SH", df, calendar_dates=cal_dates)
        assert result is not None

        start_idx, end_idx = result
        assert start_idx > 0
        assert end_idx == len(cal_dates)

        # Verify bin file has correct header
        close_bin = Path(tmpdir) / "features" / "SH600000" / "close.day.bin"
        with open(close_bin, "rb") as f:
            bin_start, bin_end = struct.unpack("ii", f.read(8))
            data = np.frombuffer(f.read(), dtype=np.float32)
        assert bin_start == start_idx
        assert bin_end == end_idx
        assert len(data) == end_idx - start_idx


def test_convert_batch():
    """Test batch conversion."""
    from src.data.qlib_adapter.converter import WindToQlibConverter

    with tempfile.TemporaryDirectory() as tmpdir:
        converter = WindToQlibConverter(qlib_dir=tmpdir)

        data = {}
        for code in ["600000.SH", "000001.SZ", "510300.SH"]:
            dates = pd.date_range("2024-01-01", periods=10)
            data[code] = pd.DataFrame({
                "close": np.random.uniform(10, 20, 10),
                "volume": np.random.randint(1000, 10000, 10).astype(float),
            }, index=dates)

        success = converter.convert_batch(data)
        assert success == 3

        # Verify by checking bin files exist
        for code in ["SH600000", "SZ000001", "SH510300"]:
            close_bin = Path(tmpdir) / "features" / code / "close.day.bin"
            assert close_bin.exists()
