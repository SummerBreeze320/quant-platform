import pytest
import pandas as pd
from unittest.mock import MagicMock, patch
from src.data_pipeline.transformer import DataTransformer
from src.data_pipeline.wind_client import WindClient
from src.data_pipeline.collector import WindDataCollector

def test_data_transformer_standardize_quotes():
    raw_df = pd.DataFrame({
        "SEC_CODE": ["000001.SZ", "000001.SZ"],
        "DATETIME": ["2026-09-11", "2026-09-12"],
        "OPEN": [10.0, 10.2],
        "HIGH": [10.5, 10.6],
        "LOW": [9.9, 10.1],
        "CLOSE": [10.3, 10.4],
        "VOLUME": [100000.0, 120000.0],
        "AMT": [1030000.0, 1248000.0],
        "ADJFACTOR": [1.0, 1.0],
        "VWAP": [10.3, 10.4]
    })
    
    transformed = DataTransformer.standardize_quotes(raw_df)
    
    assert "symbol" in transformed.columns
    assert "date" in transformed.columns
    assert "$close" in transformed.columns
    assert "$volume" in transformed.columns
    assert "$factor" in transformed.columns
    assert "$money" in transformed.columns
    assert "$vwap" in transformed.columns
    assert len(transformed) == 2
    assert transformed["symbol"].iloc[0] == "000001.SZ"

def test_data_transformer_adjust_prices():
    df = pd.DataFrame({
        "symbol": ["000001.SZ", "000001.SZ"],
        "date": ["2026-09-11", "2026-09-12"],
        "$open": [10.0, 10.0],
        "$close": [10.0, 10.0],
        "$factor": [1.0, 2.0]
    })
    
    adj_df = DataTransformer.compute_adjusted_prices(df, method="post")
    assert "$close_adj" in adj_df.columns
    assert adj_df["$close_adj"].iloc[0] == 10.0
    assert adj_df["$close_adj"].iloc[1] == 20.0

def test_wind_client_mock():
    mock_w = MagicMock()
    mock_w.isconnected.return_value = True
    mock_w.start.return_value = MagicMock(ErrorCode=0)
    
    client = WindClient(w_instance=mock_w)
    assert client.connect() is True
    assert client.is_connected() is True
    client.disconnect()
    mock_w.stop.assert_called_once()

def test_collector_get_calendar_mock():
    mock_client = MagicMock()
    mock_w = MagicMock()
    # Mock w.tdays return
    mock_data = MagicMock()
    mock_data.ErrorCode = 0
    mock_data.Data = [["2026-09-01", "2026-09-02", "2026-09-03"]]
    mock_w.tdays.return_value = mock_data
    mock_client.w = mock_w
    mock_client.is_connected.return_value = True

    collector = WindDataCollector(client=mock_client)
    days = collector.get_trade_days("2026-09-01", "2026-09-03")
    assert len(days) == 3
    assert days[0] == "2026-09-01"
