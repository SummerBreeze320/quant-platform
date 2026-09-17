import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from src.data_pipeline.qlib_dumper import QlibDumper
from src.qlib_engine.initializer import init_qlib
import qlib
from qlib.data import D

def test_qlib_dumper_and_reader(tmp_path):
    qlib_dir = tmp_path / "qlib_data"
    dumper = QlibDumper(str(qlib_dir))
    
    calendar = ["2026-09-01", "2026-09-02", "2026-09-03", "2026-09-04"]
    dumper.dump_calendars(calendar, freq="day")
    
    # Verify calendar file
    cal_file = qlib_dir / "calendars" / "day.txt"
    assert cal_file.exists()
    assert cal_file.read_text().strip().split("\n") == calendar
    
    # Instruments
    instruments = {
        "SH600000": ("2026-09-01", "2026-09-04"),
        "SZ000001": ("2026-09-01", "2026-09-04")
    }
    dumper.dump_instruments(instruments, pool="all")
    inst_file = qlib_dir / "instruments" / "all.txt"
    assert inst_file.exists()
    
    # Features DataFrame
    df = pd.DataFrame({
        "symbol": ["SZ000001"] * 4 + ["SH600000"] * 4,
        "date": calendar * 2,
        "$close": [10.0, 10.5, 10.3, 10.8, 20.0, 20.2, 19.8, 20.5],
        "$volume": [1000.0, 1200.0, 1100.0, 1500.0, 3000.0, 3200.0, 2900.0, 3100.0]
    })
    
    dumper.dump_features(df, freq="day")
    
    # Verify feature bin file
    feature_file = qlib_dir / "features" / "sz000001" / "close.day.bin"
    assert feature_file.exists()
    
    # Verify Qlib initialization and reading
    init_qlib(provider_uri=str(qlib_dir))
    
    features_df = D.features(["SZ000001"], ["$close", "$volume"], start_time="2026-09-01", end_time="2026-09-04")
    assert not features_df.empty
    assert len(features_df) == 4
    assert np.isclose(features_df["$close"].iloc[0], 10.0)
