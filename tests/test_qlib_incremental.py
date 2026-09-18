import numpy as np
import pandas as pd
import pytest

from src.data_pipeline.qlib_dumper import QlibDumper


def quotes(dates, values, **fields):
    return pd.DataFrame({
        "symbol": ["000001.SZ"] * len(dates),
        "date": dates,
        "$close": values,
        **fields,
    })


def read_close(root):
    return np.fromfile(root / "features/sz000001/close.day.bin", dtype="<f4")


def test_daily_append_and_correction_preserve_history(tmp_path):
    dumper = QlibDumper(str(tmp_path))
    dumper.dump_calendars(["2026-09-01", "2026-09-02"])
    dumper.dump_features(quotes(["2026-09-01", "2026-09-02"], [10, 11]))
    dumper.dump_calendars(["2026-09-01", "2026-09-02", "2026-09-03"])
    update = quotes(["2026-09-02", "2026-09-03"], [12, 13])
    dumper.dump_features(update)
    np.testing.assert_array_equal(read_close(tmp_path), [0, 10, 12, 13])
    before = read_close(tmp_path).tobytes()
    dumper.dump_features(update)
    assert read_close(tmp_path).tobytes() == before


def test_sparse_backfill_preserves_other_dates_and_fields(tmp_path):
    dumper = QlibDumper(str(tmp_path))
    dates = ["2026-09-01", "2026-09-02", "2026-09-03", "2026-09-04"]
    dumper.dump_calendars(dates)
    dumper.dump_features(quotes([dates[1], dates[3]], [11, 13], **{"$volume": [100, 200]}))
    volume_file = tmp_path / "features/sz000001/volume.day.bin"
    original_volume = volume_file.read_bytes()
    dumper.dump_features(quotes([dates[0], dates[2], dates[2]], [10, 99, 12]))
    np.testing.assert_array_equal(read_close(tmp_path), [0, 10, 11, 12, 13])
    assert volume_file.read_bytes() == original_volume


def test_gap_is_nan_when_appending(tmp_path):
    dumper = QlibDumper(str(tmp_path))
    dates = ["2026-09-01", "2026-09-02", "2026-09-03"]
    dumper.dump_calendars(dates)
    dumper.dump_features(quotes([dates[0]], [10]))
    dumper.dump_features(quotes([dates[2]], [12]))
    np.testing.assert_array_equal(read_close(tmp_path), [0, 10, np.nan, 12])


@pytest.mark.parametrize("calendar", [
    ["2026-08-31", "2026-09-01", "2026-09-03"],
    ["2026-09-01", "2026-09-02", "2026-09-03"],
    ["2026-09-01"],
])
def test_existing_feature_calendar_cannot_be_reindexed(tmp_path, calendar):
    dumper = QlibDumper(str(tmp_path))
    dates = ["2026-09-01", "2026-09-03"]
    dumper.dump_calendars(dates)
    dumper.dump_features(quotes(dates, [10, 12]))
    with pytest.raises(ValueError, match="append|calendar"):
        dumper.dump_calendars(calendar)
    assert dumper.load_calendar() == dates
    np.testing.assert_array_equal(read_close(tmp_path), [0, 10, 12])


def test_unknown_date_rejected_before_writing(tmp_path):
    dumper = QlibDumper(str(tmp_path))
    dumper.dump_calendars(["2026-09-01"])
    dumper.dump_features(quotes(["2026-09-01"], [10]))
    with pytest.raises(ValueError, match="calendar"):
        dumper.dump_features(quotes(["2026-09-02"], [20]))
    np.testing.assert_array_equal(read_close(tmp_path), [0, 10])


def test_failed_replace_preserves_original_feature(tmp_path, monkeypatch):
    dumper = QlibDumper(str(tmp_path))
    dumper.dump_calendars(["2026-09-01", "2026-09-02"])
    dumper.dump_features(quotes(["2026-09-01"], [10]))

    def fail_replace(*args):
        raise OSError("simulated replacement failure")

    monkeypatch.setattr("src.data_pipeline.qlib_dumper.os.replace", fail_replace)
    with pytest.raises(OSError, match="replacement failure"):
        dumper.dump_features(quotes(["2026-09-02"], [12]))
    np.testing.assert_array_equal(read_close(tmp_path), [0, 10])
    assert sorted(p.name for p in (tmp_path / "features/sz000001").iterdir()) == ["close.day.bin"]
