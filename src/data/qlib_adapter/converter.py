"""Wind DataFrame -> Qlib bin format writer.

Only handles WRITING bin files. All data READING is done through
Qlib's native API (src.core.data_access / qlib.data.D).

Qlib bin format:
- Each field stored as {field}.{freq}.bin
- Binary layout: start_index(i32) + end_index(i32) + float32 data array
- start_index/end_index are offsets into the calendar (day.txt)
- Data array length = end_index - start_index
"""
import logging
import struct
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class WindToQlibConverter:
    """Write Wind DataFrames to Qlib bin format with calendar alignment."""

    FIELD_MAP = {
        "open": "open", "high": "high", "low": "low", "close": "close",
        "volume": "volume", "amount": "amount", "pct_chg": "change",
        "turn": "turnover", "free_turn": "free_turnover", "vwap": "vwap",
        "nav": "nav", "discount": "discount",
        "pe_ttm": "pe_ttm", "pb_lf": "pb_lf", "ps_ttm": "ps_ttm",
        "dividend_yield": "dividend_yield", "total_mv": "total_mv",
        "circ_mv": "circ_mv", "roe": "roe", "roa": "roa",
        "debt_ratio": "debt_ratio", "revenue_yoy": "revenue_yoy",
        "profit_yoy": "profit_yoy",
    }

    def __init__(self, qlib_dir: str = "data/qlib_bin"):
        self.qlib_dir = Path(qlib_dir)
        self.features_dir = self.qlib_dir / "features"
        self.features_dir.mkdir(parents=True, exist_ok=True)
        self._calendar: Optional[List[str]] = None

    def load_calendar(self) -> List[str]:
        """Load trade calendar from calendars/day.txt (for writing alignment)."""
        cal_path = self.qlib_dir / "calendars" / "day.txt"
        if not cal_path.exists():
            logger.warning("Calendar file not found, data won't be aligned.")
            return []
        dates = cal_path.read_text(encoding="utf-8").strip().split("\n")
        self._calendar = dates
        return dates

    @property
    def calendar(self) -> List[str]:
        if self._calendar is None:
            self.load_calendar()
        return self._calendar or []

    def convert_stock(
        self,
        code: str,
        df: pd.DataFrame,
        freq: str = "day",
        calendar_dates: Optional[list] = None,
    ) -> Optional[Tuple[int, int]]:
        """Convert single instrument DataFrame to Qlib bin files.

        Args:
            code: Wind code like "600000.SH"
            df: DataFrame indexed by date with OHLCV columns
            freq: "day" or "1min"
            calendar_dates: global trade calendar date list.
        Returns:
            (start_index, end_index) tuple, or None on failure.
        """
        if df.empty:
            logger.warning(f"Empty data for {code}, skipping.")
            return None

        instrument_dir = self._normalize_code(code)
        stock_dir = self.features_dir / instrument_dir
        stock_dir.mkdir(parents=True, exist_ok=True)

        cal = calendar_dates or self.calendar
        if cal:
            start_idx, end_idx, aligned = self._align_to_calendar(df, cal)
        else:
            start_idx = 0
            end_idx = len(df)
            aligned = df

        n = len(aligned)
        for wind_field, qlib_field in self.FIELD_MAP.items():
            if wind_field not in aligned.columns:
                continue
            values = aligned[wind_field].values.astype(np.float32)
            values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
            bin_path = stock_dir / f"{qlib_field}.{freq}.bin"
            with open(bin_path, "wb") as f:
                f.write(struct.pack("ii", start_idx, end_idx))
                values.tofile(f)
            logger.debug(f"Written {instrument_dir}/{qlib_field}.{freq}.bin: {n} values [{start_idx}:{end_idx}]")

        logger.info(f"Converted {code} -> {instrument_dir}/: {n} rows [{start_idx}:{end_idx}]")
        return start_idx, end_idx

    def convert_batch(
        self,
        data: Dict[str, pd.DataFrame],
        freq: str = "day",
        calendar_dates: Optional[list] = None,
    ) -> int:
        """Batch convert multiple instruments."""
        success = 0
        for code, df in data.items():
            try:
                result = self.convert_stock(code, df, freq, calendar_dates)
                if result is not None:
                    success += 1
            except Exception as e:
                logger.error(f"Convert failed {code}: {e}")
        logger.info(f"Batch converted: {success}/{len(data)}")
        return success

    @staticmethod
    def _normalize_code(code: str) -> str:
        """600000.SH -> SH600000"""
        if "." not in code:
            return code
        code_part, market = code.rsplit(".", 1)
        return f"{market}{code_part}"

    def _align_to_calendar(
        self, df: pd.DataFrame, calendar_dates: list
    ) -> Tuple[int, int, pd.DataFrame]:
        """Align DataFrame to global trade calendar.

        Returns (start_index, end_index, aligned_df) where indices are
        offsets into calendar_dates.
        """
        cal_index = pd.to_datetime(calendar_dates)
        df_index = pd.to_datetime(df.index)

        first_date = np.datetime64(df_index.min())
        last_date = np.datetime64(df_index.max())

        start_idx = int(np.searchsorted(cal_index.values, first_date))
        end_idx = int(np.searchsorted(cal_index.values, last_date)) + 1

        cal_slice = cal_index[start_idx:end_idx]
        aligned = df.reindex(cal_slice)
        aligned = aligned.ffill().bfill()

        return start_idx, end_idx, aligned
