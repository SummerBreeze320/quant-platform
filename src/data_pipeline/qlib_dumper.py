import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
from src.common.logger import logger

class QlibDumper:
    """Converts structured DataFrame and metadata into Qlib's binary format."""

    def __init__(self, target_dir: str):
        self.target_dir = Path(target_dir)
        self.calendars_dir = self.target_dir / "calendars"
        self.instruments_dir = self.target_dir / "instruments"
        self.features_dir = self.target_dir / "features"
        
        self.calendars_dir.mkdir(parents=True, exist_ok=True)
        self.instruments_dir.mkdir(parents=True, exist_ok=True)
        self.features_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def normalize_symbol(symbol: str) -> str:
        """Normalizes symbol to Qlib format (e.g. 000001.SZ -> SZ000001)."""
        symbol = str(symbol).strip().upper()
        if "." in symbol:
            code, market = symbol.split(".", 1)
            return f"{market}{code}"
        return symbol

    def dump_calendars(self, dates: List[str], freq: str = "day") -> None:
        """Writes trading calendar file."""
        dates = sorted(list(set([str(d).strip() for d in dates])))
        cal_path = self.calendars_dir / f"{freq}.txt"
        with open(cal_path, "w", encoding="utf-8") as f:
            for d in dates:
                f.write(f"{d}\n")
        logger.info(f"Saved {len(dates)} dates to calendar: {cal_path}")

    def load_calendar(self, freq: str = "day") -> List[str]:
        """Loads dates from existing calendar file."""
        cal_path = self.calendars_dir / f"{freq}.txt"
        if not cal_path.exists():
            return []
        with open(cal_path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

    def dump_instruments(
        self, instruments_map: Dict[str, Tuple[str, str]], pool: str = "all"
    ) -> None:
        """
        Writes instruments file (symbol, start_date, end_date).
        instruments_map: {symbol: (start_date, end_date)}
        """
        inst_path = self.instruments_dir / f"{pool}.txt"
        with open(inst_path, "w", encoding="utf-8") as f:
            for sym, (start, end) in sorted(instruments_map.items()):
                norm_sym = self.normalize_symbol(sym)
                f.write(f"{norm_sym}\t{start}\t{end}\n")
        logger.info(f"Saved {len(instruments_map)} instruments to: {inst_path}")

    def dump_features(self, df: pd.DataFrame, freq: str = "day") -> None:
        """
        Writes feature columns in df to Qlib .bin files.
        df must contain 'symbol', 'date', and columns starting with '$'.
        """
        calendar = self.load_calendar(freq=freq)
        if not calendar:
            raise ValueError("Calendar file is missing or empty. Please dump calendar first.")
        
        date_to_idx = {d: i for i, d in enumerate(calendar)}

        df_copy = df.copy()
        df_copy["norm_symbol"] = df_copy["symbol"].apply(self.normalize_symbol)
        df_copy["date"] = pd.to_datetime(df_copy["date"]).dt.strftime("%Y-%m-%d")

        # Feature columns starting with '$'
        feature_cols = [c for c in df_copy.columns if c.startswith("$")]

        for norm_sym, group in df_copy.groupby("norm_symbol"):
            sym_dir = self.features_dir / norm_sym.lower()
            sym_dir.mkdir(parents=True, exist_ok=True)
            
            group = group.sort_values(by="date")
            valid_dates = [d for d in group["date"] if d in date_to_idx]
            if not valid_dates:
                continue

            start_date = valid_dates[0]
            end_date = valid_dates[-1]
            start_idx = date_to_idx[start_date]
            end_idx = date_to_idx[end_date]
            total_len = end_idx - start_idx + 1

            # Map date indices to values
            idx_series = group["date"].map(date_to_idx).values

            for f_col in feature_cols:
                # Strip leading '$' for file name
                field_name = f_col.lstrip("$").lower()
                bin_path = sym_dir / f"{field_name}.{freq}.bin"

                # Prepare dense array initialized to NaN
                dense_arr = np.full(total_len, np.nan, dtype=np.float32)
                val_arr = group[f_col].values.astype(np.float32)

                rel_indices = idx_series - start_idx
                dense_arr[rel_indices] = val_arr

                # Write binary: start_index (float32) followed by data (float32)
                output_data = np.hstack([[start_idx], dense_arr]).astype("<f")
                with open(bin_path, "wb") as fp:
                    output_data.tofile(fp)

        logger.info(f"Successfully dumped features for {df_copy['norm_symbol'].nunique()} symbols.")
