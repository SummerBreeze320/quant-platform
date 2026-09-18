import os
import tempfile
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
        """Write a calendar without moving indices of existing features."""
        dates = sorted(list(set([str(d).strip() for d in dates])))
        cal_path = self.calendars_dir / f"{freq}.txt"
        existing = self.load_calendar(freq)
        if any(self.features_dir.glob(f"*/*.{freq}.bin")):
            if not existing or dates[:len(existing)] != existing:
                raise ValueError("Existing feature calendar is append-only; historical indices cannot change.")
        self._atomic_write(cal_path, "".join(f"{d}\n" for d in dates).encode("utf-8"))
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
        Upserts feature values while preserving all other dates and fields.
        df must contain 'symbol', 'date', and columns starting with '$'.
        """
        if df.empty:
            return
        calendar = self.load_calendar(freq=freq)
        if not calendar:
            raise ValueError("Calendar file is missing or empty. Please dump calendar first.")
        
        date_to_idx = {d: i for i, d in enumerate(calendar)}

        df_copy = df.copy()
        df_copy["norm_symbol"] = df_copy["symbol"].apply(self.normalize_symbol)
        df_copy["date"] = pd.to_datetime(df_copy["date"]).dt.strftime("%Y-%m-%d")
        if df_copy["date"].isna().any() or not df_copy["date"].isin(calendar).all():
            raise ValueError("All feature dates must be present in the calendar.")
        # A retry or correction is an upsert; last occurrence wins.
        df_copy = df_copy.drop_duplicates(["norm_symbol", "date"], keep="last")

        # Feature columns starting with '$'
        feature_cols = [c for c in df_copy.columns if c.startswith("$")]

        for norm_sym, group in df_copy.groupby("norm_symbol"):
            sym_dir = self.features_dir / norm_sym.lower()
            sym_dir.mkdir(parents=True, exist_ok=True)
            
            group = group.sort_values(by="date")
            indices = group["date"].map(date_to_idx).to_numpy(dtype=int)

            for f_col in feature_cols:
                # Strip leading '$' for file name
                field_name = f_col.lstrip("$").lower()
                bin_path = sym_dir / f"{field_name}.{freq}.bin"

                start_idx, end_idx = int(indices.min()), int(indices.max())
                old = None
                if bin_path.exists():
                    old = np.fromfile(bin_path, dtype="<f4")
                    if (bin_path.stat().st_size % 4 or len(old) < 2
                            or not np.isfinite(old[0]) or old[0] < 0
                            or old[0] != int(old[0])
                            or int(old[0]) + len(old) - 1 > len(calendar)):
                        raise ValueError(f"Invalid existing Qlib feature file: {bin_path}")
                    old_start = int(old[0])
                    start_idx = min(start_idx, old_start)
                    end_idx = max(end_idx, old_start + len(old) - 2)

                dense_arr = np.full(end_idx - start_idx + 1, np.nan, dtype="<f4")
                if old is not None:
                    offset = old_start - start_idx
                    dense_arr[offset:offset + len(old) - 1] = old[1:]
                dense_arr[indices - start_idx] = group[f_col].to_numpy(dtype=np.float32)
                output = np.concatenate(([start_idx], dense_arr)).astype("<f4")
                self._atomic_write(bin_path, output.tobytes())

        logger.info(f"Successfully dumped features for {df_copy['norm_symbol'].nunique()} symbols.")

    @staticmethod
    def _atomic_write(path: Path, content: bytes) -> None:
        """Replace one complete file; a failed write/replace leaves the old one intact."""
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as fp:
                temp_path = Path(fp.name)
                fp.write(content)
                fp.flush()
                os.fsync(fp.fileno())
            os.replace(temp_path, path)
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)
