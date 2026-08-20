"""Qlib expression-engine factor calculator for the data pipeline.

Bridges the data layer (bin files) with the factor layer by computing
factor values via Qlib's D.features() expression engine. Supports batch
computation, caching, incremental updates, and Feature Store integration.
"""
import logging
from typing import Optional, List, Dict, Union, Tuple
from pathlib import Path
import pandas as pd
import numpy as np

from src.core import ensure_qlib, get_features, list_instruments, get_calendar

logger = logging.getLogger(__name__)


class FactorCalculator:
    """Compute factor values using Qlib's expression engine.

    Supports:
    - Alpha158-style expression factors
    - Custom factor expressions
    - Batch computation across instruments
    - Disk cache for computed factors
    - Feature Store integration (versioned, cross-experiment reuse)
    """

    DEFAULT_FACTORS: Dict[str, str] = {
        "kmid": "($close - $open) / $open",
        "klen": "($high - $low) / $open",
        "kup": "($high - $open) / $open",
        "klow": "($open - $low) / $open",
        "ksft": "($close - $low) / $open",
        "cfh": "($high - $close) / $open",
        "cfl": "($close - $low) / $open",
        "open0": "$open / Ref($close, 1) - 1",
        "high0": "$high / Ref($close, 1) - 1",
        "low0": "$low / Ref($close, 1) - 1",
        "rocp_5": "Ref($close, 5) / $close - 1",
        "rocp_10": "Ref($close, 10) / $close - 1",
        "rocp_20": "Ref($close, 20) / $close - 1",
        "rocp_60": "Ref($close, 60) / $close - 1",
        "ma_5": "Mean($close, 5) / $close - 1",
        "ma_20": "Mean($close, 20) / $close - 1",
        "ma_60": "Mean($close, 60) / $close - 1",
        "std_5": "Std($close, 5) / $close",
        "std_20": "Std($close, 20) / $close",
        "std_60": "Std($close, 60) / $close",
        "vol_std_5": "Std($volume, 5) / Mean($volume, 5)",
        "vol_std_20": "Std($volume, 20) / Mean($volume, 20)",
        "vol_ratio_5": "$volume / Mean($volume, 5)",
        "vol_ratio_20": "$volume / Mean($volume, 20)",
        "rsi_14": "100 - 100 / (1 + Mean(Max($close - Ref($close, 1), 0), 14) / Mean(Abs($close - Ref($close, 1)), 14))",
        "bb_width": "(Mean($close, 20) + 2 * Std($close, 20)) / (Mean($close, 20) - 2 * Std($close, 20)) - 1",
    }

    def __init__(
        self,
        qlib_dir: str = "data/qlib_bin",
        cache_dir: str = "data/cache/factors",
        use_feature_store: bool = True,
    ):
        ensure_qlib(qlib_dir)
        self.qlib_dir = Path(qlib_dir)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.use_feature_store = use_feature_store
        self._feature_store = None

        if use_feature_store:
            try:
                from src.data.feature_store import FeatureStore
                self._feature_store = FeatureStore()
                logger.info("FeatureStore connected")
            except Exception as e:
                logger.warning(f"FeatureStore unavailable: {e}")

    def compute(
        self,
        instruments: Union[str, List[str]],
        factor_names: Optional[List[str]] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        use_cache: bool = True,
        use_feature_store: bool = True,
    ) -> pd.DataFrame:
        """Compute factor values for given instruments.

        Args:
            instruments: Code or list of codes (Qlib format: SH600000).
            factor_names: Factor names from DEFAULT_FACTORS. None = all.
            start_time/end_time: Date range.
            use_cache: If True, load from disk cache if available.
            use_feature_store: If True, check Feature Store first.

        Returns:
            MultiIndex DataFrame (instrument, datetime) with factor columns.
        """
        if isinstance(instruments, str):
            instruments = [instruments]

        if factor_names is None:
            factor_names = list(self.DEFAULT_FACTORS.keys())

        # 1. Try Feature Store first
        if use_feature_store and self._feature_store is not None:
            fs_df = self._try_feature_store(
                factor_names, instruments, start_time, end_time
            )
            if fs_df is not None:
                return fs_df

        # 2. Check local cache
        cache_key = self._cache_key(instruments, factor_names, start_time, end_time)
        cache_path = self.cache_dir / f"{cache_key}.parquet"

        if use_cache and cache_path.exists():
            logger.info(f"Loading cached factors: {cache_path.name}")
            return pd.read_parquet(cache_path)

        # 3. Compute via Qlib
        expressions = []
        valid_names = []
        for name in factor_names:
            if name in self.DEFAULT_FACTORS:
                expressions.append(self.DEFAULT_FACTORS[name])
                valid_names.append(name)
            else:
                logger.warning(f"Unknown factor: {name}, skipping")

        if not expressions:
            return pd.DataFrame()

        logger.info(f"Computing {len(valid_names)} factors for {len(instruments)} instruments...")

        df = get_features(
            instruments=instruments,
            fields=expressions,
            start_time=start_time,
            end_time=end_time,
            disk_cache=1,
        )

        if df is not None and not df.empty:
            df.columns = valid_names
            if use_cache:
                df.to_parquet(cache_path)
                logger.info(f"Cached factors: {cache_path.name}")

            # Save to Feature Store
            if use_feature_store and self._feature_store is not None:
                self._save_to_feature_store(valid_names, expressions, df, instruments, start_time, end_time)

        return df

    def _try_feature_store(
        self,
        factor_names: List[str],
        instruments: List[str],
        start_time: Optional[str],
        end_time: Optional[str],
    ) -> Optional[pd.DataFrame]:
        """Try to load features from Feature Store."""
        try:
            df = self._feature_store.serve(
                names=factor_names,
                start_date=start_time,
                end_date=end_time,
            )
            if df is not None and not df.empty:
                logger.info(f"FeatureStore hit: {len(df)} rows for {len(factor_names)} features")
                return df
        except Exception as e:
            logger.debug(f"FeatureStore miss: {e}")
        return None

    def _save_to_feature_store(
        self,
        names: List[str],
        expressions: List[str],
        df: pd.DataFrame,
        instruments: List[str],
        start_time: Optional[str],
        end_time: Optional[str],
    ):
        """Save computed features to Feature Store."""
        date_range = (start_time or "2020-01-01", end_time or "2024-12-31")
        for name, expr in zip(names, expressions):
            if name in df.columns:
                try:
                    self._feature_store.save(
                        name=name,
                        expression=expr,
                        data=df[[name]],
                        instruments=instruments,
                        date_range=date_range,
                    )
                except Exception as e:
                    logger.debug(f"FeatureStore save failed for {name}: {e}")

    def compute_latest(
        self,
        instruments: Union[str, List[str]],
        factor_names: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Get the latest factor values (cross-sectional snapshot).

        Returns a DataFrame indexed by instrument with factor columns.
        """
        df = self.compute(instruments, factor_names, use_cache=False)
        if df.empty:
            return pd.DataFrame()

        latest = df.groupby(level="instrument").last()
        return latest

    def compute_incremental(
        self,
        instruments: Union[str, List[str]],
        factor_names: List[str],
        last_date: str,
    ) -> pd.DataFrame:
        """Compute factors incrementally since a given date.

        Used for daily updates: only compute factors for dates after last_date.
        """
        from src.core.data_access import get_calendar

        cal = get_calendar()
        if last_date in cal:
            idx = cal.index(pd.Timestamp(last_date))
            start = cal[min(idx + 1, len(cal) - 1)]
            start_time = start.strftime("%Y-%m-%d")
        else:
            start_time = last_date

        return self.compute(
            instruments, factor_names,
            start_time=start_time,
            use_cache=False,
        )

    def _cache_key(
        self,
        instruments: List[str],
        factor_names: List[str],
        start_time: Optional[str],
        end_time: Optional[str],
    ) -> str:
        import hashlib
        key_str = f"{'-'.join(sorted(instruments))}|{'-'.join(sorted(factor_names))}|{start_time}|{end_time}"
        return hashlib.md5(key_str.encode()).hexdigest()[:16]

    def list_available_factors(self) -> List[Dict]:
        """List all available factors with their expressions."""
        return [
            {"name": name, "expression": expr}
            for name, expr in self.DEFAULT_FACTORS.items()
        ]
