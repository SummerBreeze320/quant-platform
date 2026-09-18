import hashlib
import numpy as np
import pandas as pd
from typing import List, Tuple
from src.common.logger import logger

FEATURE_COLUMNS = [
    "close", "open", "high", "low", "volume",
    "ma5", "ma10", "ma20", "std20", "rsi", "bias", "volatility",
]


class DataHandler:
    """Prepares training and prediction data from Qlib, with synthetic fallback."""

    @staticmethod
    def _try_qlib_features(instruments_key: str, fields: List[str], start: str, end: str):
        try:
            from qlib.data import D
            from src.qlib_engine.initializer import is_qlib_initialized
            if not is_qlib_initialized():
                return None
            insts = D.instruments(instruments_key)
            df = D.features(insts, fields, start_time=start, end_time=end)
            if df is not None and not df.empty:
                df = df.reset_index().rename(
                    columns={"instrument": "symbol", "datetime": "date"}
                )
                df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
                return df
        except Exception as e:
            logger.warning(f"Qlib feature extraction failed, using synthetic: {e}")
        return None

    @staticmethod
    def prepare_training_data(
        feature_set: List[str],
        train_start: str,
        train_end: str,
        n_stocks: int = 30,
        n_days: int = 100,
    ) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """Returns (X_train, y_train, feature_names)."""
        feature_names = FEATURE_COLUMNS

        df = DataHandler._try_qlib_features(
            "csi300", ["$close", "$open", "$high", "$low", "$volume"],
            train_start, train_end,
        )
        if df is not None and len(df) > 20:
            for c in ["close", "open", "high", "low", "volume"]:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
            df = df.dropna(subset=["close"]).sort_values(["symbol", "date"])
            df["label"] = df.groupby("symbol")["close"].pct_change().shift(-1)
            df = df.dropna(subset=["label"])
            feature_cols = [c for c in ["close", "open", "high", "low", "volume"] if c in df.columns]
            if len(df) > 20:
                return df[feature_cols], df["label"], feature_cols

        np.random.seed(42)
        n = n_stocks * n_days
        X = pd.DataFrame(np.random.randn(n, len(feature_names)), columns=feature_names)
        w = np.random.randn(len(feature_names))
        y = X @ w + np.random.randn(n) * 0.1
        return X, y, feature_names

    @staticmethod
    def prepare_prediction_data(
        feature_names: List[str],
        date: str,
        n_stocks: int = 30,
    ) -> pd.DataFrame:
        """Returns features for a single date with a symbol column."""
        df = DataHandler._try_qlib_features(
            "csi300", ["$close", "$open", "$high", "$low", "$volume"],
            date, date,
        )
        if df is not None and not df.empty:
            for c in ["close", "open", "high", "low", "volume"]:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
            df = df.dropna()
            if not df.empty:
                return df

        seed = int(hashlib.md5(date.encode()).hexdigest()[:8], 16) % (2**31)
        rng = np.random.RandomState(seed)
        data = {fn: rng.randn(n_stocks) for fn in feature_names}
        data["symbol"] = [f"SZ{i:06d}" for i in range(n_stocks)]
        data["date"] = date
        return pd.DataFrame(data)

    @staticmethod
    def prepare_backtest_data(
        feature_names: List[str],
        start_date: str = "2026-08-01",
        end_date: str = "2026-08-31",
        n_stocks: int = 30,
    ) -> pd.DataFrame:
        """Returns DataFrame with date, symbol, feature columns, ret, industry, market_cap."""
        df = DataHandler._try_qlib_features(
            "csi300", ["$close", "$open", "$high", "$low", "$volume"],
            start_date, end_date,
        )
        if df is not None and len(df) > 20:
            for c in ["close", "open", "high", "low", "volume"]:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
            df = df.dropna(subset=["close"]).sort_values(["symbol", "date"])
            df["ret"] = df.groupby("symbol")["close"].pct_change().fillna(0)
            df["industry"] = "Unknown"
            df["market_cap"] = 1e10
            return df

        dates = pd.date_range(start_date, end_date, freq="B").strftime("%Y-%m-%d").tolist()
        symbols = [f"SZ{i:06d}" for i in range(n_stocks)]
        industries = (["Bank", "Tech", "Pharma", "Consumer", "Energy"] * (n_stocks // 5 + 1))[:n_stocks]
        market_caps = [1e10 + i * 5e8 for i in range(n_stocks)]

        records = []
        rng = np.random.RandomState(42)
        for d in dates:
            for i, s in enumerate(symbols):
                row = {
                    "date": d, "symbol": s,
                    "industry": industries[i], "market_cap": market_caps[i],
                    "ret": float(rng.normal(0.0008, 0.015)),
                }
                for fn in feature_names:
                    row[fn] = float(rng.randn())
                records.append(row)
        return pd.DataFrame(records)
