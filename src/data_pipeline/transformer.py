import pandas as pd
import numpy as np
from typing import Literal

class DataTransformer:
    """Cleans, standardizes and adjusts raw market data for Qlib."""

    FIELD_MAPPING = {
        "SEC_CODE": "symbol",
        "WINDCODE": "symbol",
        "DATETIME": "date",
        "TRADE_DATE": "date",
        "DATE": "date",
        "OPEN": "$open",
        "HIGH": "$high",
        "LOW": "$low",
        "CLOSE": "$close",
        "VOLUME": "$volume",
        "AMT": "$money",
        "AMOUNT": "$money",
        "ADJFACTOR": "$factor",
        "VWAP": "$vwap",
        "TURNOVER": "$turnover",
        "FREE_FLOAT_SHARES": "$free_shares"
    }

    @classmethod
    def standardize_quotes(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Standardizes column names and formats from raw Wind query results."""
        if df.empty:
            return pd.DataFrame()

        clean_df = df.copy()
        
        # Upper-case column names for uniform matching
        clean_df.columns = [str(col).upper() for col in clean_df.columns]
        
        # Rename according to mapping
        rename_map = {k: v for k, v in cls.FIELD_MAPPING.items() if k in clean_df.columns}
        clean_df = clean_df.rename(columns=rename_map)

        # Standardize date format
        if "date" in clean_df.columns:
            clean_df["date"] = pd.to_datetime(clean_df["date"]).dt.strftime("%Y-%m-%d")

        # Standardize symbol code (e.g., ensure 000001.SZ format)
        if "symbol" in clean_df.columns:
            clean_df["symbol"] = clean_df["symbol"].astype(str).str.strip()

        # Convert numeric feature columns
        numeric_cols = [col for col in clean_df.columns if col.startswith("$")]
        for col in numeric_cols:
            clean_df[col] = pd.to_numeric(clean_df[col], errors="coerce")

        # Fill default factor if missing
        if "$factor" not in clean_df.columns:
            clean_df["$factor"] = 1.0
        else:
            clean_df["$factor"] = clean_df["$factor"].fillna(1.0)

        # Ensure minimal required columns
        required = ["symbol", "date", "$close"]
        for r in required:
            if r not in clean_df.columns:
                raise ValueError(f"Missing required standardized column: {r}")

        return clean_df.sort_values(by=["symbol", "date"]).reset_index(drop=True)

    @classmethod
    def compute_adjusted_prices(
        cls, df: pd.DataFrame, method: Literal["post", "pre"] = "post"
    ) -> pd.DataFrame:
        """
        Computes forward-adjusted (pre) or backward-adjusted (post) prices.
        Using post-adjustment: adj_price = raw_price * adj_factor
        """
        out_df = df.copy()
        price_cols = [col for col in ["$open", "$high", "$low", "$close", "$vwap"] if col in out_df.columns]
        
        factor = out_df.get("$factor", 1.0)
        
        if method == "post":
            for col in price_cols:
                out_df[f"{col}_adj"] = out_df[col] * factor
        elif method == "pre":
            # For pre-adjustment, relative to the latest available factor per symbol
            for symbol, group in out_df.groupby("symbol"):
                latest_factor = group["$factor"].iloc[-1] if not group.empty else 1.0
                ratio = group["$factor"] / (latest_factor if latest_factor != 0 else 1.0)
                for col in price_cols:
                    out_df.loc[group.index, f"{col}_adj"] = group[col] * ratio
                    
        return out_df
