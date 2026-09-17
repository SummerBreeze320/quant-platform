from typing import Optional
import numpy as np
import pandas as pd
from src.common.logger import logger

class FactorNeutralizer:
    """
    Factor Neutralizer and Cross-Sectional Preprocessor.
    Implements MAD Winsorization, Z-Score Standardization and OLS Orthogonal Neutralization
    against Industry (Barra classification) and Size (Log Market Cap).
    """

    @staticmethod
    def winsorize_mad(series: pd.Series, n: float = 3.0) -> pd.Series:
        """
        Applies Median Absolute Deviation (MAD) winsorization to remove extreme outliers.
        Effective bounds: [median - n * 1.4826 * MAD, median + n * 1.4826 * MAD].
        """
        median = float(series.median())
        mad = float((series - median).abs().median())
        if mad < 1e-8:
            return series

        scale = 1.4826 * mad
        lower_bound = median - n * scale
        upper_bound = median + n * scale
        return series.clip(lower=lower_bound, upper=upper_bound)

    @staticmethod
    def standardize_zscore(series: pd.Series) -> pd.Series:
        """Standardizes series to mean 0 and standard deviation 1."""
        std = float(series.std())
        if std < 1e-8:
            return series - series.mean()
        return (series - series.mean()) / std

    @classmethod
    def neutralize(
        cls,
        df: pd.DataFrame,
        score_col: str = "score",
        industry_col: Optional[str] = "industry",
        market_cap_col: Optional[str] = "market_cap",
        winsorize: bool = True
    ) -> pd.DataFrame:
        """
        Neutralizes raw factor scores against industry and log market cap using OLS regression.
        The resulting residuals represent pure idiosyncratic alpha.
        """
        res_df = df.copy()
        if res_df.empty:
            res_df["score_neutral"] = []
            return res_df

        scores = res_df[score_col].astype(float)
        if winsorize:
            scores = cls.winsorize_mad(scores)

        # Build regressors design matrix
        regressors = []

        # 1. Industry One-Hot Dummies (full span without intercept)
        has_industry = industry_col and industry_col in res_df.columns
        if has_industry:
            ind_dummies = pd.get_dummies(res_df[industry_col], dtype=float)
            regressors.append(ind_dummies)

        # 2. Log Market Cap
        has_cap = market_cap_col and market_cap_col in res_df.columns
        if has_cap:
            # Handle potential non-positive values
            valid_cap = res_df[market_cap_col].clip(lower=1.0)
            log_cap = pd.DataFrame({"log_market_cap": np.log(valid_cap.astype(float))})
            regressors.append(log_cap)

        if not regressors:
            # No neutralization variables found, return standardized score
            res_df["score_neutral"] = cls.standardize_zscore(scores)
            return res_df

        if not has_industry:
            # Add constant intercept if no full-rank industry dummies
            X_df = pd.concat(regressors, axis=1)
            X_df.insert(0, "intercept", 1.0)
        else:
            X_df = pd.concat(regressors, axis=1)

        X = X_df.values
        y = scores.values

        # Perform OLS regression: y = X * beta + residuals
        # Using pseudo-inverse for numerical stability
        try:
            beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            residuals = y - X @ beta
        except Exception as e:
            logger.warning(f"Neutralization regression failed, falling back to raw: {e}")
            residuals = y

        res_df["score_neutral"] = cls.standardize_zscore(pd.Series(residuals, index=res_df.index))
        return res_df
