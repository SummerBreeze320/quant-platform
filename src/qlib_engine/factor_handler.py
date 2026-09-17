import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from scipy import stats
from src.common.logger import logger

class FactorHandler:
    """Manages Qlib Alpha expressions and statistical evaluation of quantitative factors."""

    @staticmethod
    def evaluate_factor_performance(
        df: pd.DataFrame,
        factor_col: str,
        label_col: str = "label",
        date_col: str = "date"
    ) -> Dict[str, Any]:
        """
        Computes cross-sectional Rank IC, Normal IC, ICIR, and t-stat.
        df must contain [date_col, factor_col, label_col].
        """
        clean_df = df.dropna(subset=[factor_col, label_col]).copy()
        if clean_df.empty:
            return {
                "rank_ic": 0.0,
                "normal_ic": 0.0,
                "icir": 0.0,
                "t_stat": 0.0,
                "positive_ic_ratio": 0.0,
                "n_dates": 0
            }

        daily_rank_ics = []
        daily_normal_ics = []

        for _, group in clean_df.groupby(date_col):
            if len(group) < 3:
                continue
            
            f_vals = group[factor_col].values
            l_vals = group[label_col].values
            
            # Spearman / Rank IC
            rank_ic, _ = stats.spearmanr(f_vals, l_vals)
            if not np.isnan(rank_ic):
                daily_rank_ics.append(rank_ic)

            # Pearson IC
            normal_ic, _ = stats.pearsonr(f_vals, l_vals)
            if not np.isnan(normal_ic):
                daily_normal_ics.append(normal_ic)

        if not daily_rank_ics:
            return {
                "rank_ic": 0.0,
                "normal_ic": 0.0,
                "icir": 0.0,
                "t_stat": 0.0,
                "positive_ic_ratio": 0.0,
                "n_dates": 0
            }

        rank_ic_arr = np.array(daily_rank_ics)
        mean_rank_ic = float(np.mean(rank_ic_arr))
        std_rank_ic = float(np.std(rank_ic_arr))
        n_dates = len(rank_ic_arr)
        
        icir = float(mean_rank_ic / (std_rank_ic + 1e-8))
        t_stat = float(mean_rank_ic / ((std_rank_ic / np.sqrt(n_dates)) + 1e-8))
        pos_ratio = float(np.mean(rank_ic_arr > 0))

        return {
            "rank_ic": round(mean_rank_ic, 4),
            "normal_ic": round(float(np.mean(daily_normal_ics)), 4) if daily_normal_ics else 0.0,
            "icir": round(icir, 4),
            "t_stat": round(t_stat, 4),
            "positive_ic_ratio": round(pos_ratio, 4),
            "n_dates": n_dates
        }

    @staticmethod
    def extract_features(
        instruments: List[str],
        fields: List[str],
        start_time: str,
        end_time: str
    ) -> pd.DataFrame:
        """Loads features using Qlib's D.features engine."""
        try:
            from qlib.data import D
            df = D.features(instruments, fields, start_time=start_time, end_time=end_time)
            # Reset index from (instrument, datetime) MultiIndex to standard columns
            df = df.reset_index()
            df = df.rename(columns={"instrument": "symbol", "datetime": "date"})
            df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
            return df
        except Exception as e:
            logger.error(f"Error extracting features via Qlib: {e}")
            raise
