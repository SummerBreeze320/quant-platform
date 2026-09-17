from typing import Dict, Any
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

class FactorQuantileAnalyzer:
    """
    Factor Quantile and Long-Short Spread Analyzer.
    Partitions universe cross-sectionally into quantiles (e.g. Q1 ~ Q5)
    to evaluate factor monotonicity, spread performance and decay.
    """

    @staticmethod
    def analyze(
        df: pd.DataFrame,
        n_quantiles: int = 5,
        score_col: str = "score",
        ret_col: str = "ret",
        date_col: str = "date"
    ) -> Dict[str, Any]:
        """
        Runs quantile stratification analysis across dates.
        """
        if df.empty:
            return {}

        dates = sorted(df[date_col].unique())
        quantile_labels = [f"Q{i}" for i in range(1, n_quantiles + 1)]
        
        # Store daily returns per quantile
        q_daily_returns = {q: [] for q in quantile_labels}
        long_short_daily = []
        valid_dates = []

        for d in dates:
            day_slice = df[df[date_col] == d].copy()
            if len(day_slice) < n_quantiles:
                continue

            # Rank scores descending so Q1 is highest score
            ranks = day_slice[score_col].rank(method="first", ascending=False)
            day_slice["quantile"] = pd.qcut(ranks, q=n_quantiles, labels=quantile_labels)

            day_q_means = day_slice.groupby("quantile", observed=False)[ret_col].mean()
            
            for q in quantile_labels:
                val = float(day_q_means.get(q, 0.0))
                q_daily_returns[q].append(val)

            # Long-Short: Q1 return minus Q_n return
            ls_ret = float(day_q_means.get("Q1", 0.0) - day_q_means.get(f"Q{n_quantiles}", 0.0))
            long_short_daily.append(ls_ret)
            valid_dates.append(str(d))

        # Compute summary stats for each quantile
        quantile_summary = {}
        cum_returns_list = []
        for q in quantile_labels:
            ret_series = np.array(q_daily_returns[q])
            cum_ret = float(np.prod(1.0 + ret_series) - 1.0) if len(ret_series) > 0 else 0.0
            ann_ret = float((1.0 + cum_ret) ** (252.0 / max(len(ret_series), 1)) - 1.0)
            std = float(np.std(ret_series))
            sharpe = float(np.mean(ret_series) / (std + 1e-8) * np.sqrt(252)) if std > 0 else 0.0

            quantile_summary[q] = {
                "total_return": round(cum_ret, 4),
                "annualized_return": round(ann_ret, 4),
                "sharpe_ratio": round(sharpe, 4)
            }
            cum_returns_list.append(cum_ret)

        # Monotonicity test: Spearman correlation between [1, 2, ..., n] and [-cum_returns]
        # (since Q1 should be highest return)
        corr, _ = spearmanr(list(range(1, n_quantiles + 1)), [-c for c in cum_returns_list])
        monotonicity_score = round(float(corr) if not np.isnan(corr) else 0.0, 4)

        # Long-Short spread stats
        ls_arr = np.array(long_short_daily)
        ls_cum = float(np.prod(1.0 + ls_arr) - 1.0) if len(ls_arr) > 0 else 0.0
        ls_std = float(np.std(ls_arr))
        ls_sharpe = float(np.mean(ls_arr) / (ls_std + 1e-8) * np.sqrt(252)) if ls_std > 0 else 0.0

        return {
            "n_quantiles": n_quantiles,
            "dates": valid_dates,
            "quantile_returns": quantile_summary,
            "monotonicity_score": monotonicity_score,
            "long_short_cumulative": round(ls_cum, 4),
            "long_short_sharpe": round(ls_sharpe, 4)
        }
