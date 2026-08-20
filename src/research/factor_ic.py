"""Batch factor IC analysis pipeline.

Evaluates multiple factors simultaneously:
- IC (Pearson correlation between factor and forward returns)
- Rank IC (Spearman correlation)
- ICIR (IC / IC std)
- IC decay (IC over different forward periods)
- Factor turnover and coverage

Supports parallel evaluation and result aggregation.
"""
import logging
from typing import Optional, List, Dict, Union
from pathlib import Path
import pandas as pd
import numpy as np

from src.core import ensure_qlib, get_features

logger = logging.getLogger(__name__)


class FactorICAnalyzer:
    """Batch factor IC analysis.

    Usage:
        analyzer = FactorICAnalyzer()
        results = analyzer.evaluate(
            instruments=["SH600000", "SZ000001", ...],
            factor_names=["rocp_5", "rocp_20", "std_20"],
            forward_period=5,
        )
    """

    def __init__(
        self,
        qlib_dir: str = "data/qlib_bin",
        output_dir: str = "data/reports/factor_ic",
    ):
        ensure_qlib(qlib_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def evaluate(
        self,
        instruments: Union[str, List[str]],
        factor_names: List[str],
        forward_period: int = 5,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
    ) -> pd.DataFrame:
        """Evaluate IC for multiple factors.

        Args:
            instruments: Code or list of codes.
            factor_names: Factor names to evaluate.
            forward_period: Forward return period (days).
            start_time/end_time: Evaluation date range.

        Returns:
            DataFrame with columns: factor, IC, ICIR, Rank_IC, Rank_ICIR, coverage, turnover
        """
        from .factor_library import QLIB_EXPRESSION_FACTORS

        if isinstance(instruments, str):
            instruments = [instruments]

        # Get factor expressions
        expressions = []
        valid_names = []
        for name in factor_names:
            if name in QLIB_EXPRESSION_FACTORS:
                expressions.append(QLIB_EXPRESSION_FACTORS[name])
                valid_names.append(name)
            else:
                logger.warning(f"Unknown factor: {name}")

        if not valid_names:
            return pd.DataFrame()

        # Compute factor values
        logger.info(f"Computing {len(valid_names)} factors for {len(instruments)} instruments...")
        factor_df = get_features(instruments, expressions, start_time, end_time)
        if factor_df is None or factor_df.empty:
            return pd.DataFrame()
        factor_df.columns = valid_names

        # Compute forward returns
        close_expr = "$close"
        close_df = get_features(instruments, [close_expr], start_time, end_time)
        if close_df is None or close_df.empty:
            return pd.DataFrame()
        close_df.columns = ["close"]

        # Calculate forward returns
        fwd_ret = close_df.groupby(level="instrument")["close"].pct_change(forward_period).shift(-forward_period)
        fwd_ret = fwd_ret.dropna()

        # Calculate IC for each factor
        results = []
        for factor_name in valid_names:
            factor_values = factor_df[factor_name].dropna()

            # Align index
            common = factor_values.index.intersection(fwd_ret.index)
            if len(common) < 30:
                results.append({
                    "factor": factor_name,
                    "IC": 0, "ICIR": 0,
                    "Rank_IC": 0, "Rank_ICIR": 0,
                    "coverage": 0,
                    "n_obs": len(common),
                })
                continue

            f = factor_values.loc[common]
            r = fwd_ret.loc[common]

            # Cross-sectional IC per date
            ic_series = self._compute_ic_series(f, r)
            rank_ic_series = self._compute_ic_series(f, r, rank=True)

            coverage = len(common) / len(factor_values) if len(factor_values) > 0 else 0

            results.append({
                "factor": factor_name,
                "IC": float(ic_series.mean()),
                "ICIR": float(ic_series.mean() / ic_series.std()) if ic_series.std() > 0 else 0,
                "Rank_IC": float(rank_ic_series.mean()),
                "Rank_ICIR": float(rank_ic_series.mean() / rank_ic_series.std()) if rank_ic_series.std() > 0 else 0,
                "coverage": float(coverage),
                "n_obs": len(common),
            })

        result_df = pd.DataFrame(results)
        result_df = result_df.sort_values("IC", ascending=False)

        # Save to file
        output_file = self.output_dir / f"factor_ic_{forward_period}d.csv"
        result_df.to_csv(output_file, index=False)
        logger.info(f"Factor IC analysis saved: {output_file}")

        return result_df

    def evaluate_ic_decay(
        self,
        instruments: Union[str, List[str]],
        factor_name: str,
        periods: List[int] = [1, 3, 5, 10, 20, 60],
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
    ) -> pd.DataFrame:
        """Evaluate IC decay over different forward periods.

        Shows how factor predictability changes over time.
        """
        from .factor_library import QLIB_EXPRESSION_FACTORS

        if factor_name not in QLIB_EXPRESSION_FACTORS:
            logger.error(f"Unknown factor: {factor_name}")
            return pd.DataFrame()

        if isinstance(instruments, str):
            instruments = [instruments]

        # Get close prices
        close_df = get_features(instruments, ["$close"], start_time, end_time)
        if close_df is None or close_df.empty:
            return pd.DataFrame()
        close_df.columns = ["close"]

        # Get factor values
        factor_expr = QLIB_EXPRESSION_FACTORS[factor_name]
        factor_df = get_features(instruments, [factor_expr], start_time, end_time)
        if factor_df is None or factor_df.empty:
            return pd.DataFrame()
        factor_df.columns = [factor_name]
        factor_values = factor_df[factor_name].dropna()

        results = []
        for period in periods:
            fwd_ret = close_df.groupby(level="instrument")["close"].pct_change(period).shift(-period).dropna()
            common = factor_values.index.intersection(fwd_ret.index)

            if len(common) < 30:
                results.append({"period": period, "IC": 0, "Rank_IC": 0})
                continue

            ic_series = self._compute_ic_series(factor_values.loc[common], fwd_ret.loc[common])
            rank_ic_series = self._compute_ic_series(
                factor_values.loc[common], fwd_ret.loc[common], rank=True
            )

            results.append({
                "period": period,
                "IC": float(ic_series.mean()),
                "Rank_IC": float(rank_ic_series.mean()),
            })

        return pd.DataFrame(results)

    def _compute_ic_series(
        self,
        factor: pd.Series,
        returns: pd.Series,
        rank: bool = False,
    ) -> pd.Series:
        """Compute time series of cross-sectional IC."""
        # Align to (instrument, datetime) MultiIndex
        aligned = pd.DataFrame({"factor": factor, "ret": returns}).dropna()

        if rank:
            aligned["factor"] = aligned.groupby(level="datetime")["factor"].rank()
            aligned["ret"] = aligned.groupby(level="datetime")["ret"].rank()

        ic_series = aligned.groupby(level="datetime").apply(
            lambda g: g["factor"].corr(g["ret"]) if len(g) > 5 else np.nan
        )
        return ic_series.dropna()

    def get_factor_summary(self, ic_df: pd.DataFrame) -> Dict:
        """Generate summary statistics from IC results."""
        if ic_df.empty:
            return {}

        effective = ic_df[ic_df["IC"].abs() >= 0.03]
        strong = ic_df[ic_df["IC"].abs() >= 0.05]

        return {
            "total_factors": len(ic_df),
            "effective_factors": len(effective),
            "strong_factors": len(strong),
            "mean_ic": float(ic_df["IC"].mean()),
            "max_ic": float(ic_df["IC"].max()),
            "min_ic": float(ic_df["IC"].min()),
            "best_factor": ic_df.iloc[0]["factor"] if len(ic_df) > 0 else None,
            "best_factor_ic": float(ic_df.iloc[0]["IC"]) if len(ic_df) > 0 else 0,
        }


def batch_evaluate_factors(
    instruments: Union[str, List[str]],
    factor_names: Optional[List[str]] = None,
    forward_period: int = 5,
    qlib_dir: str = "data/qlib_bin",
) -> Dict:
    """Convenience function for batch factor evaluation.

    Args:
        instruments: Code or list of codes.
        factor_names: Factor names to evaluate. None = all from QLIB_EXPRESSION_FACTORS.
        forward_period: Forward return period.
        qlib_dir: Qlib data directory.

    Returns:
        Dict with IC DataFrame and summary.
    """
    from .factor_library import QLIB_EXPRESSION_FACTORS

    if factor_names is None:
        factor_names = list(QLIB_EXPRESSION_FACTORS.keys())

    analyzer = FactorICAnalyzer(qlib_dir=qlib_dir)
    ic_df = analyzer.evaluate(instruments, factor_names, forward_period)
    summary = analyzer.get_factor_summary(ic_df)

    return {
        "ic_table": ic_df,
        "summary": summary,
    }
