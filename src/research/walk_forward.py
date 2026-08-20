"""Walk-Forward Validator — 滚动前进验证

Implements rolling forward validation with strict no-look-ahead guarantee.
Each fold's training data ends before the test data starts.
"""
import logging
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardFold:
    """One fold of walk-forward validation."""
    fold_id: int
    train_start: str
    train_end: str
    valid_start: str
    valid_end: str
    test_start: str
    test_end: str


@dataclass
class WalkForwardResult:
    """Aggregated walk-forward validation result."""
    n_folds: int
    fold_results: List[Dict[str, Any]] = field(default_factory=list)
    avg_annual_return: float = 0.0
    avg_sharpe: float = 0.0
    avg_max_drawdown: float = 0.0
    return_consistency: float = 0.0
    ic_stability: float = 0.0
    passed: bool = False


class WalkForwardValidator:
    """Rolling forward validation with no-look-ahead guarantee.

    Splits a date range into multiple folds where each fold has:
    - train: earliest segment (grows or rolls forward)
    - valid: middle segment for hyperparameter tuning
    - test: out-of-sample segment (strictly after train+valid)

    No future data leaks into training at any fold.
    """

    def __init__(
        self,
        train_pct: float = 0.6,
        valid_pct: float = 0.2,
        n_folds: int = 5,
        min_train_days: int = 252,
    ):
        self.train_pct = train_pct
        self.valid_pct = valid_pct
        self.n_folds = n_folds
        self.min_train_days = min_train_days

    def split(
        self, start_date: str, end_date: str
    ) -> List[WalkForwardFold]:
        """Generate walk-forward folds from date range.

        Args:
            start_date: Backtest start date (YYYY-MM-DD)
            end_date: Backtest end date (YYYY-MM-DD)

        Returns:
            List of WalkForwardFold objects
        """
        dates = pd.date_range(start_date, end_date, freq="D")
        n_total = len(dates)
        if n_total < self.min_train_days:
            logger.warning(
                f"Only {n_total} days available, need at least {self.min_train_days}"
            )
            return []

        train_size = int(n_total * self.train_pct)
        valid_size = int(n_total * self.valid_pct)
        test_size = n_total - train_size - valid_size

        if test_size < 21:
            logger.warning("Test segment too short, reducing n_folds")
            self.n_folds = max(1, test_size // 21)

        fold_size = max(1, test_size // self.n_folds)
        folds = []

        for i in range(self.n_folds):
            t_end_idx = train_size + i * fold_size
            v_start_idx = t_end_idx
            v_end_idx = v_start_idx + valid_size
            te_start_idx = v_end_idx
            te_end_idx = min(te_start_idx + fold_size, n_total)

            if te_end_idx <= te_start_idx:
                break

            fold = WalkForwardFold(
                fold_id=i,
                train_start=dates[0].strftime("%Y-%m-%d"),
                train_end=dates[min(t_end_idx - 1, n_total - 1)].strftime("%Y-%m-%d"),
                valid_start=dates[min(v_start_idx, n_total - 1)].strftime("%Y-%m-%d"),
                valid_end=dates[min(v_end_idx - 1, n_total - 1)].strftime("%Y-%m-%d"),
                test_start=dates[min(te_start_idx, n_total - 1)].strftime("%Y-%m-%d"),
                test_end=dates[min(te_end_idx - 1, n_total - 1)].strftime("%Y-%m-%d"),
            )
            folds.append(fold)
            logger.info(
                f"Fold {i}: train={fold.train_start}→{fold.train_end} "
                f"valid={fold.valid_start}→{fold.valid_end} "
                f"test={fold.test_start}→{fold.test_end}"
            )

        return folds

    def validate(
        self,
        backtest_fn,
        start_date: str,
        end_date: str,
        metrics_keys: Optional[List[str]] = None,
    ) -> WalkForwardResult:
        """Run walk-forward validation on a backtest function.

        Args:
            backtest_fn: callable(start_date, end_date) -> dict with 'metrics'
            start_date: Overall start date
            end_date: Overall end date
            metrics_keys: Which metrics to aggregate

        Returns:
            WalkForwardResult with aggregated statistics
        """
        if metrics_keys is None:
            metrics_keys = ["annual_return", "sharpe_ratio", "max_drawdown"]

        folds = self.split(start_date, end_date)
        if not folds:
            return WalkForwardResult(n_folds=0, passed=False)

        result = WalkForwardResult(n_folds=len(folds))
        fold_metrics = []

        for fold in folds:
            try:
                bt_result = backtest_fn(fold.test_start, fold.test_end)
                m = bt_result.get("metrics", {}) if isinstance(bt_result, dict) else {}
                fold_metrics.append(m)
                result.fold_results.append({
                    "fold_id": fold.fold_id,
                    "test_period": f"{fold.test_start}→{fold.test_end}",
                    "metrics": m,
                })
                logger.info(
                    f"Fold {fold.fold_id}: "
                    f"annual={m.get('annual_return', 0):.4f} "
                    f"sharpe={m.get('sharpe_ratio', 0):.4f}"
                )
            except Exception as e:
                logger.warning(f"Fold {fold.fold_id} failed: {e}")
                result.fold_results.append({
                    "fold_id": fold.fold_id,
                    "error": str(e),
                })

        if fold_metrics:
            for key in metrics_keys:
                values = [m.get(key, 0) or 0 for m in fold_metrics]
                avg = float(np.mean(values))
                std = float(np.std(values))
                if key == "annual_return":
                    result.avg_annual_return = avg
                elif key == "sharpe_ratio":
                    result.avg_sharpe = avg
                elif key == "max_drawdown":
                    result.avg_max_drawdown = avg

            returns = [m.get("annual_return", 0) or 0 for m in fold_metrics]
            positive = sum(1 for r in returns if r > 0)
            result.return_consistency = positive / len(returns) if returns else 0

            sharpes = [m.get("sharpe_ratio", 0) or 0 for m in fold_metrics]
            if len(sharpes) > 1 and np.mean(sharpes) != 0:
                result.ic_stability = 1.0 - min(
                    1.0, float(np.std(sharpes) / abs(np.mean(sharpes)))
                )

            result.passed = (
                result.avg_sharpe > 0
                and result.return_consistency >= 0.6
                and result.avg_max_drawdown > -0.25
            )

        logger.info(
            f"Walk-Forward: {result.n_folds} folds, "
            f"avg_sharpe={result.avg_sharpe:.4f} "
            f"consistency={result.return_consistency:.0%} "
            f"passed={result.passed}"
        )
        return result
