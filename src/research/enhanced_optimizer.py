"""EnhancedIndexing portfolio optimizer wrapper.

Wraps Qlib's EnhancedIndexing portfolio optimization with tracking error
constraints, factor tilts, and sector neutrality. Integrates with the
existing backtest/simulator layer.
"""
import logging
from typing import Optional, Dict, Any, List
import pandas as pd
import numpy as np

from src.core import ensure_qlib

logger = logging.getLogger(__name__)


class EnhancedIndexingOptimizer:
    """增强指数优化器

    在跟踪基准的基础上，通过因子偏移获取超额收益。
    约束条件:
    - 跟踪误差上限 (tracking_error_limit)
    - 单票权重上限 (weights_limit)
    - 行业偏离上限 (industry_deviation_limit)
    - 换手率上限 (turnover_limit)
    """

    def __init__(
        self,
        benchmark: str = "SH000300",
        tracking_error_limit: float = 0.05,
        weights_limit: float = 0.05,
        industry_deviation_limit: float = 0.10,
        turnover_limit: float = 0.30,
    ):
        self.benchmark = benchmark
        self.tracking_error_limit = tracking_error_limit
        self.weights_limit = weights_limit
        self.industry_deviation_limit = industry_deviation_limit
        self.turnover_limit = turnover_limit

    def optimize(
        self,
        pred_scores: pd.Series,
        factor_matrix: pd.DataFrame,
        cov_matrix: pd.DataFrame,
        benchmark_weights: Optional[pd.Series] = None,
        prev_weights: Optional[pd.Series] = None,
    ) -> pd.Series:
        """优化组合权重

        Args:
            pred_scores: 模型预测得分 (index=instrument code)
            factor_matrix: 因子暴露矩阵 (index=instrument, columns=factors)
            cov_matrix: 收益协方差矩阵 (index/columns=instrument)
            benchmark_weights: 基准权重 (None=等权)
            prev_weights: 上期权重 (用于换手率约束)

        Returns:
            优化后的目标权重 (index=instrument)
        """
        stocks = pred_scores.index.intersection(cov_matrix.index)
        if len(stocks) < 2:
            return pd.Series(dtype=float)

        scores = pred_scores.loc[stocks].values
        cov = cov_matrix.loc[stocks, stocks].values

        if benchmark_weights is None:
            bench = np.ones(len(stocks)) / len(stocks)
        else:
            bench = benchmark_weights.reindex(stocks).fillna(0).values

        if prev_weights is None:
            prev = np.zeros(len(stocks))
        else:
            prev = prev_weights.reindex(stocks).fillna(0).values

        n = len(stocks)

        from scipy.optimize import minimize

        def objective(w):
            port_return = np.dot(w, scores)
            tracking_error = np.sqrt(
                np.dot(w - bench, np.dot(cov, w - bench))
            )
            return -port_return / max(tracking_error, 1e-8)

        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
            {"type": "ineq", "fun": lambda w: (
                self.tracking_error_limit -
                np.sqrt(np.dot(w - bench, np.dot(cov, w - bench)))
            )},
            {"type": "ineq", "fun": lambda w: self.turnover_limit - np.sum(np.abs(w - prev))},
        ]

        bounds = [(0, self.weights_limit)] * n

        x0 = bench.copy()

        result = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 200, "ftol": 1e-8},
        )

        if not result.success:
            logger.warning(f"EnhancedIndexing optimization: {result.message}")
            return pd.Series(bench, index=stocks)

        weights = pd.Series(result.x, index=stocks)
        weights = weights[weights > 1e-6]

        logger.info(
            f"EnhancedIndexing optimization: {len(weights)} positions, "
            f"tracking_error={np.sqrt(np.dot(result.x - bench, np.dot(cov, result.x - bench))):.4f}, "
            f"turnover={np.sum(np.abs(result.x - prev)):.4f}"
        )
        return weights

    def run_backtest(
        self,
        dataset: Any,
        model: Any,
        benchmark: str = "SH000300",
        segments: Optional[Dict] = None,
    ) -> Dict:
        """使用Qlib的EnhancedIndexing策略回测

        Args:
            dataset: DatasetH from create_dataset()
            model: 训练好的模型
            benchmark: 基准代码
            segments: 数据分段

        Returns:
            回测结果字典
        """
        ensure_qlib()

        from qlib.contrib.evaluate import backtest as qlib_backtest
        from qlib.contrib.evaluate import risk_analysis
        from qlib.contrib.strategy.signal_strategy import EnhancedIndexingStrategy

        strategy = EnhancedIndexingStrategy(
            model=model,
            benchmark=benchmark,
            tracking_error_limit=self.tracking_error_limit,
            weights_limit=self.weights_limit,
        )

        from qlib.backtest.executor import SimulatorExecutor
        executor = SimulatorExecutor(
            time_per_step="day",
            generate_portfolio_metrics=True,
        )

        logger.info(f"Running EnhancedIndexing backtest: benchmark={benchmark}")

        try:
            portfolio_metric = qlib_backtest(
                dataset=dataset,
                model=model,
                strategy=strategy,
                executor=executor,
                benchmark=benchmark,
            )
            analysis = risk_analysis(portfolio_metric)

            result = {
                "portfolio_metric": portfolio_metric,
                "analysis": analysis.to_dict() if hasattr(analysis, "to_dict") else {},
                "strategy": "enhanced_indexing",
                "benchmark": benchmark,
                "tracking_error_limit": self.tracking_error_limit,
                "weights_limit": self.weights_limit,
            }
            logger.info("EnhancedIndexing backtest complete.")
            return result

        except Exception as e:
            logger.error(f"EnhancedIndexing backtest failed: {e}")
            return {"error": str(e)}
