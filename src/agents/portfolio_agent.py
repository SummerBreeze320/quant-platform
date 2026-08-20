"""PortfolioAgent — 组合空间搜索

在组合空间中搜索最优持仓配置，输出目标持仓方案。
使用真实Qlib市场数据计算协方差矩阵和组合权重。

包装现有模块:
- src.research.enhanced_optimizer (EnhancedIndexingOptimizer)
- src.core (get_ohlcv, list_instruments)
"""
import logging
from typing import Optional
import numpy as np
import pandas as pd
from .base import BaseAgent
from .contracts import PortfolioSearchRequest, PortfolioConfig

logger = logging.getLogger(__name__)


class PortfolioAgent(BaseAgent):
    """组合空间搜索Agent

    搜索空间:
    - 持仓数量: 5/10/20/50只
    - 权重方案: 等权/评分加权/增强指数/最小方差
    - 调仓频率: 日/周/月

    输入: PortfolioSearchRequest
    输出: PortfolioConfig (最优持仓配置 + 真实权重)
    """

    def __init__(self, config: Optional[dict] = None):
        super().__init__(name="PortfolioAgent", config=config)
        self._returns_cache: Optional[pd.DataFrame] = None

    def run(self, input_data: PortfolioSearchRequest) -> PortfolioConfig:
        """执行组合空间搜索"""
        from src.core import ensure_qlib, list_instruments

        ensure_qlib()
        instruments = list_instruments(as_list=True)

        returns_df = self._load_returns(
            instruments, input_data.start_date, input_data.end_date
        )

        if returns_df is None or returns_df.empty or returns_df.shape[1] < 3:
            logger.warning("Insufficient returns data, using default config")
            return PortfolioConfig(
                n_holdings=10,
                weight_scheme="enhanced_indexing",
                rebalance_freq="W",
                max_position=input_data.max_position,
                tracking_error_limit=input_data.tracking_error_limit,
            )

        best_config = None
        best_score = -1.0

        for n in input_data.n_holdings_range:
            if n > returns_df.shape[1]:
                continue
            for scheme in input_data.weight_schemes:
                for freq in input_data.rebalance_freq:
                    score = self._evaluate_config_real(
                        n, scheme, freq, returns_df, input_data
                    )
                    if score > best_score:
                        best_score = score
                        best_config = PortfolioConfig(
                            n_holdings=n,
                            weight_scheme=scheme,
                            rebalance_freq=freq,
                            max_position=input_data.max_position,
                            tracking_error_limit=input_data.tracking_error_limit,
                        )

        if best_config is None:
            best_config = PortfolioConfig(
                n_holdings=min(10, returns_df.shape[1]),
                weight_scheme="enhanced_indexing",
                rebalance_freq="W",
                max_position=input_data.max_position,
                tracking_error_limit=input_data.tracking_error_limit,
            )

        weights = self._compute_weights(best_config, returns_df, input_data)
        best_config.weights = weights

        logger.info(
            f"PortfolioAgent selected: {best_config.n_holdings} holdings, "
            f"{best_config.weight_scheme}, {best_config.rebalance_freq} "
            f"(score={best_score:.4f}, {len(weights)} weights)"
        )
        return best_config

    def _load_returns(
        self, instruments: list, start_date: Optional[str], end_date: Optional[str]
    ) -> Optional[pd.DataFrame]:
        """加载真实收益率数据"""
        from src.core import get_ohlcv

        all_returns = {}
        for code in instruments:
            try:
                df = get_ohlcv(code, start_time=start_date, end_time=end_date)
                if df is not None and len(df) > 20:
                    rets = df["close"].pct_change().dropna()
                    if len(rets) > 10:
                        all_returns[code] = rets
            except Exception:
                continue

        if not all_returns:
            return None

        df = pd.DataFrame(all_returns)
        df = df.dropna(how="all")
        df = df.fillna(0)
        self._returns_cache = df
        logger.info(f"Loaded returns: {df.shape[0]} days x {df.shape[1]} instruments")
        return df

    def _evaluate_config_real(
        self,
        n_holdings: int,
        scheme: str,
        freq: str,
        returns_df: pd.DataFrame,
        request: PortfolioSearchRequest,
    ) -> float:
        """使用真实市场数据评估组合配置"""
        cols = list(returns_df.columns[:n_holdings])
        sub_returns = returns_df[cols]
        cov = sub_returns.cov().values

        if scheme == "min_variance":
            port_var = self._min_variance_score(cov)
            score = 1.0 / (1.0 + port_var * 100)
        elif scheme == "enhanced_indexing":
            score = self._diversification_ratio(cov, n_holdings)
        elif scheme == "equal_weight":
            score = self._diversification_ratio(cov, n_holdings) * 0.9
        else:
            score = self._diversification_ratio(cov, n_holdings) * 0.85

        if n_holdings == 10:
            score += 0.08
        elif n_holdings == 20:
            score += 0.06
        elif n_holdings == 5:
            score += 0.04

        if freq == "W":
            score += 0.05
        elif freq == "M":
            score += 0.03

        if request.tracking_error_limit > 0:
            score += 0.02

        return score

    @staticmethod
    def _min_variance_score(cov: np.ndarray) -> float:
        """计算最小方差组合的方差"""
        n = cov.shape[0]
        try:
            inv = np.linalg.pinv(cov)
            ones = np.ones(n)
            w = inv @ ones / (ones @ inv @ ones)
            return float(w @ cov @ w)
        except Exception:
            return float(np.mean(np.diag(cov)))

    @staticmethod
    def _diversification_ratio(cov: np.ndarray, n: int) -> float:
        """计算分散化比率 (越高越好)"""
        avg_vol = float(np.mean(np.sqrt(np.diag(cov))))
        if avg_vol <= 0:
            return 0.5
        port_var = float(np.sum(cov)) / (n * n)
        port_vol = np.sqrt(max(port_var, 0))
        if port_vol <= 0:
            return 0.5
        return min(avg_vol / port_vol / n, 1.0)

    def _compute_weights(
        self,
        config: PortfolioConfig,
        returns_df: pd.DataFrame,
        request: PortfolioSearchRequest,
    ) -> dict:
        """为最优配置计算真实权重"""
        n = config.n_holdings
        cols = list(returns_df.columns[:n])

        if config.weight_scheme == "equal_weight":
            w = 1.0 / n
            return {c: round(w, 4) for c in cols}

        if config.weight_scheme == "min_variance":
            return self._compute_min_variance_weights(returns_df[cols])

        if config.weight_scheme == "enhanced_indexing":
            return self._compute_enhanced_indexing_weights(
                returns_df[cols], config, request
            )

        if config.weight_scheme == "score_weighted":
            return self._compute_score_weighted_weights(cols, request)

        w = 1.0 / n
        return {c: round(w, 4) for c in cols}

    @staticmethod
    def _compute_min_variance_weights(returns_sub: pd.DataFrame) -> dict:
        """使用scipy计算最小方差权重"""
        from scipy.optimize import minimize

        cov = returns_sub.cov().values
        n = cov.shape[0]

        def portfolio_variance(w):
            return w @ cov @ w

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [(0.0, 0.15)] * n
        x0 = np.ones(n) / n

        try:
            result = minimize(
                portfolio_variance, x0, method="SLSQP",
                bounds=bounds, constraints=constraints,
            )
            if result.success:
                weights = result.x
            else:
                weights = x0
        except Exception:
            weights = x0

        cols = list(returns_sub.columns)
        return {c: round(float(w), 4) for c, w in zip(cols, weights) if w > 0.001}

    @staticmethod
    def _compute_enhanced_indexing_weights(
        returns_sub: pd.DataFrame, config: PortfolioConfig, request: PortfolioSearchRequest
    ) -> dict:
        """使用EnhancedIndexingOptimizer计算权重"""
        try:
            from src.research.enhanced_optimizer import EnhancedIndexingOptimizer

            optimizer = EnhancedIndexingOptimizer(
                benchmark=request.benchmark,
                tracking_error_limit=config.tracking_error_limit,
                weights_limit=config.max_position,
            )
            cov_matrix = returns_sub.cov()
            pred_scores = pd.Series(1.0, index=returns_sub.columns)
            weights = optimizer.optimize(
                pred_scores=pred_scores,
                factor_matrix=cov_matrix,
                cov_matrix=cov_matrix,
            )
            if weights is not None and len(weights) > 0:
                return {c: round(float(w), 4) for c, w in weights.items() if w > 0.001}
        except Exception as e:
            logger.warning(f"EnhancedIndexingOptimizer failed: {e}")

        n = len(returns_sub.columns)
        w = 1.0 / n
        return {c: round(w, 4) for c in returns_sub.columns}

    @staticmethod
    def _compute_score_weighted_weights(cols: list, request: PortfolioSearchRequest) -> dict:
        """使用因子IC分数作为权重"""
        if request.factor_set and request.factor_set.ic_scores:
            ic_scores = request.factor_set.ic_scores
            total_ic = sum(abs(v) for v in ic_scores.values())
            if total_ic > 0:
                weights = {}
                for c in cols:
                    score = abs(ic_scores.get(c, 0.01))
                    weights[c] = round(score / total_ic, 4)
                total_w = sum(weights.values())
                if total_w > 0:
                    return {c: w / total_w for c, w in weights.items()}

        w = 1.0 / len(cols)
        return {c: round(w, 4) for c in cols}
