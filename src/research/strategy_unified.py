"""Unified strategy layer bridging custom strategies and Qlib's strategy framework.

Provides a common BaseStrategy interface that can wrap both our custom
ETF/stock strategies and Qlib's TopkDropout/EnhancedIndexing strategies.
"""
import logging
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, Union
import numpy as np
import pandas as pd

from src.core import ensure_qlib, get_ohlcv, list_instruments

logger = logging.getLogger(__name__)


class BaseStrategy(ABC):
    """Unified strategy interface.

    A strategy takes market data and produces trade decisions (signals,
    weights, or orders). This base class standardizes the interface so
    strategies can be used interchangeably in backtest and live trading.
    """

    def __init__(self, name: str = "base"):
        self.name = name
        self._fitted = False

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate trading signals from market data.

        Args:
            data: OHLCV DataFrame indexed by datetime.

        Returns:
            pd.Series of signals: 1=long, -1=short, 0=flat.
        """
        ...

    def generate_weights(self, data: pd.DataFrame) -> pd.Series:
        """Generate portfolio weights from market data.

        Default implementation converts signals to equal-weight positions.
        Override for custom weighting schemes.

        Returns:
            pd.Series of weights summing to 1.0 (for long-only).
        """
        signals = self.generate_signals(data)
        long_count = (signals > 0).sum()
        if long_count == 0:
            return pd.Series(dtype=float)
        weights = signals[signals > 0] / long_count
        return weights

    def fit(self, data: pd.DataFrame) -> None:
        """Optional training step (e.g. for ML-based strategies)."""
        self._fitted = True

    @property
    def fitted(self) -> bool:
        return self._fitted


class TopkDropoutStrategy(BaseStrategy):
    """Top-K dropout strategy: select top-K stocks by prediction score, drop N each rebalance.

    Wraps Qlib's TopkDropoutStrategy for use with model predictions.
    """

    def __init__(self, topk: int = 50, n_drop: int = 5, name: str = "topk_dropout"):
        super().__init__(name=name)
        self.topk = topk
        self.n_drop = n_drop
        self._model = None
        self._scores = None

    def set_predictions(self, scores: pd.Series):
        """Set pre-computed prediction scores from a trained model."""
        self._scores = scores

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        if self._scores is None:
            raise RuntimeError("Call set_predictions() before generate_signals()")
        ranked = self._scores.groupby(level="datetime").rank(ascending=False, pct=True)
        signals = (ranked <= self.topk / len(ranked)).astype(float)
        signals.name = "signal"
        return signals


class EnhancedIndexingStrategy(BaseStrategy):
    """Enhanced indexing: track benchmark with active factor tilts.

    Uses portfolio optimization to maintain benchmark tracking while
    tilting towards high-alpha stocks.
    """

    def __init__(
        self,
        benchmark: str = "SH000300",
        tracking_error_limit: float = 0.05,
        name: str = "enhanced_indexing",
    ):
        super().__init__(name=name)
        self.benchmark = benchmark
        self.tracking_error_limit = tracking_error_limit
        self._scores = None

    def set_predictions(self, scores: pd.Series):
        self._scores = scores

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        if self._scores is None:
            raise RuntimeError("Call set_predictions() before generate_signals()")
        signals = self._scores.copy()
        signals[signals > 0] = 1.0
        signals[signals <= 0] = 0.0
        return signals


class ETFRotationStrategy(BaseStrategy):
    """ETF rotation: rotate among ETFs based on momentum/trend signals.

    Adapts our existing ETF strategies to the unified interface.
    """

    def __init__(
        self,
        etf_codes: List[str],
        lookback: int = 20,
        top_k: int = 3,
        name: str = "etf_rotation",
    ):
        super().__init__(name=name)
        self.etf_codes = etf_codes
        self.lookback = lookback
        self.top_k = top_k

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        returns = data["close"].pct_change(self.lookback)
        signals = pd.Series(0, index=data.index)
        if len(returns) > 0:
            top = returns.nlargest(min(self.top_k, len(returns)))
            signals[top.index] = 1.0
        return signals

    def generate_weights(self, data: pd.DataFrame) -> pd.Series:
        returns = data["close"].pct_change(self.lookback)
        valid = returns.dropna()
        if len(valid) == 0:
            return pd.Series(dtype=float)
        top = valid.nlargest(min(self.top_k, len(valid)))
        weights = top / top.sum()
        return weights


class ETFSignalAdapter(BaseStrategy):
    """Adapter to wrap Qlib-native ETF strategies into unified interface.

    Uses Qlib expression engine for signal generation instead of
    the old custom ETFStrategyBase classes.
    """

    def __init__(self, etf_codes: List[str], lookback: int = 20, name: str = "etf_adapter"):
        super().__init__(name=name)
        self._etf_codes = etf_codes
        self._lookback = lookback
        self._etf_strategy = ETFRotationStrategy(
            etf_codes=etf_codes, lookback=lookback, top_k=3,
        )

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        return self._etf_strategy.generate_signals(data)


class FactorSignalAdapter(BaseStrategy):
    """Adapter to wrap Qlib expression factors into unified strategy interface.

    Uses the factor_library's QLIB_EXPRESSION_FACTORS for signal generation.
    """

    def __init__(
        self,
        factor_name: str = "rocp_20",
        threshold: float = 0.0,
        name: str = "factor_adapter",
    ):
        super().__init__(name=name)
        self._factor_name = factor_name
        self.threshold = threshold

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        from src.research.factor_library import QLIB_EXPRESSION_FACTORS
        expr = QLIB_EXPRESSION_FACTORS.get(self._factor_name, "Ref($close, 5) / $close - 1")
        # For single-instrument data, compute factor directly
        if "close" in data.columns:
            close = data["close"]
            if expr == "Ref($close, 5) / $close - 1":
                values = close.shift(5) / close - 1
            elif expr == "Mean($close, 20) / $close - 1":
                values = close.rolling(20).mean() / close - 1
            elif expr == "Std($close, 20) / $close":
                values = close.rolling(20).std() / close
            else:
                values = close.pct_change(20)
        else:
            values = pd.Series(0, index=data.index)

        signals = pd.Series(0, index=data.index)
        signals[values > self.threshold] = 1.0
        signals[values < -self.threshold] = -1.0
        return signals


def create_strategy(
    strategy_type: str = "topk_dropout",
    **kwargs,
) -> BaseStrategy:
    """Factory function to create a strategy by type.

    Args:
        strategy_type: "topk_dropout", "enhanced_indexing", "etf_rotation",
                       "etf_adapter", or "factor_adapter".
    """
    strategies = {
        "topk_dropout": TopkDropoutStrategy,
        "enhanced_indexing": EnhancedIndexingStrategy,
        "etf_rotation": ETFRotationStrategy,
    }

    if strategy_type in strategies:
        return strategies[strategy_type](**kwargs)
    elif strategy_type == "etf_adapter":
        return ETFSignalAdapter(**kwargs)
    elif strategy_type == "factor_adapter":
        return FactorSignalAdapter(**kwargs)
    else:
        raise ValueError(f"Unknown strategy type: {strategy_type}")
