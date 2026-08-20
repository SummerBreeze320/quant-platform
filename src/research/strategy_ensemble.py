"""Strategy Ensemble — 多策略集成池

Dynamically switches sub-strategies based on market state.
Allocates weights across multiple approved strategies.
"""
import logging
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from src.research.strategy_registry import StrategyRegistry, StrategyEntry
from src.agents.contracts import MarketState

logger = logging.getLogger(__name__)


@dataclass
class EnsembleAllocation:
    """Allocation result for a single rebalance."""
    timestamp: str
    market_state: str
    strategy_weights: Dict[str, float]
    total_positions: int
    signal: str


class StrategyEnsemble:
    """Multi-strategy ensemble pool.

    Features:
    - Dynamic sub-strategy switching by market state
    - Weight allocation based on strategy scores
    - Fallback to best overall strategy when no market-specific match
    - Tracking of allocation history
    """

    DEFAULT_WEIGHTS = {
        MarketState.BULL: {"score_weighted": 1.0, "fallback": 1.0},
        MarketState.BEAR: {"score_weighted": 0.3, "fallback": 1.0},
        MarketState.SIDEWAYS: {"score_weighted": 0.6, "fallback": 1.0},
        MarketState.HIGH_VOL: {"score_weighted": 0.2, "fallback": 1.0},
        MarketState.LOW_VOL: {"score_weighted": 0.8, "fallback": 1.0},
    }

    def __init__(self, registry: Optional[StrategyRegistry] = None):
        self.registry = registry or StrategyRegistry()
        self.allocation_history: List[EnsembleAllocation] = []

    def select_strategies(self, market_state: MarketState) -> List[StrategyEntry]:
        """Select candidate strategies for current market state."""
        candidates = self.registry.list_by_market_state(market_state.value)

        if not candidates:
            candidates = self.registry.list_approved()
            if not candidates:
                logger.warning("No approved strategies available")
                return []
            logger.info(
                f"No market-specific strategies for {market_state.value}, "
                f"using {len(candidates)} general strategies"
            )

        top_n = min(5, len(candidates))
        return sorted(candidates, key=lambda s: s.score, reverse=True)[:top_n]

    def allocate_weights(
        self, strategies: List[StrategyEntry], market_state: MarketState
    ) -> Dict[str, float]:
        """Allocate weights across strategies based on scores.

        Uses softmax-weighted allocation with market-state adjustment.
        """
        if not strategies:
            return {}

        if len(strategies) == 1:
            return {strategies[0].strategy_id: 1.0}

        scores = np.array([s.score for s in strategies])
        scores = np.maximum(scores, 0.01)

        exp_scores = np.exp(scores / max(np.max(scores), 0.01) * 3.0)
        weights = exp_scores / exp_scores.sum()

        alpha = self.DEFAULT_WEIGHTS.get(market_state, {}).get("score_weighted", 0.5)
        equal_w = np.ones(len(strategies)) / len(strategies)
        final_w = alpha * weights + (1 - alpha) * equal_w

        final_w = final_w / final_w.sum()

        return {
            s.strategy_id: round(float(w), 4)
            for s, w in zip(strategies, final_w)
        }

    def rebalance(
        self, market_state: MarketState
    ) -> EnsembleAllocation:
        """Rebalance ensemble for current market state.

        Args:
            market_state: Current market environment

        Returns:
            EnsembleAllocation with strategy weights
        """
        strategies = self.select_strategies(market_state)
        weights = self.allocate_weights(strategies, market_state)

        allocation = EnsembleAllocation(
            timestamp=datetime.now().isoformat(),
            market_state=market_state.value,
            strategy_weights=weights,
            total_positions=len(strategies),
            signal=self._generate_signal(market_state, strategies),
        )

        self.allocation_history.append(allocation)
        logger.info(
            f"Ensemble rebalanced for {market_state.value}: "
            f"{len(strategies)} strategies, "
            f"weights={list(weights.values())[:3]}..."
        )
        return allocation

    def _generate_signal(
        self, market_state: MarketState, strategies: List[StrategyEntry]
    ) -> str:
        """Generate a trading signal based on market state and strategies."""
        if not strategies:
            return "hold"

        if market_state == MarketState.BULL:
            return "overweight"
        elif market_state == MarketState.BEAR:
            return "underweight"
        elif market_state == MarketState.HIGH_VOL:
            return "reduce"
        elif market_state == MarketState.LOW_VOL:
            return "normal"
        else:
            return "normal"

    def get_current_positions(self) -> Dict[str, float]:
        """Get current strategy weight allocation (from last rebalance)."""
        if not self.allocation_history:
            return {}
        return self.allocation_history[-1].strategy_weights

    def performance_summary(self) -> Dict[str, Any]:
        """Get ensemble performance summary."""
        if not self.allocation_history:
            return {
                "total_rebalances": 0,
                "strategies_used": 0,
            }

        all_weights = []
        for a in self.allocation_history:
            all_weights.extend(a.strategy_weights.values())

        return {
            "total_rebalances": len(self.allocation_history),
            "strategies_used": len(
                set(
                    sid
                    for a in self.allocation_history
                    for sid in a.strategy_weights
                )
            ),
            "avg_strategies_per_rebalance": round(
                np.mean([a.total_positions for a in self.allocation_history]), 1
            ),
            "market_states_seen": list(
                set(a.market_state for a in self.allocation_history)
            ),
        }
