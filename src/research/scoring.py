"""Scoring Function — 综合评分模块

Computes a composite score for strategy evaluation and comparison.
Configurable weights for different metrics.
"""
import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass

import numpy as np

from src.agents.contracts import BacktestMetrics, InvestmentObjective

logger = logging.getLogger(__name__)


@dataclass
class ScoringWeights:
    """Weights for composite scoring formula."""
    annual_return: float = 0.4
    sharpe: float = 0.3
    drawdown: float = 0.2
    turnover: float = 0.1

    @classmethod
    def from_objective(cls, obj: InvestmentObjective) -> "ScoringWeights":
        """Adjust weights based on investment objective."""
        if obj.goal == "max_return":
            return cls(annual_return=0.5, sharpe=0.2, drawdown=0.2, turnover=0.1)
        elif obj.goal == "risk_adjusted":
            return cls(annual_return=0.2, sharpe=0.5, drawdown=0.2, turnover=0.1)
        elif obj.goal == "low_turnover":
            return cls(annual_return=0.3, sharpe=0.3, drawdown=0.2, turnover=0.2)
        return cls()


class ScoringFunction:
    """Composite scoring function for strategy evaluation.

    Score = w1*annual_return + w2*sharpe - w3*|max_drawdown| - w4*turnover

    Additional penalties:
    - Constraint violations (drawdown > limit, turnover > limit)
    - Low win rate
    - Poor IC stability
    """

    def __init__(self, weights: Optional[ScoringWeights] = None):
        self.weights = weights or ScoringWeights()

    def compute(
        self,
        metrics: BacktestMetrics,
        objective: Optional[InvestmentObjective] = None,
    ) -> float:
        """Compute composite score.

        Args:
            metrics: Backtest metrics
            objective: Investment objective for constraint checking

        Returns:
            Composite score in [-1, 1] range (higher is better)
        """
        if objective:
            self.weights = ScoringWeights.from_objective(objective)

        ar = metrics.annual_return or 0
        sr = metrics.sharpe_ratio or 0
        dd = abs(metrics.max_drawdown or 0)
        to = metrics.turnover or 0

        base_score = (
            self.weights.annual_return * ar
            + self.weights.sharpe * sr
            - self.weights.drawdown * dd
            - self.weights.turnover * to
        )

        penalty = 0.0

        if objective and objective.constraints:
            max_dd = objective.constraints.get("max_drawdown", 0.20)
            max_to = objective.constraints.get("max_turnover", 0.50)

            if dd > max_dd:
                penalty += (dd - max_dd) * 2.0

            if to > max_to:
                penalty += (to - max_to) * 1.5

        if metrics.win_rate and metrics.win_rate < 0.4:
            penalty += (0.4 - metrics.win_rate) * 0.5

        if metrics.ic_mean and abs(metrics.ic_mean) < 0.01:
            penalty += 0.05

        score = base_score - penalty
        score = float(np.clip(score, -1.0, 1.0))

        logger.debug(
            f"Score: base={base_score:.4f} penalty={penalty:.4f} → {score:.4f} "
            f"(ar={ar:.4f} sr={sr:.4f} dd={dd:.4f} to={to:.4f})"
        )
        return round(score, 4)

    def rank(
        self,
        results: Dict[str, BacktestMetrics],
        objective: Optional[InvestmentObjective] = None,
    ) -> list:
        """Rank multiple experiment results by score.

        Args:
            results: Dict of experiment_id -> metrics
            objective: Investment objective

        Returns:
            List of (experiment_id, score) sorted by score descending
        """
        scored = [
            (eid, self.compute(m, objective))
            for eid, m in results.items()
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def explain(
        self,
        metrics: BacktestMetrics,
        objective: Optional[InvestmentObjective] = None,
    ) -> Dict[str, Any]:
        """Explain score breakdown for debugging."""
        if objective:
            self.weights = ScoringWeights.from_objective(objective)

        ar = metrics.annual_return or 0
        sr = metrics.sharpe_ratio or 0
        dd = abs(metrics.max_drawdown or 0)
        to = metrics.turnover or 0

        components = {
            "annual_return": round(self.weights.annual_return * ar, 4),
            "sharpe": round(self.weights.sharpe * sr, 4),
            "drawdown_penalty": round(-self.weights.drawdown * dd, 4),
            "turnover_penalty": round(-self.weights.turnover * to, 4),
        }

        penalty = 0.0
        if objective and objective.constraints:
            max_dd = objective.constraints.get("max_drawdown", 0.20)
            max_to = objective.constraints.get("max_turnover", 0.50)
            if dd > max_dd:
                penalty += (dd - max_dd) * 2.0
            if to > max_to:
                penalty += (to - max_to) * 1.5

        total = sum(components.values()) - penalty
        return {
            "components": components,
            "constraint_penalty": round(penalty, 4),
            "total_score": round(float(np.clip(total, -1.0, 1.0)), 4),
            "weights": {
                "annual_return": self.weights.annual_return,
                "sharpe": self.weights.sharpe,
                "drawdown": self.weights.drawdown,
                "turnover": self.weights.turnover,
            },
        }
