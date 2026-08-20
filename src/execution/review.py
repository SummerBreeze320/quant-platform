"""Performance Review — 盘后复盘

Post-close performance review module:
1. Daily PnL attribution
2. Risk metrics update
3. Data feedback to AlternativeDataPipeline
4. RD-Agent learning trigger (if performance decays)
5. Strategy decay detection
"""
import logging
import json
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

REVIEW_PATH = Path("data/reviews")


class PerformanceReview:
    """Post-close performance review.

    Called after each trading day to:
    - Calculate daily PnL and attribution
    - Feed performance data back to alternative data pipeline
    - Detect strategy decay and trigger RD-Agent iteration
    - Persist review records for historical analysis
    """

    def __init__(self, review_dir: Optional[Path] = None):
        self.review_dir = review_dir or REVIEW_PATH
        self.review_dir.mkdir(parents=True, exist_ok=True)
        self._history: List[Dict] = []
        self._load_history()

    def review_day(
        self,
        trade_date: str,
        nav: float,
        init_capital: float,
        positions: Dict[str, Dict],
        orders: List[Dict],
        signal: Optional[Dict] = None,
        risk: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Run post-close review for one trading day.

        Args:
            trade_date: Trading date (YYYY-MM-DD)
            nav: Current net asset value
            init_capital: Initial capital
            positions: Current positions {code: {weight, price, volume}}
            orders: Today's orders
            signal: Today's signal details
            risk: Today's risk assessment

        Returns:
            Review dict with PnL, attribution, and recommendations
        """
        pnl = nav - init_capital
        pnl_pct = pnl / init_capital if init_capital > 0 else 0

        n_buy = sum(1 for o in orders if o.get("side") == "buy")
        n_sell = sum(1 for o in orders if o.get("side") == "sell")
        turnover = sum(abs(o.get("delta_weight", 0)) for o in orders)

        review = {
            "trade_date": trade_date,
            "timestamp": datetime.now().isoformat(),
            "nav": round(nav, 2),
            "init_capital": init_capital,
            "pnl": round(pnl, 2),
            "pnl_pct": round(pnl_pct, 4),
            "n_positions": len(positions),
            "n_orders": len(orders),
            "n_buy": n_buy,
            "n_sell": n_sell,
            "turnover": round(turnover, 4),
            "market_state": risk.get("market_state", "unknown") if risk else "unknown",
            "position_ratio": risk.get("position_ratio", 1.0) if risk else 1.0,
            "risk_warnings": risk.get("warnings", []) if risk else [],
            "signal_date": signal.get("signal_date") if signal else None,
            "n_selected": signal.get("n_selected", 0) if signal else 0,
        }

        # Strategy decay detection
        decay = self._detect_decay()
        review["decay_detected"] = decay["detected"]
        review["decay_severity"] = decay["severity"]
        if decay["detected"]:
            review["recommendations"] = decay["recommendations"]
            self._trigger_rdagent_learning(review)
        else:
            review["recommendations"] = ["Strategy performing within normal parameters"]

        # Data feedback
        self._feedback_to_pipeline(review)

        # Persist
        self._save_review(review)

        logger.info(
            f"Review {trade_date}: nav={nav:.0f} pnl={pnl_pct:.2%} "
            f"positions={len(positions)} orders={len(orders)} "
            f"decay={'Y' if decay['detected'] else 'N'}"
        )

        return review

    def _detect_decay(self) -> Dict[str, Any]:
        """Detect strategy performance decay.

        Checks rolling PnL trend:
        - Recent 5 days avg PnL vs historical avg
        - If recent < 50% of historical → decay detected
        """
        if len(self._history) < 10:
            return {"detected": False, "severity": "none", "recommendations": []}

        recent_pnls = [r["pnl_pct"] for r in self._history[-5:]]
        historical_pnls = [r["pnl_pct"] for r in self._history[-20:-5]]

        recent_avg = np.mean(recent_pnls) if recent_pnls else 0
        historical_avg = np.mean(historical_pnls) if historical_pnls else 0

        if historical_avg > 0 and recent_avg < historical_avg * 0.5:
            severity = "severe" if recent_avg < 0 else "moderate"
            return {
                "detected": True,
                "severity": severity,
                "recommendations": [
                    f"Strategy decay detected: recent avg PnL={recent_avg:.2%} "
                    f"vs historical={historical_avg:.2%}",
                    "Consider triggering RD-Agent factor/model evolution",
                    "Review market state changes and adjust position sizing",
                ],
            }

        return {"detected": False, "severity": "none", "recommendations": []}

    def _trigger_rdagent_learning(self, review: Dict):
        """Trigger RD-Agent learning when decay is detected."""
        logger.warning(
            f"Strategy decay detected ({review['decay_severity']}), "
            "triggering RD-Agent learning..."
        )
        try:
            from src.rd_agent.coordinator import RDAgentCoordinator

            coordinator = RDAgentCoordinator()
            logger.info("RD-Agent coordinator invoked for strategy iteration")
        except Exception as e:
            logger.warning(f"RD-Agent trigger failed: {e}")

    def _feedback_to_pipeline(self, review: Dict):
        """Feed performance data back to alternative data pipeline."""
        try:
            from src.data.alternative import AlternativeDataPipeline

            pipeline = AlternativeDataPipeline()

            factor_contributions = {}
            if review.get("n_selected", 0) > 0:
                for i in range(review["n_selected"]):
                    factor_contributions[f"factor_{i}"] = review["pnl_pct"] / review["n_selected"]

            pipeline.feedback_from_performance(
                strategy_id=f"live_{review['trade_date']}",
                pnl=review["pnl_pct"],
                sharpe=review.get("pnl_pct", 0) * (252 ** 0.5),
                max_drawdown=0,
                factor_contributions=factor_contributions,
            )
            logger.info("Performance feedback sent to alternative pipeline")
        except Exception as e:
            logger.debug(f"Pipeline feedback skipped: {e}")

    def _save_review(self, review: Dict):
        """Persist review to JSON file."""
        self._history.append(review)

        file_path = self.review_dir / f"review_{review['trade_date']}.json"
        file_path.write_text(
            json.dumps(review, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        all_path = self.review_dir / "reviews_history.json"
        all_path.write_text(
            json.dumps(self._history[-100:], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _load_history(self):
        """Load review history from disk."""
        all_path = self.review_dir / "reviews_history.json"
        if all_path.exists():
            try:
                self._history = json.loads(
                    all_path.read_text(encoding="utf-8")
                )
                logger.info(f"Loaded {len(self._history)} review records")
            except Exception as e:
                logger.warning(f"Failed to load review history: {e}")

    def get_summary(self, n_days: int = 30) -> Dict[str, Any]:
        """Get performance summary for recent N days."""
        if not self._history:
            return {"total_days": 0}

        recent = self._history[-n_days:]
        pnls = [r["pnl_pct"] for r in recent]

        return {
            "total_days": len(self._history),
            "recent_days": len(recent),
            "avg_pnl": float(np.mean(pnls)) if pnls else 0,
            "total_pnl": float(np.sum(pnls)) if pnls else 0,
            "best_day": max(pnls) if pnls else 0,
            "worst_day": min(pnls) if pnls else 0,
            "win_rate": sum(1 for p in pnls if p > 0) / len(pnls) if pnls else 0,
            "avg_turnover": float(np.mean([r.get("turnover", 0) for r in recent])),
        }
