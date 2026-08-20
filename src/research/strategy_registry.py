"""Strategy Registry — 策略注册表

Manages strategy lifecycle: registration, versioning, market-state tagging.
Persists to JSON file for cross-session reuse.
"""
import logging
import json
from typing import List, Dict, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

REGISTRY_PATH = Path("data/strategy_registry.json")


@dataclass
class StrategyEntry:
    """One registered strategy."""
    strategy_id: str
    name: str
    version: str
    factor_set: List[str]
    model_type: str
    model_params: Dict
    n_holdings: int
    weight_scheme: str
    rebalance_freq: str
    market_states: List[str]
    annual_return: float
    sharpe_ratio: float
    max_drawdown: float
    score: float
    approved: bool
    overfitting_score: float
    created_at: str
    status: str = "active"
    metadata: Dict = field(default_factory=dict)


class StrategyRegistry:
    """Strategy registry with JSON persistence.

    Supports:
    - register: add new strategy with metrics + review
    - get: retrieve by ID
    - list_approved: all approved strategies
    - list_by_market_state: filter by market environment
    - deactivate: soft-delete a strategy
    - version_bump: create new version of existing strategy
    """

    def __init__(self, registry_path: Optional[Path] = None):
        self.path = registry_path or REGISTRY_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._strategies: Dict[str, StrategyEntry] = {}
        self._load()

    def _load(self):
        """Load registry from JSON file."""
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                for sid, sdata in data.items():
                    self._strategies[sid] = StrategyEntry(**sdata)
                logger.info(f"Loaded {len(self._strategies)} strategies from registry")
            except Exception as e:
                logger.warning(f"Failed to load registry: {e}")

    def _save(self):
        """Persist registry to JSON file."""
        data = {sid: asdict(s) for sid, s in self._strategies.items()}
        self.path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info(f"Saved {len(self._strategies)} strategies to registry")

    def register(
        self,
        name: str,
        factor_set: List[str],
        model_type: str,
        model_params: Dict,
        n_holdings: int,
        weight_scheme: str,
        rebalance_freq: str,
        market_states: List[str],
        annual_return: float,
        sharpe_ratio: float,
        max_drawdown: float,
        score: float,
        approved: bool,
        overfitting_score: float,
        metadata: Optional[Dict] = None,
    ) -> str:
        """Register a new strategy. Returns strategy_id."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        strategy_id = f"strat_{timestamp}"

        existing = [s for s in self._strategies.values() if s.name == name]
        version = f"v{len(existing) + 1}"

        entry = StrategyEntry(
            strategy_id=strategy_id,
            name=name,
            version=version,
            factor_set=factor_set,
            model_type=model_type,
            model_params=model_params,
            n_holdings=n_holdings,
            weight_scheme=weight_scheme,
            rebalance_freq=rebalance_freq,
            market_states=market_states,
            annual_return=round(annual_return, 4),
            sharpe_ratio=round(sharpe_ratio, 4),
            max_drawdown=round(max_drawdown, 4),
            score=round(score, 4),
            approved=approved,
            overfitting_score=round(overfitting_score, 4),
            created_at=datetime.now().isoformat(),
            metadata=metadata or {},
        )

        self._strategies[strategy_id] = entry
        self._save()
        logger.info(
            f"Registered strategy {strategy_id} ({name} {version}): "
            f"score={score:.4f} approved={approved}"
        )
        return strategy_id

    def get(self, strategy_id: str) -> Optional[StrategyEntry]:
        """Retrieve a strategy by ID."""
        return self._strategies.get(strategy_id)

    def list_approved(self) -> List[StrategyEntry]:
        """List all approved and active strategies."""
        return [
            s for s in self._strategies.values()
            if s.approved and s.status == "active"
        ]

    def list_by_market_state(self, market_state: str) -> List[StrategyEntry]:
        """List approved strategies suitable for a market state."""
        return [
            s for s in self._strategies.values()
            if s.approved and s.status == "active"
            and (market_state in s.market_states or "all" in s.market_states)
        ]

    def list_all(self) -> List[StrategyEntry]:
        """List all strategies."""
        return list(self._strategies.values())

    def deactivate(self, strategy_id: str):
        """Soft-delete a strategy."""
        if strategy_id in self._strategies:
            self._strategies[strategy_id].status = "inactive"
            self._save()
            logger.info(f"Deactivated strategy {strategy_id}")

    def best_for_market(self, market_state: str) -> Optional[StrategyEntry]:
        """Get the best strategy for a given market state."""
        candidates = self.list_by_market_state(market_state)
        if not candidates:
            return None
        return max(candidates, key=lambda s: s.score)

    def summary(self) -> Dict:
        """Get registry summary statistics."""
        all_strats = list(self._strategies.values())
        approved = [s for s in all_strats if s.approved and s.status == "active"]
        return {
            "total": len(all_strats),
            "approved": len(approved),
            "active": len([s for s in all_strats if s.status == "active"]),
            "best_score": max(s.score for s in approved) if approved else 0,
            "best_sharpe": max(s.sharpe_ratio for s in approved) if approved else 0,
        }
