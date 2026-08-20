"""Search Space — 四层策略空间定义

Defines the four-layer strategy space:
1. Factor layer: categories, max_factors, IC thresholds
2. Model layer: types, hyperparameter grids
3. Portfolio layer: n_holdings, weight_schemes, rebalance_freq
4. Market state layer: bull/bear/sideways/high_vol/low_vol

Provides sampling and enumeration utilities for strategy exploration.
"""
import logging
import random
from typing import List, Dict, Any, Optional, Iterator
from dataclasses import dataclass, field

from src.agents.contracts import (
    MarketState,
    FactorSearchRequest,
    ModelSearchRequest,
    PortfolioSearchRequest,
)

logger = logging.getLogger(__name__)


@dataclass
class FactorLayer:
    """Factor space definition."""
    categories: List[str] = field(default_factory=lambda: [
        "price_volume", "momentum", "volatility", "fundamental", "alternative"
    ])
    max_factors_range: List[int] = field(default_factory=lambda: [10, 20, 30])
    ic_threshold_range: List[float] = field(default_factory=lambda: [0.01, 0.02, 0.03])
    icir_threshold_range: List[float] = field(default_factory=lambda: [0.2, 0.3, 0.5])

    def to_request(self, start_date: str, end_date: str) -> FactorSearchRequest:
        return FactorSearchRequest(
            categories=self.categories,
            max_factors=random.choice(self.max_factors_range),
            ic_threshold=random.choice(self.ic_threshold_range),
            icir_threshold=random.choice(self.icir_threshold_range),
            start_date=start_date,
            end_date=end_date,
            universe="all",
        )


@dataclass
class ModelLayer:
    """Model space definition."""
    model_types: List[str] = field(default_factory=lambda: [
        "lightgbm", "xgboost", "transformer", "lstm"
    ])
    param_search: bool = True
    cv_folds_options: List[int] = field(default_factory=lambda: [3, 5, 10])

    def to_request(
        self,
        start_date: str,
        end_date: str,
        factor_set=None,
    ) -> ModelSearchRequest:
        return ModelSearchRequest(
            model_types=self.model_types,
            param_search=self.param_search,
            cv_folds=random.choice(self.cv_folds_options),
            factor_set=factor_set,
            universe="all",
            start_date=start_date,
            end_date=end_date,
        )


@dataclass
class PortfolioLayer:
    """Portfolio space definition."""
    n_holdings_range: List[int] = field(default_factory=lambda: [5, 10, 20, 50])
    weight_schemes: List[str] = field(default_factory=lambda: [
        "equal_weight", "score_weighted", "enhanced_indexing", "min_variance"
    ])
    rebalance_freqs: List[str] = field(default_factory=lambda: ["D", "W", "M"])
    max_position_range: List[float] = field(default_factory=lambda: [0.05, 0.10, 0.15])
    tracking_error_range: List[float] = field(default_factory=lambda: [0.03, 0.05, 0.08])

    def to_request(
        self,
        benchmark: str = "SH000300",
        factor_set=None,
        model_setting=None,
        start_date: str = "",
        end_date: str = "",
    ) -> PortfolioSearchRequest:
        return PortfolioSearchRequest(
            n_holdings_range=self.n_holdings_range,
            weight_schemes=self.weight_schemes,
            rebalance_freq=self.rebalance_freqs,
            max_position=random.choice(self.max_position_range),
            tracking_error_limit=random.choice(self.tracking_error_range),
            benchmark=benchmark,
            factor_set=factor_set,
            model_setting=model_setting,
            start_date=start_date,
            end_date=end_date,
        )


@dataclass
class MarketStateLayer:
    """Market state space definition."""
    states: List[str] = field(default_factory=lambda: [
        MarketState.BULL.value,
        MarketState.BEAR.value,
        MarketState.SIDEWAYS.value,
        MarketState.HIGH_VOL.value,
        MarketState.LOW_VOL.value,
    ])
    position_ratio_map: Dict[str, float] = field(default_factory=lambda: {
        MarketState.BULL.value: 1.0,
        MarketState.SIDEWAYS.value: 0.5,
        MarketState.LOW_VOL.value: 0.7,
        MarketState.BEAR.value: 0.2,
        MarketState.HIGH_VOL.value: 0.2,
    })

    def get_position_ratio(self, state: str) -> float:
        return self.position_ratio_map.get(state, 0.5)


@dataclass
class StrategyConfig:
    """One point in the four-layer strategy space."""
    factor: FactorSearchRequest
    model: ModelSearchRequest
    portfolio: PortfolioSearchRequest
    market_states: List[str]


class SearchSpace:
    """Four-layer strategy space with sampling and enumeration.

    Layers:
    1. Factor: which factor categories, how many, IC thresholds
    2. Model: which model types, hyperparameter search
    3. Portfolio: n_holdings, weight scheme, rebalance frequency
    4. Market state: which environments this strategy targets

    Provides:
    - sample(): Random sample from the space (for exploration)
    - enumerate(): Systematic enumeration (for small subspaces)
    - cardinality(): Total number of configurations
    """

    def __init__(
        self,
        factor_layer: Optional[FactorLayer] = None,
        model_layer: Optional[ModelLayer] = None,
        portfolio_layer: Optional[PortfolioLayer] = None,
        market_layer: Optional[MarketStateLayer] = None,
    ):
        self.factor_layer = factor_layer or FactorLayer()
        self.model_layer = model_layer or ModelLayer()
        self.portfolio_layer = portfolio_layer or PortfolioLayer()
        self.market_layer = market_layer or MarketStateLayer()

    def sample(
        self,
        start_date: str = "2020-01-01",
        end_date: str = "2024-12-31",
        benchmark: str = "SH000300",
        n_market_states: int = 3,
    ) -> StrategyConfig:
        """Sample a random configuration from the strategy space."""
        factor_req = self.factor_layer.to_request(start_date, end_date)

        model_req = self.model_layer.to_request(
            start_date, end_date, factor_set=None
        )

        portfolio_req = self.portfolio_layer.to_request(
            benchmark=benchmark,
            factor_set=None,
            model_setting=None,
            start_date=start_date,
            end_date=end_date,
        )

        states = random.sample(
            self.market_layer.states,
            min(n_market_states, len(self.market_layer.states)),
        )

        config = StrategyConfig(
            factor=factor_req,
            model=model_req,
            portfolio=portfolio_req,
            market_states=states,
        )

        logger.debug(
            f"Sampled config: "
            f"max_factors={factor_req.max_factors} "
            f"models={model_req.model_types} "
            f"holdings={portfolio_req.n_holdings_range} "
            f"states={states}"
        )
        return config

    def enumerate_configs(
        self,
        start_date: str = "2020-01-01",
        end_date: str = "2024-12-31",
        benchmark: str = "SH000300",
    ) -> Iterator[StrategyConfig]:
        """Enumerate all configurations (for small subspaces).

        Note: Full enumeration may be very large. Use with reduced layers.
        """
        for max_f in self.factor_layer.max_factors_range:
            for ic_t in self.factor_layer.ic_threshold_range:
                for model_type in self.model_layer.model_types:
                    for n_hold in self.portfolio_layer.n_holdings_range:
                        for ws in self.portfolio_layer.weight_schemes:
                            for rb in self.portfolio_layer.rebalance_freqs:
                                for mp in self.portfolio_layer.max_position_range:
                                    for te in self.portfolio_layer.tracking_error_range:
                                        factor_req = FactorSearchRequest(
                                            categories=self.factor_layer.categories,
                                            max_factors=max_f,
                                            ic_threshold=ic_t,
                                            icir_threshold=self.factor_layer.icir_threshold_range[0],
                                            start_date=start_date,
                                            end_date=end_date,
                                            universe="all",
                                        )
                                        model_req = ModelSearchRequest(
                                            model_types=[model_type],
                                            param_search=self.model_layer.param_search,
                                            cv_folds=self.model_layer.cv_folds_options[0],
                                            start_date=start_date,
                                            end_date=end_date,
                                            universe="all",
                                        )
                                        portfolio_req = PortfolioSearchRequest(
                                            n_holdings_range=[n_hold],
                                            weight_schemes=[ws],
                                            rebalance_freq=[rb],
                                            max_position=mp,
                                            tracking_error_limit=te,
                                            benchmark=benchmark,
                                            start_date=start_date,
                                            end_date=end_date,
                                        )
                                        yield StrategyConfig(
                                            factor=factor_req,
                                            model=model_req,
                                            portfolio=portfolio_req,
                                            market_states=self.market_layer.states,
                                        )

    def cardinality(self) -> int:
        """Total number of configurations in the space."""
        n_factor = (
            len(self.factor_layer.max_factors_range)
            * len(self.factor_layer.ic_threshold_range)
        )
        n_model = len(self.model_layer.model_types)
        n_portfolio = (
            len(self.portfolio_layer.n_holdings_range)
            * len(self.portfolio_layer.weight_schemes)
            * len(self.portfolio_layer.rebalance_freqs)
            * len(self.portfolio_layer.max_position_range)
            * len(self.portfolio_layer.tracking_error_range)
        )
        total = n_factor * n_model * n_portfolio
        logger.info(
            f"Search space cardinality: {total} "
            f"(factor={n_factor} model={n_model} portfolio={n_portfolio})"
        )
        return total

    def reduced_space(
        self,
        max_configs: int = 50,
    ) -> "SearchSpace":
        """Create a reduced search space for faster exploration."""
        return SearchSpace(
            factor_layer=FactorLayer(
                categories=self.factor_layer.categories[:3],
                max_factors_range=self.factor_layer.max_factors_range[:2],
                ic_threshold_range=self.factor_layer.ic_threshold_range[:2],
            ),
            model_layer=ModelLayer(
                model_types=self.model_layer.model_types[:2],
                cv_folds_options=[5],
            ),
            portfolio_layer=PortfolioLayer(
                n_holdings_range=self.portfolio_layer.n_holdings_range[:2],
                weight_schemes=self.portfolio_layer.weight_schemes[:2],
                rebalance_freqs=["W"],
                max_position_range=[0.10],
                tracking_error_range=[0.05],
            ),
            market_layer=self.market_layer,
        )
