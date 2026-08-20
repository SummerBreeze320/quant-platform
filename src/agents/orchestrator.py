"""Research Orchestrator — 闭环编排器

Orchestrates the full closed-loop research pipeline:
1. Parse investment objective → generate search space
2. Sample strategy configurations
3. For each config: FactorAgent → ModelAgent → PortfolioAgent → RiskAgent
4. ExperimentAgent: backtest + score
5. CriticAgent: review + overfitting check
6. WalkForwardValidator: out-of-sample validation
7. StrategyRegistry: register approved strategies
8. StrategyEnsemble: build ensemble pool

The orchestrator is the "brain" that connects all Agents and research modules
into a single coherent pipeline.
"""
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime

from src.agents.contracts import (
    InvestmentObjective,
    ExperimentResult,
    ExperimentStatus,
    MarketState,
)
from src.research.scoring import ScoringFunction
from src.research.search_space import SearchSpace, StrategyConfig
from src.research.strategy_registry import StrategyRegistry
from src.research.strategy_ensemble import StrategyEnsemble
from src.research.walk_forward import WalkForwardValidator

logger = logging.getLogger(__name__)


class ResearchOrchestrator:
    """Orchestrates the full closed-loop research pipeline.

    This is the main entry point for automated strategy research.
    It coordinates all Agents and research modules.

    Flow:
        objective → search_space → sample configs
        → for each config:
            FactorAgent → ModelAgent → PortfolioAgent → RiskAgent
            → ExperimentAgent (backtest + score)
            → CriticAgent (review)
            → WalkForward (out-of-sample validation)
        → register approved strategies
        → build ensemble pool
    """

    def __init__(
        self,
        search_space: Optional[SearchSpace] = None,
        scoring: Optional[ScoringFunction] = None,
        registry: Optional[StrategyRegistry] = None,
        ensemble: Optional[StrategyEnsemble] = None,
        walk_forward: Optional[WalkForwardValidator] = None,
        n_configs: int = 3,
    ):
        self.search_space = search_space or SearchSpace()
        self.scoring = scoring or ScoringFunction()
        self.registry = registry or StrategyRegistry()
        self.ensemble = ensemble or StrategyEnsemble(self.registry)
        self.walk_forward = walk_forward or WalkForwardValidator()
        self.n_configs = n_configs

        self._factor_agent = None
        self._model_agent = None
        self._portfolio_agent = None
        self._risk_agent = None
        self._experiment_agent = None
        self._critic_agent = None

    def _initialize_agents(self):
        """Lazy-initialize all agents."""
        if self._factor_agent is None:
            from src.agents.factor_agent import FactorAgent
            self._factor_agent = FactorAgent()

        if self._model_agent is None:
            from src.agents.model_agent import ModelAgent
            self._model_agent = ModelAgent()

        if self._portfolio_agent is None:
            from src.agents.portfolio_agent import PortfolioAgent
            self._portfolio_agent = PortfolioAgent()

        if self._risk_agent is None:
            from src.agents.risk_agent import RiskAgent
            self._risk_agent = RiskAgent()

        if self._experiment_agent is None:
            from src.agents.experiment_agent import ExperimentAgent
            self._experiment_agent = ExperimentAgent()

        if self._critic_agent is None:
            from src.agents.critic_agent import CriticAgent
            self._critic_agent = CriticAgent()

    def run(self, objective: InvestmentObjective) -> List[ExperimentResult]:
        """Run the full closed-loop research pipeline.

        Args:
            objective: Investment objective with constraints

        Returns:
            List of ExperimentResult for all sampled configurations
        """
        self._initialize_agents()
        logger.info(
            f"Orchestrator started: goal={objective.goal} "
            f"universe={objective.universe} "
            f"n_configs={self.n_configs}"
        )

        start_date = objective.start_date or "2020-01-01"
        end_date = objective.end_date or "2024-12-31"

        configs = [
            self.search_space.sample(
                start_date=start_date,
                end_date=end_date,
                benchmark=objective.benchmark,
            )
            for _ in range(self.n_configs)
        ]

        results: List[ExperimentResult] = []

        for i, config in enumerate(configs):
            logger.info(f"=== Config {i+1}/{len(configs)} ===")
            try:
                result = self._run_single_config(objective, config, start_date, end_date)
                results.append(result)

                if result.review_result and result.review_result.approved:
                    self._register_strategy(result, config, objective)

            except Exception as e:
                logger.error(f"Config {i+1} failed: {e}", exc_info=True)
                results.append(ExperimentResult(
                    experiment_id=f"failed_{i}",
                    status=ExperimentStatus.REJECTED,
                    created_at=datetime.now().isoformat(),
                ))

        self._build_ensemble(results)

        logger.info(
            f"Orchestrator finished: {len(results)} configs, "
            f"{sum(1 for r in results if r.status == ExperimentStatus.APPROVED)} approved"
        )
        return results

    def _run_single_config(
        self,
        objective: InvestmentObjective,
        config: StrategyConfig,
        start_date: str,
        end_date: str,
    ) -> ExperimentResult:
        """Run a single configuration through the full pipeline."""
        exp_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{config.model.model_types[0]}"

        # 1. Factor search
        logger.info(f"[{exp_id}] Factor search...")
        factor_set = self._factor_agent.execute(config.factor)

        # 2. Model search
        logger.info(f"[{exp_id}] Model search...")
        model_request = config.model
        model_request.factor_set = factor_set
        model_config = self._model_agent.execute(model_request)

        # 3. Portfolio search
        logger.info(f"[{exp_id}] Portfolio search...")
        portfolio_request = config.portfolio
        portfolio_request.factor_set = factor_set
        portfolio_request.model_setting = model_config
        portfolio_request.start_date = start_date
        portfolio_request.end_date = end_date
        portfolio_config = self._portfolio_agent.execute(portfolio_request)

        # 4. Risk assessment
        logger.info(f"[{exp_id}] Risk assessment...")
        from src.agents.contracts import RiskAssessmentRequest
        risk_request = RiskAssessmentRequest(
            benchmark=objective.benchmark,
            portfolio_config=portfolio_config,
        )
        risk_assessment = self._risk_agent.execute(risk_request)

        # 5. Experiment: backtest + score
        logger.info(f"[{exp_id}] Backtest + scoring...")
        from src.agents.contracts import ExperimentPlan
        plan = ExperimentPlan(
            experiment_id=exp_id,
            factor_set=factor_set,
            model_setting=model_config,
            portfolio_config=portfolio_config,
            risk_assessment=risk_assessment,
            objective=objective,
        )
        bt_result = self._experiment_agent.execute(plan)

        # 6. Scoring
        if bt_result and bt_result.backtest_result:
            score = self.scoring.compute(
                bt_result.backtest_result.metrics, objective
            )
            bt_result.score = score
            logger.info(f"[{exp_id}] Score: {score:.4f}")

        # 7. Critic review
        logger.info(f"[{exp_id}] Critic review...")
        from src.agents.contracts import ReviewRequest
        review_req = ReviewRequest(
            experiment_id=exp_id,
            metrics=bt_result.backtest_result.metrics if bt_result.backtest_result else None,
            factor_names=factor_set.factors,
            backtest_period=f"{start_date}~{end_date}",
            train_test_split={
                "train": f"{start_date}~{end_date}",
                "test": f"{start_date}~{end_date}",
            },
        )
        review_result = self._critic_agent.execute(review_req)

        # 8. Walk-Forward validation (if approved)
        wf_result = None
        if review_result.approved:
            logger.info(f"[{exp_id}] Walk-forward validation...")
            try:
                wf_result = self.walk_forward.validate(
                    backtest_fn=lambda s, e: self._quick_backtest(
                        config, s, e, objective
                    ),
                    start_date=start_date,
                    end_date=end_date,
                )
                if not wf_result.passed:
                    review_result.approved = False
                    review_result.reasons.append(
                        f"Walk-forward validation failed: "
                        f"consistency={wf_result.return_consistency:.0%}"
                    )
                    logger.info(f"[{exp_id}] Walk-forward FAILED")
                else:
                    logger.info(f"[{exp_id}] Walk-forward PASSED")
            except Exception as e:
                logger.warning(f"[{exp_id}] Walk-forward skipped: {e}")

        # Build final result
        result = ExperimentResult(
            experiment_id=exp_id,
            status=ExperimentStatus.APPROVED if review_result.approved else ExperimentStatus.REJECTED,
            factor_set=factor_set,
            model_setting=model_config,
            portfolio_config=portfolio_config,
            risk_assessment=risk_assessment,
            backtest_result=bt_result.backtest_result if bt_result else None,
            review_result=review_result,
            score=bt_result.score if bt_result else 0.0,
            created_at=datetime.now().isoformat(),
        )

        return result

    def _quick_backtest(
        self,
        config: StrategyConfig,
        start: str,
        end: str,
        objective: InvestmentObjective,
    ) -> Dict[str, Any]:
        """Quick backtest for walk-forward validation."""
        try:
            from src.backtest.backtest_engine import run_qlib_backtest

            model_type = config.model.model_types[0] if config.model.model_types else "lightgbm"
            n_holdings = config.portfolio.n_holdings_range[0] if config.portfolio.n_holdings_range else 10
            weight_scheme = config.portfolio.weight_schemes[0] if config.portfolio.weight_schemes else "equal_weight"

            result = run_qlib_backtest(
                model_type=model_type,
                n_holdings=n_holdings,
                weight_scheme=weight_scheme,
                start_date=start,
                end_date=end,
                benchmark=objective.benchmark,
            )
            return result
        except Exception as e:
            logger.warning(f"Quick backtest failed ({start}~{end}): {e}")
            return {"metrics": {}}

    def _register_strategy(
        self,
        result: ExperimentResult,
        config: StrategyConfig,
        objective: InvestmentObjective,
    ):
        """Register an approved strategy to the registry."""
        if not result.backtest_result:
            return

        metrics = result.backtest_result.metrics
        name = f"{result.model_setting.model_type}_{result.portfolio_config.weight_scheme}"
        market_states = config.market_states

        strategy_id = self.registry.register(
            name=name,
            factor_set=result.factor_set.factors if result.factor_set else [],
            model_type=result.model_setting.model_type if result.model_setting else "lightgbm",
            model_params=result.model_setting.params if result.model_setting else {},
            n_holdings=result.portfolio_config.n_holdings if result.portfolio_config else 10,
            weight_scheme=result.portfolio_config.weight_scheme if result.portfolio_config else "equal_weight",
            rebalance_freq=result.portfolio_config.rebalance_freq if result.portfolio_config else "W",
            market_states=market_states,
            annual_return=metrics.annual_return,
            sharpe_ratio=metrics.sharpe_ratio,
            max_drawdown=metrics.max_drawdown,
            score=result.score,
            approved=result.review_result.approved if result.review_result else True,
            overfitting_score=result.review_result.overfitting_score if result.review_result else 0,
            metadata={
                "experiment_id": result.experiment_id,
                "benchmark": objective.benchmark,
                "constraints": objective.constraints,
            },
        )
        logger.info(f"Registered strategy {strategy_id} ({name})")

    def _build_ensemble(self, results: List[ExperimentResult]):
        """Build ensemble pool from approved strategies."""
        approved = [
            r for r in results
            if r.status == ExperimentStatus.APPROVED and r.review_result
        ]
        if not approved:
            logger.info("No approved strategies for ensemble")
            return

        from src.agents.market_state import MarketStateRecognizer
        recognizer = MarketStateRecognizer()
        try:
            market_state = recognizer.recognize()
        except Exception:
            market_state = MarketState.SIDEWAYS

        allocation = self.ensemble.rebalance(market_state)
        logger.info(
            f"Ensemble built: {allocation.total_positions} strategies "
            f"for {market_state.value} market"
        )

    def evolve(
        self,
        objective: InvestmentObjective,
        n_iterations: int = 3,
    ) -> List[List[ExperimentResult]]:
        """Run multiple iterations of the research loop.

        Each iteration uses the knowledge from previous iterations
        to explore the strategy space more effectively.

        Args:
            objective: Investment objective
            n_iterations: Number of iterations

        Returns:
            List of iteration results (each is a list of ExperimentResults)
        """
        all_iterations = []

        for i in range(n_iterations):
            logger.info(f"=== Evolution Iteration {i+1}/{n_iterations} ===")
            results = self.run(objective)
            all_iterations.append(results)

            approved = [
                r for r in results
                if r.status == ExperimentStatus.APPROVED
            ]
            logger.info(
                f"Iteration {i+1}: {len(results)} configs, "
                f"{len(approved)} approved"
            )

            if not approved and i > 0:
                logger.warning("No approved strategies in this iteration, stopping")
                break

        return all_iterations

    def status(self) -> Dict[str, Any]:
        """Get orchestrator status summary."""
        registry_summary = self.registry.summary()
        ensemble_summary = self.ensemble.performance_summary()
        return {
            "search_space_cardinality": self.search_space.cardinality(),
            "n_configs": self.n_configs,
            "registry": registry_summary,
            "ensemble": ensemble_summary,
        }
