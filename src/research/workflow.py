"""End-to-end research workflow.

Chains the full research pipeline:
1. Create DataHandler (Alpha158/Alpha360/Fundamental)
2. Create Dataset with train/valid/test segments
3. Train model (with MLflow tracking)
4. Run backtest (TopkDropout / EnhancedIndexing / nested)
5. Generate report with metrics

Usage:
    from src.research import ResearchWorkflow, run_full_workflow
    result = run_full_workflow(
        handler_type="alpha158",
        model_name="lightgbm",
        strategy_type="topk_dropout",
    )
"""
import logging
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
import pandas as pd

from src.core import ensure_qlib
from src.research.walk_forward import WalkForwardValidator, WalkForwardResult
from src.research.scoring import ScoringFunction, ScoringWeights

logger = logging.getLogger(__name__)


@dataclass
class WorkflowConfig:
    """Configuration for the research workflow."""
    handler_type: str = "alpha158"
    model_name: str = "lightgbm"
    strategy_type: str = "topk_dropout"

    start_time: str = "2015-01-01"
    end_time: str = "2024-12-31"
    fit_start_time: str = "2015-01-01"
    fit_end_time: str = "2019-12-31"

    train_start: str = "2015-01-01"
    train_end: str = "2019-12-31"
    valid_start: str = "2020-01-01"
    valid_end: str = "2021-12-31"
    test_start: str = "2022-01-01"
    test_end: str = "2024-12-31"

    instruments: str = "csi300"
    benchmark: str = "SH000300"

    topk: int = 50
    n_drop: int = 5
    tracking_error_limit: float = 0.05
    weights_limit: float = 0.05

    experiment_name: str = "quant_platform"
    use_mlflow: bool = True
    override_params: Optional[Dict] = None


@dataclass
class WorkflowResult:
    """Result of the research workflow."""
    config: WorkflowConfig
    handler: Any = None
    dataset: Any = None
    model: Any = None
    predictions: Any = None
    backtest_result: Dict = field(default_factory=dict)
    metrics: Dict = field(default_factory=dict)
    report: Dict = field(default_factory=dict)
    mlflow_run_id: Optional[str] = None
    success: bool = False
    error: Optional[str] = None


class ResearchWorkflow:
    """Orchestrates the full research pipeline."""

    def __init__(self, config: Optional[WorkflowConfig] = None):
        self.config = config or WorkflowConfig()

    def step1_create_handler(self) -> Any:
        """Step 1: Create DataHandler."""
        from .handlers import get_handler

        logger.info(f"[Step 1/5] Creating handler: {self.config.handler_type}")
        handler = get_handler(
            handler_type=self.config.handler_type,
            start_time=self.config.start_time,
            end_time=self.config.end_time,
            fit_start_time=self.config.fit_start_time,
            fit_end_time=self.config.fit_end_time,
            instruments=self.config.instruments,
        )
        return handler

    def step2_create_dataset(self, handler: Any) -> Any:
        """Step 2: Create Dataset with train/valid/test segments."""
        from .handlers import create_dataset

        segments = {
            "train": (self.config.train_start, self.config.train_end),
            "valid": (self.config.valid_start, self.config.valid_end),
            "test": (self.config.test_start, self.config.test_end),
        }

        logger.info(f"[Step 2/5] Creating dataset: segments={list(segments.keys())}")
        dataset = create_dataset(handler=handler, segments=segments)
        return dataset

    def step3_train_model(self, dataset: Any) -> Dict:
        """Step 3: Train model with MLflow tracking."""
        from .model_zoo import create_model, train_model

        logger.info(f"[Step 3/5] Training model: {self.config.model_name}")
        model = create_model(
            model_name=self.config.model_name,
            override_params=self.config.override_params,
        )

        result = train_model(
            model=model,
            dataset=dataset,
            experiment_name=self.config.experiment_name,
            run_name=f"{self.config.model_name}_{self.config.handler_type}",
            use_mlflow=self.config.use_mlflow,
        )
        return result

    def step4_run_backtest(self, dataset: Any, model: Any) -> Dict:
        """Step 4: Run backtest based on strategy type."""
        logger.info(f"[Step 4/5] Running backtest: {self.config.strategy_type}")

        if self.config.strategy_type == "topk_dropout":
            from .simulator import run_backtest_daily
            return run_backtest_daily(
                dataset=dataset,
                model=model,
                topk=self.config.topk,
                n_drop=self.config.n_drop,
                benchmark=self.config.benchmark,
            )

        elif self.config.strategy_type == "enhanced_indexing":
            from .enhanced_optimizer import EnhancedIndexingOptimizer
            optimizer = EnhancedIndexingOptimizer(
                benchmark=self.config.benchmark,
                tracking_error_limit=self.config.tracking_error_limit,
                weights_limit=self.config.weights_limit,
            )
            return optimizer.run_backtest(
                dataset=dataset,
                model=model,
                benchmark=self.config.benchmark,
            )

        elif self.config.strategy_type == "nested":
            from .simulator import run_nested_backtest
            return run_nested_backtest(
                dataset=dataset,
                model=model,
                topk=self.config.topk,
                n_drop=self.config.n_drop,
                benchmark=self.config.benchmark,
            )

        else:
            raise ValueError(f"Unknown strategy type: {self.config.strategy_type}")

    def step5_generate_report(self, backtest_result: Dict, train_result: Dict) -> Dict:
        """Step 5: Generate summary report."""
        logger.info("[Step 5/5] Generating report")

        analysis = backtest_result.get("analysis", {})
        train_metrics = train_result.get("metrics", {})

        report = {
            "config": {
                "handler": self.config.handler_type,
                "model": self.config.model_name,
                "strategy": self.config.strategy_type,
                "instruments": self.config.instruments,
                "benchmark": self.config.benchmark,
            },
            "model_metrics": train_metrics,
            "backtest": {},
            "mlflow": train_result.get("mlflow", {}),
        }

        if "returns" in analysis:
            returns = analysis["returns"]
            report["backtest"]["annual_return"] = returns.get("annualized_return", 0)
            report["backtest"]["cumulative_return"] = returns.get("total_return", 0)

        if "risk" in analysis:
            risk = analysis["risk"]
            report["backtest"]["sharpe"] = risk.get("information_ratio", 0)
            report["backtest"]["max_drawdown"] = risk.get("max_drawdown", 0)

        if "ic" in analysis:
            ic = analysis["ic"]
            report["backtest"]["ic"] = ic.get("IC", 0)
            report["backtest"]["icir"] = ic.get("ICIR", 0)

        return report

    def run_with_walk_forward(self, n_folds: int = 5) -> Tuple[WorkflowResult, WalkForwardResult]:
        """Run workflow with walk-forward validation.

        Args:
            n_folds: Number of walk-forward folds

        Returns:
            Tuple of (workflow_result, walk_forward_result)
        """
        wf = WalkForwardValidator(n_folds=n_folds)

        def _bt_fn(start: str, end: str) -> Dict:
            old_start, old_end = self.config.test_start, self.config.test_end
            old_fstart, old_fend = self.config.fit_start_time, self.config.fit_end_time
            self.config.test_start = start
            self.config.test_end = end
            self.config.fit_start_time = self.config.train_start
            self.config.fit_end_time = start
            try:
                result = self.run()
                return {"metrics": result.report.get("backtest", {})}
            finally:
                self.config.test_start = old_start
                self.config.test_end = old_end
                self.config.fit_start_time = old_fstart
                self.config.fit_end_time = old_fend

        wf_result = wf.validate(
            backtest_fn=_bt_fn,
            start_date=self.config.test_start,
            end_date=self.config.test_end,
        )
        wf_result_obj = wf.validate(
            backtest_fn=_bt_fn,
            start_date=self.config.test_start,
            end_date=self.config.test_end,
        )
        result = self.run()
        return result, wf_result_obj

    def score_result(self, result: WorkflowResult) -> float:
        """Score a workflow result using the composite scoring function."""
        from src.agents.contracts import BacktestMetrics

        bt = result.report.get("backtest", {})
        metrics = BacktestMetrics(
            annual_return=bt.get("annual_return", 0),
            sharpe_ratio=bt.get("sharpe", 0),
            max_drawdown=bt.get("max_drawdown", 0),
            ic_mean=bt.get("ic", 0),
            icir=bt.get("icir", 0),
        )
        scorer = ScoringFunction()
        score = scorer.compute(metrics)
        logger.info(f"Workflow score: {score:.4f}")
        return score

    def run(self) -> WorkflowResult:
        """Run the full research workflow."""
        result = WorkflowResult(config=self.config)
        ensure_qlib()

        try:
            # Step 1: Create handler
            result.handler = self.step1_create_handler()

            # Step 2: Create dataset
            result.dataset = self.step2_create_dataset(result.handler)

            # Step 3: Train model
            train_result = self.step3_train_model(result.dataset)
            result.model = train_result.get("model")
            result.predictions = train_result.get("predictions")
            result.metrics = train_result.get("metrics", {})
            result.mlflow_run_id = train_result.get("mlflow", {}).get("run_id")

            # Step 4: Run backtest
            result.backtest_result = self.step4_run_backtest(
                result.dataset, result.model
            )

            # Step 5: Generate report
            result.report = self.step5_generate_report(
                result.backtest_result, train_result
            )

            result.success = True
            logger.info("Research workflow completed successfully.")

        except Exception as e:
            logger.error(f"Research workflow failed: {e}")
            result.error = str(e)

        return result


def run_full_workflow(
    handler_type: str = "alpha158",
    model_name: str = "lightgbm",
    strategy_type: str = "topk_dropout",
    instruments: str = "csi300",
    **kwargs,
) -> WorkflowResult:
    """Convenience function to run the full research workflow.

    Args:
        handler_type: "alpha158", "alpha360", or "fundamental"
        model_name: Model name from model zoo (lightgbm, lstm, etc.)
        strategy_type: "topk_dropout", "enhanced_indexing", or "nested"
        instruments: "csi300", "csi500", "all", or custom list
        **kwargs: Override any WorkflowConfig field

    Returns:
        WorkflowResult with all intermediate results.
    """
    config = WorkflowConfig(
        handler_type=handler_type,
        model_name=model_name,
        strategy_type=strategy_type,
        instruments=instruments,
        **kwargs,
    )
    workflow = ResearchWorkflow(config=config)
    return workflow.run()
