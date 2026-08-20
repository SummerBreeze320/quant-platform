"""SimulatorExecutor wrapper for Qlib-native backtesting.

Provides nested backtesting capability (daily + minute-level) using
Qlib's execution simulator framework.
"""
import logging
from typing import Optional, Dict, Any, List, Union
from pathlib import Path

import pandas as pd
import numpy as np

from src.core import ensure_qlib, get_ohlcv, list_instruments

logger = logging.getLogger(__name__)


def run_backtest_daily(
    dataset: Any,
    model: Any,
    topk: int = 50,
    n_drop: int = 5,
    benchmark: str = "SH000300",
) -> Dict:
    """Run a standard daily backtest using Qlib's backtest_daily.

    This is the simplest backtest mode: daily rebalancing with TopkDropout strategy.

    Args:
        dataset: DatasetH from create_dataset().
        model: Trained model from create_model().
        topk: Number of top stocks to hold.
        n_drop: Number of stocks to drop each rebalance.
        benchmark: Benchmark code for comparison.

    Returns:
        Dict with analysis results, indicators, and portfolio analysis.
    """
    ensure_qlib()

    from qlib.contrib.evaluate import backtest_daily, alpha

    logger.info(f"Running daily backtest: topk={topk}, n_drop={n_drop}, bench={benchmark}")

    result = backtest_daily(
        dataset=dataset,
        topk=topk,
        n_drop=n_drop,
    )

    analysis = result.get("analysis", {})
    indicators = result.get("indicators", {})
    portfolio = result.get("port_analysis_record", {})

    report = {
        "returns": analysis.get("return", {}),
        "cost": analysis.get("cost", {}),
        "bench": analysis.get("bench", {}),
        "risk": analysis.get("risk", {}),
        "indicators": indicators,
    }

    label_data = dataset.prepare("test", col_set=["label"])
    if isinstance(label_data, tuple):
        y_test = label_data[1]
    else:
        y_test = label_data

    pred = model.predict(dataset=dataset, segment="test")
    if isinstance(pred, pd.DataFrame):
        pred = pred.iloc[:, 0]

    try:
        ic_score = alpha(pred)
        report["ic"] = {
            "IC": float(ic_score.loc["IC", "mean"]),
            "ICIR": float(ic_score.loc["ICIR", "mean"]),
            "Rank IC": float(ic_score.loc["Rank IC", "mean"]),
            "Rank ICIR": float(ic_score.loc["Rank ICIR", "mean"]),
        }
    except Exception as e:
        logger.warning(f"IC calculation failed: {e}")
        report["ic"] = {}

    logger.info("Daily backtest complete.")
    return {"result": result, "analysis": report}


def run_backtest_with_executor(
    dataset: Any,
    model: Any,
    strategy: Any = None,
    executor_config: Optional[Dict] = None,
    benchmark: str = "SH000300",
) -> Dict:
    """Run backtest with custom SimulatorExecutor configuration.

    Supports nested execution (e.g. daily + intraday) via Qlib's
    nested executor framework.

    Args:
        dataset: DatasetH instance.
        model: Trained model.
        strategy: Custom strategy (must have generate_trade_decision method).
                  If None, uses TopkDropoutStrategy.
        executor_config: Dict with executor settings.
        benchmark: Benchmark code.
    """
    ensure_qlib()

    from qlib.contrib.evaluate import backtest
    from qlib.contrib.evaluate import risk_analysis

    if strategy is None:
        from qlib.contrib.strategy.signal_strategy import TopkDropoutStrategy
        strategy = TopkDropoutStrategy(topk=50, n_drop=5)

    if executor_config is None:
        executor_config = {
            "time_per_step": "day",
            "generate_portfolio_metrics": True,
        }

    from qlib.backtest.executor import SimulatorExecutor
    executor = SimulatorExecutor(**executor_config)

    logger.info("Running backtest with SimulatorExecutor...")
    portfolio_metric = backtest(
        dataset=dataset,
        model=model,
        strategy=strategy,
        executor=executor,
        benchmark=benchmark,
    )

    analysis = risk_analysis(portfolio_metric)
    logger.info("Backtest with executor complete.")

    return {
        "portfolio_metric": portfolio_metric,
        "analysis": analysis.to_dict() if hasattr(analysis, "to_dict") else analysis,
    }


def run_nested_backtest(
    dataset: Any,
    model: Any,
    topk: int = 50,
    n_drop: int = 5,
    benchmark: str = "SH000300",
    inner_executor: str = "twap",
) -> Dict:
    """Run nested backtest with daily rebalancing + intraday execution.

    Daily level decides target weights, inner executor (TWAP/VWAP)
    executes over intraday time slices.

    Args:
        dataset: DatasetH instance.
        model: Trained model.
        topk/n_drop: TopkDropout parameters.
        benchmark: Benchmark code.
        inner_executor: "twap" or "vwap" for intraday execution.
    """
    ensure_qlib()

    from qlib.backtest.executor import SimulatorExecutor
    from qlib.contrib.evaluate import backtest as qlib_backtest
    from qlib.contrib.evaluate import risk_analysis

    from qlib.contrib.strategy.signal_strategy import TopkDropoutStrategy
    strategy = TopkDropoutStrategy(topk=topk, n_drop=n_drop)

    inner_config = {
        "time_per_step": "1min",
        "generate_portfolio_metrics": True,
        "trade_cost": {"buy": 0.0003, "sell": 0.0013},
    }
    outer_config = {
        "time_per_step": "day",
        "generate_portfolio_metrics": True,
        "inner_strategy": {
            "class": "TWAPStrategy" if inner_executor == "twap" else "VWAPStrategy",
        },
    }

    logger.info(f"Running nested backtest: daily + {inner_executor} intraday")

    try:
        portfolio_metric = qlib_backtest(
            dataset=dataset,
            model=model,
            strategy=strategy,
            executor=SimulatorExecutor(**outer_config),
            benchmark=benchmark,
        )
        analysis = risk_analysis(portfolio_metric)
        return {
            "portfolio_metric": portfolio_metric,
            "analysis": analysis.to_dict() if hasattr(analysis, "to_dict") else analysis,
        }
    except Exception as e:
        logger.error(f"Nested backtest failed: {e}, falling back to daily")
        return run_backtest_daily(dataset, model, topk, n_drop, benchmark)


def run_strategy_backtest_qlib(
    strategy,
    code: str = "SH600000",
    qlib_dir: str = "data/qlib_bin",
    init_cash: float = 1_000_000,
) -> Dict:
    """Run a custom strategy through the backtest engine using Qlib data.

    This bridges our unified strategy interface with the existing BacktestEngine.
    """
    from src.backtest.backtest_engine import BacktestEngine

    engine = BacktestEngine(qlib_dir=qlib_dir)

    if hasattr(strategy, "_etf_strategy"):
        return engine.run_strategy_backtest(
            strategy=strategy._etf_strategy,
            code=code,
            init_cash=init_cash,
        )
    elif hasattr(strategy, "_factor"):
        df = get_ohlcv(code)
        if df.empty:
            return {"error": f"No data for {code}"}
        signals = strategy.generate_signals(df)
        return engine.run_simple_backtest(
            signals=pd.DataFrame({code: signals}),
            init_cash=init_cash,
        )
    else:
        df = get_ohlcv(code)
        if df.empty:
            return {"error": f"No data for {code}"}
        signals = strategy.generate_signals(df)
        return engine.run_simple_backtest(
            signals=pd.DataFrame({code: signals}),
            init_cash=init_cash,
        )
