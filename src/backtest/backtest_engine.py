"""Qlib backtest engine wrapper.

Supports two modes:
1. Qlib-native backtest: uses Qlib's Alpha158 + LightGBM pipeline.
2. Strategy backtest: runs our custom strategy classes (ETF trend,
   mean reversion, etc.) on data read from local Qlib bin files.

The strategy backtest mode works without Wind or Qlib installed —
it reads bin files directly via the converter.
"""
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any

from .metrics import calc_all_metrics
from src.core import get_field, get_ohlcv, list_instruments, ensure_qlib

logger = logging.getLogger(__name__)


class BacktestEngine:
    """Backtest engine supporting both Qlib-native and custom strategies."""

    def __init__(self, qlib_dir: str = "data/qlib_bin"):
        self.qlib_dir = qlib_dir
        self._qlib_inited = False

    def init_qlib(self):
        """Initialize Qlib for native backtests."""
        if self._qlib_inited:
            return
        try:
            ensure_qlib(self.qlib_dir)
            self._qlib_inited = True
            logger.info(f"Qlib initialized with data: {self.qlib_dir}")
        except ImportError:
            logger.warning("Qlib not installed. Only strategy backtest mode available.")
            raise
        except Exception as e:
            logger.error(f"Qlib init failed: {e}")
            raise

    def run_qlib_backtest(
        self,
        strategy_config: dict,
        start_date: str,
        end_date: str,
        benchmark: str = "SH000300",
        init_cash: float = 1_000_000,
    ) -> dict:
        """Run Qlib-native backtest with Alpha158 + LightGBM.

        Requires pyqlib installed and data in Qlib bin format.
        """
        self.init_qlib()
        try:
            from qlib.contrib.model.gbdt import LGBModel
            from qlib.contrib.data.dataset import DatasetH
            from qlib.contrib.data.handler import Alpha158
            from qlib.contrib.evaluate import backtest_daily

            handler_config = {
                "start_time": start_date,
                "end_time": end_date,
                "fit_start_time": start_date,
                "fit_end_time": end_date,
                "instruments": "all",
            }
            handler = Alpha158(**handler_config)

            dataset = DatasetH(
                handler=handler,
                segments={
                    "train": (start_date, end_date),
                    "valid": (start_date, end_date),
                },
            )

            model = LGBModel(
                loss="mse",
                num_leaves=64,
                learning_rate=0.05,
                num_boost_round=200,
            )
            model.fit(dataset)
            pred = model.predict(dataset)
            logger.info(f"Qlib prediction shape: {pred.shape}")

            result = backtest_daily(
                start_time=start_date,
                end_time=end_date,
                benchmark=benchmark,
                account=init_cash,
            )
            return self._format_result(
                result, strategy_config, start_date, end_date, init_cash
            )
        except ImportError as e:
            logger.error(f"Qlib module not available: {e}")
            return {"error": f"Qlib module not available: {e}"}
        except Exception as e:
            logger.error(f"Qlib backtest failed: {e}")
            return {"error": str(e)}

    def run_strategy_backtest(
        self,
        strategy,
        code: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        init_cash: float = 1_000_000,
        commission: float = 0.0003,
        field: str = "close",
    ) -> dict:
        """Run a custom strategy on data from local Qlib bin files.

        Args:
            strategy: Strategy instance with generate_signal(df) method
            code: Instrument code like "SH600000" or "SH510300"
            start_date: Optional start date filter
            end_date: Optional end date filter
            init_cash: Initial capital
            commission: Commission rate per trade
            field: Price field to use for returns

        Returns:
            Backtest result dict with metrics, equity curve, etc.
        """
        ensure_qlib(self.qlib_dir)

        # Read OHLCV data from bin files
        df = self._read_ohlcv(code)
        if df is None or df.empty:
            return {"error": f"No data found for {code}"}

        # Filter date range
        if start_date:
            df = df[df.index >= pd.Timestamp(start_date)]
        if end_date:
            df = df[df.index <= pd.Timestamp(end_date)]
        if df.empty:
            return {"error": f"No data in date range for {code}"}

        # Generate signals
        signal = strategy.generate_signal(df)

        # Simulate positions based on signals
        position = signal.replace(0, np.nan).ffill().fillna(0)
        position = position.clip(-1, 1)

        # Calculate returns
        price = df[field] if field in df.columns else df["close"]
        daily_returns = position.shift(1) * price.pct_change()
        daily_returns = daily_returns.fillna(0)

        # Apply transaction costs on signal changes
        trades = position.diff().abs()
        cost = trades * commission
        daily_returns = daily_returns - cost

        # Build equity curve
        equity = (1 + daily_returns).cumprod()
        total_return = equity.iloc[-1] - 1

        metrics = calc_all_metrics(daily_returns)

        result = {
            "strategy": getattr(strategy, "name", "unknown"),
            "code": code,
            "metrics": metrics,
            "equity_curve": equity,
            "daily_returns": daily_returns,
            "positions": position,
            "signals": signal,
            "init_cash": init_cash,
            "final_value": init_cash * (1 + total_return),
            "data_points": len(df),
            "start_date": str(df.index[0].date()),
            "end_date": str(df.index[-1].date()),
        }
        logger.info(
            f"Strategy backtest [{result['strategy']}]: "
            f"annual={metrics['annual_return']}, "
            f"sharpe={metrics['sharpe_ratio']}, "
            f"max_dd={metrics['max_drawdown']}"
        )
        return result

    def run_multi_asset_backtest(
        self,
        strategy,
        codes: list,
        price_data: dict,
        init_cash: float = 1_000_000,
        commission: float = 0.0003,
    ) -> dict:
        """Run strategy across multiple assets (e.g., sector rotation).

        Args:
            strategy: Strategy with generate_portfolio_signals method
            codes: List of instrument codes
            price_data: {code: DataFrame} dict

        Returns:
            Aggregated backtest result.
        """
        signals = strategy.generate_portfolio_signals(price_data)

        all_returns = []
        for code in codes:
            if code not in signals or code not in price_data:
                continue
            sig = signals[code]
            price = price_data[code]["close"]
            pos = sig.clip(0, 1)
            ret = pos.shift(1) * price.pct_change()
            all_returns.append(ret)

        if not all_returns:
            return {"error": "No valid signals generated"}

        portfolio_returns = pd.concat(all_returns, axis=1).sum(axis=1) / len(codes)
        portfolio_returns = portfolio_returns.fillna(0)

        equity = (1 + portfolio_returns).cumprod()
        metrics = calc_all_metrics(portfolio_returns)

        return {
            "strategy": getattr(strategy, "name", "multi_asset"),
            "metrics": metrics,
            "equity_curve": equity,
            "daily_returns": portfolio_returns,
            "init_cash": init_cash,
            "final_value": init_cash * equity.iloc[-1],
            "n_assets": len(codes),
        }

    def _read_ohlcv(self, code: str) -> Optional[pd.DataFrame]:
        """Read OHLCV data from Qlib bin files for a single instrument."""
        df = get_ohlcv(code)
        if df is None or df.empty:
            return None

        # Drop rows where all values are 0 (unlisted periods)
        df = df[(df != 0).any(axis=1)]
        return df

    def _format_result(
        self, qlib_result, config, start, end, cash
    ) -> dict:
        """Format Qlib-native backtest result."""
        try:
            if isinstance(qlib_result, dict):
                returns = qlib_result.get("return", pd.Series(dtype=float))
            else:
                returns = qlib_result
            if isinstance(returns, pd.Series) and not returns.empty:
                metrics = calc_all_metrics(returns)
            else:
                metrics = {}
        except Exception:
            metrics = {}
        return {
            "metrics": metrics,
            "config": config,
            "start_date": start,
            "end_date": end,
            "init_cash": cash,
            "raw_result": str(qlib_result)[:500],
        }

    def run_qlib_backtest(
        self,
        strategy_config: dict,
        start_date: str,
        end_date: str,
        benchmark: str = "SH000300",
        init_cash: float = 1_000_000,
    ) -> dict:
        """Qlib-native backtest: Alpha158 + LightGBM → predict → top-k portfolio.

        Does NOT depend on torch. Uses LightGBM for model training and
        a manual vectorized backtest for portfolio simulation.
        """
        self.init_qlib()

        model_type = strategy_config.get("model_type", "lightgbm")
        n_holdings = strategy_config.get("n_holdings", 10)
        weight_scheme = strategy_config.get("weight_scheme", "equal")
        factor_names = strategy_config.get("factor_names", [])
        model_params = strategy_config.get("model_params", {})

        from qlib.contrib.data.handler import Alpha158
        from qlib.data.dataset import DatasetH
        from qlib.data.dataset.processor import CSZScoreNorm, DropnaLabel, CSRankNorm

        segments = self._compute_segments(start_date, end_date)
        train_end = segments["train"][1]
        valid_start = segments["valid"][0]
        valid_end = segments["valid"][1]
        test_start = segments["test"][0]

        handler = Alpha158(
            instruments="all",
            start_time=start_date,
            end_time=end_date,
            fit_start_time=start_date,
            fit_end_time=train_end,
            infer_processors=[CSZScoreNorm(fields_group="feature")],
            learn_processors=[DropnaLabel(), CSRankNorm(fields_group="label")],
            label=["Ref($close, -2) / Ref($close, -1) - 1"],
        )
        dataset = DatasetH(handler=handler, segments=segments)
        logger.info(f"Dataset created: {segments}")

        if model_type == "lightgbm":
            from qlib.contrib.model.gbdt import LGBModel
            model = LGBModel(
                loss="mse",
                num_leaves=model_params.get("num_leaves", 64),
                learning_rate=model_params.get("learning_rate", 0.05),
                num_boost_round=model_params.get("n_estimators", 200),
                colsample_bytree=model_params.get("colsample_bytree", 0.8),
                subsample=model_params.get("subsample", 0.8),
                lambda_l1=model_params.get("lambda_l1", 0.1),
                lambda_l2=model_params.get("lambda_l2", 0.1),
            )
        elif model_type == "xgboost":
            from qlib.contrib.model.xgboost import XGBModel
            model = XGBModel(
                objective="reg:squarederror",
                max_depth=model_params.get("max_depth", 6),
                learning_rate=model_params.get("learning_rate", 0.05),
                n_estimators=model_params.get("n_estimators", 200),
                subsample=model_params.get("subsample", 0.8),
                colsample_bytree=model_params.get("colsample_bytree", 0.8),
            )
        else:
            from qlib.contrib.model.gbdt import LGBModel
            model = LGBModel(loss="mse", num_leaves=64, learning_rate=0.05,
                             num_boost_round=200)

        model.fit(dataset)
        logger.info(f"Model {model_type} trained on train segment")

        pred = model.predict(dataset, segment="test")
        if pred is None or (hasattr(pred, "empty") and pred.empty):
            logger.warning("Empty predictions, trying valid segment")
            pred = model.predict(dataset, segment="valid")

        if pred is None or (hasattr(pred, "empty") and pred.empty):
            return {"error": "No predictions generated", "metrics": {}}

        portfolio_returns = self._simulate_portfolio(
            pred, n_holdings, weight_scheme, start_date, end_date
        )

        if portfolio_returns is None or portfolio_returns.empty:
            return {"error": "Portfolio simulation failed", "metrics": {}}

        metrics = calc_all_metrics(portfolio_returns)
        equity = (1 + portfolio_returns).cumprod()
        drawdown = self._calc_drawdown(equity)

        result = {
            "metrics": metrics,
            "config": strategy_config,
            "start_date": start_date,
            "end_date": end_date,
            "init_cash": init_cash,
            "final_value": init_cash * float(equity.iloc[-1]),
            "net_values": (equity * init_cash).tolist(),
            "drawdowns": (drawdown * 100).tolist(),
            "daily_returns": portfolio_returns,
            "n_predictions": len(pred),
            "model_type": model_type,
        }
        logger.info(
            f"Qlib backtest [{model_type}]: "
            f"annual={metrics.get('annual_return', 0):.4f} "
            f"sharpe={metrics.get('sharpe_ratio', 0):.4f} "
            f"max_dd={metrics.get('max_drawdown', 0):.4f}"
        )
        return result

    @staticmethod
    def _compute_segments(start: str, end: str) -> dict:
        """Split date range into 60% train / 20% valid / 20% test."""
        dates = pd.date_range(start, end, freq="D")
        n = len(dates)
        train_cut = int(n * 0.6)
        valid_cut = int(n * 0.8)
        train_end = dates[train_cut].strftime("%Y-%m-%d")
        valid_start = dates[min(train_cut + 1, n - 1)].strftime("%Y-%m-%d")
        valid_end = dates[valid_cut].strftime("%Y-%m-%d")
        test_start = dates[min(valid_cut + 1, n - 1)].strftime("%Y-%m-%d")
        return {
            "train": (start, train_end),
            "valid": (valid_start, valid_end),
            "test": (test_start, end),
        }

    @staticmethod
    def _calc_drawdown(equity: pd.Series) -> pd.Series:
        """Calculate drawdown series from equity curve."""
        peak = equity.cummax()
        return (equity - peak) / peak

    def _simulate_portfolio(
        self,
        predictions: pd.Series,
        n_holdings: int,
        weight_scheme: str,
        start_date: str,
        end_date: str,
    ) -> Optional[pd.Series]:
        """Simulate a top-k long-only portfolio from model predictions.

        For each day, pick top n_holdings stocks by prediction score,
        assign equal or min-variance weights, and compute daily returns.
        """
        if predictions is None or len(predictions) == 0:
            return None

        if isinstance(predictions, pd.DataFrame):
            if "score" in predictions.columns:
                pred_col = predictions["score"]
            else:
                pred_col = predictions.iloc[:, 0]
        else:
            pred_col = predictions

        pred_df = pred_col.unstack(level="instrument") if pred_col.index.nlevels > 1 else pred_col.unstack()

        if pred_df.isna().all().all():
            return None

        daily_ranks = pred_df.rank(axis=1, ascending=False, method="first")
        top_mask = daily_ranks <= n_holdings

        from src.core import get_ohlcv

        all_codes = list(pred_df.columns)
        returns_data = {}
        for code in all_codes:
            ohlcv = get_ohlcv(code)
            if ohlcv is not None and not ohlcv.empty:
                ret = ohlcv["close"].pct_change()
                returns_data[code] = ret

        if not returns_data:
            return None

        all_returns = pd.DataFrame(returns_data)
        all_returns = all_returns.reindex(columns=pred_df.columns)
        all_returns = all_returns.loc[
            all_returns.index.intersection(pred_df.index)
        ]
        top_mask = top_mask.loc[
            top_mask.index.intersection(all_returns.index)
        ]

        if weight_scheme == "equal":
            weights = top_mask.astype(float).div(
                top_mask.sum(axis=1).replace(0, 1), axis=0
            )
        else:
            weights = top_mask.astype(float).div(
                top_mask.sum(axis=1).replace(0, 1), axis=0
            )

        portfolio_returns = (weights.shift(1) * all_returns).sum(axis=1)
        portfolio_returns = portfolio_returns.fillna(0)
        portfolio_returns = portfolio_returns[
            (portfolio_returns.index >= pd.Timestamp(start_date))
            & (portfolio_returns.index <= pd.Timestamp(end_date))
        ]
        return portfolio_returns

    # Backwards-compatible alias
    def run_backtest(self, strategy_config: dict, start_date: str, end_date: str,
                     benchmark: str = "SH000300", init_cash: float = 1_000_000,
                     cost: dict = None) -> dict:
        """Legacy entry point — delegates to Qlib-native backtest."""
        return self.run_qlib_backtest(
            strategy_config, start_date, end_date, benchmark, init_cash
        )

    def run_simple_backtest(
        self,
        signals: pd.DataFrame,
        price: pd.DataFrame,
        init_cash: float = 1_000_000,
        commission: float = 0.0003,
    ) -> dict:
        """Simplified vectorized backtest without Qlib dependency.

        Args:
            signals: Position signals DataFrame (1=long, -1=short, 0=flat)
            price: Price DataFrame aligned with signals
            init_cash: Initial capital
            commission: Commission rate per unit traded

        Returns:
            Result dict with metrics and equity curve.
        """
        position = signals.replace(0, np.nan).ffill().fillna(0)
        position = position.clip(-1, 1)

        daily_returns = position.shift(1) * price.pct_change()
        daily_returns = daily_returns.fillna(0)

        # Transaction costs
        trades = position.diff().abs()
        cost = trades * commission
        daily_returns = daily_returns - cost

        cumulative = (1 + daily_returns).cumprod()
        total_return = cumulative.iloc[-1] - 1

        if isinstance(daily_returns, pd.DataFrame):
            portfolio_returns = daily_returns.sum(axis=1)
        else:
            portfolio_returns = daily_returns

        metrics = calc_all_metrics(portfolio_returns)
        result = {
            "metrics": metrics,
            "cumulative_return": cumulative,
            "daily_returns": portfolio_returns,
            "positions": position,
            "init_cash": init_cash,
            "final_value": init_cash * (1 + total_return),
        }
        logger.info(
            f"Simple backtest: annual={metrics['annual_return']}, "
            f"sharpe={metrics['sharpe_ratio']}, "
            f"max_dd={metrics['max_drawdown']}"
        )
        return result
