"""Qlib模型预测信号生成器

使用Qlib训练好的模型生成实盘交易信号：
1. 加载Alpha158/360 Handler计算最新因子
2. 使用训练好的模型生成预测得分
3. 截面选股(Top-K) → 目标持仓权重
4. 组合优化(最大化夏普/最小化跟踪误差)

集成到实盘管线: Qlib predict → 信号 → 调度 → 网关 → 执行
"""
import logging
import pickle
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

import pandas as pd
import numpy as np

from src.core import ensure_qlib, get_ohlcv, list_instruments

logger = logging.getLogger(__name__)


class QlibSignalGenerator:
    """基于Qlib模型预测的信号生成器

    与自定义因子SignalGenerator互补:
    - SignalGenerator: 自定义alpha因子 → 多因子合成 → 选股
    - QlibSignalGenerator: Qlib模型 → 预测得分 → 选股

    Usage:
        gen = QlibSignalGenerator(model_path="data/models/lightgbm_v1.pkl")
        signal = gen.generate(codes=["SH600000", ...], topk=5)
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        model: Optional[Any] = None,
        handler_type: str = "alpha158",
        topk: int = 5,
        n_drop: int = 1,
        max_position: float = 0.15,
        qlib_dir: str = "data/qlib_bin",
    ):
        ensure_qlib(qlib_dir)
        self.handler_type = handler_type
        self.topk = topk
        self.n_drop = n_drop
        self.max_position = max_position
        self._model = model
        self._handler = None
        self._dataset = None

        if model_path:
            self.load_model(model_path)

    def load_model(self, model_path: str):
        """加载训练好的模型"""
        path = Path(model_path)
        if not path.exists():
            logger.error(f"Model file not found: {model_path}")
            return

        try:
            with open(path, "rb") as f:
                self._model = pickle.load(f)
            logger.info(f"Model loaded: {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")

    def set_model(self, model: Any):
        """直接设置模型实例"""
        self._model = model
        logger.info(f"Model set: {model.__class__.__name__}")

    def _prepare_handler(
        self,
        instruments: str = "all",
        start_time: str = "2020-01-01",
        end_time: Optional[str] = None,
    ):
        """准备数据处理器"""
        if end_time is None:
            end_time = "2024-12-31"

        from src.research.handlers import get_handler, create_dataset

        self._handler = get_handler(
            handler_type=self.handler_type,
            instruments=instruments,
            start_time=start_time,
            end_time=end_time,
        )

        segments = {
            "train": ("2020-01-01", "2022-12-31"),
            "valid": ("2023-01-01", "2023-12-31"),
            "test": ("2024-01-01", "2024-12-31"),
        }
        self._dataset = create_dataset(self._handler, segments)
        logger.info(f"Handler prepared: {self.handler_type}, instruments={instruments}")

    def _auto_train(self):
        """自动训练 LightGBM 模型（当未加载预训练模型时）"""
        from qlib.contrib.model.gbdt import LGBModel

        logger.info("Auto-training LightGBM model (no pre-trained model loaded)...")
        model = LGBModel(
            loss="mse",
            num_leaves=64,
            learning_rate=0.05,
            num_boost_round=200,
            colsample_bytree=0.8,
            subsample=0.8,
            lambda_l1=0.1,
            lambda_l2=0.1,
        )
        model.fit(self._dataset)
        self._model = model
        logger.info(f"LightGBM auto-trained successfully: {model.__class__.__name__}")

    def predict(self, instruments: str = "all") -> pd.Series:
        """生成模型预测得分

        Args:
            instruments: 股票池

        Returns:
            pd.Series of prediction scores (index=MultiIndex(instrument, datetime))
        """
        self._prepare_handler(instruments)

        if self._model is None:
            self._auto_train()

        logger.info("Generating predictions...")
        pred = self._model.predict(dataset=self._dataset, segment="test")

        if isinstance(pred, pd.DataFrame):
            pred = pred.iloc[:, 0]

        logger.info(f"Predictions: {len(pred)} records")
        return pred

    def generate(
        self,
        codes: Optional[List[str]] = None,
        instruments: str = "all",
        as_of_date: Optional[str] = None,
        objective: str = "equal_weight",
    ) -> Dict:
        """生成交易信号

        Args:
            codes: 指定股票池(None则使用Qlib instruments)
            instruments: Qlib instruments参数("all"/"csi300"/...)
            as_of_date: 信号日期(None则用最新)
            objective: 权重优化目标

        Returns:
            信号字典 {signal_date, target_weights, scores, metadata}
        """
        # 1. 生成预测
        pred = self.predict(instruments)

        # 2. 获取最新截面得分
        if as_of_date:
            latest_date = pd.Timestamp(as_of_date)
            if latest_date not in pred.index.get_level_values("datetime"):
                dates = pred.index.get_level_values("datetime")
                latest_date = dates[dates <= latest_date][-1]
        else:
            latest_date = pred.index.get_level_values("datetime").max()

        latest_scores = pred.xs(latest_date, level="datetime") if isinstance(
            pred.index, pd.MultiIndex
        ) else pred.tail(1)

        if len(latest_scores) < 3:
            return {"error": "有效预测不足"}

        # 3. 选股
        top_stocks = latest_scores.nlargest(min(self.topk, len(latest_scores)))

        # 4. 权重分配
        if objective == "equal_weight":
            weights = pd.Series(
                1.0 / len(top_stocks),
                index=top_stocks.index,
            )
        elif objective == "score_weighted":
            positive_scores = top_stocks.clip(lower=0)
            if positive_scores.sum() > 0:
                weights = positive_scores / positive_scores.sum()
            else:
                weights = pd.Series(
                    1.0 / len(top_stocks),
                    index=top_stocks.index,
                )
        elif objective == "max_sharpe":
            from src.research.enhanced_optimizer import EnhancedIndexingOptimizer
            optimizer = EnhancedIndexingOptimizer(weights_limit=self.max_position)
            returns_data = {}
            for code in top_stocks.index:
                df = get_ohlcv(str(code))
                if df is not None and not df.empty:
                    returns_data[str(code)] = df["close"].pct_change()
            returns_df = pd.DataFrame(returns_data).tail(60)
            if not returns_df.empty:
                cov_matrix = returns_df.cov()
                factor_matrix = pd.DataFrame(index=top_stocks.index)
                weights = optimizer.optimize(
                    pred_scores=top_stocks,
                    factor_matrix=factor_matrix,
                    cov_matrix=cov_matrix,
                )
            else:
                weights = pd.Series(
                    1.0 / len(top_stocks),
                    index=top_stocks.index,
                )
        else:
            weights = pd.Series(
                1.0 / len(top_stocks),
                index=top_stocks.index,
            )

        weights = weights[weights > 1e-6]

        signal = {
            "signal_date": latest_date.strftime("%Y-%m-%d") if hasattr(latest_date, 'strftime') else str(latest_date),
            "generated_at": datetime.now().isoformat(),
            "target_weights": {str(k): float(v) for k, v in weights.items()},
            "prediction_scores": {str(k): float(v) for k, v in top_stocks.items()},
            "n_selected": len(weights),
            "n_universe": len(latest_scores),
            "model": self._model.__class__.__name__ if self._model else "unknown",
            "handler": self.handler_type,
            "objective": objective,
        }

        logger.info(
            f"Qlib signal generated: {signal['n_selected']} positions, "
            f"date={signal['signal_date']}"
        )
        return signal

    @property
    def last_signal(self) -> Optional[Dict]:
        return getattr(self, "_last_signal", None)

    def _combine_factors(
        self,
        model_scores: pd.Series,
        alternative_scores: Optional[pd.Series] = None,
        fundamental_scores: Optional[pd.Series] = None,
        alpha_weights: Optional[Dict[str, float]] = None,
    ) -> pd.Series:
        """Combine multiple factor sources into a single Alpha Score.

        Alpha = w1*price_volume(model) + w2*fundamental + w3*alternative

        The model prediction captures price-volume + momentum signals.
        Fundamental and alternative scores are optional enhancements.

        Args:
            model_scores: Qlib model prediction scores (price+volume+momentum)
            alternative_scores: Alternative data scores (news, sentiment, etc.)
            fundamental_scores: Fundamental factor scores (PE, PB, ROE, etc.)
            alpha_weights: Custom fusion weights. Defaults to standard weights.

        Returns:
            pd.Series of combined Alpha Scores indexed by instrument.
        """
        from src.research.factor_library import compute_alpha_score

        return compute_alpha_score(
            price_volume_score=model_scores,
            fundamental_score=fundamental_scores,
            alternative_score=alternative_scores,
            weights=alpha_weights,
        )

    def generate_with_alpha(
        self,
        codes: Optional[List[str]] = None,
        instruments: str = "all",
        as_of_date: Optional[str] = None,
        use_alternative: bool = True,
        use_fundamental: bool = True,
    ) -> Dict:
        """Generate trading signal with multi-source Alpha Score fusion.

        Extends generate() by combining model predictions with
        alternative data and fundamental factors.

        Args:
            codes: Stock pool (None = use Qlib instruments)
            instruments: Qlib instruments parameter
            as_of_date: Signal date
            use_alternative: Include alternative data scores
            use_fundamental: Include fundamental factor scores

        Returns:
            Signal dict with alpha_score details.
        """
        # 1. Generate model predictions
        pred = self.predict(instruments)

        if as_of_date:
            latest_date = pd.Timestamp(as_of_date)
            if latest_date not in pred.index.get_level_values("datetime"):
                dates = pred.index.get_level_values("datetime")
                latest_date = dates[dates <= latest_date][-1]
        else:
            latest_date = pred.index.get_level_values("datetime").max()

        latest_scores = pred.xs(latest_date, level="datetime") if isinstance(
            pred.index, pd.MultiIndex
        ) else pred.tail(1)

        if len(latest_scores) < 3:
            return {"error": "Insufficient predictions"}

        model_scores = latest_scores.iloc[:, 0] if isinstance(latest_scores, pd.DataFrame) else latest_scores

        # 2. Get alternative data scores (optional)
        alt_scores = None
        if use_alternative:
            try:
                from src.research.factor_library import compute_alternative_factors
                from src.core import list_instruments

                all_codes = codes or list_instruments(as_list=True)
                alt_df = compute_alternative_factors(
                    codes=all_codes,
                    end_date=latest_date.strftime("%Y-%m-%d") if hasattr(latest_date, "strftime") else str(latest_date),
                    lookback=20,
                )
                if alt_df is not None and not alt_df.empty:
                    alt_scores = alt_df.mean(axis=1)
                    alt_scores.name = "alternative"
            except Exception as e:
                logger.warning(f"Alternative data unavailable: {e}")

        # 3. Get fundamental scores (optional)
        fund_scores = None
        if use_fundamental:
            try:
                from src.data.factor_calculator import FactorCalculator
                fc = FactorCalculator()
                fund_factors = ["pe_ratio", "pb_ratio", "roe", "roa"]
                fund_df = fc.compute_latest(
                    instruments=codes or "all",
                    factor_names=fund_factors,
                )
                if fund_df is not None and not fund_df.empty:
                    fund_scores = fund_df.mean(axis=1)
                    fund_scores.name = "fundamental"
            except Exception as e:
                logger.warning(f"Fundamental data unavailable: {e}")

        # 4. Combine into Alpha Score
        alpha_scores = self._combine_factors(
            model_scores=model_scores,
            alternative_scores=alt_scores,
            fundamental_scores=fund_scores,
        )

        # 5. Select top-K stocks
        top_stocks = alpha_scores.nlargest(min(self.topk, len(alpha_scores)))

        # 6. Equal weight (can be extended to score-weighted)
        weights = pd.Series(
            1.0 / len(top_stocks),
            index=top_stocks.index,
        )

        signal = {
            "signal_date": latest_date.strftime("%Y-%m-%d") if hasattr(latest_date, "strftime") else str(latest_date),
            "generated_at": datetime.now().isoformat(),
            "target_weights": {str(k): float(v) for k, v in weights.items()},
            "alpha_scores": {str(k): float(v) for k, v in top_stocks.items()},
            "model_scores": {str(k): float(v) for k, v in model_scores.reindex(top_stocks.index).items()},
            "n_selected": len(weights),
            "n_universe": len(alpha_scores),
            "model": self._model.__class__.__name__ if self._model else "unknown",
            "alpha_components": {
                "has_alternative": alt_scores is not None,
                "has_fundamental": fund_scores is not None,
                "n_alt_factors": len(alt_scores) if alt_scores is not None else 0,
                "n_fund_factors": len(fund_scores) if fund_scores is not None else 0,
            },
        }

        self._last_signal = signal
        logger.info(
            f"Alpha signal generated: {signal['n_selected']} positions, "
            f"date={signal['signal_date']}, "
            f"alt={'Y' if signal['alpha_components']['has_alternative'] else 'N'} "
            f"fund={'Y' if signal['alpha_components']['has_fundamental'] else 'N'}"
        )
        return signal
