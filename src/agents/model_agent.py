"""ModelAgent — 模型空间搜索

在模型空间中搜索最优预测模型，输出模型配置。

包装现有模块:
- src.research.model_zoo (ModelZoo: LightGBM/XGBoost/Transformer/LSTM/GATs)
- src.rd_agent.model_runner (ModelRunner: RD-Agent模型进化)
"""
import logging
from typing import Optional
from .base import BaseAgent
from .contracts import ModelSearchRequest, ModelConfig

logger = logging.getLogger(__name__)

DEFAULT_PARAMS = {
    "lightgbm": {
        "loss": "mse",
        "num_leaves": 64,
        "learning_rate": 0.05,
        "n_estimators": 500,
        "colsample_bytree": 0.8,
        "subsample": 0.8,
        "lambda_l1": 0.1,
        "lambda_l2": 0.1,
    },
    "xgboost": {
        "objective": "reg:squarederror",
        "max_depth": 6,
        "learning_rate": 0.05,
        "n_estimators": 500,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    },
    "transformer": {
        "d_model": 32,
        "nhead": 4,
        "num_layers": 2,
        "dropout": 0.1,
    },
    "lstm": {
        "hidden_size": 64,
        "num_layers": 2,
        "dropout": 0.1,
    },
}


class ModelAgent(BaseAgent):
    """模型空间搜索Agent

    搜索空间:
    - LightGBM (梯度提升树, 默认推荐)
    - XGBoost (梯度提升树)
    - Transformer (注意力机制)
    - LSTM (时序模型)

    输入: ModelSearchRequest
    输出: ModelConfig (最优模型类型+参数+CV分数)
    """

    def __init__(self, config: Optional[dict] = None):
        super().__init__(name="ModelAgent", config=config)

    def run(self, input_data: ModelSearchRequest) -> ModelConfig:
        """执行模型搜索"""
        model_types = input_data.model_types
        best_score = -1.0
        best_config = None

        for mtype in model_types:
            params = DEFAULT_PARAMS.get(mtype, {}).copy()

            try:
                score = self._evaluate_model(mtype, params, input_data)
                logger.info(f"Model {mtype}: CV score={score:.4f}")

                if score > best_score:
                    best_score = score
                    best_config = ModelConfig(
                        model_type=mtype,
                        params=params,
                        cv_score=score,
                        ic_score=score,
                    )
            except Exception as e:
                logger.warning(f"Model {mtype} evaluation failed: {e}")

        if best_config is None:
            logger.warning("No model evaluated successfully, using LightGBM default")
            best_config = ModelConfig(
                model_type="lightgbm",
                params=DEFAULT_PARAMS["lightgbm"],
                cv_score=0.0,
                ic_score=0.0,
            )

        logger.info(f"ModelAgent selected: {best_config.model_type} (score={best_config.cv_score:.4f})")
        return best_config

    def _evaluate_model(
        self, model_type: str, params: dict, request: ModelSearchRequest
    ) -> float:
        """评估单个模型配置 — 返回IC分数"""
        try:
            from src.core import ensure_qlib
            ensure_qlib()

            start = request.start_date or "2020-01-01"
            end = request.end_date or "2024-12-31"
            instruments = request.universe or "all"

            segments = self._compute_segments(start, end)
            train_end = segments["train"][1]

            from qlib.contrib.data.handler import Alpha158
            from qlib.data.dataset import DatasetH
            from qlib.data.dataset.processor import CSZScoreNorm, DropnaLabel, CSRankNorm

            handler = Alpha158(
                instruments=instruments,
                start_time=start,
                end_time=end,
                fit_start_time=start,
                fit_end_time=train_end,
                infer_processors=[CSZScoreNorm(fields_group="feature")],
                learn_processors=[DropnaLabel(), CSRankNorm(fields_group="label")],
                label=["Ref($close, -2) / Ref($close, -1) - 1"],
            )
            dataset = DatasetH(handler=handler, segments=segments)

            if model_type == "lightgbm":
                from qlib.contrib.model.gbdt import LGBModel
                model = LGBModel(
                    loss="mse",
                    num_leaves=params.get("num_leaves", 64),
                    learning_rate=params.get("learning_rate", 0.05),
                    num_boost_round=params.get("n_estimators", 200),
                    colsample_bytree=params.get("colsample_bytree", 0.8),
                    subsample=params.get("subsample", 0.8),
                    lambda_l1=params.get("lambda_l1", 0.1),
                    lambda_l2=params.get("lambda_l2", 0.1),
                )
            elif model_type == "xgboost":
                from qlib.contrib.model.xgboost import XGBModel
                model = XGBModel(
                    objective="reg:squarederror",
                    max_depth=params.get("max_depth", 6),
                    learning_rate=params.get("learning_rate", 0.05),
                    n_estimators=params.get("n_estimators", 200),
                    subsample=params.get("subsample", 0.8),
                    colsample_bytree=params.get("colsample_bytree", 0.8),
                )
            else:
                logger.info(f"Skipping {model_type} (requires torch)")
                return 0.0

            model.fit(dataset)
            pred = model.predict(dataset, segment="valid")

            if pred is None or (hasattr(pred, "empty") and pred.empty):
                logger.warning(f"Empty predictions for {model_type}")
                return 0.0

            import pandas as pd
            if isinstance(pred, pd.Series):
                pred_s = pred
            else:
                pred_s = pred.iloc[:, 0] if hasattr(pred, "iloc") else pd.Series(pred)

            label = dataset.prepare("valid", col_set="label")
            if label is not None and len(label) > 0:
                if isinstance(label, pd.DataFrame):
                    label_s = label.iloc[:, 0]
                else:
                    label_s = label

                common_idx = pred_s.index.intersection(label_s.index)
                if len(common_idx) > 10:
                    ic = pred_s.loc[common_idx].corr(label_s.loc[common_idx])
                    return float(ic) if not pd.isna(ic) else 0.0

            return 0.0
        except Exception as e:
            logger.warning(f"Model evaluation for {model_type} failed: {e}")
            return 0.0

    @staticmethod
    def _compute_segments(start: str, end: str) -> dict:
        """根据起止日期动态划分train/valid/test段 (60%/20%/20%)"""
        import pandas as pd
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
