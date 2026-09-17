from pathlib import Path
import pickle
from typing import Optional, Dict, Any, List, Union
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import Ridge
from src.common.logger import logger
from src.qlib_engine.model_trainer import BaseModelTrainer, get_model_trainer

class EnsembleTrainer(BaseModelTrainer):
    """
    Multi-model Ensemble Trainer for quantitative investment.
    Supports:
      - 'weighted': Weighted average of base model predictions.
      - 'rank': Cross-sectional percentile ranking fusion (scale-invariant, robust).
      - 'stacking': Two-stage Meta-Learner (Ridge Regression) trained on validation predictions.
    """

    def __init__(
        self,
        method: str = "weighted",
        base_models: Optional[List[Union[str, BaseModelTrainer, Dict[str, Any]]]] = None,
        weights: Optional[Dict[str, float]] = None,
        meta_learner_params: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None
    ):
        params_dict = params or {}
        # Allow params dict to override method and base_models
        if "method" in params_dict:
            method = params_dict["method"]
        if "base_models" in params_dict:
            base_models = params_dict["base_models"]
        if "weights" in params_dict:
            weights = params_dict["weights"]

        super().__init__(model_type="Ensemble", params=params_dict)
        self.method = method.lower()
        if self.method not in ["weighted", "rank", "stacking"]:
            raise ValueError(f"Unsupported ensemble method: '{method}'. Supported: ['weighted', 'rank', 'stacking']")

        self.base_model_specs = base_models or ["LightGBM", "XGBoost"]
        self.base_trainers: List[BaseModelTrainer] = []
        self.weights = weights or {}
        self.meta_learner: Optional[Ridge] = None
        self.meta_learner_params = meta_learner_params or {"alpha": 1.0}

    def _init_base_trainers(self):
        self.base_trainers = []
        for spec in self.base_model_specs:
            if isinstance(spec, str):
                self.base_trainers.append(get_model_trainer(spec))
            elif isinstance(spec, BaseModelTrainer):
                self.base_trainers.append(spec)
            elif isinstance(spec, dict):
                m_type = spec.get("model_type", "LightGBM")
                m_params = spec.get("params", {})
                self.base_trainers.append(get_model_trainer(m_type, params=m_params))
            else:
                raise ValueError(f"Invalid base model spec: {spec}")

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: Optional[pd.DataFrame] = None,
        y_valid: Optional[pd.Series] = None,
        **kwargs
    ) -> None:
        self.feature_names = list(X_train.columns)
        self._init_base_trainers()
        n_models = len(self.base_trainers)
        logger.info(f"Training Ensemble with method='{self.method}' across {n_models} base models...")

        # If weights not fully defined, use equal weighting
        if not self.weights or len(self.weights) != n_models:
            equal_w = 1.0 / n_models
            self.weights = {trainer.model_type: equal_w for trainer in self.base_trainers}
        else:
            total_w = sum(self.weights.values())
            self.weights = {k: v / total_w for k, v in self.weights.items()}

        valid_preds = []
        for trainer in self.base_trainers:
            logger.info(f"Fitting base trainer: {trainer.model_type}...")
            trainer.fit(X_train, y_train, X_valid=X_valid, y_valid=y_valid, **kwargs)
            if self.method == "stacking":
                eval_X = X_valid if X_valid is not None else X_train
                pred = trainer.predict(eval_X)
                valid_preds.append(pred)

        if self.method == "stacking":
            meta_X = np.column_stack(valid_preds)
            meta_y = y_valid.values if y_valid is not None else y_train.values
            logger.info(f"Fitting Stacking Meta-Learner (Ridge) on shape {meta_X.shape}...")
            self.meta_learner = Ridge(**self.meta_learner_params)
            self.meta_learner.fit(meta_X, meta_y)

        logger.info("Ensemble training completed.")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.base_trainers:
            raise ValueError("Ensemble is not fitted yet.")

        preds_list = []
        for trainer in self.base_trainers:
            p = trainer.predict(X)
            if self.method == "rank":
                # Convert to percentile rank in [0, 1]
                p = (rankdata(p) - 1.0) / (len(p) - 1.0 + 1e-8)
            preds_list.append(p)

        if self.method in ["weighted", "rank"]:
            final_pred = np.zeros_like(preds_list[0], dtype=float)
            for trainer, p in zip(self.base_trainers, preds_list):
                w = self.weights.get(trainer.model_type, 1.0 / len(self.base_trainers))
                final_pred += w * p
            return final_pred
        elif self.method == "stacking":
            meta_X = np.column_stack(preds_list)
            return self.meta_learner.predict(meta_X)

    def get_feature_importances(self) -> Dict[str, float]:
        if not self.base_trainers:
            return {}
        agg_importances: Dict[str, float] = {feat: 0.0 for feat in self.feature_names}

        for trainer in self.base_trainers:
            w = self.weights.get(trainer.model_type, 1.0 / len(self.base_trainers))
            sub_imps = trainer.get_feature_importances()
            for feat, imp in sub_imps.items():
                if feat in agg_importances:
                    agg_importances[feat] += w * imp

        total_sum = sum(agg_importances.values()) + 1e-8
        return {
            feat: round(float(v / total_sum), 4)
            for feat, v in agg_importances.items()
        }

    def save(self, file_path: str) -> None:
        target_path = Path(file_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_type": "Ensemble",
            "method": self.method,
            "feature_names": self.feature_names,
            "params": self.params,
            "weights": self.weights,
            "meta_learner": self.meta_learner,
            "meta_learner_params": self.meta_learner_params,
            "base_trainers": self.base_trainers
        }
        with open(target_path, "wb") as f:
            pickle.dump(payload, f)
        logger.info(f"Ensemble model saved to {file_path}")

    @classmethod
    def load(cls, file_path: str) -> "EnsembleTrainer":
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        trainer = cls(
            method=data.get("method", "weighted"),
            weights=data.get("weights"),
            meta_learner_params=data.get("meta_learner_params"),
            params=data.get("params")
        )
        trainer.feature_names = data.get("feature_names", [])
        trainer.base_trainers = data.get("base_trainers", [])
        trainer.meta_learner = data.get("meta_learner")
        logger.info(f"Ensemble model loaded from {file_path}")
        return trainer
