import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, Any, List
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from src.common.logger import logger

class BaseModelTrainer(ABC):
    """Abstract base class defining unified interface for quantitative models."""

    def __init__(
        self,
        model_type: str,
        params: Optional[Dict[str, Any]] = None
    ):
        self.model_type = model_type
        self.params = params or {}
        self.model: Any = None
        self.feature_names: List[str] = []

    @abstractmethod
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: Optional[pd.DataFrame] = None,
        y_valid: Optional[pd.Series] = None,
        **kwargs
    ) -> None:
        """Trains the model with training and optional validation sets."""
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generates cross-sectional alpha predictions."""
        pass

    @abstractmethod
    def get_feature_importances(self) -> Dict[str, float]:
        """Returns normalized feature importance dictionary."""
        pass

    def save(self, file_path: str) -> None:
        """Serializes model and metadata to disk."""
        target_path = Path(file_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_type": self.model_type,
            "model": self.model,
            "feature_names": self.feature_names,
            "params": self.params
        }
        with open(target_path, "wb") as f:
            pickle.dump(payload, f)
        logger.info(f"{self.model_type} model saved to {file_path}")

    @classmethod
    def load(cls, file_path: str) -> "BaseModelTrainer":
        """Loads serialized model from disk."""
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        model_type = data.get("model_type", "LightGBM")
        m_type = model_type.upper().strip()
        if m_type in ["ENSEMBLE", "STACKING", "BLENDING"]:
            from src.qlib_engine.ensemble import EnsembleTrainer
            return EnsembleTrainer.load(file_path)
        elif m_type in ["LSTM", "PYTORCH_LSTM"]:
            return PyTorchLSTMTrainer.load(file_path)
        trainer = get_model_trainer(model_type, params=data.get("params"))
        trainer.model = data["model"]
        trainer.feature_names = data["feature_names"]
        logger.info(f"{model_type} model loaded from {file_path}")
        return trainer

class LightGBMTrainer(BaseModelTrainer):
    """LightGBM GBDT implementation for fast quantitative tabular learning."""

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        default_params = {
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "feature_fraction": 0.8,
            "verbose": -1,
            "random_state": 42
        }
        if params:
            default_params.update(params)
        super().__init__(model_type="LightGBM", params=default_params)
        self.model: Optional[lgb.LGBMRegressor] = None

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: Optional[pd.DataFrame] = None,
        y_valid: Optional[pd.Series] = None,
        n_estimators: int = 100,
        **kwargs
    ) -> None:
        self.feature_names = list(X_train.columns)
        self.model = lgb.LGBMRegressor(
            n_estimators=n_estimators,
            **self.params
        )
        eval_X = X_valid if X_valid is not None and y_valid is not None else None
        eval_y = y_valid if X_valid is not None and y_valid is not None else None
        logger.info(f"Fitting LightGBM on {X_train.shape[0]} samples with {X_train.shape[1]} features...")
        self.model.fit(X_train, y_train, eval_X=eval_X, eval_y=eval_y)
        logger.info("LightGBM training completed.")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("LightGBM model is not fitted yet.")
        return self.model.predict(X[self.feature_names])

    def get_feature_importances(self) -> Dict[str, float]:
        if self.model is None:
            return {}
        importances = self.model.feature_importances_
        total = sum(importances) + 1e-8
        return {
            name: round(float(imp / total), 4)
            for name, imp in zip(self.feature_names, importances)
        }

class XGBoostTrainer(BaseModelTrainer):
    """XGBoost GBDT implementation providing robust gradient boosting with regularization."""

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        default_params = {
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "random_state": 42,
            "verbosity": 0
        }
        if params:
            default_params.update(params)
        super().__init__(model_type="XGBoost", params=default_params)
        self.model: Optional[xgb.XGBRegressor] = None

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: Optional[pd.DataFrame] = None,
        y_valid: Optional[pd.Series] = None,
        n_estimators: int = 100,
        **kwargs
    ) -> None:
        self.feature_names = list(X_train.columns)
        self.model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            **self.params
        )
        eval_set = [(X_valid, y_valid)] if X_valid is not None and y_valid is not None else None
        logger.info(f"Fitting XGBoost on {X_train.shape[0]} samples with {X_train.shape[1]} features...")
        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            verbose=False
        )
        logger.info("XGBoost training completed.")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("XGBoost model is not fitted yet.")
        return self.model.predict(X[self.feature_names])

    def get_feature_importances(self) -> Dict[str, float]:
        if self.model is None:
            return {}
        importances = self.model.feature_importances_
        total = sum(importances) + 1e-8
        return {
            name: round(float(imp / total), 4)
            for name, imp in zip(self.feature_names, importances)
        }

class _LSTMRegressorModule(nn.Module):
    """PyTorch LSTM architecture for sequential alpha factor modeling."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 32,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expected input shape: (batch_size, seq_len, input_dim) or (batch_size, input_dim)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out.squeeze(-1)

class PyTorchLSTMTrainer(BaseModelTrainer):
    """PyTorch-based LSTM deep sequence model for financial time-series predictions."""

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        default_params = {
            "hidden_size": 32,
            "num_layers": 2,
            "dropout": 0.1,
            "lr": 0.005,
            "batch_size": 64,
            "epochs": 30,
            "weight_decay": 1e-4
        }
        if params:
            default_params.update(params)
        super().__init__(model_type="LSTM", params=default_params)
        self.model: Optional[_LSTMRegressorModule] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: Optional[pd.DataFrame] = None,
        y_valid: Optional[pd.Series] = None,
        **kwargs
    ) -> None:
        self.feature_names = list(X_train.columns)
        input_dim = len(self.feature_names)
        
        self.model = _LSTMRegressorModule(
            input_dim=input_dim,
            hidden_dim=int(self.params.get("hidden_size", 32)),
            num_layers=int(self.params.get("num_layers", 2)),
            dropout=float(self.params.get("dropout", 0.1))
        ).to(self.device)

        X_tensor = torch.tensor(X_train.values, dtype=torch.float32)
        y_tensor = torch.tensor(y_train.values, dtype=torch.float32)
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(
            dataset,
            batch_size=int(self.params.get("batch_size", 64)),
            shuffle=True
        )

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=float(self.params.get("lr", 0.005)),
            weight_decay=float(self.params.get("weight_decay", 1e-4))
        )
        criterion = nn.MSELoss()

        epochs = int(self.params.get("epochs", 30))
        logger.info(f"Training PyTorch LSTM on {X_train.shape[0]} samples for {epochs} epochs on {self.device}...")
        self.model.train()
        for epoch in range(epochs):
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                output = self.model(batch_x)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()

        logger.info("PyTorch LSTM training completed.")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("PyTorch LSTM model is not fitted yet.")
        self.model.eval()
        X_tensor = torch.tensor(X[self.feature_names].values, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            preds = self.model(X_tensor).cpu().numpy()
        return preds

    def get_feature_importances(self) -> Dict[str, float]:
        if self.model is None:
            return {}
        # Use first-layer LSTM weights magnitude as proxy for input feature importance
        weight_ih = self.model.lstm.weight_ih_l0.abs().mean(dim=0).detach().cpu().numpy()
        total = float(np.sum(weight_ih) + 1e-8)
        return {
            name: round(float(w / total), 4)
            for name, w in zip(self.feature_names, weight_ih)
        }

    def save(self, file_path: str) -> None:
        target_path = Path(file_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_type": "LSTM",
            "feature_names": self.feature_names,
            "params": self.params,
            "state_dict": self.model.state_dict() if self.model else None
        }
        with open(target_path, "wb") as f:
            pickle.dump(payload, f)
        logger.info(f"PyTorch LSTM saved to {file_path}")

    @classmethod
    def load(cls, file_path: str) -> "PyTorchLSTMTrainer":
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        trainer = cls(params=data.get("params"))
        trainer.feature_names = data.get("feature_names", [])
        if data.get("state_dict") is not None and trainer.feature_names:
            trainer.model = _LSTMRegressorModule(
                input_dim=len(trainer.feature_names),
                hidden_dim=int(trainer.params.get("hidden_size", 32)),
                num_layers=int(trainer.params.get("num_layers", 2)),
                dropout=float(trainer.params.get("dropout", 0.1))
            ).to(trainer.device)
            trainer.model.load_state_dict(data["state_dict"])
            trainer.model.eval()
        logger.info(f"PyTorch LSTM loaded from {file_path}")
        return trainer

def get_model_trainer(
    model_type: str = "LightGBM",
    params: Optional[Dict[str, Any]] = None
) -> BaseModelTrainer:
    """Factory function creating quantitative model trainers based on model_type."""
    m_type = model_type.upper().strip()
    if m_type in ["LIGHTGBM", "LGB"]:
        return LightGBMTrainer(params=params)
    elif m_type in ["XGBOOST", "XGB"]:
        return XGBoostTrainer(params=params)
    elif m_type in ["LSTM", "PYTORCH_LSTM", "PYTORCH"]:
        return PyTorchLSTMTrainer(params=params)
    elif m_type in ["ENSEMBLE", "STACKING", "BLENDING"]:
        from src.qlib_engine.ensemble import EnsembleTrainer
        return EnsembleTrainer(params=params)
    else:
        raise ValueError(f"Unsupported model_type: '{model_type}'. Supported: ['LightGBM', 'XGBoost', 'LSTM', 'Ensemble']")

class ModelTrainer:
    """Polymorphic facade preserving backward compatibility with legacy ModelTrainer callers."""

    def __new__(
        cls,
        model_type: str = "LightGBM",
        params: Optional[Dict[str, Any]] = None
    ) -> BaseModelTrainer:
        return get_model_trainer(model_type=model_type, params=params)

    @classmethod
    def load(cls, file_path: str) -> BaseModelTrainer:
        """Dispatches load to appropriate subclass based on serialized metadata."""
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        model_type = data.get("model_type", "LightGBM")
        m_type = model_type.upper().strip()
        if m_type in ["LSTM", "PYTORCH_LSTM"]:
            return PyTorchLSTMTrainer.load(file_path)
        elif m_type in ["ENSEMBLE", "STACKING", "BLENDING"]:
            from src.qlib_engine.ensemble import EnsembleTrainer
            return EnsembleTrainer.load(file_path)
        else:
            return BaseModelTrainer.load(file_path)
