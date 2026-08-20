"""Model zoo: unified interface to Qlib's model library + MLflow tracking.

Wraps Qlib contrib models (LightGBM, LSTM, Transformer, GATs, etc.)
with experiment tracking via MLflow.
"""
import logging
from typing import Optional, Dict, Any, List
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)

MODEL_REGISTRY: Dict[str, Dict] = {
    "lightgbm": {
        "class": "LGBModel",
        "module": "qlib.contrib.model.gbdt",
        "type": "tree",
        "params": {"loss": "mse", "colsample_bytree": 0.8879, "learning_rate": 0.0421, "subsample": 0.8789, "lambda_l1": 2.7415, "lambda_l2": 5.5623, "max_depth": 8, "num_leaves": 400, "num_threads": 20},
    },
    "linear": {
        "class": "LinearModel",
        "module": "qlib.contrib.model.linear",
        "type": "linear",
        "params": {},
    },
    "xgboost": {
        "class": "XGBModel",
        "module": "qlib.contrib.model.xgboost",
        "type": "tree",
        "params": {"loss": "mse", "colsample_bytree": 0.8879, "learning_rate": 0.0421, "subsample": 0.8789, "lambda_l1": 2.7415, "lambda_l2": 5.5623, "max_depth": 8},
    },
    "catboost": {
        "class": "CatBoostModel",
        "module": "qlib.contrib.model.catboost_model",
        "type": "tree",
        "params": {"iterations": 1000, "learning_rate": 0.03, "depth": 8},
    },
    "lstm": {
        "class": "LSTM",
        "module": "qlib.contrib.model.pytorch_lstm_ts",
        "type": "deep",
        "params": {"d_feat": 6, "hidden_size": 64, "num_layers": 2, "dropout": 0.0, "n_epochs": 50, "lr": 1e-3, "early_stop": 5, "batch_size": 800, "metric": "loss"},
    },
    "alstm": {
        "class": "ALSTM",
        "module": "qlib.contrib.model.pytorch_alstm_ts",
        "type": "deep",
        "params": {"d_feat": 6, "hidden_size": 64, "num_layers": 2, "dropout": 0.0, "n_epochs": 50, "lr": 1e-3, "early_stop": 5, "batch_size": 800, "metric": "loss"},
    },
    "gru": {
        "class": "GRU",
        "module": "qlib.contrib.model.pytorch_gru_ts",
        "type": "deep",
        "params": {"d_feat": 6, "hidden_size": 64, "num_layers": 2, "dropout": 0.0, "n_epochs": 50, "lr": 1e-3, "early_stop": 5, "batch_size": 800, "metric": "loss"},
    },
    "transformer": {
        "class": "TransformerModel",
        "module": "qlib.contrib.model.pytorch_transformer",
        "type": "deep",
        "params": {"d_feat": 6, "embed_dim": 64, "depth": 4, "output_dim": 1, "num_heads": 2, "n_epochs": 50, "lr": 1e-4, "early_stop": 5, "batch_size": 800, "metric": "loss"},
    },
    "gats": {
        "class": "GATs",
        "module": "qlib.contrib.model.pytorch_gats_ts",
        "type": "deep",
        "params": {"d_feat": 6, "hidden_size": 64, "num_layers": 2, "dropout": 0.0, "n_epochs": 50, "lr": 1e-3, "early_stop": 5, "batch_size": 800, "metric": "loss"},
    },
    "dnn": {
        "class": "DNNModelPytorch",
        "module": "qlib.contrib.model.pytorch_nn",
        "type": "deep",
        "params": {"d_feat": 6, "hidden_size": 128, "num_layers": 3, "dropout": 0.2, "n_epochs": 50, "lr": 1e-3, "early_stop": 5, "batch_size": 800, "metric": "loss"},
    },
}


def list_models(model_type: Optional[str] = None) -> List[Dict]:
    """List available models in the zoo.

    Args:
        model_type: Filter by "tree", "linear", or "deep". None = all.
    """
    models = []
    for name, info in MODEL_REGISTRY.items():
        entry = {"name": name, "type": info["type"], "class": info["class"]}
        if model_type and info["type"] != model_type:
            continue
        models.append(entry)
    return models


def create_model(
    model_name: str = "lightgbm",
    override_params: Optional[Dict] = None,
) -> Any:
    """Create a model instance by name.

    Args:
        model_name: Key in MODEL_REGISTRY (lightgbm, lstm, transformer, etc.)
        override_params: Override default hyperparameters.
    """
    if model_name not in MODEL_REGISTRY:
        available = list(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model '{model_name}'. Available: {available}")

    info = MODEL_REGISTRY[model_name]
    params = {**info["params"]}
    if override_params:
        params.update(override_params)

    module_path = info["module"]
    class_name = info["class"]

    import importlib
    module = importlib.import_module(module_path)
    model_cls = getattr(module, class_name)

    try:
        model = model_cls(**params)
    except TypeError:
        model = model_cls()

    logger.info(f"Model created: {model_name} ({class_name}), params={params}")
    return model


def train_model(
    model: Any,
    dataset: Any,
    experiment_name: str = "default",
    run_name: Optional[str] = None,
    use_mlflow: bool = True,
) -> Dict:
    """Train a model and optionally log to MLflow.

    Args:
        model: Model instance from create_model().
        dataset: DatasetH from create_dataset().
        experiment_name: MLflow experiment name.
        run_name: MLflow run name.
        use_mlflow: If True, log params/metrics to MLflow.

    Returns:
        Dict with model, predictions, and MLflow run info.
    """
    import pickle

    train_data = dataset.prepare("train", col_set=["feature", "label"])
    valid_data = dataset.prepare("valid", col_set=["feature", "label"])

    x_train, y_train = train_data
    x_valid, y_valid = valid_data

    mlflow_info = {}
    if use_mlflow:
        try:
            import mlflow
            mlflow.set_experiment(experiment_name)
            mlflow.start_run(run_name=run_name or model.__class__.__name__)

            model_name = model.__class__.__name__
            mlflow.log_params({"model_class": model_name})
            if hasattr(model, "learning_rate"):
                mlflow.log_param("learning_rate", model.learning_rate)
            mlflow_info["experiment"] = experiment_name
            mlflow_info["run_id"] = mlflow.active_run().info.run_id
        except ImportError:
            logger.warning("MLflow not installed, skipping experiment tracking.")
            use_mlflow = False

    logger.info(f"Training {model.__class__.__name__}...")
    model.fit(x_train, y_train)
    logger.info("Training complete.")

    pred = model.predict(dataset=dataset, segment="valid")
    if isinstance(pred, pd.DataFrame):
        pred = pred.iloc[:, 0]

    score = pd.Series(pred.values.ravel(), index=pred.index.get_level_values("instrument"))
    score = score.groupby("datetime").rank(pct=True) - 0.5

    metrics = {"mean_prediction": float(pred.mean()), "pred_std": float(pred.std())}
    if use_mlflow:
        mlflow.log_metrics(metrics)
        mlflow.end_run()

    return {
        "model": model,
        "predictions": pred,
        "metrics": metrics,
        "mlflow": mlflow_info,
    }


def backtest_model(
    model: Any,
    dataset: Any,
    topk: int = 50,
    n_drop: int = 5,
    benchmark: str = "SH000300",
) -> Dict:
    """Run a standard Qlib backtest with the trained model.

    Uses TopkDropoutStrategy with daily rebalancing.
    """
    from qlib.contrib.evaluate import backtest_daily, alpha

    topk_drop = create_topk_dropout_strategy(topk=topk, n_drop=n_drop)
    result = backtest_daily(
        dataset=dataset,
        topk=topk,
        n_drop=n_drop,
    )

    analysis = result["analysis"]
    report_dict = {
        "returns": analysis.get("return", {}),
        "cost": analysis.get("cost", {}),
        "bench": analysis.get("bench", {}),
        "risk": analysis.get("risk", {}),
        "ic": alpha(pred_data=dataset.prepare("test", col_set="label")[1]),
    }

    return {
        "result": result,
        "analysis": report_dict,
        "indicators": result.get("indicators", {}),
    }


def create_topk_dropout_strategy(topk: int = 50, n_drop: int = 5):
    """Create Qlib TopkDropoutStrategy."""
    from qlib.contrib.strategy.signal_strategy import TopkDropoutStrategy
    return TopkDropoutStrategy(topk=topk, n_drop=n_drop)
