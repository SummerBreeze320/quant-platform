import pytest
import numpy as np
import pandas as pd
from src.qlib_engine.factor_handler import FactorHandler
from src.qlib_engine.model_trainer import (
    ModelTrainer,
    BaseModelTrainer,
    LightGBMTrainer,
    XGBoostTrainer,
    PyTorchLSTMTrainer,
    get_model_trainer
)
from src.qlib_engine.backtest import BacktestEngine

def test_factor_metrics_calculation():
    # Synthetic factor values and forward returns
    np.random.seed(42)
    n = 100
    dates = ["2026-09-01"] * 50 + ["2026-09-02"] * 50
    symbols = [f"SZ{i:06d}" for i in range(50)] * 2
    
    # Factor is positively correlated with return
    factors = np.random.randn(n)
    returns = factors * 0.1 + np.random.randn(n) * 0.05
    
    df = pd.DataFrame({
        "date": dates,
        "symbol": symbols,
        "factor": factors,
        "label": returns
    })
    
    metrics = FactorHandler.evaluate_factor_performance(df, factor_col="factor", label_col="label")
    assert "rank_ic" in metrics
    assert "icir" in metrics
    assert metrics["rank_ic"] > 0.3  # Strong positive correlation

def test_model_trainer_fit_and_predict(tmp_path):
    np.random.seed(42)
    X_train = pd.DataFrame(np.random.randn(100, 4), columns=["f1", "f2", "f3", "f4"])
    y_train = pd.Series(X_train["f1"] * 2 + np.random.randn(100) * 0.1)
    
    trainer = ModelTrainer(model_type="LightGBM")
    trainer.fit(X_train, y_train)
    
    preds = trainer.predict(X_train)
    assert len(preds) == 100
    assert np.corrcoef(preds, y_train)[0, 1] > 0.7
    
    # Test save and load
    save_path = tmp_path / "model.pkl"
    trainer.save(str(save_path))
    assert save_path.exists()
    
    loaded_trainer = ModelTrainer.load(str(save_path))
    loaded_preds = loaded_trainer.predict(X_train)
    np.testing.assert_allclose(preds, loaded_preds)

def test_backtest_engine_simulation():
    # 5 dates, 10 stocks
    dates = pd.date_range("2026-09-01", periods=5).strftime("%Y-%m-%d").tolist()
    symbols = [f"SZ{i:06d}" for i in range(10)]
    
    records = []
    for d in dates:
        for s in symbols:
            records.append({
                "date": d,
                "symbol": s,
                "score": float(int(s[-2:])),  # higher index -> higher score
                "ret": 0.01  # constant 1% daily return
            })
    pred_df = pd.DataFrame(records)
    
    engine = BacktestEngine(top_k=3, commission_rate=0.0002, stamp_tax_rate=0.0005)
    report = engine.run_backtest(pred_df)
    
    assert "annualized_return" in report
    assert "sharpe_ratio" in report
    assert "max_drawdown" in report
    assert report["annualized_return"] > 0
    assert len(report["cumulative_returns"]) == 5

def test_xgboost_trainer_fit_and_predict(tmp_path):
    import xgboost as xgb
    np.random.seed(42)
    X_train = pd.DataFrame(np.random.randn(100, 4), columns=["f1", "f2", "f3", "f4"])
    y_train = pd.Series(X_train["f1"] * 2 + np.random.randn(100) * 0.1)
    
    trainer = ModelTrainer(model_type="XGBoost")
    assert isinstance(trainer, XGBoostTrainer)
    trainer.fit(X_train, y_train, n_estimators=50)
    assert isinstance(trainer.model, xgb.XGBRegressor)
    
    preds = trainer.predict(X_train)
    assert len(preds) == 100
    assert np.corrcoef(preds, y_train)[0, 1] > 0.7
    
    importances = trainer.get_feature_importances()
    assert "f1" in importances
    assert importances["f1"] > importances["f2"]
    
    save_path = tmp_path / "xgb_model.pkl"
    trainer.save(str(save_path))
    assert save_path.exists()
    
    loaded = ModelTrainer.load(str(save_path))
    assert isinstance(loaded, XGBoostTrainer)
    loaded_preds = loaded.predict(X_train)
    np.testing.assert_allclose(preds, loaded_preds, rtol=1e-4)

def test_pytorch_lstm_trainer_fit_and_predict(tmp_path):
    import torch.nn as nn
    np.random.seed(42)
    X_train = pd.DataFrame(np.random.randn(120, 4), columns=["f1", "f2", "f3", "f4"])
    y_train = pd.Series(X_train["f1"] * 1.5 + X_train["f2"] * 0.5 + np.random.randn(120) * 0.1)
    
    trainer = ModelTrainer(model_type="LSTM", params={"hidden_size": 16, "num_layers": 1, "epochs": 20, "lr": 0.02})
    assert isinstance(trainer, PyTorchLSTMTrainer)
    trainer.fit(X_train, y_train)
    assert isinstance(trainer.model, nn.Module)
    
    preds = trainer.predict(X_train)
    assert len(preds) == 120
    assert not np.isnan(preds).any()
    
    save_path = tmp_path / "lstm_model.pkl"
    trainer.save(str(save_path))
    assert save_path.exists()
    
    loaded = ModelTrainer.load(str(save_path))
    assert isinstance(loaded, PyTorchLSTMTrainer)
    loaded_preds = loaded.predict(X_train)
    np.testing.assert_allclose(preds, loaded_preds, atol=1e-4)

def test_model_trainer_factory_dispatch():
    lgb_t = get_model_trainer(model_type="LightGBM")
    xgb_t = get_model_trainer(model_type="XGBoost")
    lstm_t = get_model_trainer(model_type="LSTM")
    assert isinstance(lgb_t, LightGBMTrainer)
    assert isinstance(xgb_t, XGBoostTrainer)
    assert isinstance(lstm_t, PyTorchLSTMTrainer)
