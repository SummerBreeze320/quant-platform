import pytest
import numpy as np
import pandas as pd
from src.qlib_engine.ensemble import EnsembleTrainer
from src.qlib_engine.model_trainer import get_model_trainer, ModelTrainer

@pytest.fixture
def sample_data():
    np.random.seed(42)
    n = 120
    X_train = pd.DataFrame(np.random.randn(n, 4), columns=["f1", "f2", "f3", "f4"])
    y_train = pd.Series(X_train["f1"] * 2.0 + X_train["f2"] * 1.0 + np.random.randn(n) * 0.1)

    X_valid = pd.DataFrame(np.random.randn(40, 4), columns=["f1", "f2", "f3", "f4"])
    y_valid = pd.Series(X_valid["f1"] * 2.0 + X_valid["f2"] * 1.0 + np.random.randn(40) * 0.1)

    return X_train, y_train, X_valid, y_valid

def test_ensemble_weighted_average(sample_data):
    X_train, y_train, X_valid, y_valid = sample_data
    trainer = EnsembleTrainer(
        method="weighted",
        base_models=["LightGBM", "XGBoost"],
        weights={"LightGBM": 0.5, "XGBoost": 0.5}
    )
    trainer.fit(X_train, y_train, X_valid=X_valid, y_valid=y_valid, n_estimators=30)
    preds = trainer.predict(X_valid)

    assert len(preds) == len(X_valid)
    assert not np.isnan(preds).any()
    # Correlation with true target should be high
    corr = np.corrcoef(preds, y_valid)[0, 1]
    assert corr > 0.7

    importances = trainer.get_feature_importances()
    assert "f1" in importances
    assert importances["f1"] > importances["f3"]

def test_ensemble_rank_fusion(sample_data):
    X_train, y_train, X_valid, y_valid = sample_data
    trainer = EnsembleTrainer(
        method="rank",
        base_models=["LightGBM", "XGBoost"]
    )
    trainer.fit(X_train, y_train, X_valid=X_valid, y_valid=y_valid, n_estimators=30)
    preds = trainer.predict(X_valid)

    assert len(preds) == len(X_valid)
    # Rank normalized values should fall strictly in [0, 1]
    assert np.all(preds >= 0.0)
    assert np.all(preds <= 1.0)
    corr = np.corrcoef(preds, y_valid)[0, 1]
    assert corr > 0.65

def test_ensemble_stacking_meta_learner(sample_data):
    X_train, y_train, X_valid, y_valid = sample_data
    trainer = EnsembleTrainer(
        method="stacking",
        base_models=["LightGBM", "XGBoost"]
    )
    trainer.fit(X_train, y_train, X_valid=X_valid, y_valid=y_valid, n_estimators=30)
    preds = trainer.predict(X_valid)

    assert len(preds) == len(X_valid)
    assert not np.isnan(preds).any()
    corr = np.corrcoef(preds, y_valid)[0, 1]
    assert corr > 0.7

def test_ensemble_save_and_load(sample_data, tmp_path):
    X_train, y_train, X_valid, y_valid = sample_data
    trainer = EnsembleTrainer(
        method="weighted",
        base_models=["LightGBM", "XGBoost"]
    )
    trainer.fit(X_train, y_train, X_valid=X_valid, y_valid=y_valid, n_estimators=20)
    preds = trainer.predict(X_valid)

    save_path = tmp_path / "ensemble_model.pkl"
    trainer.save(str(save_path))
    assert save_path.exists()

    loaded = EnsembleTrainer.load(str(save_path))
    loaded_preds = loaded.predict(X_valid)
    np.testing.assert_allclose(preds, loaded_preds, rtol=1e-4)

def test_ensemble_factory_dispatch():
    trainer = get_model_trainer("Ensemble", params={"method": "rank", "base_models": ["LightGBM", "XGBoost"]})
    assert isinstance(trainer, EnsembleTrainer)
    assert trainer.method == "rank"

    # Facade via ModelTrainer
    facade_trainer = ModelTrainer(model_type="Ensemble", params={"method": "stacking"})
    assert isinstance(facade_trainer, EnsembleTrainer)
    assert facade_trainer.method == "stacking"
