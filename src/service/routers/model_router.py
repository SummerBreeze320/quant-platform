from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from src.common.db import get_db
from src.common.redis_client import get_redis
from src.common.logger import logger
from src.models.model_registry import ModelRegistry
from src.service.schemas.model_schema import ModelTrainRequest, ModelResponse
from src.qlib_engine.data_handler import DataHandler
from src.qlib_engine.model_trainer import get_model_trainer

router = APIRouter(prefix="/model", tags=["AI Models"])

@router.get("/list", response_model=List[ModelResponse])
def list_models(db: Session = Depends(get_db)):
    """Lists all trained quantitative models in the registry."""
    return db.query(ModelRegistry).order_by(ModelRegistry.id.desc()).all()

@router.post("/train", response_model=ModelResponse)
def train_model(request: ModelTrainRequest, db: Session = Depends(get_db)):
    """Registers, trains, saves and activates a new quantitative model."""
    existing = db.query(ModelRegistry).filter_by(model_name=request.model_name).first()
    if existing:
        raise HTTPException(status_code=400, detail="Model name already exists.")

    model_entry = ModelRegistry(
        model_name=request.model_name,
        model_type=request.model_type,
        feature_set=request.feature_set,
        train_start=request.train_start,
        train_end=request.train_end,
        model_path=f"data/models/{request.model_name}.pkl",
        metrics={"status": "TRAINING"},
        is_active=False
    )
    db.add(model_entry)
    db.commit()
    db.refresh(model_entry)

    try:
        X_train, y_train, feature_names = DataHandler.prepare_training_data(
            feature_set=request.feature_set,
            train_start=request.train_start,
            train_end=request.train_end,
        )
        trainer = get_model_trainer(request.model_type, params=request.params)
        trainer.fit(X_train, y_train)
        trainer.save(model_entry.model_path)

        model_entry.metrics = {
            "status": "TRAINED",
            "n_samples": int(len(X_train)),
            "n_features": int(len(feature_names)),
            "feature_names": feature_names,
            "model_type": request.model_type,
        }
        model_entry.is_active = True

        db.query(ModelRegistry).filter(
            ModelRegistry.id != model_entry.id,
            ModelRegistry.is_active == True,
        ).update({"is_active": False})
        db.commit()
        db.refresh(model_entry)
        logger.info(f"Model '{request.model_name}' trained and saved to {model_entry.model_path}.")
    except Exception as e:
        logger.error(f"Model training failed: {e}")
        model_entry.metrics = {"status": "FAILED", "error": str(e)}
        db.commit()
        db.refresh(model_entry)
        raise HTTPException(status_code=500, detail=f"Training failed: {e}")

    return model_entry

@router.get("/predict/latest")
def get_latest_predictions():
    """Fetches the latest cached market cross-sectional predictions from Redis."""
    redis_client = get_redis()
    pred_data = redis_client.get_json("predictions:latest")
    if not pred_data:
        return {"date": "latest", "predictions": []}
    return pred_data
