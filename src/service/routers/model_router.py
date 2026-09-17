from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from src.common.db import get_db
from src.common.redis_client import get_redis
from src.models.model_registry import ModelRegistry
from src.service.schemas.model_schema import ModelTrainRequest, ModelResponse

router = APIRouter(prefix="/model", tags=["AI Models"])

@router.get("/list", response_model=List[ModelResponse])
def list_models(db: Session = Depends(get_db)):
    """Lists all trained quantitative models in the registry."""
    return db.query(ModelRegistry).order_by(ModelRegistry.id.desc()).all()

@router.post("/train", response_model=ModelResponse)
def train_model(request: ModelTrainRequest, db: Session = Depends(get_db)):
    """Registers and schedules a new LightGBM model training job."""
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
        metrics={"status": "INITIALIZED"},
        is_active=False
    )
    db.add(model_entry)
    db.commit()
    db.refresh(model_entry)
    return model_entry

@router.get("/predict/latest")
def get_latest_predictions():
    """Fetches the latest cached market cross-sectional predictions from Redis."""
    redis_client = get_redis()
    # Try finding today's or default predictions
    pred_data = redis_client.get_json("predictions:latest")
    if not pred_data:
        return {"date": "latest", "predictions": []}
    return pred_data
