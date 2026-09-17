from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from src.common.db import get_db
from src.models.factor import FactorMetadata
from src.service.schemas.factor_schema import (
    FactorRegisterRequest,
    FactorResponse,
    FactorMetricsResponse,
)

router = APIRouter(prefix="/factor", tags=["Factor Catalog"])

@router.get("/list", response_model=List[FactorResponse])
def list_factors(
    category: Optional[str] = None,
    active_only: bool = True,
    db: Session = Depends(get_db)
):
    """Lists registered Alpha factors in the system."""
    query = db.query(FactorMetadata)
    if active_only:
        query = query.filter(FactorMetadata.is_active == True)
    if category:
        query = query.filter(FactorMetadata.category == category)
    return query.order_by(FactorMetadata.id.desc()).all()

@router.post("/register", response_model=FactorResponse)
def register_factor(
    request: FactorRegisterRequest,
    db: Session = Depends(get_db)
):
    """Manually registers a new Alpha factor formula into the catalog."""
    existing = db.query(FactorMetadata).filter_by(name=request.name).first()
    if existing:
        raise HTTPException(status_code=400, detail=f"Factor with name '{request.name}' already exists.")

    factor = FactorMetadata(
        name=request.name,
        expression=request.expression,
        category=request.category,
        description=request.description,
        created_by="manual",
        is_active=True
    )
    db.add(factor)
    db.commit()
    db.refresh(factor)
    return factor

@router.get("/{factor_id}/metrics", response_model=FactorMetricsResponse)
def get_factor_metrics(
    factor_id: int,
    db: Session = Depends(get_db)
):
    """Retrieves detailed performance metrics for a specific factor."""
    factor = db.query(FactorMetadata).filter_by(id=factor_id).first()
    if not factor:
        raise HTTPException(status_code=404, detail="Factor not found.")
    
    return {
        "id": factor.id,
        "name": factor.name,
        "ic_mean": factor.ic_mean,
        "icir": factor.icir,
        "t_stat": factor.t_stat,
        "extra_metrics": factor.extra_metrics
    }
