from typing import Optional, Dict, Any
from pydantic import BaseModel, ConfigDict

class FactorRegisterRequest(BaseModel):
    name: str
    expression: str
    category: str = "custom"
    description: Optional[str] = None

class FactorResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    expression: str
    category: str
    description: Optional[str] = None
    ic_mean: Optional[float] = None
    icir: Optional[float] = None
    t_stat: Optional[float] = None
    created_by: str
    is_active: bool

class FactorMetricsResponse(BaseModel):
    id: int
    name: str
    ic_mean: Optional[float] = None
    icir: Optional[float] = None
    t_stat: Optional[float] = None
    extra_metrics: Optional[Dict[str, Any]] = None
