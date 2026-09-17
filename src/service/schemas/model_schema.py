from typing import Optional, List, Dict, Any
from pydantic import BaseModel, ConfigDict

class ModelTrainRequest(BaseModel):
    model_name: str
    model_type: str = "LightGBM"
    feature_set: List[str]
    train_start: str = "2020-01-01"
    train_end: str = "2025-12-31"
    params: Optional[Dict[str, Any]] = None

class ModelResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    model_name: str
    model_type: str
    feature_set: List[str]
    train_start: str
    train_end: str
    model_path: str
    metrics: Optional[Dict[str, Any]] = None
    is_active: bool

class BacktestRunRequest(BaseModel):
    model_id: Optional[int] = None
    strategy_name: str = "Top30_EqualWeight"
    top_k: int = 30
    benchmark: str = "000300.SH"
    commission_rate: float = 0.0002
    stamp_tax_rate: float = 0.0005
    slippage: float = 0.0005
    neutralize: bool = False
    industry_neutral_allocation: bool = False
    use_convex_optimizer: bool = False
    risk_aversion: float = 1.0
    max_stock_weight: float = 0.05
    industry_tolerance: float = 0.01
    style_tolerance: float = 0.10
    turnover_penalty: float = 0.0

class BacktestResponse(BaseModel):
    strategy_name: str
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    n_trading_days: int
    cumulative_returns: Dict[str, float]

class ConvexOptimizationRequest(BaseModel):
    alpha_scores: Dict[str, float]
    exposures: Dict[str, Dict[str, float]]
    factor_cov: Optional[Dict[str, Dict[str, float]]] = None
    specific_var: Optional[Dict[str, float]] = None
    benchmark_weights: Optional[Dict[str, float]] = None
    prev_weights: Optional[Dict[str, float]] = None
    risk_aversion: float = 1.0
    max_stock_weight: float = 0.05
    industry_tolerance: Optional[float] = 0.01
    style_tolerance: Optional[float] = 0.10
    turnover_penalty: float = 0.0
    industry_factors: Optional[List[str]] = None
    style_factors: Optional[List[str]] = None

class ConvexOptimizationResponse(BaseModel):
    status: str
    expected_alpha: float
    predicted_tracking_error: float
    turnover: float
    active_exposures: Dict[str, float]
    weights: Dict[str, float]
