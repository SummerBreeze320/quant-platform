from enum import Enum, IntEnum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

class RiskAction(str, Enum):
    PASS = "PASS"
    WARN = "WARN"
    REJECT = "REJECT"

class CircuitBreakerLevel(IntEnum):
    NORMAL = 0
    YELLOW_WARN = 1
    ORANGE_RESTRICT_BUY = 2
    RED_HALT = 3

class AlertLevel(str, Enum):
    INFO = "INFO"
    WARN = "WARN"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class OrderRiskRequest(BaseModel):
    account_id: str = "default"
    order_id: Optional[str] = None
    symbol: str
    direction: str
    price: float
    volume: int
    current_position: int = 0
    total_equity: float
    available_cash: float
    adv_20: Optional[float] = None
    timestamp: Optional[str] = None

class OrderRiskResult(BaseModel):
    passed: bool
    action: RiskAction
    violations: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    metrics: Dict[str, Any] = Field(default_factory=dict)

class PortfolioRiskRequest(BaseModel):
    account_id: str = "default"
    total_equity: float
    positions: Dict[str, float] = Field(default_factory=dict)
    high_watermark: Optional[float] = None
    benchmark_weights: Optional[Dict[str, float]] = None
    factor_exposures: Optional[Dict[str, Dict[str, float]]] = None

class PortfolioRiskResult(BaseModel):
    account_id: str
    current_equity: float
    high_watermark: float
    drawdown: float
    circuit_breaker_level: CircuitBreakerLevel
    max_stock_weight: float
    top5_weight: float
    leverage_ratio: float
    active_risk: Optional[Dict[str, float]] = None
    alerts: List[str] = Field(default_factory=list)

class CircuitBreakerState(BaseModel):
    account_id: str
    level: CircuitBreakerLevel
    high_watermark: float
    current_equity: float
    max_drawdown: float
    last_triggered_at: Optional[str] = None
    trigger_reason: Optional[str] = None

class RiskAlertEvent(BaseModel):
    event_id: str
    timestamp: str
    level: AlertLevel
    account_id: str
    rule_name: str
    message: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
