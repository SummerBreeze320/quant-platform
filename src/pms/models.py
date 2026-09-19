from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from src.execution_engine.models import Position

class StrategyType(str, Enum):
    ALPHA = "ALPHA"
    INDEX_ENHANCEMENT = "INDEX_ENHANCEMENT"
    SECTOR_ROTATION = "SECTOR_ROTATION"
    MOMENTUM = "MOMENTUM"

class AllocationMethod(str, Enum):
    EQUAL_WEIGHT = "EQUAL_WEIGHT"
    FIXED_BUDGET = "FIXED_BUDGET"
    RISK_PARITY = "RISK_PARITY"
    SHARPE_WEIGHTED = "SHARPE_WEIGHTED"

class StrategyAccount(BaseModel):
    strategy_id: str
    name: str
    strategy_type: StrategyType = StrategyType.ALPHA
    allocated_budget: float = 0.0
    current_cash: float = 0.0
    total_equity: float = 0.0
    positions: Dict[str, Position] = Field(default_factory=dict)
    is_active: bool = True
    updated_at: str = Field(default_factory=lambda: datetime.now().isoformat())

class MasterAccount(BaseModel):
    master_id: str = "master_default"
    reserve_cash: float = 0.0
    total_equity: float = 0.0
    strategies: Dict[str, StrategyAccount] = Field(default_factory=dict)
    updated_at: str = Field(default_factory=lambda: datetime.now().isoformat())

class CashTransfer(BaseModel):
    strategy_id: str
    transfer_amount: float
    reason: str = ""
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

class CapitalAllocationPlan(BaseModel):
    method: AllocationMethod
    target_ratios: Dict[str, float]
    transfers: List[CashTransfer] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())

class ConsolidatedPosition(BaseModel):
    symbol: str
    total_volume: int = 0
    total_market_value: float = 0.0
    total_cost: float = 0.0
    unrealized_pnl: float = 0.0
    weight_in_master: float = 0.0
    contributing_strategies: List[str] = Field(default_factory=list)

class ConsolidatedPortfolio(BaseModel):
    master_id: str = "master_default"
    total_equity: float = 0.0
    total_cash: float = 0.0
    total_market_value: float = 0.0
    leverage_ratio: float = 0.0
    positions: Dict[str, ConsolidatedPosition] = Field(default_factory=dict)
    max_stock_concentration: float = 0.0
    top5_concentration: float = 0.0
    circuit_breaker_level: int = 0
    macro_alerts: List[str] = Field(default_factory=list)
    updated_at: str = Field(default_factory=lambda: datetime.now().isoformat())

class StrategyPerformance(BaseModel):
    strategy_id: str
    name: str
    allocated_budget: float
    current_equity: float
    cumulative_return: float
    contribution_to_master: float

class DailyNavRecord(BaseModel):
    """逐日单位净值与收益率快照"""
    date: str
    account_id: str
    equity: float
    nav: float = 1.0
    daily_return: float = 0.0
    benchmark_return: float = 0.0
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
