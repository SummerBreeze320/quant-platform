from enum import Enum
from datetime import datetime
from typing import Optional, Dict, List
from pydantic import BaseModel, Field

class OrderDirection(str, Enum):
    BUY = "BUY"
    SELL = "SELL"

class OrderType(str, Enum):
    LIMIT = "LIMIT"
    MARKET = "MARKET"

class OrderStatus(str, Enum):
    PENDING = "PENDING"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    REJECTED = "REJECTED"
    CANCELLED = "CANCELLED"

class AlgoType(str, Enum):
    DIRECT = "DIRECT"
    TWAP = "TWAP"
    VWAP = "VWAP"

class Position(BaseModel):
    symbol: str
    total_volume: int = 0
    available_volume: int = 0
    frozen_volume: int = 0
    avg_cost: float = 0.0
    market_value: float = 0.0

class AccountState(BaseModel):
    account_id: str = "default"
    strategy_id: Optional[str] = None
    total_equity: float = 0.0
    available_cash: float = 0.0
    positions: Dict[str, Position] = Field(default_factory=dict)
    updated_at: str = Field(default_factory=lambda: datetime.now().isoformat())

class Order(BaseModel):
    order_id: str
    account_id: str = "default"
    strategy_id: Optional[str] = None
    symbol: str
    direction: OrderDirection
    order_type: OrderType = OrderType.LIMIT
    price: float
    volume: int
    filled_volume: int = 0
    filled_amount: float = 0.0
    status: OrderStatus = OrderStatus.PENDING
    algo_type: AlgoType = AlgoType.DIRECT
    parent_order_id: Optional[str] = None
    reject_reason: Optional[str] = None
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())

class Trade(BaseModel):
    trade_id: str
    order_id: str
    account_id: str = "default"
    symbol: str
    direction: OrderDirection
    price: float
    volume: int
    amount: float
    commission: float = 0.0
    stamp_tax: float = 0.0
    executed_at: str = Field(default_factory=lambda: datetime.now().isoformat())

class RebalancePlan(BaseModel):
    account_id: str
    target_weights: Dict[str, float]
    sell_orders: List[Order] = Field(default_factory=list)
    buy_orders: List[Order] = Field(default_factory=list)
