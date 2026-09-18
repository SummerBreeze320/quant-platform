from datetime import datetime
from enum import Enum
from typing import Dict, List
from pydantic import BaseModel, Field

class SignalDirection(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    FLAT = "FLAT"

class MarketTick(BaseModel):
    """L1/L2 实时行情快照"""
    symbol: str
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    last_price: float
    volume: int                     # 累计成交量
    turnover: float                 # 累计成交额
    bid_prices: List[float] = Field(default_factory=list)  # 买一至买五价格
    bid_volumes: List[int] = Field(default_factory=list)   # 买一至买五委托量
    ask_prices: List[float] = Field(default_factory=list)  # 卖一至卖五价格
    ask_volumes: List[int] = Field(default_factory=list)   # 卖一至卖五委托量

class IndicatorValue(BaseModel):
    """流式指标输出包"""
    symbol: str
    indicator_name: str
    value: float
    details: Dict[str, float] = Field(default_factory=dict)
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

class SignalEvent(BaseModel):
    """实时量化高频交易信号"""
    signal_id: str
    strategy_id: str
    symbol: str
    direction: SignalDirection
    price: float
    volume: int = 100               # 建议触发股数 (按 A 股 100 股整手)
    strength: float = 1.0           # 信号置信度 (0.0 ~ 1.0)
    reason: str                     # 触发原因
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
