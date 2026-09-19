from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from src.market_feed.models import MarketTick, SignalEvent


class BaseRealtimeStrategy(ABC):
    """日内高频流式行情策略抽象基类"""

    def __init__(self, strategy_id: str, symbol: str):
        self.strategy_id = strategy_id
        self.symbol = symbol

    @abstractmethod
    def on_tick(self, tick: MarketTick) -> Optional[SignalEvent]:
        """接收行情 Tick 驱动内部逻辑并决定是否输出交易信号"""
        pass

    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """获取当前策略内部状态、水位与持仓"""
        pass

    @abstractmethod
    def reset(self) -> None:
        """重置日内状态"""
        pass
