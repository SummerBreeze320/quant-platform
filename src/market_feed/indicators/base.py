from abc import ABC, abstractmethod
from src.market_feed.models import MarketTick, IndicatorValue

class BaseStreamingIndicator(ABC):
    @abstractmethod
    def update(self, tick: MarketTick) -> IndicatorValue:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass
