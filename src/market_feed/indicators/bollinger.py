import math
from collections import deque
from src.market_feed.models import MarketTick, IndicatorValue
from src.market_feed.indicators.base import BaseStreamingIndicator

class StreamingBollingerBands(BaseStreamingIndicator):
    """滑动窗口实时布林带 (O(1) 增量维护)"""

    def __init__(self, window: int = 20, num_std: float = 2.0):
        self.window = window
        self.num_std = num_std
        self.reset()

    def reset(self) -> None:
        self.prices = deque(maxlen=self.window)

    def update(self, tick: MarketTick) -> IndicatorValue:
        self.prices.append(tick.last_price)
        n = len(self.prices)
        mean = sum(self.prices) / n

        if n > 1:
            variance = sum((p - mean) ** 2 for p in self.prices) / (n - 1)
            std = math.sqrt(variance)
        else:
            std = 0.0

        if std > 0:
            upper = round(mean + self.num_std * std, 4)
            lower = round(mean - self.num_std * std, 4)
            pct_b = round((tick.last_price - lower) / (upper - lower), 4) if upper > lower else 0.5
        else:
            upper = round(mean, 4)
            lower = round(mean, 4)
            pct_b = 0.5

        band_width = round((upper - lower) / mean, 6) if mean > 0 else 0.0

        return IndicatorValue(
            symbol=tick.symbol,
            indicator_name="BOLLINGER",
            value=round(mean, 4),
            details={
                "mid": round(mean, 4),
                "upper": upper,
                "lower": lower,
                "std": round(std, 4),
                "pct_b": pct_b,
                "band_width": band_width,
            },
            timestamp=tick.timestamp,
        )
