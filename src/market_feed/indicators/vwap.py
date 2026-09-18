from typing import Optional
from src.market_feed.models import MarketTick, IndicatorValue
from src.market_feed.indicators.base import BaseStreamingIndicator

class StreamingVWAP(BaseStreamingIndicator):
    """增量流式 VWAP (成交量加权平均价)"""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.total_volume = 0.0
        self.total_turnover = 0.0
        self.last_raw_volume: Optional[int] = None
        self.last_raw_turnover: Optional[float] = None
        self.current_vwap = 0.0

    def update(self, tick: MarketTick) -> IndicatorValue:
        if self.last_raw_volume is None or self.last_raw_turnover is None:
            delta_vol = tick.volume if tick.volume > 0 else 1
            delta_to = tick.turnover if tick.turnover > 0 else (tick.last_price * delta_vol)
        else:
            delta_vol = max(0, tick.volume - self.last_raw_volume)
            delta_to = max(0.0, tick.turnover - self.last_raw_turnover)

        self.last_raw_volume = tick.volume
        self.last_raw_turnover = tick.turnover

        self.total_volume += delta_vol
        self.total_turnover += delta_to

        if self.total_volume > 0:
            self.current_vwap = round(self.total_turnover / self.total_volume, 4)
        else:
            self.current_vwap = tick.last_price

        bias = round((tick.last_price - self.current_vwap) / self.current_vwap, 6) if self.current_vwap > 0 else 0.0

        return IndicatorValue(
            symbol=tick.symbol,
            indicator_name="VWAP",
            value=self.current_vwap,
            details={"vwap": self.current_vwap, "bias": bias},
            timestamp=tick.timestamp,
        )
