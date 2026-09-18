from typing import Optional, List
from src.market_feed.models import MarketTick, IndicatorValue
from src.market_feed.indicators.base import BaseStreamingIndicator

class OrderFlowImbalance(BaseStreamingIndicator):
    """微观五档盘口订单流不平衡度 (Cont et al. OFI)"""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.prev_bid_price: Optional[float] = None
        self.prev_bid_volume: Optional[int] = None
        self.prev_ask_price: Optional[float] = None
        self.prev_ask_volume: Optional[int] = None

    def update(self, tick: MarketTick) -> IndicatorValue:
        b_p = tick.bid_prices[0] if tick.bid_prices else tick.last_price
        b_v = tick.bid_volumes[0] if tick.bid_volumes else 0
        a_p = tick.ask_prices[0] if tick.ask_prices else tick.last_price
        a_v = tick.ask_volumes[0] if tick.ask_volumes else 0

        if self.prev_bid_price is None or self.prev_ask_price is None:
            self.prev_bid_price = b_p
            self.prev_bid_volume = b_v
            self.prev_ask_price = a_p
            self.prev_ask_volume = a_v
            return IndicatorValue(
                symbol=tick.symbol,
                indicator_name="OFI",
                value=0.0,
                details={"ofi_raw": 0.0, "normalized_ofi": 0.0},
                timestamp=tick.timestamp
            )

        # 买方挂单流变动
        if b_p > self.prev_bid_price:
            delta_wb = b_v
        elif b_p == self.prev_bid_price:
            delta_wb = b_v - self.prev_bid_volume
        else:
            delta_wb = -self.prev_bid_volume

        # 卖方挂单流变动
        if a_p > self.prev_ask_price:
            delta_wa = -self.prev_ask_volume
        elif a_p == self.prev_ask_price:
            delta_wa = a_v - self.prev_ask_volume
        else:
            delta_wa = a_v

        ofi_raw = float(delta_wb - delta_wa)
        total_depth = max(1.0, float(b_v + a_v))
        norm_ofi = round(max(-1.0, min(1.0, ofi_raw / total_depth)), 4)

        self.prev_bid_price = b_p
        self.prev_bid_volume = b_v
        self.prev_ask_price = a_p
        self.prev_ask_volume = a_v

        return IndicatorValue(
            symbol=tick.symbol,
            indicator_name="OFI",
            value=norm_ofi,
            details={"ofi_raw": ofi_raw, "normalized_ofi": norm_ofi},
            timestamp=tick.timestamp
        )
