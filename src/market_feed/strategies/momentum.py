import uuid
from typing import Dict, Any, Optional
from src.market_feed.models import MarketTick, SignalEvent, SignalDirection
from src.market_feed.indicators import StreamingVWAP, StreamingBollingerBands, OrderFlowImbalance
from src.market_feed.strategies.base import BaseRealtimeStrategy


class IntradayMomentumBreakoutStrategy(BaseRealtimeStrategy):
    """
    日内微观动量突破策略 (Intraday Momentum Breakout Strategy):
    结合微观订单流失衡 (OFI)、成交量加权均价 (VWAP) 以及布林带上轨突破，
    捕捉短线大单驱动的主升浪，并内置最高点动态追踪止损 (Trailing Stop) 与目标止盈保护。
    """

    def __init__(
        self,
        strategy_id: str,
        symbol: str,
        ofi_threshold: float = 0.50,
        trailing_stop_pct: float = 0.008,
        take_profit_pct: float = 0.02,
        order_volume: int = 100,
    ):
        super().__init__(strategy_id=strategy_id, symbol=symbol)
        self.ofi_threshold = ofi_threshold
        self.trailing_stop_pct = trailing_stop_pct
        self.take_profit_pct = take_profit_pct
        self.order_volume = max(100, (order_volume // 100) * 100)

        # 流式技术指标
        self.vwap = StreamingVWAP()
        self.bollinger = StreamingBollingerBands(window=10, num_std=1.5)
        self.ofi = OrderFlowImbalance()

        # 持仓状态追踪
        self.in_position = False
        self.entry_price = 0.0
        self.peak_price = 0.0
        self.position_volume = 0
        self.total_trades = 0

    def on_tick(self, tick: MarketTick) -> Optional[SignalEvent]:
        if tick.symbol != self.symbol or tick.last_price <= 0:
            return None

        # 1. 推进微观指标更新
        res_vwap = self.vwap.update(tick)
        res_bb = self.bollinger.update(tick)
        res_ofi = self.ofi.update(tick)

        vwap_val = res_vwap.value
        pct_b = res_bb.details.get("pct_b", 0.5)
        ofi_val = res_ofi.value

        # 2. 持仓中：执行追踪止损与固定止盈检测
        if self.in_position:
            self.peak_price = max(self.peak_price, tick.last_price)
            exit_reason = ""

            # (a) 固定目标止盈检测
            if tick.last_price >= self.entry_price * (1.0 + self.take_profit_pct):
                exit_reason = f"触发固定目标止盈 ({tick.last_price:.2f} >= 入场价 {self.entry_price:.2f} * {1.0 + self.take_profit_pct:.3f})"
            # (b) 峰值追踪动态止损检测
            elif tick.last_price <= self.peak_price * (1.0 - self.trailing_stop_pct):
                exit_reason = f"触发极值追踪止损 ({tick.last_price:.2f} <= 峰值 {self.peak_price:.2f} 回撤 {self.trailing_stop_pct*100:.1f}%)"

            if exit_reason:
                vol = self.position_volume
                self.in_position = False
                self.entry_price = 0.0
                self.peak_price = 0.0
                self.position_volume = 0
                self.total_trades += 1
                return SignalEvent(
                    signal_id=f"sig_mom_exit_{uuid.uuid4().hex[:8]}",
                    strategy_id=self.strategy_id,
                    symbol=self.symbol,
                    direction=SignalDirection.SELL,
                    price=tick.last_price,
                    volume=vol,
                    strength=0.9,
                    reason=exit_reason,
                    timestamp=tick.timestamp,
                )

        # 3. 空仓中：检测量价共振与订单流突破入场条件
        elif not self.in_position:
            # 突破条件：布林带上轨突破 + 站稳 VWAP + OFI 买压激增
            if pct_b > 1.0 and tick.last_price >= vwap_val and ofi_val >= self.ofi_threshold:
                self.in_position = True
                self.entry_price = tick.last_price
                self.peak_price = tick.last_price
                self.position_volume = self.order_volume
                self.total_trades += 1
                return SignalEvent(
                    signal_id=f"sig_mom_entry_{uuid.uuid4().hex[:8]}",
                    strategy_id=self.strategy_id,
                    symbol=self.symbol,
                    direction=SignalDirection.BUY,
                    price=tick.last_price,
                    volume=self.order_volume,
                    strength=0.95,
                    reason=f"量价突破共振 (%b={pct_b:.2f}, OFI={ofi_val:.2f} >= {self.ofi_threshold}, 价高于VWAP {vwap_val:.2f})",
                    timestamp=tick.timestamp,
                )

        return None

    def get_state(self) -> Dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "symbol": self.symbol,
            "in_position": self.in_position,
            "entry_price": self.entry_price,
            "peak_price": self.peak_price,
            "position_volume": self.position_volume,
            "total_trades": self.total_trades,
            "vwap": self.vwap.current_vwap,
            "bollinger_pct_b": self.bollinger.current_pct_b,
            "ofi": self.ofi.current_ofi,
        }

    def reset(self) -> None:
        self.in_position = False
        self.entry_price = 0.0
        self.peak_price = 0.0
        self.position_volume = 0
        self.total_trades = 0
        self.vwap.reset()
        self.bollinger.reset()
        self.ofi.reset()
