import time
import uuid
from typing import Dict, Optional
from src.market_feed.models import MarketTick, SignalEvent, SignalDirection
from src.market_feed.indicators import StreamingVWAP, StreamingBollingerBands, OrderFlowImbalance

class CooldownManager:
    """高频防抖防刷单冷却管理器"""

    def __init__(self, cooldown_seconds: float = 3.0):
        self.cooldown_seconds = cooldown_seconds
        self._last_trigger_times: Dict[str, float] = {}

    def is_cooling_down(self, symbol: str, direction: SignalDirection) -> bool:
        key = f"{symbol}:{direction.value}"
        now = time.time()
        last_time = self._last_trigger_times.get(key, 0.0)
        if now - last_time < self.cooldown_seconds:
            return True
        return False

    def record_trigger(self, symbol: str, direction: SignalDirection) -> None:
        key = f"{symbol}:{direction.value}"
        self._last_trigger_times[key] = time.time()

class SignalEngine:
    """实时高频与微观盘口信号触发器"""

    def __init__(self, default_strategy_id: str = "hft_momentum", cooldown_seconds: float = 3.0):
        self.default_strategy_id = default_strategy_id
        self.cooldown = CooldownManager(cooldown_seconds=cooldown_seconds)
        self.vwaps: Dict[str, StreamingVWAP] = {}
        self.bollingers: Dict[str, StreamingBollingerBands] = {}
        self.ofis: Dict[str, OrderFlowImbalance] = {}

    def _get_or_create_indicators(self, symbol: str):
        if symbol not in self.vwaps:
            self.vwaps[symbol] = StreamingVWAP()
            self.bollingers[symbol] = StreamingBollingerBands(window=10, num_std=1.5)
            self.ofis[symbol] = OrderFlowImbalance()
        return self.vwaps[symbol], self.bollingers[symbol], self.ofis[symbol]

    def process_tick(self, tick: MarketTick) -> Optional[SignalEvent]:
        vwap_ind, bb_ind, ofi_ind = self._get_or_create_indicators(tick.symbol)

        res_vwap = vwap_ind.update(tick)
        res_bb = bb_ind.update(tick)
        res_ofi = ofi_ind.update(tick)

        vwap_val = res_vwap.value
        pct_b = res_bb.details.get("pct_b", 0.5)
        norm_ofi = res_ofi.value

        direction: Optional[SignalDirection] = None
        reason: str = ""

        # 规则 1: 布林带上突破 + 站稳 VWAP 强买方
        if pct_b > 1.0 and tick.last_price >= vwap_val:
            direction = SignalDirection.BUY
            reason = f"布林带上轨突破 (%b={pct_b:.2f}) 且高于VWAP ({vwap_val:.2f})"
        # 规则 2: 布林带下跌破 + 跌破 VWAP
        elif pct_b < 0.0 and tick.last_price <= vwap_val:
            direction = SignalDirection.SELL
            reason = f"布林带下轨跌破 (%b={pct_b:.2f}) 且低于VWAP ({vwap_val:.2f})"
        # 规则 3: OFI 买压激增 (买卖挂单极度不平衡)
        elif norm_ofi > 0.65:
            direction = SignalDirection.BUY
            reason = f"微观盘口OFI买压聚集 (OFI={norm_ofi:.2f})"
        elif norm_ofi < -0.65:
            direction = SignalDirection.SELL
            reason = f"微观盘口OFI抛压显著 (OFI={norm_ofi:.2f})"

        if direction is not None:
            # 冷却检查
            if self.cooldown.is_cooling_down(tick.symbol, direction):
                return None

            self.cooldown.record_trigger(tick.symbol, direction)
            return SignalEvent(
                signal_id=f"sig_{uuid.uuid4().hex[:8]}",
                strategy_id=self.default_strategy_id,
                symbol=tick.symbol,
                direction=direction,
                price=tick.last_price,
                volume=100,
                strength=0.9,
                reason=reason,
                timestamp=tick.timestamp
            )

        return None
