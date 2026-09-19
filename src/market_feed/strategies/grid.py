import uuid
from typing import Dict, Any, Optional, List
from src.market_feed.models import MarketTick, SignalEvent, SignalDirection
from src.market_feed.strategies.base import BaseRealtimeStrategy


class DynamicGridStrategy(BaseRealtimeStrategy):
    """
    日内自适应动态网格策略 (Dynamic Grid Trading Strategy):
    在基准锚定价格周围构建分档多空网格，逢低分批吸筹建仓，反弹逢高获利平仓，
    并内置单票最大持仓上限拦截与下行硬止损保护。
    """

    def __init__(
        self,
        strategy_id: str,
        symbol: str,
        base_price: Optional[float] = None,
        grid_num_levels: int = 5,
        grid_step_pct: float = 0.01,
        order_volume_per_grid: int = 100,
        max_position_volume: int = 500,
        stop_loss_pct: float = 0.05,
    ):
        super().__init__(strategy_id=strategy_id, symbol=symbol)
        self.base_price = base_price
        self.grid_num_levels = max(1, grid_num_levels)
        self.grid_step_pct = max(0.001, grid_step_pct)
        self.order_volume_per_grid = max(100, (order_volume_per_grid // 100) * 100)
        self.max_position_volume = max(self.order_volume_per_grid, max_position_volume)
        self.stop_loss_pct = stop_loss_pct

        # 状态记录
        self.current_position = 0
        self.buy_levels_filled: List[bool] = [False] * self.grid_num_levels
        self.total_triggers = 0
        self.stop_loss_triggered = False

    def on_tick(self, tick: MarketTick) -> Optional[SignalEvent]:
        if tick.symbol != self.symbol or tick.last_price <= 0:
            return None

        # 1. 自动初始化基准锚定价格
        if self.base_price is None or self.base_price <= 0:
            self.base_price = tick.last_price

        # 2. 检查硬止损保护
        stop_loss_price = self.base_price * (1.0 - self.stop_loss_pct)
        if tick.last_price <= stop_loss_price and self.current_position > 0:
            vol_to_close = self.current_position
            self.current_position = 0
            self.buy_levels_filled = [False] * self.grid_num_levels
            self.stop_loss_triggered = True
            self.total_triggers += 1
            return SignalEvent(
                signal_id=f"sig_grid_sl_{uuid.uuid4().hex[:8]}",
                strategy_id=self.strategy_id,
                symbol=self.symbol,
                direction=SignalDirection.SELL,
                price=tick.last_price,
                volume=vol_to_close,
                strength=1.0,
                reason=f"网格下行击穿止损线 ({tick.last_price:.2f} <= {stop_loss_price:.2f}) 强制全平",
                timestamp=tick.timestamp,
            )

        if self.stop_loss_triggered:
            # 止损触发后进入冷静锁定，等待显式 reset 或反弹重置
            return None

        # 3. 逢低买入网格检测 (从浅入深)
        for i in range(self.grid_num_levels):
            buy_price = self.base_price * (1.0 - (i + 1) * self.grid_step_pct)
            if tick.last_price <= buy_price and not self.buy_levels_filled[i]:
                # 检查最大持仓上限
                if self.current_position + self.order_volume_per_grid <= self.max_position_volume:
                    self.buy_levels_filled[i] = True
                    self.current_position += self.order_volume_per_grid
                    self.total_triggers += 1
                    return SignalEvent(
                        signal_id=f"sig_grid_b_{uuid.uuid4().hex[:8]}",
                        strategy_id=self.strategy_id,
                        symbol=self.symbol,
                        direction=SignalDirection.BUY,
                        price=tick.last_price,
                        volume=self.order_volume_per_grid,
                        strength=0.8,
                        reason=f"网格买入触发 第{i+1}档 ({tick.last_price:.2f} <= {buy_price:.2f})",
                        timestamp=tick.timestamp,
                    )

        # 4. 反弹逢高获利卖出网格检测 (从深档向浅档释放)
        for i in reversed(range(self.grid_num_levels)):
            if self.buy_levels_filled[i]:
                # 回弹目标卖出价设为当前档上方一格
                sell_target = self.base_price * (1.0 - i * self.grid_step_pct)
                if tick.last_price >= sell_target and self.current_position >= self.order_volume_per_grid:
                    self.buy_levels_filled[i] = False
                    self.current_position -= self.order_volume_per_grid
                    self.total_triggers += 1
                    return SignalEvent(
                        signal_id=f"sig_grid_s_{uuid.uuid4().hex[:8]}",
                        strategy_id=self.strategy_id,
                        symbol=self.symbol,
                        direction=SignalDirection.SELL,
                        price=tick.last_price,
                        volume=self.order_volume_per_grid,
                        strength=0.8,
                        reason=f"网格止盈卖出触发 第{i+1}档回补 ({tick.last_price:.2f} >= {sell_target:.2f})",
                        timestamp=tick.timestamp,
                    )

        return None

    def get_state(self) -> Dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "symbol": self.symbol,
            "base_price": self.base_price,
            "current_position": self.current_position,
            "max_position_volume": self.max_position_volume,
            "buy_levels_filled": self.buy_levels_filled,
            "total_triggers": self.total_triggers,
            "stop_loss_triggered": self.stop_loss_triggered,
        }

    def reset(self) -> None:
        self.base_price = None
        self.current_position = 0
        self.buy_levels_filled = [False] * self.grid_num_levels
        self.total_triggers = 0
        self.stop_loss_triggered = False
