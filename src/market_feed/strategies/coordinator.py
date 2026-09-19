from typing import Dict, List, Optional, Any
from src.common.logger import logger
from src.market_feed.models import MarketTick, SignalEvent
from src.market_feed.router import SignalRouter
from src.market_feed.strategies.base import BaseRealtimeStrategy


class RealtimeStrategyCoordinator:
    """
    日内实时策略协调调度器：
    统一聚合管理多个日内高频策略实例，监听分发 Tick 行情流，
    并将生成的买卖交易信号自动路由至 SignalRouter 执行风控校验与柜台报单撮合。
    """

    def __init__(self, router: Optional[SignalRouter] = None):
        self.router = router
        self.strategies: Dict[str, BaseRealtimeStrategy] = {}

    def register_strategy(self, strategy: BaseRealtimeStrategy) -> None:
        self.strategies[strategy.strategy_id] = strategy
        logger.info(f"[RealtimeCoordinator] Registered strategy '{strategy.strategy_id}' on {strategy.symbol}")

    def on_tick(self, tick: MarketTick) -> List[SignalEvent]:
        """接收实时行情 Tick，分发至对应标的策略并路由触发信号"""
        triggered_signals: List[SignalEvent] = []

        for strat in self.strategies.values():
            if strat.symbol == tick.symbol:
                try:
                    sig = strat.on_tick(tick)
                    if sig is not None:
                        triggered_signals.append(sig)
                        if self.router is not None:
                            self.router.route_signal(sig)
                except Exception as e:
                    logger.error(f"[RealtimeCoordinator] Error in strategy {strat.strategy_id} on_tick: {e}")

        return triggered_signals

    def get_strategy_states(self) -> Dict[str, Dict[str, Any]]:
        return {sid: s.get_state() for sid, s in self.strategies.items()}

    def reset_all(self) -> None:
        for s in self.strategies.values():
            s.reset()
