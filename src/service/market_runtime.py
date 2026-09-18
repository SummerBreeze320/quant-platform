"""Market processing and WebSocket connections owned by one application."""
import asyncio
from threading import Lock

from fastapi import WebSocket

from src.market_feed.models import MarketTick
from src.market_feed.replay import TickReplayEngine
from src.market_feed.router import SignalRouter
from src.market_feed.signal_engine import SignalEngine
from src.market_feed.stream_bus import StreamBus


class ConnectionManager:
    def __init__(self):
        self._connections = {}
        self._lock = Lock()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        with self._lock:
            self._connections[websocket] = (asyncio.get_running_loop(), asyncio.Lock())

    def disconnect(self, websocket: WebSocket):
        with self._lock:
            self._connections.pop(websocket, None)

    async def _send(self, websocket, send_lock, payload):
        try:
            async with send_lock:
                await websocket.send_json(payload)
        except Exception:
            self.disconnect(websocket)

    def broadcast(self, payload: dict):
        with self._lock:
            connections = list(self._connections.items())
        for websocket, (loop, send_lock) in connections:
            coro = self._send(websocket, send_lock, payload)
            try:
                # Sync endpoints run in worker threads; never create another loop
                # for a WebSocket that belongs to the ASGI connection's loop.
                asyncio.run_coroutine_threadsafe(coro, loop)
            except RuntimeError:
                coro.close()
                self.disconnect(websocket)


from src.risk_engine.circuit_breaker import CircuitBreakerManager
from src.risk_engine.alert import RiskAlertManager
from src.risk_engine.models import CircuitBreakerLevel, AlertLevel
from typing import Optional

class MarketRuntime:
    def __init__(
        self,
        broker,
        risk_checker,
        lock,
        circuit_breaker: Optional[CircuitBreakerManager] = None,
        alert_manager: Optional[RiskAlertManager] = None,
    ):
        self.lock = lock
        self.broker = broker
        self.circuit_breaker = circuit_breaker
        self.alert_manager = alert_manager
        self.connections = ConnectionManager()
        self.signal_engine = SignalEngine(default_strategy_id="hft_stream_01", cooldown_seconds=1.0)
        self.signal_router = SignalRouter(broker=broker, risk_checker=risk_checker)
        self.signal_router.register_strategy("hft_stream_01", auto_execute=True)
        self.bus = StreamBus(stream_key="market_stream:ticks")
        self.bus.subscribe(self.on_tick)
        self.replay_engine = TickReplayEngine(stream_bus=self.bus)

    def on_tick(self, tick: MarketTick):
        with self.lock:
            # 1. 优先广播原始行情 Tick，通知所有实时行情监听端
            self.connections.broadcast({"type": "TICK", "data": tick.model_dump(mode="json")})

            # 2. 动态盯市重估持仓与盈亏
            if hasattr(self.broker, "update_market_price"):
                updated_accounts = self.broker.update_market_price(tick.symbol, tick.last_price)
                for acc in updated_accounts:
                    # 联动动态熔断监测
                    if self.circuit_breaker is not None:
                        cb_state = self.circuit_breaker.update_equity(acc.account_id, acc.total_equity)
                        if cb_state.level > CircuitBreakerLevel.NORMAL:
                            self.connections.broadcast({
                                "type": "CIRCUIT_BREAKER",
                                "data": cb_state.model_dump(mode="json")
                            })
                            if self.alert_manager and cb_state.trigger_reason:
                                alert_lvl = AlertLevel.CRITICAL if cb_state.level == CircuitBreakerLevel.RED_HALT else AlertLevel.WARN
                                self.alert_manager.record_alert(
                                    level=alert_lvl,
                                    account_id=acc.account_id,
                                    rule_name="DynamicDrawdownCircuitBreaker",
                                    message=cb_state.trigger_reason,
                                    metadata={"equity": acc.total_equity, "max_drawdown": cb_state.max_drawdown}
                                )
                    # 广播持仓/净值估值更新
                    self.connections.broadcast({
                        "type": "PORTFOLIO_UPDATE",
                        "data": {
                            "account_id": acc.account_id,
                            "total_equity": acc.total_equity,
                            "available_cash": acc.available_cash,
                            "symbol": tick.symbol,
                            "last_price": tick.last_price,
                            "total_unrealized_pnl": sum(p.unrealized_pnl for p in acc.positions.values()),
                        }
                    })

            # 3. 信号引擎与撮合路由
            signal = self.signal_engine.process_tick(tick)
            if signal:
                self.signal_router.route_signal(signal)
                self.connections.broadcast({"type": "SIGNAL", "data": signal.model_dump(mode="json")})
