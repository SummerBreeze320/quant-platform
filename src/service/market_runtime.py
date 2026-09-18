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


class MarketRuntime:
    def __init__(self, broker, risk_checker, lock):
        self.lock = lock
        self.connections = ConnectionManager()
        self.signal_engine = SignalEngine(default_strategy_id="hft_stream_01", cooldown_seconds=1.0)
        self.signal_router = SignalRouter(broker=broker, risk_checker=risk_checker)
        self.signal_router.register_strategy("hft_stream_01", auto_execute=True)
        self.bus = StreamBus(stream_key="market_stream:ticks")
        self.bus.subscribe(self.on_tick)
        self.replay_engine = TickReplayEngine(stream_bus=self.bus)

    def on_tick(self, tick: MarketTick):
        with self.lock:
            signal = self.signal_engine.process_tick(tick)
            if signal:
                self.signal_router.route_signal(signal)
                self.connections.broadcast({"type": "SIGNAL", "data": signal.model_dump(mode="json")})
            self.connections.broadcast({"type": "TICK", "data": tick.model_dump(mode="json")})
