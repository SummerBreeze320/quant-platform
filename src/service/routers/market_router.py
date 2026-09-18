from typing import List, Optional
from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect, Query, HTTPException
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool
from src.market_feed.models import MarketTick
from src.service.runtime import ServiceRuntime, get_runtime

router = APIRouter(prefix="/market", tags=["Market Feed & Streaming Signals"])

class ReplayRequest(BaseModel):
    ticks: List[MarketTick]
    speed: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)

class StartFeedRequest(BaseModel):
    source: str = Field(default="SIMULATION", description="行情源类型: QMT, PUBLIC, SIMULATION")
    symbols: List[str] = Field(default_factory=lambda: ["600000.SH", "000001.SZ"], description="初始订阅标的列表")
    interval_seconds: Optional[float] = Field(default=0.5, ge=0.01, description="采样或轮询间隔秒数")

class SubscribeFeedRequest(BaseModel):
    symbols: List[str] = Field(..., min_length=1, description="标的代码列表")

@router.post("/tick")
def receive_tick(tick: MarketTick, runtime: ServiceRuntime = Depends(get_runtime)):
    with runtime.lock:
        msg_id = runtime.market.bus.publish_tick(tick)
        return {"status": "SUCCESS", "msg_id": msg_id}

@router.post("/replay")
def replay_ticks(req: ReplayRequest, runtime: ServiceRuntime = Depends(get_runtime)):
    # Each tick is processed under the shared lock; do not hold it while sleeping.
    count = runtime.market.replay_engine.replay(req.ticks, speed=req.speed)
    return {"status": "SUCCESS", "replayed_count": count}

@router.post("/feed/start")
def start_market_feed(req: StartFeedRequest, runtime: ServiceRuntime = Depends(get_runtime)):
    """启动实时行情源驱动服务 (QMT / 公网降级 / 波动仿真)"""
    with runtime.lock:
        try:
            status = runtime.market.live_feed.start(
                source=req.source,
                symbols=req.symbols,
                interval_seconds=req.interval_seconds,
            )
            return {"status": "SUCCESS", "feed": status}
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to start feed: {e}")

@router.post("/feed/stop")
def stop_market_feed(runtime: ServiceRuntime = Depends(get_runtime)):
    """停止实时行情源驱动服务"""
    with runtime.lock:
        status = runtime.market.live_feed.stop()
        return {"status": "SUCCESS", "feed": status}

@router.get("/feed/status")
def get_market_feed_status(runtime: ServiceRuntime = Depends(get_runtime)):
    """查询当前实时行情驱动器状态、TPS 与订阅标的"""
    with runtime.lock:
        status = runtime.market.live_feed.get_status()
        return {"status": "SUCCESS", "feed": status}

@router.post("/feed/subscribe")
def subscribe_feed_symbols(req: SubscribeFeedRequest, runtime: ServiceRuntime = Depends(get_runtime)):
    """追加订阅标的代码"""
    with runtime.lock:
        status = runtime.market.live_feed.subscribe(req.symbols)
        return {"status": "SUCCESS", "feed": status}

@router.post("/feed/unsubscribe")
def unsubscribe_feed_symbols(req: SubscribeFeedRequest, runtime: ServiceRuntime = Depends(get_runtime)):
    """退订标的代码"""
    with runtime.lock:
        status = runtime.market.live_feed.unsubscribe(req.symbols)
        return {"status": "SUCCESS", "feed": status}

@router.get("/indicators")
def get_indicators(symbol: str = Query(..., description="标的代码"), runtime: ServiceRuntime = Depends(get_runtime)):
    with runtime.lock:
        return runtime.market.signal_engine.get_indicator_snapshot(symbol)

@router.get("/signals")
def get_recent_signals(limit: int = Query(50, ge=1, le=200), runtime: ServiceRuntime = Depends(get_runtime)):
    with runtime.lock:
        return runtime.market.signal_router.get_recent_signals(limit=limit)

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    runtime = websocket.app.state.runtime
    market = await run_in_threadpool(lambda: runtime.market)
    await market.connections.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        market.connections.disconnect(websocket)
