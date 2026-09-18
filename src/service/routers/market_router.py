from typing import List
from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect, Query
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool
from src.market_feed.models import MarketTick
from src.service.runtime import ServiceRuntime, get_runtime

router = APIRouter(prefix="/market", tags=["Market Feed & Streaming Signals"])

class ReplayRequest(BaseModel):
    ticks: List[MarketTick]
    speed: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)

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
