"""FastAPI backend main entry.

Serves the interactive dashboard at / and API endpoints at /api/*
and /api/qlib/* (offline endpoints that don't need MySQL).

Phase 3 additions:
- Modular routers: rdagent, mlflow, trading_live
- Enhanced WebSocket with real-time push (market data, risk alerts, trade updates)
"""
import logging
import asyncio
import json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pathlib import Path

from .routes import router as api_router, qlib_router
from .research_routes import router as research_router
from .routers import rdagent_router, mlflow_router, trading_live_router, alternative_router

logger = logging.getLogger(__name__)

app = FastAPI(title="Quant Platform", version="0.3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files (CSS/JS)
static_dir = Path(__file__).parent.parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Include routers
app.include_router(api_router)
app.include_router(qlib_router)
app.include_router(research_router)
app.include_router(rdagent_router)
app.include_router(mlflow_router)
app.include_router(trading_live_router)
app.include_router(alternative_router)


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Serve the main dashboard HTML page."""
    template_path = Path(__file__).parent.parent / "templates" / "index.html"
    if template_path.exists():
        return HTMLResponse(template_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Dashboard template not found</h1><p>Run: python -m src mock-data first</p>")


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "version": "0.3.0",
        "layers": [
            "data", "factor", "model", "strategy",
            "backtest", "execution", "dashboard",
        ],
        "modules": [
            "qlib", "rd_agent", "mlflow", "research_workflow",
            "unified_risk", "enhanced_indexing", "alternative_data",
        ],
    }


class ConnectionManager:
    """Enhanced WebSocket connection manager.

    Supports topic-based subscriptions:
    - market: Real-time market data
    - risk: Risk alerts
    - trades: Trade execution updates
    - system: System status
    """

    def __init__(self):
        self.active: dict[WebSocket, set] = {}

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active[ws] = {"market", "risk", "trades", "system"}
        await ws.send_json({"type": "welcome", "message": "connected"})

    def disconnect(self, ws: WebSocket):
        if ws in self.active:
            del self.active[ws]

    async def broadcast(self, message: dict, topic: str = "system"):
        """Broadcast to all connections subscribed to a topic."""
        disconnected = []
        for ws, topics in self.active.items():
            if topic in topics:
                try:
                    await ws.send_json({**message, "topic": topic})
                except Exception:
                    disconnected.append(ws)
        for ws in disconnected:
            self.disconnect(ws)

    async def subscribe(self, ws: WebSocket, topics: list):
        if ws in self.active:
            self.active[ws] = set(topics)


manager = ConnectionManager()


@app.websocket("/ws/realtime")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            try:
                msg = json.loads(data)
                msg_type = msg.get("type", "")

                if msg_type == "subscribe":
                    topics = msg.get("topics", [])
                    await manager.subscribe(websocket, topics)
                    await websocket.send_json({
                        "type": "subscribed",
                        "topics": topics,
                    })
                elif msg_type == "ping":
                    await websocket.send_json({"type": "pong"})
                else:
                    await websocket.send_json({
                        "type": "echo",
                        "data": msg,
                    })
            except json.JSONDecodeError:
                await websocket.send_json({
                    "type": "error",
                    "message": "Invalid JSON",
                })
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
