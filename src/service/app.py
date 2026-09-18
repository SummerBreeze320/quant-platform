from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.common.config import get_settings
from src.common.db import init_db
from src.common.logger import logger
from src.qlib_engine.initializer import init_qlib
from src.tasks.scheduler import get_scheduler
from src.service.runtime import ServiceRuntime
from src.service.routers import (
    data_router,
    factor_router,
    model_router,
    backtest_router,
    agent_router,
    optimizer_router,
    risk_router,
    execution_router,
    pms_router,
    market_router,
    tasks_router,
)

settings = get_settings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context: handles startup and shutdown logic."""
    logger.info("Initializing database tables...")
    try:
        init_db()
    except Exception as e:
        logger.warning(f"Could not connect to PostgreSQL on startup (Docker might be starting): {e}")

    logger.info("Initializing Qlib engine...")
    try:
        init_qlib()
    except Exception as e:
        logger.warning(f"Qlib engine startup notice: {e}")

    logger.info("Starting background scheduler...")
    scheduler = get_scheduler()
    scheduler.start()

    yield

    logger.info("Shutting down background scheduler...")
    scheduler.shutdown()
    logger.info("QuantCopliot service stopped.")

def create_app(session_factory=None) -> FastAPI:
    """Factory creating and configuring the FastAPI application."""
    app = FastAPI(
        title="QuantCopliot",
        description="Pure Python Quantitative Service Platform based on Microsoft Qlib, WindPy and RD-Agent",
        version="0.1.0",
        lifespan=lifespan
    )
    app.state.runtime = ServiceRuntime(session_factory=session_factory)


    # CORS configuration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Health check endpoint
    @app.get("/health", tags=["System"])
    def health_check():
        return {
            "status": "healthy",
            "service": "QuantCopliot",
            "environment": settings.ENV,
            "version": "0.1.0"
        }

    # Register API v1 routes
    app.include_router(data_router, prefix="/api/v1")
    app.include_router(factor_router, prefix="/api/v1")
    app.include_router(model_router, prefix="/api/v1")
    app.include_router(backtest_router, prefix="/api/v1")
    app.include_router(agent_router, prefix="/api/v1")
    app.include_router(optimizer_router, prefix="/api/v1")
    app.include_router(risk_router, prefix="/api/v1")
    app.include_router(execution_router, prefix="/api/v1")
    app.include_router(pms_router, prefix="/api/v1")
    app.include_router(market_router, prefix="/api/v1")
    app.include_router(tasks_router, prefix="/api/v1")

    return app

app = create_app()
