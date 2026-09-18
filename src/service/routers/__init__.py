from src.service.routers.data_router import router as data_router
from src.service.routers.factor_router import router as factor_router
from src.service.routers.model_router import router as model_router
from src.service.routers.backtest_router import router as backtest_router
from src.service.routers.agent_router import router as agent_router
from src.service.routers.optimizer_router import router as optimizer_router
from src.service.routers.risk_router import router as risk_router
from src.service.routers.execution_router import router as execution_router
from src.service.routers.pms_router import router as pms_router
from src.service.routers.market_router import router as market_router
from src.service.routers.tasks_router import router as tasks_router

__all__ = [
    "data_router",
    "factor_router",
    "model_router",
    "backtest_router",
    "agent_router",
    "optimizer_router",
    "risk_router",
    "execution_router",
    "pms_router",
    "market_router",
    "tasks_router",
]
