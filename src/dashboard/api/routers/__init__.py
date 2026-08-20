"""Modular API routers for Phase 3.

7 modular routers:
- market: Market data, instruments, K-line
- backtest: Strategy backtesting, multi-factor
- research: Model zoo, factor library, training
- trading: Live trading simulation, Qlib predict
- risk: Risk analysis, stress testing
- rdagent: RD-Agent factor mining, model evolution
- alternative: Alternative data (news, announcements, reports, macro)
"""
from .rdagent import router as rdagent_router
from .mlflow_router import router as mlflow_router
from .trading_live import router as trading_live_router
from .alternative import router as alternative_router
