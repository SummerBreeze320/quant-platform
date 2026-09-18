from typing import Optional, List, Dict, Any
from fastapi import APIRouter, Query
from pydantic import BaseModel
from src.risk_engine.models import (
    OrderRiskRequest, OrderRiskResult,
    PortfolioRiskRequest, PortfolioRiskResult,
    CircuitBreakerState, RiskAlertEvent, AlertLevel
)
from src.risk_engine.circuit_breaker import CircuitBreakerManager
from src.risk_engine.alert import RiskAlertManager
from src.risk_engine.pre_trade import PreTradeRiskChecker
from src.risk_engine.post_trade import PostTradeRiskMonitor

router = APIRouter(prefix="/risk", tags=["Risk Management"])

# Singleton instances for API workers
global_cb = CircuitBreakerManager()
global_alert = RiskAlertManager()
global_checker = PreTradeRiskChecker(cb_manager=global_cb, alert_manager=global_alert)
global_monitor = PostTradeRiskMonitor(cb_manager=global_cb, alert_manager=global_alert)

class BlacklistRequest(BaseModel):
    action: str  # "add", "remove", "list"
    symbols: List[str] = []

class ResetRequest(BaseModel):
    account_id: str = "default"
    reset_watermark: bool = True

@router.post("/check_order", response_model=OrderRiskResult)
def check_order(req: OrderRiskRequest, fail_fast: bool = False):
    return global_checker.check_order(req, fail_fast=fail_fast)

@router.post("/check_portfolio", response_model=PortfolioRiskResult)
def check_portfolio(req: PortfolioRiskRequest):
    return global_monitor.analyze_portfolio(req)

@router.get("/circuit_breaker/status", response_model=CircuitBreakerState)
def get_circuit_breaker_status(account_id: str = Query("default")):
    return global_cb.get_state(account_id)

@router.post("/circuit_breaker/reset", response_model=CircuitBreakerState)
def reset_circuit_breaker(req: ResetRequest):
    return global_cb.reset(account_id=req.account_id, reset_watermark=req.reset_watermark)

@router.get("/alerts", response_model=List[RiskAlertEvent])
def get_alerts(limit: int = Query(50, ge=1, le=200), level: Optional[str] = Query(None)):
    alert_lvl = None
    if level:
        for member in AlertLevel:
            if member.value == level or member.name == level:
                alert_lvl = member
                break
    return global_alert.get_recent_alerts(limit=limit, level=alert_lvl)

@router.post("/blacklist")
def manage_blacklist(req: BlacklistRequest):
    if req.action == "add":
        for s in req.symbols:
            global_checker.add_blacklist(s)
    elif req.action == "remove":
        for s in req.symbols:
            global_checker.remove_blacklist(s)
    return {"blacklist": sorted(list(global_checker.get_blacklist()))}
