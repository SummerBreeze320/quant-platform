from typing import Optional, List, Dict, Any
from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from src.risk_engine.models import (
    OrderRiskRequest, OrderRiskResult,
    PortfolioRiskRequest, PortfolioRiskResult,
    CircuitBreakerState, RiskAlertEvent, AlertLevel
)
from src.service.runtime import ServiceRuntime, get_runtime

router = APIRouter(prefix="/risk", tags=["Risk Management"])

class BlacklistRequest(BaseModel):
    action: str  # "add", "remove", "list"
    symbols: List[str] = []

class ResetRequest(BaseModel):
    account_id: str = "default"
    reset_watermark: bool = True

@router.post("/check_order", response_model=OrderRiskResult)
def check_order(req: OrderRiskRequest, fail_fast: bool = False, runtime: ServiceRuntime = Depends(get_runtime)):
    with runtime.lock:
        return runtime.risk_checker.check_order(req, fail_fast=fail_fast)

@router.post("/check_portfolio", response_model=PortfolioRiskResult)
def check_portfolio(req: PortfolioRiskRequest, runtime: ServiceRuntime = Depends(get_runtime)):
    with runtime.lock:
        return runtime.risk_monitor.analyze_portfolio(req)

@router.get("/circuit_breaker/status", response_model=CircuitBreakerState)
def get_circuit_breaker_status(account_id: str = Query("default"), runtime: ServiceRuntime = Depends(get_runtime)):
    with runtime.lock:
        return runtime.circuit_breaker.get_state(account_id).model_copy(deep=True)

@router.post("/circuit_breaker/reset", response_model=CircuitBreakerState)
def reset_circuit_breaker(req: ResetRequest, runtime: ServiceRuntime = Depends(get_runtime)):
    with runtime.lock:
        return runtime.circuit_breaker.reset(account_id=req.account_id, reset_watermark=req.reset_watermark)

@router.get("/alerts", response_model=List[RiskAlertEvent])
def get_alerts(limit: int = Query(50, ge=1, le=200), level: Optional[str] = Query(None), runtime: ServiceRuntime = Depends(get_runtime)):
    alert_lvl = None
    if level:
        for member in AlertLevel:
            if member.value == level or member.name == level:
                alert_lvl = member
                break
    with runtime.lock:
        return runtime.alert_manager.get_recent_alerts(limit=limit, level=alert_lvl)

@router.post("/blacklist")
def manage_blacklist(req: BlacklistRequest, runtime: ServiceRuntime = Depends(get_runtime)):
    with runtime.lock:
        if req.action == "add":
            for s in req.symbols:
                runtime.risk_checker.add_blacklist(s)
        elif req.action == "remove":
            for s in req.symbols:
                runtime.risk_checker.remove_blacklist(s)
        return {"blacklist": sorted(runtime.risk_checker.get_blacklist())}
