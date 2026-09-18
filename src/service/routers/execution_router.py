from typing import Optional, Dict, Any, List
from fastapi import APIRouter, Depends, Query, HTTPException
from pydantic import BaseModel, Field
from src.execution_engine import (
    ExecutionCoordinator, PaperBroker, QmtBrokerGateway, AlgoType, AccountState, Trade
)
from src.service.runtime import ServiceRuntime, get_runtime

router = APIRouter(prefix="/execution", tags=["Execution Engine"])

class RebalanceRequest(BaseModel):
    account_id: str = "default"
    target_weights: Dict[str, float]
    current_prices: Dict[str, float]
    algo_type: str = "DIRECT"
    execution_mode: str = Field(default="SYNC", description="执行模式: SYNC (即时执行) 或 ASYNC_SCHEDULED (异步切片执行)")
    interval_seconds: float = Field(default=0.0, ge=0.0, description="时间切片执行间隔秒数")

class SettleRequest(BaseModel):
    account_id: str = "default"

@router.post("/rebalance")
def execute_rebalance(req: RebalanceRequest, runtime: ServiceRuntime = Depends(get_runtime)):
    algo = AlgoType(req.algo_type.upper()) if req.algo_type.upper() in AlgoType._value2member_map_ else AlgoType.DIRECT
    with runtime.lock:
        return runtime.coordinator.execute_rebalance(
            account_id=req.account_id,
            target_weights=req.target_weights,
            current_prices=req.current_prices,
            algo_type=algo,
            execution_mode=req.execution_mode,
            interval_seconds=req.interval_seconds,
        )

@router.get("/account", response_model=AccountState)
def get_account(account_id: str = Query("default"), runtime: ServiceRuntime = Depends(get_runtime)):
    with runtime.lock:
        return runtime.broker.get_account(account_id).model_copy(deep=True)

@router.post("/settle")
def settle_overnight(req: SettleRequest, runtime: ServiceRuntime = Depends(get_runtime)):
    with runtime.lock:
        runtime.broker.settle_overnight(req.account_id)
    return {"status": "SUCCESS", "message": f"Overnight settlement completed for {req.account_id}"}

@router.get("/trades")
def get_trades(
    account_id: str = Query("default"),
    limit: int = Query(100, ge=1, le=500),
    runtime: ServiceRuntime = Depends(get_runtime)
):
    with runtime.lock:
        if runtime.storage is not None:
            db_trades = runtime.storage.get_historical_trades(account_id=account_id, limit=limit)
            if db_trades:
                return db_trades
        return [t.model_copy(deep=True) for t in runtime.broker.trades if t.account_id == account_id][:limit]

@router.get("/orders")
def get_orders(
    account_id: str = Query("default"),
    limit: int = Query(100, ge=1, le=500),
    runtime: ServiceRuntime = Depends(get_runtime)
):
    with runtime.lock:
        if runtime.storage is not None:
            return runtime.storage.get_historical_orders(account_id=account_id, limit=limit)
        return []

@router.get("/gateway/status")
def get_gateway_status(runtime: ServiceRuntime = Depends(get_runtime)):
    gw = runtime.coordinator.gateway
    is_qmt = isinstance(gw, QmtBrokerGateway)
    return {
        "gateway_type": "QMT" if is_qmt else "PAPER",
        "is_connected": getattr(gw, "is_connected", True),
        "mock_mode": getattr(gw, "mock_mode", True),
        "mini_qmt_path": getattr(gw, "mini_qmt_path", None),
        "account_id": getattr(gw, "account_id", "default"),
    }

@router.get("/tasks")
def list_execution_tasks(
    account_id: Optional[str] = Query(None),
    runtime: ServiceRuntime = Depends(get_runtime)
):
    return [t.model_dump() for t in runtime.scheduler.list_tasks(account_id=account_id)]

@router.get("/tasks/{task_id}")
def get_execution_task(task_id: str, runtime: ServiceRuntime = Depends(get_runtime)):
    task = runtime.scheduler.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found.")
    return task.model_dump()

@router.post("/tasks/{task_id}/cancel")
def cancel_execution_task(task_id: str, runtime: ServiceRuntime = Depends(get_runtime)):
    success = runtime.scheduler.cancel_task(task_id)
    if not success:
        raise HTTPException(status_code=400, detail=f"Cannot cancel task {task_id} (not found or already terminal).")
    return {"status": "SUCCESS", "task_id": task_id, "message": "Task cancelled."}

class MarkToMarketRequest(BaseModel):
    symbol: str
    price: float = Field(..., gt=0.0, description="标的最新市场价格")

@router.post("/mark_to_market")
def mark_to_market(req: MarkToMarketRequest, runtime: ServiceRuntime = Depends(get_runtime)):
    with runtime.lock:
        if not hasattr(runtime.broker, "update_market_price"):
            raise HTTPException(status_code=400, detail="Current broker gateway does not support mark_to_market.")
        updated_accounts = runtime.broker.update_market_price(req.symbol, req.price)
        res = []
        for acc in updated_accounts:
            cb_state = runtime.circuit_breaker.update_equity(acc.account_id, acc.total_equity)
            pos_info = acc.positions.get(req.symbol)
            res.append({
                "account_id": acc.account_id,
                "total_equity": acc.total_equity,
                "available_cash": acc.available_cash,
                "circuit_breaker_level": cb_state.level.value,
                "position": pos_info.model_dump() if pos_info else None,
            })
        return {"status": "SUCCESS", "symbol": req.symbol, "price": req.price, "updated_accounts": res}

@router.get("/positions")
def get_positions(
    account_id: Optional[str] = Query(None, description="指定账户ID；缺省返回所有账户持仓"),
    runtime: ServiceRuntime = Depends(get_runtime)
):
    with runtime.lock:
        accounts = [runtime.broker.get_account(account_id)] if account_id else runtime.broker.get_all_accounts()
        results = []
        for acc in accounts:
            for sym, pos in acc.positions.items():
                p_dict = pos.model_dump()
                p_dict["account_id"] = acc.account_id
                results.append(p_dict)
        return results
