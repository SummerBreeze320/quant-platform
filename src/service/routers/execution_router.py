from typing import Optional, Dict, Any, List
from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field
from src.execution_engine import (
    ExecutionCoordinator, PaperBroker, AlgoType, AccountState, Trade
)
from src.service.runtime import ServiceRuntime, get_runtime

router = APIRouter(prefix="/execution", tags=["Execution Engine"])

class RebalanceRequest(BaseModel):
    account_id: str = "default"
    target_weights: Dict[str, float]
    current_prices: Dict[str, float]
    algo_type: str = "DIRECT"

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
            algo_type=algo
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

@router.get("/trades", response_model=List[Trade])
def get_trades(account_id: str = Query("default"), runtime: ServiceRuntime = Depends(get_runtime)):
    with runtime.lock:
        return [t.model_copy(deep=True) for t in runtime.broker.trades if t.account_id == account_id]
