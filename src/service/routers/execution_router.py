from typing import Optional, Dict, Any, List
from fastapi import APIRouter, Query
from pydantic import BaseModel, Field
from src.execution_engine import (
    ExecutionCoordinator, PaperBroker, AlgoType, AccountState, Trade
)
from src.risk_engine import PreTradeRiskChecker, CircuitBreakerManager

router = APIRouter(prefix="/execution", tags=["Execution Engine"])

global_broker = PaperBroker()
global_coordinator = ExecutionCoordinator(gateway=global_broker)

class RebalanceRequest(BaseModel):
    account_id: str = "default"
    target_weights: Dict[str, float]
    current_prices: Dict[str, float]
    algo_type: str = "DIRECT"

class SettleRequest(BaseModel):
    account_id: str = "default"

@router.post("/rebalance")
def execute_rebalance(req: RebalanceRequest):
    algo = AlgoType(req.algo_type.upper()) if req.algo_type.upper() in AlgoType._value2member_map_ else AlgoType.DIRECT
    return global_coordinator.execute_rebalance(
        account_id=req.account_id,
        target_weights=req.target_weights,
        current_prices=req.current_prices,
        algo_type=algo
    )

@router.get("/account", response_model=AccountState)
def get_account(account_id: str = Query("default")):
    return global_broker.get_account(account_id)

@router.post("/settle")
def settle_overnight(req: SettleRequest):
    global_broker.settle_overnight(req.account_id)
    return {"status": "SUCCESS", "message": f"Overnight settlement completed for {req.account_id}"}

@router.get("/trades", response_model=List[Trade])
def get_trades(account_id: str = Query("default")):
    return [t for t in global_broker.trades if t.account_id == account_id]
