from typing import Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from src.pms.models import (
    StrategyType,
    AllocationMethod,
    MasterAccount,
    StrategyAccount,
    CapitalAllocationPlan,
    ConsolidatedPortfolio,
    StrategyPerformance,
)
from src.service.runtime import ServiceRuntime, get_runtime

router = APIRouter(prefix="/pms", tags=["Portfolio Management System"])

class StrategyCreateRequest(BaseModel):
    strategy_id: str
    name: str
    strategy_type: StrategyType = StrategyType.ALPHA
    initial_budget: float = Field(default=0.0, ge=0.0)

class AllocationRunRequest(BaseModel):
    method: AllocationMethod = AllocationMethod.EQUAL_WEIGHT
    investable_ratio: float = Field(default=0.95, ge=0.1, le=1.0)
    base_ratios: Optional[Dict[str, float]] = None
    volatilities: Optional[Dict[str, float]] = None
    sharpes: Optional[Dict[str, float]] = None
    apply: bool = False

@router.get("/master", response_model=MasterAccount)
def get_master_account(runtime: ServiceRuntime = Depends(get_runtime)):
    with runtime.lock:
        mgr = runtime.pms_manager
        mgr.sync_from_broker()
        return mgr.master.model_copy(deep=True)

@router.post("/strategies", response_model=StrategyAccount)
def register_strategy(req: StrategyCreateRequest, runtime: ServiceRuntime = Depends(get_runtime)):
    with runtime.lock:
        try:
            strat = runtime.pms_manager.register_strategy(
                strategy_id=req.strategy_id,
                name=req.name,
                strategy_type=req.strategy_type,
                initial_budget=req.initial_budget,
            )
            return strat.model_copy(deep=True)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

@router.get("/strategies", response_model=List[StrategyPerformance])
def list_strategies(runtime: ServiceRuntime = Depends(get_runtime)):
    with runtime.lock:
        mgr = runtime.pms_manager
        mgr.sync_from_broker()
        return mgr.get_performance_summary()

@router.post("/allocate", response_model=CapitalAllocationPlan)
def run_allocation(req: AllocationRunRequest, runtime: ServiceRuntime = Depends(get_runtime)):
    with runtime.lock:
        mgr = runtime.pms_manager
        mgr.sync_from_broker()
        current_equities = mgr.get_strategy_equities()
        if not current_equities:
            return CapitalAllocationPlan(method=req.method, target_ratios={})

        plan = mgr.allocator.generate_plan(
            master_equity=mgr.master.total_equity,
            current_equities=current_equities,
            method=req.method,
            investable_ratio=req.investable_ratio,
            base_ratios=req.base_ratios,
            volatilities=req.volatilities,
            sharpes=req.sharpes,
        )
        if req.apply:
            try:
                mgr.apply_allocation_plan(plan)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=f"执行调资失败: {e}")
        return plan

@router.get("/lookthrough", response_model=ConsolidatedPortfolio)
def get_lookthrough_portfolio(runtime: ServiceRuntime = Depends(get_runtime)):
    with runtime.lock:
        mgr = runtime.pms_manager
        mgr.sync_from_broker()
        return runtime.aggregator.aggregate(mgr.master)
