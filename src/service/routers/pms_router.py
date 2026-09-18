from typing import Dict, List, Optional
from fastapi import APIRouter, HTTPException, Query
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
from src.pms.manager import PortfolioManager
from src.pms.aggregator import PortfolioAggregator
from src.execution_engine.gateway.paper_broker import PaperBroker
from src.risk_engine.circuit_breaker import CircuitBreakerManager

router = APIRouter(prefix="/pms", tags=["Portfolio Management System"])

# 单例实例
_broker = PaperBroker()
_circuit_breaker = CircuitBreakerManager()
_pms_manager = PortfolioManager(
    master_id="master_default",
    initial_reserve=10_000_000.0,
    broker=_broker,
)
_aggregator = PortfolioAggregator(
    circuit_breaker=_circuit_breaker,
    max_stock_concentration=0.15,
    max_leverage_ratio=1.0,
)

def get_pms_manager() -> PortfolioManager:
    return _pms_manager

def get_aggregator() -> PortfolioAggregator:
    return _aggregator

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
def get_master_account():
    mgr = get_pms_manager()
    mgr.sync_from_broker()
    mgr.refresh_total_equity()
    return mgr.master

@router.post("/strategies", response_model=StrategyAccount)
def register_strategy(req: StrategyCreateRequest):
    mgr = get_pms_manager()
    try:
        strat = mgr.register_strategy(
            strategy_id=req.strategy_id,
            name=req.name,
            strategy_type=req.strategy_type,
            initial_budget=req.initial_budget,
        )
        return strat
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/strategies", response_model=List[StrategyPerformance])
def list_strategies():
    mgr = get_pms_manager()
    mgr.sync_from_broker()
    return mgr.get_performance_summary()

@router.post("/allocate", response_model=CapitalAllocationPlan)
def run_allocation(req: AllocationRunRequest):
    mgr = get_pms_manager()
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
def get_lookthrough_portfolio():
    mgr = get_pms_manager()
    mgr.sync_from_broker()
    aggregator = get_aggregator()
    return aggregator.aggregate(mgr.master)
