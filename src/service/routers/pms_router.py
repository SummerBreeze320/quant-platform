from typing import Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
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

@router.get("/transfers")
def get_transfers(
    strategy_id: Optional[str] = Query(None, description="策略ID筛选"),
    limit: int = Query(100, ge=1, le=500),
    runtime: ServiceRuntime = Depends(get_runtime)
):
    with runtime.lock:
        if runtime.storage is not None:
            return runtime.storage.get_historical_transfers(strategy_id=strategy_id, limit=limit)
        return []


class RecordNavRequest(BaseModel):
    date: Optional[str] = Field(default=None, description="交易日 YYYY-MM-DD，缺省为今日")
    benchmark_return: float = Field(default=0.0, description="基准指数当日收益率")
    account_id: Optional[str] = Field(default=None, description="账户ID，缺省留存master与所有策略")


@router.get("/analytics")
def get_risk_analytics(
    account_id: str = Query("master", description="策略ID或'master'"),
    risk_free_rate: float = Query(0.02, ge=0.0, le=0.2, description="年化无风险利率"),
    runtime: ServiceRuntime = Depends(get_runtime)
):
    """
    获取指定策略或母账户的多维量化风险指标：
    Sharpe, Sortino, Max Drawdown, Calmar, Beta, Alpha, Win Rate, PnL Ratio
    """
    with runtime.lock:
        mgr = runtime.pms_manager
        metrics = mgr.get_risk_analytics(account_id=account_id, risk_free_rate=risk_free_rate)
        return {"status": "SUCCESS", "account_id": account_id, "metrics": metrics.model_dump()}


@router.get("/analytics/monthly")
def get_monthly_returns(
    account_id: str = Query("master", description="策略ID或'master'"),
    runtime: ServiceRuntime = Depends(get_runtime)
):
    """
    获取指定账户的月度收益率热力图数据与年度汇总
    """
    with runtime.lock:
        mgr = runtime.pms_manager
        matrix = mgr.get_monthly_returns_matrix(account_id=account_id)
        return {"status": "SUCCESS", "account_id": account_id, "data": matrix}


@router.get("/analytics/attribution")
def get_brinson_attribution(
    strategy_id: Optional[str] = Query(None, description="指定策略ID，缺省为母账户穿透组合"),
    runtime: ServiceRuntime = Depends(get_runtime)
):
    """
    执行 Brinson-Fachler 行业绩效归因：
    分解配置效应 (Allocation)、选股效应 (Selection) 与交互效应 (Interaction)
    """
    with runtime.lock:
        mgr = runtime.pms_manager
        attr = mgr.get_brinson_attribution(strategy_id=strategy_id)
        return {"status": "SUCCESS", "strategy_id": strategy_id or "consolidated_master", "attribution": attr}


@router.post("/nav/record")
def record_daily_nav(
    req: RecordNavRequest,
    runtime: ServiceRuntime = Depends(get_runtime)
):
    """
    留存当日单位净值与资产快照
    """
    from datetime import datetime
    trade_date = req.date or datetime.now().strftime("%Y-%m-%d")
    with runtime.lock:
        mgr = runtime.pms_manager
        if req.account_id:
            rec = mgr.record_daily_nav(
                account_id=req.account_id,
                date=trade_date,
                benchmark_return=req.benchmark_return,
            )
            return {"status": "SUCCESS", "records": {req.account_id: rec.model_dump()}}
        else:
            recs = mgr.record_all_daily_nav(
                date=trade_date,
                benchmark_return=req.benchmark_return,
            )
            return {"status": "SUCCESS", "records": {k: v.model_dump() for k, v in recs.items()}}


@router.get("/nav/history")
def get_nav_history(
    account_id: str = Query("master", description="策略ID或'master'"),
    runtime: ServiceRuntime = Depends(get_runtime)
):
    """
    获取历史逐日净值序列
    """
    with runtime.lock:
        mgr = runtime.pms_manager
        history = mgr.get_nav_history(account_id)
        return {"status": "SUCCESS", "account_id": account_id, "history": [h.model_dump() for h in history]}

