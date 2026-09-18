from typing import Optional, Dict, Any, List
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from src.service.runtime import ServiceRuntime, get_runtime
from src.tasks.pipeline import PremarketRebalancePipeline, DailySettlementPipeline
from src.tasks.scheduler import get_scheduler

router = APIRouter(prefix="/tasks", tags=["Scheduled Tasks & Pipeline"])


class RebalancePipelineRequest(BaseModel):
    strategy_id: str = "hft_stream_01"
    trade_date: Optional[str] = None
    top_k: int = Field(default=10, ge=1, le=100, description="选取预测排行Top-K个股")
    max_stock_weight: float = Field(default=0.08, ge=0.01, le=0.5, description="单票持仓上限")
    algo_type: str = Field(default="DIRECT", description="算法类型: DIRECT, TWAP, VWAP")
    execution_mode: str = Field(default="SYNC", description="执行模式: SYNC (即时) 或 ASYNC_SCHEDULED (切片)")
    interval_seconds: float = Field(default=0.0, ge=0.0, description="切片时间步长")
    current_prices_override: Optional[Dict[str, float]] = None


class SettlementPipelineRequest(BaseModel):
    account_id: Optional[str] = Field(default=None, description="结算账户ID，缺省结算所有账户")
    trade_date: Optional[str] = None


@router.post("/rebalance")
def run_rebalance_pipeline(
    req: RebalancePipelineRequest,
    runtime: ServiceRuntime = Depends(get_runtime),
):
    """
    一键手动触发盘前策略自动调仓流水线：
    Alpha预测提取 -> 凸优化求解目标权重 -> 执行引擎建仓/调仓
    """
    return PremarketRebalancePipeline.run(
        strategy_id=req.strategy_id,
        trade_date=req.trade_date,
        top_k=req.top_k,
        max_stock_weight=req.max_stock_weight,
        algo_type=req.algo_type,
        execution_mode=req.execution_mode,
        interval_seconds=req.interval_seconds,
        runtime=runtime,
        current_prices_override=req.current_prices_override,
    )


@router.post("/settle")
def run_settlement_pipeline(
    req: SettlementPipelineRequest,
    runtime: ServiceRuntime = Depends(get_runtime),
):
    """
    一键手动触发盘后日终交收清算流水线：
    解冻 T+1 可用股票 -> 同步结转 PMS 组合层各策略净值
    """
    return DailySettlementPipeline.run(
        account_id=req.account_id,
        trade_date=req.trade_date,
        runtime=runtime,
    )


@router.get("/jobs")
def list_scheduled_jobs():
    """
    查询后台 APScheduler 定时任务列表及其下次运行时间
    """
    scheduler = get_scheduler()
    jobs = scheduler.get_jobs()
    result = []
    for job in jobs:
        next_run = getattr(job, "next_run_time", None)
        result.append({
            "id": getattr(job, "id", str(job)),
            "name": getattr(job, "name", str(job)),
            "next_run_time": next_run.isoformat() if next_run else None,
            "trigger": str(getattr(job, "trigger", "")),
        })
    return result


@router.post("/jobs/{job_id}/run")
def trigger_job(job_id: str):
    """
    立即手动触发指定的后台定时任务
    """
    scheduler = get_scheduler()
    job = scheduler.scheduler.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")
    job.modify(next_run_time=None)  # APScheduler runs immediately if modified or function called
    try:
        job.func()
        return {"status": "SUCCESS", "job_id": job_id, "message": f"Job {job_id} executed successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Job {job_id} execution failed: {e}")
