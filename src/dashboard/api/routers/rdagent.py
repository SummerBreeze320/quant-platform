"""RD-Agent API router.

Endpoints for:
- Factor mining (fin_factor scenario)
- Model evolution (fin_model scenario)
- Evolution loop coordination
- Workspace history and status
"""
import logging
from typing import Optional, Dict, Any
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/rdagent", tags=["rdagent"])


class FactorMiningRequest(BaseModel):
    max_iterations: int = Field(5, description="最大迭代次数")
    min_ic: float = Field(0.03, description="IC筛选阈值")


class ModelEvolutionRequest(BaseModel):
    max_iterations: int = Field(5, description="最大迭代次数")
    base_model: str = Field("lightgbm", description="基线模型")
    target_metric: str = Field("IC", description="优化目标")


class EvolutionLoopRequest(BaseModel):
    max_rounds: int = Field(3, description="进化轮次")
    factor_iterations: int = Field(5, description="每轮因子挖掘迭代")
    model_iterations: int = Field(5, description="每轮模型进化迭代")
    min_factor_ic: float = Field(0.03, description="因子IC阈值")


@router.get("/status")
async def get_status():
    """获取RD-Agent配置和状态"""
    from src.rd_agent import RDAgentConfig
    config = RDAgentConfig()
    return {
        "llm_backend": config.llm_backend,
        "llm_model": config.llm_model,
        "work_dir": config.work_dir,
        "max_iterations": config.max_iterations,
    }


@router.post("/factors/mine")
async def mine_factors(
    req: FactorMiningRequest,
    background_tasks: BackgroundTasks,
):
    """启动因子挖掘任务(异步)"""
    from src.rd_agent import FactorRunner
    runner = FactorRunner()

    def run_task():
        result = runner.run_factor_mining(
            max_iterations=req.max_iterations,
        )
        if result.get("success"):
            best = runner.get_best_factors(result, min_ic=req.min_ic)
            logger.info(f"Factor mining complete: {len(best)} effective factors")

    background_tasks.add_task(run_task)

    return {
        "status": "started",
        "message": f"因子挖掘已启动({req.max_iterations}轮)",
        "max_iterations": req.max_iterations,
    }


@router.post("/models/evolve")
async def evolve_models(
    req: ModelEvolutionRequest,
    background_tasks: BackgroundTasks,
):
    """启动模型进化任务(异步)"""
    from src.rd_agent import ModelRunner
    runner = ModelRunner()

    def run_task():
        result = runner.run_model_evolution(
            max_iterations=req.max_iterations,
            base_model=req.base_model,
            target_metric=req.target_metric,
        )
        if result.get("success"):
            config = runner.get_model_config(result)
            logger.info(f"Model evolution complete: best config={config}")

    background_tasks.add_task(run_task)

    return {
        "status": "started",
        "message": f"模型进化已启动({req.max_iterations}轮, base={req.base_model})",
    }


@router.post("/evolution/loop")
async def run_evolution_loop(
    req: EvolutionLoopRequest,
    background_tasks: BackgroundTasks,
):
    """启动完整进化循环(异步)"""
    from src.rd_agent import RDAgentCoordinator
    coordinator = RDAgentCoordinator()

    def run_task():
        result = coordinator.run_evolution_loop(
            max_rounds=req.max_rounds,
            factor_iterations=req.factor_iterations,
            model_iterations=req.model_iterations,
            min_factor_ic=req.min_factor_ic,
        )
        logger.info(f"Evolution loop complete: best IC={result.get('best_metric', 0):.4f}")

    background_tasks.add_task(run_task)

    return {
        "status": "started",
        "message": f"进化循环已启动({req.max_rounds}轮)",
    }


@router.get("/history")
async def get_history():
    """获取RD-Agent进化历史"""
    from src.rd_agent import RDAgentCoordinator
    coordinator = RDAgentCoordinator()
    workspace = coordinator.workspace

    return {
        "total_rounds": len(workspace.get("rounds", [])),
        "total_factors": len(workspace.get("factors", [])),
        "total_models": len(workspace.get("models", [])),
        "best_config": workspace.get("best_config"),
        "last_run": workspace.get("last_run"),
        "rounds": workspace.get("rounds", [])[-10:],
    }


@router.get("/best-config")
async def get_best_config():
    """获取历史最优配置"""
    from src.rd_agent import RDAgentCoordinator
    coordinator = RDAgentCoordinator()
    config = coordinator.get_best_config()

    if not config:
        return {"config": None, "message": "暂无进化结果"}

    return {"config": config}
