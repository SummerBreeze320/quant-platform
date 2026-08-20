"""Research API routes: model zoo, factor library, handlers, backtest, MLflow.

Phase 2 API endpoints for the research layer.
"""
import logging
import tempfile
from pathlib import Path
from typing import Optional, List, Dict, Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/research", tags=["research"])

QLIB_DIR = str(Path(__file__).resolve().parents[2] / "data" / "qlib_bin")


class HandlerRequest(BaseModel):
    handler_type: str = Field("alpha158", description="alpha158, alpha360, or fundamental")
    start_time: str = "2010-01-01"
    end_time: str = "2024-12-31"
    fit_start_time: str = "2010-01-01"
    fit_end_time: str = "2018-12-31"
    instruments: str = "csi300"


class TrainRequest(BaseModel):
    model_name: str = "lightgbm"
    handler_type: str = "alpha158"
    start_time: str = "2010-01-01"
    end_time: str = "2024-12-31"
    fit_start_time: str = "2010-01-01"
    fit_end_time: str = "2018-12-31"
    instruments: str = "csi300"
    experiment_name: str = "quant_platform"
    override_params: Optional[Dict[str, Any]] = None


class BacktestRequest(BaseModel):
    model_name: str = "lightgbm"
    handler_type: str = "alpha158"
    topk: int = 50
    n_drop: int = 5
    benchmark: str = "SH000300"
    start_time: str = "2010-01-01"
    end_time: str = "2024-12-31"
    fit_start_time: str = "2010-01-01"
    fit_end_time: str = "2018-12-31"
    instruments: str = "csi300"


class FactorComputeRequest(BaseModel):
    instruments: List[str] = Field(["SH600000"])
    factor_names: Optional[List[str]] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None


@router.get("/models")
async def list_models(model_type: Optional[str] = None):
    """List available models in the model zoo."""
    from src.research import list_models as _list_models
    return {"models": _list_models(model_type=model_type)}


@router.get("/factors")
async def list_factors(category: Optional[str] = None):
    """List available factors in the factor library."""
    from src.research import list_factors as _list_factors
    return {"factors": _list_factors(category=category)}


@router.get("/strategies")
async def list_strategies():
    """List available unified strategy types."""
    return {
        "strategies": [
            {"name": "topk_dropout", "description": "Top-K dropout stock selection"},
            {"name": "enhanced_indexing", "description": "Enhanced indexing with factor tilts"},
            {"name": "etf_rotation", "description": "ETF momentum rotation"},
        ]
    }


@router.post("/factors/compute")
async def compute_factors(req: FactorComputeRequest):
    """Compute factor values for given instruments."""
    from src.core import ensure_qlib
    from src.research import compute_factors

    try:
        ensure_qlib(QLIB_DIR)
        df = compute_factors(
            instruments=req.instruments,
            factor_names=req.factor_names,
            start_time=req.start_time,
            end_time=req.end_time,
        )
        if df is None or df.empty:
            return {"error": "No data", "data": []}
        df_reset = df.reset_index()
        records = df_reset.head(500).to_dict(orient="records")
        return {
            "total": len(df),
            "returned": len(records),
            "columns": list(df.columns),
            "data": records,
        }
    except Exception as e:
        logger.error(f"Factor computation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/model/train")
async def train_model(req: TrainRequest):
    """Train a model and log to MLflow."""
    from src.core import ensure_qlib
    from src.research import get_handler, create_dataset, create_model, train_model as _train_model

    try:
        ensure_qlib(QLIB_DIR)

        handler = get_handler(
            handler_type=req.handler_type,
            start_time=req.start_time,
            end_time=req.end_time,
            fit_start_time=req.fit_start_time,
            fit_end_time=req.fit_end_time,
            instruments=req.instruments,
        )
        dataset = create_dataset(handler)
        model = create_model(req.model_name, req.override_params)

        result = _train_model(
            model=model,
            dataset=dataset,
            experiment_name=req.experiment_name,
            run_name=f"{req.model_name}_{req.handler_type}",
            use_mlflow=True,
        )

        return {
            "status": "trained",
            "model": req.model_name,
            "handler": req.handler_type,
            "metrics": result.get("metrics", {}),
            "mlflow": result.get("mlflow", {}),
        }
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/backtest")
async def run_backtest(req: BacktestRequest):
    """Run a full backtest: train model → backtest with TopkDropout."""
    from src.core import ensure_qlib
    from src.research import get_handler, create_dataset, create_model, train_model, run_backtest_daily

    try:
        ensure_qlib(QLIB_DIR)

        handler = get_handler(
            handler_type=req.handler_type,
            start_time=req.start_time,
            end_time=req.end_time,
            fit_start_time=req.fit_start_time,
            fit_end_time=req.fit_end_time,
            instruments=req.instruments,
        )
        dataset = create_dataset(handler)
        model = create_model(req.model_name)
        train_result = train_model(model, dataset)
        trained_model = train_result["model"]

        result = run_backtest_daily(
            dataset=dataset,
            model=trained_model,
            topk=req.topk,
            n_drop=req.n_drop,
            benchmark=req.benchmark,
        )

        analysis = result.get("analysis", {})
        return {
            "status": "complete",
            "model": req.model_name,
            "topk": req.topk,
            "n_drop": req.n_drop,
            "analysis": analysis,
        }
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/mlflow/runs")
async def list_mlflow_runs(experiment: str = "quant_platform", limit: int = 20):
    """List recent MLflow runs."""
    try:
        import mlflow
        mlflow.set_experiment(experiment)
        runs = mlflow.search_runs(max_results=limit)
        if runs.empty:
            return {"runs": []}
        runs_dict = runs.head(limit).to_dict(orient="records")
        return {"runs": runs_dict, "total": len(runs)}
    except ImportError:
        return {"error": "MLflow not installed"}
    except Exception as e:
        return {"error": str(e)}


@router.get("/mlflow/experiments")
async def list_mlflow_experiments():
    """List all MLflow experiments."""
    try:
        import mlflow
        experiments = mlflow.search_experiments()
        return {
            "experiments": [
                {
                    "id": e.experiment_id,
                    "name": e.name,
                    "artifact_location": e.artifact_location,
                    "status": e.lifecycle_stage,
                }
                for e in experiments
            ]
        }
    except ImportError:
        return {"error": "MLflow not installed"}
    except Exception as e:
        return {"error": str(e)}


@router.get("/health")
async def research_health():
    """Check research layer health."""
    from src.core import ensure_qlib, is_initialized
    checks = {
        "qlib_initialized": is_initialized(),
        "mlflow_available": False,
        "lightgbm_available": False,
        "torch_available": False,
    }
    try:
        import mlflow
        checks["mlflow_available"] = True
    except ImportError:
        pass
    try:
        import lightgbm
        checks["lightgbm_available"] = True
    except ImportError:
        pass
    try:
        import torch
        checks["torch_available"] = True
    except ImportError:
        pass
    return checks
