"""MLflow UI embedding router.

Provides:
- MLflow experiment/run listing
- MLflow UI iframe URL (for embedding in dashboard)
- Model artifact download links
"""
import logging
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/mlflow", tags=["mlflow"])


@router.get("/experiments")
async def list_experiments():
    """列出所有MLflow实验"""
    try:
        import mlflow
        experiments = mlflow.search_experiments()
        return {
            "experiments": [
                {
                    "experiment_id": e.experiment_id,
                    "name": e.name,
                    "artifact_location": e.artifact_location,
                    "tags": e.tags,
                }
                for e in experiments
            ]
        }
    except ImportError:
        return {"error": "MLflow未安装", "experiments": []}
    except Exception as e:
        return {"error": str(e), "experiments": []}


@router.get("/runs")
async def list_runs(
    experiment_name: str = "quant_platform",
    max_results: int = 20,
):
    """列出实验下的运行记录"""
    try:
        import mlflow
        import pandas as pd

        runs = mlflow.search_runs(
            experiment_names=[experiment_name],
            max_results=max_results,
        )

        if runs.empty:
            return {"runs": []}

        results = []
        for _, row in runs.iterrows():
            results.append({
                "run_id": row.get("run_id", ""),
                "status": row.get("status", ""),
                "start_time": str(row.get("start_time", "")),
                "model_class": row.get("params.model_class", ""),
                "metrics": {
                    k.replace("metrics.", ""): float(v)
                    for k, v in row.items()
                    if k.startswith("metrics.") and pd.notna(v)
                },
            })

        return {"runs": results}
    except ImportError:
        return {"error": "MLflow未安装", "runs": []}
    except Exception as e:
        return {"error": str(e), "runs": []}


@router.get("/ui-url")
async def get_mlflow_ui_url():
    """获取MLflow UI的嵌入URL

    Dashboard前端通过iframe嵌入此URL显示MLflow UI。
    需要先启动MLflow server: mlflow server --host 0.0.0.0 --port 5000
    """
    return {
        "url": "http://localhost:5000",
        "tracking_uri": "http://localhost:5000",
        "instructions": "启动MLflow server: mlflow server --host 0.0.0.0 --port 5000",
    }


@router.get("/models")
async def list_registered_models():
    """列出MLflow注册的模型"""
    try:
        import mlflow
        client = mlflow.tracking.MlflowClient()
        models = client.search_registered_models()

        return {
            "models": [
                {
                    "name": m.name,
                    "versions": [
                        {
                            "version": v.version,
                            "stage": v.current_stage,
                            "run_id": v.run_id,
                        }
                        for v in m.latest_versions
                    ],
                }
                for m in models
            ]
        }
    except ImportError:
        return {"error": "MLflow未安装", "models": []}
    except Exception as e:
        return {"error": str(e), "models": []}


class RegisterModelRequest(BaseModel):
    run_id: str
    model_name: str
    artifact_path: str = "model"


@router.post("/models/register")
async def register_model(req: RegisterModelRequest):
    """注册模型到MLflow Model Registry"""
    try:
        import mlflow
        model_uri = f"runs:/{req.run_id}/{req.artifact_path}"
        result = mlflow.register_model(
            name=req.model_name,
            model_uri=model_uri,
        )
        return {
            "success": True,
            "model_name": result.name,
            "version": result.version,
        }
    except ImportError:
        raise HTTPException(status_code=500, detail="MLflow未安装")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
