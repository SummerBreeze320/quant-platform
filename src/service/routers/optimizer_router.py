from fastapi import APIRouter, HTTPException
import pandas as pd
import numpy as np
from src.qlib_engine.optimizer import ConvexOptimizer
from src.service.schemas.model_schema import ConvexOptimizationRequest, ConvexOptimizationResponse

router = APIRouter(prefix="/optimizer", tags=["Portfolio Optimizer"])

@router.post("/optimize", response_model=ConvexOptimizationResponse)
def run_portfolio_optimization(request: ConvexOptimizationRequest):
    """
    Solves QP Convex Optimization problem with Barra structural risk and style/industry bounds.
    """
    if not request.alpha_scores:
        raise HTTPException(status_code=400, detail="alpha_scores cannot be empty.")

    symbols = list(request.alpha_scores.keys())
    alpha_series = pd.Series(request.alpha_scores)

    # Reconstruct exposures DataFrame
    exp_df = pd.DataFrame(request.exposures).T.reindex(symbols).fillna(0.0)
    factor_names = list(exp_df.columns)
    if not factor_names:
        raise HTTPException(status_code=400, detail="Exposures factor columns cannot be empty.")

    # Reconstruct factor covariance matrix or use sensible default
    if request.factor_cov:
        fcov_df = pd.DataFrame(request.factor_cov).reindex(index=factor_names, columns=factor_names).fillna(0.0)
    else:
        fcov_df = pd.DataFrame(np.eye(len(factor_names)) * 0.0004, index=factor_names, columns=factor_names)

    # Reconstruct specific variance or default
    if request.specific_var:
        svar_series = pd.Series(request.specific_var).reindex(symbols).fillna(0.0005)
    else:
        svar_series = pd.Series(0.0005, index=symbols)

    benchmark_series = pd.Series(request.benchmark_weights).reindex(symbols).fillna(0.0) if request.benchmark_weights else None
    prev_series = pd.Series(request.prev_weights).reindex(symbols).fillna(0.0) if request.prev_weights else None

    optimizer = ConvexOptimizer(
        risk_aversion=request.risk_aversion,
        max_stock_weight=request.max_stock_weight,
        industry_tolerance=request.industry_tolerance,
        style_tolerance=request.style_tolerance,
        turnover_penalty=request.turnover_penalty
    )

    result = optimizer.optimize(
        alpha=alpha_series,
        exposures=exp_df,
        factor_cov=fcov_df,
        specific_var=svar_series,
        benchmark_weights=benchmark_series,
        prev_weights=prev_series,
        industry_factors=request.industry_factors,
        style_factors=request.style_factors
    )

    # Filter out negligible weights (< 1e-5)
    weights_dict = {
        s: float(round(w, 6))
        for s, w in result["weights"].items()
        if w > 1e-5
    }

    return ConvexOptimizationResponse(
        status=result["status"],
        expected_alpha=result["expected_alpha"],
        predicted_tracking_error=result["predicted_tracking_error"],
        turnover=result["turnover"],
        active_exposures={k: round(v, 6) for k, v in result["active_exposures"].items()},
        weights=weights_dict
    )
