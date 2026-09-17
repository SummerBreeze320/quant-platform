from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session
from src.common.db import get_db
from src.models.backtest_record import BacktestRecord
from src.qlib_engine.backtest import BacktestEngine
from src.qlib_engine.attribution import BrinsonAttribution
from src.qlib_engine.factor_analysis import FactorQuantileAnalyzer
from src.qlib_engine.report_generator import QuantReportGenerator
from src.service.schemas.model_schema import BacktestRunRequest, BacktestResponse
import pandas as pd
import numpy as np

router = APIRouter(prefix="/backtest", tags=["Portfolio Backtest"])

@router.post("/run", response_model=BacktestResponse)
def run_backtest_simulation(request: BacktestRunRequest, db: Session = Depends(get_db)):
    """Runs a simulated portfolio backtest given strategy parameters."""
    engine = BacktestEngine(
        top_k=request.top_k,
        benchmark=request.benchmark,
        commission_rate=request.commission_rate,
        stamp_tax_rate=request.stamp_tax_rate,
        slippage=request.slippage,
        neutralize=request.neutralize,
        industry_neutral_allocation=request.industry_neutral_allocation,
        use_convex_optimizer=request.use_convex_optimizer,
        risk_aversion=request.risk_aversion,
        max_stock_weight=request.max_stock_weight,
        industry_tolerance=request.industry_tolerance,
        style_tolerance=request.style_tolerance,
        turnover_penalty=request.turnover_penalty
    )

    # Generate benchmark simulation data
    dates = pd.date_range("2026-08-01", periods=20, freq="B").strftime("%Y-%m-%d").tolist()
    symbols = [f"SZ{i:06d}" for i in range(50)]
    industries = ["Bank", "Tech", "Pharma", "Consumer", "Energy"] * 10
    market_caps = [1e10 + i * 5e8 for i in range(50)]
    records = []
    np.random.seed(42)
    for d in dates:
        for s, ind, cap in zip(symbols, industries, market_caps):
            records.append({
                "date": d,
                "symbol": s,
                "score": float(np.random.randn()),
                "ret": float(np.random.normal(0.0008, 0.015)),
                "industry": ind,
                "market_cap": cap
            })
    pred_df = pd.DataFrame(records)

    report = engine.run_backtest(pred_df)

    record = BacktestRecord(
        model_id=request.model_id,
        strategy_name=request.strategy_name,
        start_date=dates[0],
        end_date=dates[-1],
        benchmark=request.benchmark,
        annualized_return=report["annualized_return"],
        sharpe_ratio=report["sharpe_ratio"],
        max_drawdown=report["max_drawdown"],
        win_rate=report["win_rate"],
        report_details=report
    )
    db.add(record)
    db.commit()

    return {
        "strategy_name": request.strategy_name,
        "total_return": report["total_return"],
        "annualized_return": report["annualized_return"],
        "sharpe_ratio": report["sharpe_ratio"],
        "max_drawdown": report["max_drawdown"],
        "win_rate": report["win_rate"],
        "n_trading_days": report["n_trading_days"],
        "cumulative_returns": report["cumulative_returns"]
    }

@router.post("/report/html", response_class=HTMLResponse)
def generate_backtest_html_report(request: BacktestRunRequest, db: Session = Depends(get_db)):
    """Runs simulation and returns a complete standalone HTML performance & Brinson attribution report."""
    engine = BacktestEngine(
        top_k=request.top_k,
        benchmark=request.benchmark,
        commission_rate=request.commission_rate,
        stamp_tax_rate=request.stamp_tax_rate,
        slippage=request.slippage,
        neutralize=request.neutralize,
        industry_neutral_allocation=request.industry_neutral_allocation
    )

    dates = pd.date_range("2026-08-01", periods=20, freq="B").strftime("%Y-%m-%d").tolist()
    symbols = [f"SZ{i:06d}" for i in range(50)]
    industries = ["Bank", "Tech", "Pharma", "Consumer", "Energy"] * 10
    market_caps = [1e10 + i * 5e8 for i in range(50)]
    records = []
    np.random.seed(42)
    for d in dates:
        for s, ind, cap in zip(symbols, industries, market_caps):
            records.append({
                "date": d,
                "symbol": s,
                "score": float(np.random.randn()),
                "ret": float(np.random.normal(0.0008, 0.015)),
                "industry": ind,
                "market_cap": cap
            })
    pred_df = pd.DataFrame(records)

    report = engine.run_backtest(pred_df)
    report["strategy_name"] = request.strategy_name

    # Run Brinson attribution on latest slice
    last_date = dates[-1]
    last_slice = pred_df[pred_df["date"] == last_date].head(request.top_k).copy()
    last_slice["weight_p"] = 1.0 / len(last_slice)
    last_slice["ret_p"] = last_slice["ret"]

    bench_weights = {"Bank": 0.25, "Tech": 0.25, "Pharma": 0.20, "Consumer": 0.15, "Energy": 0.15}
    bench_returns = {"Bank": 0.001, "Tech": 0.002, "Pharma": -0.001, "Consumer": 0.0015, "Energy": -0.0005}
    brinson_res = BrinsonAttribution.calculate(last_slice, bench_weights, bench_returns)

    # Run Quantile analysis
    quantile_res = FactorQuantileAnalyzer.analyze(pred_df, n_quantiles=5)

    html = QuantReportGenerator.generate_html(
        perf_summary=report,
        brinson_data=brinson_res,
        quantile_data=quantile_res.get("quantile_returns")
    )
    return HTMLResponse(content=html)
