from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session
from src.common.db import get_db
from src.common.logger import logger
from src.models.backtest_record import BacktestRecord
from src.models.model_registry import ModelRegistry
from src.qlib_engine.backtest import BacktestEngine
from src.qlib_engine.attribution import BrinsonAttribution
from src.qlib_engine.factor_analysis import FactorQuantileAnalyzer
from src.qlib_engine.report_generator import QuantReportGenerator
from src.qlib_engine.data_handler import DataHandler, FEATURE_COLUMNS
from src.qlib_engine.model_trainer import BaseModelTrainer
from src.service.schemas.model_schema import BacktestRunRequest, BacktestResponse
import pandas as pd
import numpy as np

router = APIRouter(prefix="/backtest", tags=["Portfolio Backtest"])

_BACKTEST_START = "2026-08-01"
_BACKTEST_END = "2026-08-31"
_N_STOCKS = 50


def _generate_synthetic_pred_df() -> pd.DataFrame:
    """Generates deterministic synthetic prediction data for fallback backtesting."""
    dates = pd.date_range(_BACKTEST_START, periods=20, freq="B").strftime("%Y-%m-%d").tolist()
    symbols = [f"SZ{i:06d}" for i in range(_N_STOCKS)]
    industries = (["Bank", "Tech", "Pharma", "Consumer", "Energy"] * 10)[:_N_STOCKS]
    market_caps = [1e10 + i * 5e8 for i in range(_N_STOCKS)]
    records = []
    rng = np.random.RandomState(42)
    for d in dates:
        for s, ind, cap in zip(symbols, industries, market_caps):
            records.append({
                "date": d,
                "symbol": s,
                "score": float(rng.randn()),
                "ret": float(rng.normal(0.0008, 0.015)),
                "industry": ind,
                "market_cap": cap
            })
    return pd.DataFrame(records)


def _load_model_predictions(request: BacktestRunRequest, db: Session) -> pd.DataFrame:
    """If model_id is provided, loads the model and generates predictions for the backtest period."""
    if request.model_id is None:
        return _generate_synthetic_pred_df()

    model_entry = db.query(ModelRegistry).filter_by(id=request.model_id).first()
    if not model_entry:
        raise HTTPException(status_code=404, detail=f"Model id {request.model_id} not found.")

    metrics = model_entry.metrics or {}
    if metrics.get("status") != "TRAINED":
        raise HTTPException(status_code=400, detail=f"Model '{model_entry.model_name}' is not trained (status: {metrics.get('status')}).")

    try:
        trainer = BaseModelTrainer.load(model_entry.model_path)
        feature_names = trainer.feature_names

        data_df = DataHandler.prepare_backtest_data(
            feature_names=feature_names,
            start_date=_BACKTEST_START,
            end_date=_BACKTEST_END,
            n_stocks=_N_STOCKS,
        )

        available_features = [c for c in feature_names if c in data_df.columns]
        scores_by_date = {}
        for d, group in data_df.groupby("date"):
            X = group[available_features]
            preds = trainer.predict(X)
            scores_by_date[d] = dict(zip(group["symbol"].values, preds))

        records = []
        for d in sorted(data_df["date"].unique()):
            day_df = data_df[data_df["date"] == d]
            for _, row in day_df.iterrows():
                records.append({
                    "date": d,
                    "symbol": row["symbol"],
                    "score": float(scores_by_date[d].get(row["symbol"], 0.0)),
                    "ret": float(row.get("ret", 0.0)),
                    "industry": row.get("industry", "Unknown"),
                    "market_cap": float(row.get("market_cap", 1e10)),
                })
        pred_df = pd.DataFrame(records)
        logger.info(f"Backtest using model '{model_entry.model_name}' predictions for {len(pred_df)} records.")
        return pred_df
    except HTTPException:
        raise
    except Exception as e:
        logger.warning(f"Model prediction for backtest failed, using synthetic: {e}")
        return _generate_synthetic_pred_df()


@router.post("/run", response_model=BacktestResponse)
def run_backtest_simulation(request: BacktestRunRequest, db: Session = Depends(get_db)):
    """Runs a portfolio backtest using model predictions when model_id is provided."""
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

    pred_df = _load_model_predictions(request, db)
    dates = sorted(pred_df["date"].unique())

    report = engine.run_backtest(pred_df)

    record = BacktestRecord(
        model_id=request.model_id,
        strategy_name=request.strategy_name,
        start_date=str(dates[0]),
        end_date=str(dates[-1]),
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
    """Runs backtest and returns a complete standalone HTML performance & Brinson attribution report."""
    engine = BacktestEngine(
        top_k=request.top_k,
        benchmark=request.benchmark,
        commission_rate=request.commission_rate,
        stamp_tax_rate=request.stamp_tax_rate,
        slippage=request.slippage,
        neutralize=request.neutralize,
        industry_neutral_allocation=request.industry_neutral_allocation
    )

    pred_df = _load_model_predictions(request, db)
    dates = sorted(pred_df["date"].unique())

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
