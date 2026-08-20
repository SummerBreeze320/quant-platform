"""Dashboard API routes.

Two tiers of endpoints:
1. /api/qlib/* — offline endpoints that read from local Qlib bin files
   and work without MySQL/Wind. These power the interactive dashboard.
2. /api/* — database-backed endpoints for strategy/backtest/factor
   persistence (requires MySQL).
"""
import logging
import json
import math
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Optional, List
from pathlib import Path
from src.core import (
    ensure_qlib,
    get_field,
    get_features,
    list_instruments as _list_instruments,
    get_ohlcv as _get_ohlcv,
)
from src.core.data_access import normalize_code

logger = logging.getLogger(__name__)


def _safe_float(v):
    """Convert numpy/pandas floats to JSON-safe Python floats."""
    if v is None:
        return None
    try:
        f = float(v)
        if math.isnan(f) or math.isinf(f):
            return 0.0
        return round(f, 6)
    except (TypeError, ValueError):
        return 0.0

# Router for offline (no-DB) endpoints — mounted at /api/qlib
qlib_router = APIRouter(prefix="/api/qlib", tags=["offline"])

# Router for DB-backed endpoints — mounted at /api
router = APIRouter(prefix="/api", tags=["database"])

QLIB_DIR = "data/qlib_bin"

# Instrument code → friendly name mapping for A-share stocks and ETFs
INSTRUMENT_NAMES = {
    # Stocks
    "SH600000": "浦发银行", "SH600519": "贵州茅台", "SH600036": "招商银行",
    "SH600276": "恒瑞医药", "SH600030": "中信证券", "SH601318": "中国平安",
    "SH601166": "兴业银行", "SH600887": "伊利股份", "SH600016": "民生银行",
    "SH601288": "农业银行", "SZ000001": "平安银行", "SZ000002": "万科A",
    "SZ000333": "美的集团", "SZ000651": "格力电器", "SZ000858": "五粮液",
    "SZ002594": "比亚迪", "SZ000568": "泸州老窖", "SZ002415": "海康威视",
    "SZ000725": "京东方A", "SZ002230": "科大讯飞", "SZ300750": "宁德时代",
    "SZ300059": "东方财富", "SZ300015": "爱尔眼科", "SZ300760": "迈瑞医疗",
    "SZ300124": "汇川技术",
    # ETFs
    "SH510300": "沪深300ETF", "SH510050": "上证50ETF", "SH510500": "中证500ETF",
    "SH588000": "科创50ETF", "SH588050": "科创芯片ETF", "SZ159915": "创业板ETF",
    "SZ159949": "创业板50ETF", "SH512100": "中证1000ETF", "SH512760": "半导体ETF",
    "SH515790": "光伏ETF", "SH510310": "沪深300ETF易方达", "SH515330": "新能源车ETF",
    "SH512010": "医药ETF", "SH512200": "房地产ETF", "SH515050": "5G通信ETF",
    "SH512880": "证券ETF", "SH515950": "央企ETF", "SH516160": "新能源ETF",
    "SH513100": "纳指ETF", "SH513050": "中概互联ETF",
}


# ─── Pydantic models ───

class StrategyCreate(BaseModel):
    name: str
    type: str
    config: dict = {}
    description: str = ""

class BacktestRequest(BaseModel):
    strategy_id: int
    start_date: str
    end_date: str
    benchmark: str = "SH000300"
    init_cash: float = 1000000

class StrategyBacktestRequest(BaseModel):
    strategy_type: str
    code: str
    params: dict = {}
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    init_cash: float = 1000000

class FactorEvalRequest(BaseModel):
    code: str
    forward_period: int = 10

class MultiFactorBacktestRequest(BaseModel):
    codes: List[str] = []
    combiner_method: str = "ic_weighted"
    objective: str = "max_sharpe"
    top_n: int = 10
    rebalance_freq: str = "W"
    max_position: float = 0.15
    init_cash: float = 1000000
    forward_period: int = 5
    start_date: Optional[str] = None
    end_date: Optional[str] = None

class PortfolioOptimizeRequest(BaseModel):
    scores: dict
    objective: str = "max_sharpe"
    top_n: int = 10
    max_position: float = 0.05
    lookback: int = 60

class SignalGenRequest(BaseModel):
    codes: List[str] = []
    combiner_method: str = "ic_weighted"
    top_n: int = 5
    max_position: float = 0.15
    objective: str = "max_sharpe"
    as_of_date: Optional[str] = None

class TradingSimRequest(BaseModel):
    codes: List[str] = []
    initial_cash: float = 1000000
    combiner_method: str = "ic_weighted"
    top_n: int = 5
    max_position: float = 0.15
    start_date: Optional[str] = None
    end_date: Optional[str] = None


# ─── Offline endpoints (no DB needed) ───

@qlib_router.get("/instruments")
async def list_instruments():
    """List all instruments with friendly names."""
    ensure_qlib(QLIB_DIR)
    raw = _list_instruments()
    instruments = [
        {"code": code, "name": INSTRUMENT_NAMES.get(code, code)}
        for code in raw
    ]
    return {
        "total": len(instruments),
        "instruments": instruments,
    }


@qlib_router.get("/strategies")
async def list_available_strategies():
    """List all available strategy types with metadata."""
    strategies = [
        {
            "id": "ma_trend", "name": "均线趋势", "category": "ETF趋势",
            "params": {"short_window": 5, "long_window": 20},
            "description": "5日/20日均线金叉买入、死叉卖出",
        },
        {
            "id": "breakout", "name": "N日突破", "category": "ETF趋势",
            "params": {"window": 20},
            "description": "突破N日最高价买入、跌破最低价卖出",
        },
        {
            "id": "dual_ma", "name": "三均线", "category": "ETF趋势",
            "params": {"w1": 5, "w2": 10, "w3": 20},
            "description": "5>10>20日均线多头排列时持有",
        },
        {
            "id": "donchian", "name": "唐奇安通道", "category": "ETF趋势",
            "params": {"entry_window": 20, "exit_window": 10},
            "description": "入场20日突破、出场10日低点",
        },
        {
            "id": "bollinger", "name": "布林带回归", "category": "均值回归",
            "params": {"window": 20, "num_std": 2.0},
            "description": "跌破下轨买入、突破上轨卖出",
        },
        {
            "id": "momentum_stock", "name": "动量因子", "category": "股票因子",
            "params": {"window": 20},
            "description": "过去N日收益率动量因子选股",
        },
    ]
    return {"strategies": strategies}


@qlib_router.post("/backtest")
async def run_strategy_backtest(req: StrategyBacktestRequest):
    """Run a strategy backtest using local Qlib bin data (no DB needed)."""
    from src.backtest.backtest_engine import BacktestEngine
    from src.research.strategy_unified import create_strategy, BaseStrategy

    params = req.params or {}

    # Map frontend strategy names to unified strategy types
    strategy_map = {
        "ma_trend": "etf_rotation",
        "breakout": "etf_rotation",
        "dual_ma": "etf_rotation",
        "donchian": "etf_rotation",
        "bollinger": "etf_rotation",
        "momentum_stock": "topk_dropout",
        "topk_dropout": "topk_dropout",
        "enhanced_indexing": "enhanced_indexing",
        "etf_rotation": "etf_rotation",
    }

    strategy_type = strategy_map.get(req.strategy_type, "etf_rotation")

    if strategy_type == "topk_dropout":
        strategy = create_strategy(
            "topk_dropout",
            topk=params.get("topk", 5),
            n_drop=params.get("n_drop", 1),
        )
    elif strategy_type == "enhanced_indexing":
        strategy = create_strategy(
            "enhanced_indexing",
            benchmark=params.get("benchmark", "SH000300"),
            tracking_error_limit=params.get("tracking_error_limit", 0.05),
        )
    else:
        etf_codes = [req.code] if req.code else ["SH510300"]
        strategy = create_strategy(
            "etf_rotation",
            etf_codes=etf_codes,
            lookback=params.get("lookback", 20),
            top_k=params.get("top_k", 1),
        )

    engine = BacktestEngine(qlib_dir=QLIB_DIR)
    result = engine.run_strategy_backtest(
        strategy=strategy,
        code=req.code,
        start_date=req.start_date,
        end_date=req.end_date,
        init_cash=req.init_cash,
    )

    if "error" in result:
        raise HTTPException(400, result["error"])

    # Serialize for JSON (sanitize NaN/Inf)
    equity_data = []
    for date, val in result["equity_curve"].items():
        equity_data.append({
            "date": date.strftime("%Y-%m-%d"),
            "value": _safe_float(val),
        })

    daily_returns_data = []
    for date, val in result["daily_returns"].items():
        daily_returns_data.append({
            "date": date.strftime("%Y-%m-%d"),
            "value": _safe_float(val),
        })

    # Buy & hold benchmark: normalize price to start at init_cash
    benchmark_data = []
    prices = result.get("signals", pd.Series(dtype=float))
    # Read close price directly for benchmark
    try:
        ensure_qlib(QLIB_DIR)
        close_series = get_field(req.code, "close")
        if close_series is not None:
            # Align to equity curve dates
            eq_index = result["equity_curve"].index
            close_aligned = close_series.reindex(eq_index).ffill().bfill()
            if len(close_aligned) > 0 and close_aligned.iloc[0] > 0:
                ratio = result["init_cash"] / close_aligned.iloc[0]
                bench_values = (close_aligned * ratio).values
                for i, (date, _) in enumerate(result["equity_curve"].items()):
                    benchmark_data.append({
                        "date": date.strftime("%Y-%m-%d"),
                        "value": _safe_float(bench_values[i]),
                    })
    except Exception as e:
        logger.warning(f"Benchmark calc failed: {e}")

    # Drawdown series
    drawdown_data = []
    eq_values = [d["value"] for d in equity_data]
    peak = eq_values[0] if eq_values else 1.0
    for i, v in enumerate(eq_values):
        peak = max(peak, v)
        dd = (v - peak) / peak if peak > 0 else 0.0
        drawdown_data.append({
            "date": equity_data[i]["date"],
            "value": _safe_float(dd),
        })

    # Benchmark metrics
    bench_final = benchmark_data[-1]["value"] if benchmark_data else 0
    bench_return = (bench_final - result["init_cash"]) / result["init_cash"] if result["init_cash"] else 0

    metrics = {}
    for k, v in result["metrics"].items():
        metrics[k] = _safe_float(v)
    metrics["benchmark_return"] = _safe_float(bench_return)
    metrics["excess_return"] = _safe_float(metrics.get("total_return", 0) - bench_return)

    return {
        "strategy": result["strategy"],
        "code": result["code"],
        "code_name": INSTRUMENT_NAMES.get(result["code"], result["code"]),
        "metrics": metrics,
        "equity_curve": equity_data,
        "benchmark_curve": benchmark_data,
        "drawdown_curve": drawdown_data,
        "daily_returns": daily_returns_data,
        "init_cash": _safe_float(result["init_cash"]),
        "final_value": _safe_float(result["final_value"]),
        "benchmark_final": _safe_float(bench_final),
        "data_points": result["data_points"],
        "start_date": result["start_date"],
        "end_date": result["end_date"],
    }


@qlib_router.post("/factors/evaluate")
async def evaluate_factors(req: FactorEvalRequest):
    """Evaluate Qlib expression factors on a single instrument."""
    from src.research.factor_library import QLIB_EXPRESSION_FACTORS, FACTOR_CATEGORIES

    ensure_qlib(QLIB_DIR)

    # Read OHLCV
    df = _get_ohlcv(req.code)
    if df is None or df.empty:
        raise HTTPException(400, f"No data for {req.code}")

    df = df[(df != 0).any(axis=1)]
    if len(df) < 30:
        raise HTTPException(400, f"Insufficient data for {req.code}: {len(df)} rows")

    # Use momentum + volatility factors from Qlib expression library
    factor_names = FACTOR_CATEGORIES["momentum"] + FACTOR_CATEGORIES["volatility"][:3]

    forward_return = df["close"].pct_change(req.forward_period).shift(-req.forward_period)

    results = []
    for fname in factor_names:
        expr = QLIB_EXPRESSION_FACTORS.get(fname)
        if not expr:
            continue

        # Compute factor value using pandas (simplified Qlib expression)
        close = df["close"]
        open_ = df.get("open", close)
        high = df.get("high", close)
        low = df.get("low", close)
        volume = df.get("volume", pd.Series(1, index=df.index))

        try:
            if "Ref($close, 5)" in expr and "/ $close - 1" in expr:
                fv = close.shift(5) / close - 1
            elif "Ref($close, 10)" in expr and "/ $close - 1" in expr:
                fv = close.shift(10) / close - 1
            elif "Ref($close, 20)" in expr and "/ $close - 1" in expr:
                fv = close.shift(20) / close - 1
            elif "Ref($close, 60)" in expr and "/ $close - 1" in expr:
                fv = close.shift(60) / close - 1
            elif "Mean($close, 5)" in expr:
                fv = close.rolling(5).mean() / close - 1
            elif "Mean($close, 20)" in expr:
                fv = close.rolling(20).mean() / close - 1
            elif "Mean($close, 60)" in expr:
                fv = close.rolling(60).mean() / close - 1
            elif "Std($close, 5)" in expr:
                fv = close.rolling(5).std() / close
            elif "Std($close, 20)" in expr:
                fv = close.rolling(20).std() / close
            elif "Std($close, 60)" in expr:
                fv = close.rolling(60).std() / close
            else:
                fv = close.pct_change(20)

            category = "momentum" if fname.startswith(("rocp", "ma")) else "volatility"

            valid = fv.notna() & forward_return.notna()
            if valid.sum() < 10:
                results.append({
                    "name": fname,
                    "category": category,
                    "ic": 0.0, "rank_ic": 0.0, "valid_count": int(valid.sum()),
                })
                continue

            ic = _safe_float(fv[valid].corr(forward_return[valid]))
            rank_ic = _safe_float(fv[valid].rank().corr(forward_return[valid].rank()))
            results.append({
                "name": fname,
                "category": category,
                "ic": ic, "rank_ic": rank_ic, "valid_count": int(valid.sum()),
            })
        except Exception as e:
            logger.warning(f"Factor {fname} failed: {e}")
            results.append({
                "name": fname,
                "category": "unknown",
                "ic": 0.0, "rank_ic": 0.0, "valid_count": 0,
            })

    return {"factors": results, "forward_period": req.forward_period}


@qlib_router.get("/price/{code}")
async def get_price_preview(code: str, field: str = "close"):
    """Get price preview for an instrument."""
    ensure_qlib(QLIB_DIR)

    series = get_field(code, field)
    if series is None:
        raise HTTPException(404, f"No data for {code}")

    data = []
    for date, val in series.items():
        data.append({"date": date.strftime("%Y-%m-%d"), "value": round(float(val), 4)})

    return {
        "code": code,
        "field": field,
        "data": data[-250:],  # last 250 days
    }


@qlib_router.get("/ohlcv/{code}")
async def get_ohlcv(code: str):
    """Get OHLCV data for charting."""
    ensure_qlib(QLIB_DIR)

    df = _get_ohlcv(code)
    if df is None or df.empty:
        raise HTTPException(404, f"No data for {code}")

    df = df[(df != 0).any(axis=1)]

    ohlcv = []
    for date, row in df.iterrows():
        ohlcv.append({
            "date": date.strftime("%Y-%m-%d"),
            "open": round(float(row.get("open", 0)), 4),
            "high": round(float(row.get("high", 0)), 4),
            "low": round(float(row.get("low", 0)), 4),
            "close": round(float(row.get("close", 0)), 4),
            "volume": round(float(row.get("volume", 0)), 0),
        })
    # Return last 500 bars for performance
    return {"code": code, "data": ohlcv[-500:]}


@qlib_router.get("/health")
async def offline_health():
    """Check if local data is available (no DB needed)."""
    ensure_qlib(QLIB_DIR)
    instruments = _list_instruments()

    cal_path = Path(QLIB_DIR) / "calendars" / "day.txt"
    cal_count = 0
    cal_range = ""
    if cal_path.exists():
        lines = cal_path.read_text(encoding="utf-8").strip().split("\n")
        cal_count = len(lines)
        if cal_count > 0:
            cal_range = f"{lines[0]} ~ {lines[-1]}"

    inst_path = Path(QLIB_DIR) / "instruments"
    inst_files = list(inst_path.glob("*.txt")) if inst_path.exists() else []

    return {
        "data_available": len(instruments) > 0,
        "total_instruments": len(instruments),
        "calendar_days": cal_count,
        "calendar_range": cal_range,
        "instrument_files": [f.name for f in inst_files],
        "sample_instruments": instruments[:20],
    }


@qlib_router.get("/factors/library")
async def list_factor_library():
    """List all available Qlib expression factors with metadata."""
    from src.research.factor_library import list_factors

    factors = list_factors()
    return {"factors": factors}


@qlib_router.post("/multi-factor/backtest")
async def run_multi_factor_backtest(req: MultiFactorBacktestRequest):
    """Run multi-factor portfolio backtest using Qlib expression factors."""
    from src.execution.signal_generator import SignalGenerator
    from src.backtest.backtest_engine import BacktestEngine

    # Auto-select all stock instruments if not specified
    if not req.codes:
        ensure_qlib(QLIB_DIR)
        all_inst = _list_instruments()
        req.codes = [c for c in all_inst if c.startswith("SH6") or c.startswith("SZ0")][:20]

    # Generate signal using Qlib factor-based SignalGenerator
    gen = SignalGenerator(
        qlib_dir=QLIB_DIR,
        combiner_method=req.combiner_method,
        top_n=req.top_n,
        max_position=req.max_position,
        forward_period=req.forward_period,
    )

    signal = gen.generate(
        codes=req.codes,
        as_of_date=req.end_date,
        objective=req.objective,
    )

    if "error" in signal:
        raise HTTPException(400, signal["error"])

    # Run backtest with the target weights
    engine = BacktestEngine(qlib_dir=QLIB_DIR)
    target_weights = signal.get("target_weights", {})

    # Use first selected code for single-instrument backtest demo
    if target_weights:
        primary_code = list(target_weights.keys())[0]
        result = engine.run_simple_backtest(
            signals=pd.DataFrame({primary_code: 1.0}, index=[signal["signal_date"]]),
            init_cash=req.init_cash,
        )
    else:
        raise HTTPException(400, "No target weights generated")

    if "error" in result:
        raise HTTPException(400, result["error"])

    # Serialize results
    equity_data = []
    for date, val in result.get("equity_curve", pd.Series(dtype=float)).items():
        equity_data.append({
            "date": date.strftime("%Y-%m-%d") if hasattr(date, "strftime") else str(date),
            "nav": _safe_float(val),
        })

    daily_returns_data = []
    for date, val in result.get("daily_returns", pd.Series(dtype=float)).items():
        daily_returns_data.append({
            "date": date.strftime("%Y-%m-%d") if hasattr(date, "strftime") else str(date),
            "value": _safe_float(val),
        })

    # Drawdown
    nav_values = [d["nav"] for d in equity_data]
    drawdown_data = []
    peak = nav_values[0] if nav_values else 1.0
    for i, v in enumerate(nav_values):
        peak = max(peak, v)
        dd = (v - peak) / peak if peak > 0 else 0.0
        drawdown_data.append({
            "date": equity_data[i]["date"],
            "value": _safe_float(dd),
        })

    # Factor weights from signal
    factor_weights = {
        k: _safe_float(v) for k, v in signal.get("factor_weights", {}).items()
    }

    metrics = {
        "total_return": _safe_float((nav_values[-1] / nav_values[0] - 1) if len(nav_values) > 1 else 0),
        "n_rebalances": 1,
        "n_stocks": len(target_weights),
    }

    return {
        "metrics": metrics,
        "equity_curve": equity_data,
        "drawdown_curve": drawdown_data,
        "daily_returns": daily_returns_data,
        "trades_summary": [{"date": signal["signal_date"], "n_trades": len(target_weights)}],
        "cost_breakdown": [],
        "factor_weights": factor_weights,
        "n_stocks": len(target_weights),
        "n_rebalances": 1,
        "target_weights": {k: _safe_float(v) for k, v in target_weights.items()},
        "signal_date": signal["signal_date"],
    }


@qlib_router.post("/portfolio/optimize")
async def optimize_portfolio(req: PortfolioOptimizeRequest):
    """Optimize portfolio weights from factor scores using EnhancedIndexing."""
    from src.research.enhanced_optimizer import EnhancedIndexingOptimizer

    scores = pd.Series(req.scores)
    scores = scores.dropna()

    if len(scores) < 3:
        raise HTTPException(400, "Need at least 3 stocks")

    # Load returns for covariance
    ensure_qlib(QLIB_DIR)
    all_returns = {}
    for code in scores.index:
        close = get_field(code, "close")
        if close is not None and len(close) > 20:
            all_returns[code] = close.pct_change()

    if len(all_returns) < 3:
        raise HTTPException(400, "Not enough price data")

    returns_df = pd.DataFrame(all_returns).tail(req.lookback).dropna(axis=1, how="all")
    common = scores.index.intersection(returns_df.columns)
    scores = scores.loc[common]
    returns_df = returns_df[common]

    optimizer = EnhancedIndexingOptimizer(weights_limit=req.max_position)
    cov_matrix = returns_df.cov()
    factor_matrix = pd.DataFrame(index=scores.index)

    weights = optimizer.optimize(
        pred_scores=scores,
        factor_matrix=factor_matrix,
        cov_matrix=cov_matrix,
    )

    # Non-zero weights
    weights = weights[weights > 1e-6].sort_values(ascending=False)

    return {
        "weights": {k: _safe_float(v) for k, v in weights.items()},
        "n_holdings": len(weights),
        "objective": req.objective,
    }


@qlib_router.post("/risk/analyze")
async def analyze_risk(req: MultiFactorBacktestRequest):
    """Run risk analysis using Qlib factor-based signal and unified risk control."""
    from src.execution.signal_generator import SignalGenerator
    from src.execution.unified_risk import UnifiedRiskController
    from src.backtest.risk_metrics import calc_all_risk_metrics

    if not req.codes:
        ensure_qlib(QLIB_DIR)
        all_inst = _list_instruments()
        req.codes = [c for c in all_inst if c.startswith("SH6") or c.startswith("SZ0")][:20]

    # Generate signal
    gen = SignalGenerator(
        qlib_dir=QLIB_DIR,
        combiner_method=req.combiner_method,
        top_n=req.top_n,
        max_position=req.max_position,
    )
    signal = gen.generate(codes=req.codes, objective=req.objective)

    if "error" in signal:
        raise HTTPException(400, signal["error"])

    # Unified risk check
    risk = UnifiedRiskController()
    risk_passed, risk_msg = risk.signal_check(
        signal.get("target_weights", {}),
        signal.get("factor_scores", {}),
    )

    # Calculate risk metrics from price data
    all_returns = {}
    for code in signal.get("target_weights", {}):
        close = get_field(code, "close")
        if close is not None and len(close) > 20:
            all_returns[code] = close.pct_change()

    if all_returns:
        returns_df = pd.DataFrame(all_returns)
        weights = pd.Series(signal.get("target_weights", {}))
        common = weights.index.intersection(returns_df.columns)
        if len(common) > 0:
            port_returns = (returns_df[common] * weights.loc[common]).sum(axis=1).dropna()
            risk_metrics = calc_all_risk_metrics(port_returns)
        else:
            risk_metrics = {}
    else:
        risk_metrics = {}

    def _safe_dict(d):
        if isinstance(d, dict):
            return {k: _safe_dict(v) for k, v in d.items()}
        elif isinstance(d, list):
            return [_safe_dict(v) for v in d]
        else:
            return _safe_float(d) if isinstance(d, (int, float, np.floating, np.integer)) else d

    return {
        "performance": {
            "total_return": risk_metrics.get("total_return", 0),
            "annual_return": risk_metrics.get("annual_return", 0),
            "sharpe_ratio": risk_metrics.get("sharpe_ratio", 0),
        },
        "drawdown": {
            "max_drawdown": risk_metrics.get("max_drawdown", 0),
            "calmar_ratio": risk_metrics.get("calmar_ratio", 0),
        },
        "risk_alerts": risk.get_alerts(),
        "risk_passed": risk_passed,
        "risk_message": risk_msg if not risk_passed else "OK",
        "advanced_metrics": _safe_dict(risk_metrics),
        "target_weights": {k: _safe_float(v) for k, v in signal.get("target_weights", {}).items()},
        "signal_date": signal.get("signal_date", ""),
    }


@qlib_router.post("/trading/signal")
async def generate_trading_signal(req: SignalGenRequest):
    """生成交易信号（目标持仓）"""
    from src.execution.signal_generator import SignalGenerator

    if not req.codes:
        ensure_qlib(QLIB_DIR)
        all_inst = _list_instruments()
        req.codes = [c for c in all_inst if c.startswith("SH6") or c.startswith("SZ0")][:15]

    gen = SignalGenerator(
        qlib_dir=QLIB_DIR,
        combiner_method=req.combiner_method,
        top_n=req.top_n,
        max_position=req.max_position,
    )
    signal = gen.generate(
        codes=req.codes,
        as_of_date=req.as_of_date,
        objective=req.objective,
    )

    if "error" in signal:
        raise HTTPException(400, signal["error"])

    # 序列化
    target_weights = {
        k: _safe_float(v) for k, v in signal.get("target_weights", {}).items()
    }
    factor_weights = {
        k: _safe_float(v) for k, v in signal.get("factor_weights", {}).items()
    }

    return {
        "signal_date": signal["signal_date"],
        "generated_at": signal["generated_at"],
        "target_weights": target_weights,
        "factor_weights": factor_weights,
        "n_selected": signal["n_selected"],
        "n_universe": signal["n_universe"],
        "combiner_method": signal["combiner_method"],
        "objective": signal["objective"],
    }


@qlib_router.post("/trading/simulate")
async def run_trading_simulation(req: TradingSimRequest):
    """运行模拟盘交易

    完整的信号→调度→执行→监控链路模拟。
    """
    from src.execution.signal_generator import SignalGenerator
    from src.execution.trading_gateway import SimulatedBroker, OrderManager
    from src.execution.portfolio_scheduler import PortfolioScheduler
    from src.execution.trade_logger import TradeLogger
    from src.execution.monitor import TradingMonitor
    from src.execution.risk_manager import RiskLimits

    # 1. 加载行情数据
    ensure_qlib(QLIB_DIR)
    all_inst = _list_instruments()
    if not req.codes:
        req.codes = [c for c in all_inst if c.startswith("SH6") or c.startswith("SZ0")][:15]

    # 加载所有股票价格数据
    price_data = {}
    for code in req.codes:
        try:
            df = _get_ohlcv(code)
            if df is not None and not df.empty and len(df) > 20:
                price_data[code] = df
        except Exception:
            pass

    if not price_data:
        raise HTTPException(400, "无可用行情数据")

    # 2. 初始化交易系统组件
    trade_logger = TradeLogger("data/trade_logs")
    broker = SimulatedBroker(
        initial_cash=req.initial_cash,
        price_data=price_data,
        logger=trade_logger,
    )
    broker.connect()
    order_manager = OrderManager(broker, trade_logger)
    scheduler = PortfolioScheduler(broker, trade_logger=trade_logger)
    monitor = TradingMonitor(trade_logger)

    # 3. 确定回测日期范围
    sample_df = list(price_data.values())[0]
    all_dates = sample_df.index
    if req.start_date:
        all_dates = all_dates[all_dates >= pd.Timestamp(req.start_date)]
    if req.end_date:
        all_dates = all_dates[all_dates <= pd.Timestamp(req.end_date)]

    # 每月调仓
    date_series = pd.Series(all_dates, index=all_dates)
    rebalance_dates = date_series.groupby(date_series.index.to_period("M")).last().values

    # 4. 逐日模拟
    results = []
    nav_history = []

    for date in all_dates:
        broker.set_current_date(date)

        # 调仓日
        is_rebalance = any(pd.Timestamp(d) == date for d in rebalance_dates)

        if is_rebalance:
            # 生成信号
            gen = SignalGenerator(
                qlib_dir=QLIB_DIR,
                combiner_method=req.combiner_method,
                top_n=req.top_n,
                max_position=req.max_position,
            )
            signal = gen.generate(
                codes=req.codes,
                as_of_date=date.strftime("%Y-%m-%d"),
            )

            if "error" not in signal:
                target_weights = signal["target_weights"]
                account = broker.get_account()
                total_value = account["total_value"]

                # 生成调仓计划
                orders = scheduler.generate_rebalance_plan(
                    target_weights, total_value
                )

                # 执行调仓
                exec_result = scheduler.execute_rebalance(orders, order_manager)

                # 更新监控
                monitor.update_state(
                    nav=total_value,
                    target_weights=target_weights,
                    data_time=pd.Timestamp(date).to_pydatetime(),
                    broker_connected=True,
                )

                results.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "action": "rebalance",
                    "n_orders": exec_result["submitted"],
                    "n_skipped": exec_result["skipped"],
                    "total_value": exec_result["total_value"],
                    "nav": total_value,
                })

        # 记录净值
        account = broker.get_account()
        nav_history.append({
            "date": date.strftime("%Y-%m-%d"),
            "nav": account["total_value"],
            "cash": account["cash"],
            "pnl": account["total_pnl"],
            "pnl_pct": account["total_pnl_pct"],
        })

    # 5. 最终状态
    final_account = broker.get_account()
    positions = broker.get_position()

    # 订单统计
    order_summary = order_manager.get_order_summary()

    # 监控状态
    monitor_status = monitor.get_status()

    # 日志汇总
    daily_log = trade_logger.get_daily_summary()

    return {
        "nav_history": nav_history,
        "rebalance_log": results,
        "final_account": {
            "cash": _safe_float(final_account["cash"]),
            "total_value": _safe_float(final_account["total_value"]),
            "position_value": _safe_float(final_account["position_value"]),
            "total_pnl": _safe_float(final_account["total_pnl"]),
            "total_pnl_pct": _safe_float(final_account["total_pnl_pct"]),
        },
        "positions": {
            stock: {
                "volume": _safe_float(pos["volume"]),
                "cost_price": _safe_float(pos["cost_price"]),
                "current_price": _safe_float(pos["current_price"]),
                "market_value": _safe_float(pos["market_value"]),
                "pnl": _safe_float(pos["pnl"]),
                "pnl_pct": _safe_float(pos["pnl_pct"]),
            }
            for stock, pos in positions.items()
        },
        "order_summary": order_summary,
        "monitor_status": {
            "nav": _safe_float(monitor_status["nav"]),
            "current_drawdown": _safe_float(monitor_status["current_drawdown"]),
            "n_alerts": monitor_status["n_alerts"],
            "n_critical": monitor_status["n_critical"],
            "n_warnings": monitor_status["n_warnings"],
        },
        "daily_log_summary": daily_log,
        "n_rebalances": len(results),
    }


@qlib_router.get("/trading/algorithms")
async def list_execution_algorithms():
    """列出可用的执行算法"""
    algos = [
        {"name": "TWAP", "description": "时间加权平均价格，均匀拆单"},
        {"name": "VWAP", "description": "成交量加权平均价格，按成交量分布拆单"},
        {"name": "Iceberg", "description": "冰山订单，隐藏真实交易量"},
        {"name": "IS", "description": "执行 shortfall，平衡冲击成本和时间风险"},
    ]
    return {"algorithms": algos}


@qlib_router.post("/trading/algo/simulate")
async def simulate_execution_algo(algo: str, total_volume: float, slices: int):
    """模拟执行算法拆单"""
    from src.execution.execution_algo import create_algo, AlgoConfig

    config = AlgoConfig(
        total_volume=total_volume,
        stock="DEMO",
        side="buy",
        slices=slices,
    )
    try:
        algo_obj = create_algo(algo, config)
        slice_plan = algo_obj.generate_slices()
        return {
            "algorithm": algo,
            "total_volume": total_volume,
            "n_slices": len(slice_plan),
            "slices": [
                {"id": s["slice_id"], "volume": _safe_float(s["volume"]),
                 "time_offset": s["time_offset"]}
                for s in slice_plan
            ],
        }
    except Exception as e:
        raise HTTPException(400, f"算法模拟失败: {str(e)}")


# ─── Database-backed endpoints ───

@router.get("/strategies")
async def list_strategies():
    from src.utils.db import DatabaseManager
    try:
        db = DatabaseManager()
        db.connect()
        df = db.query_df("SELECT id, name, type, status, description, created_at FROM strategy ORDER BY created_at DESC")
        return {"strategies": df.to_dict(orient="records")}
    except Exception as e:
        return {"strategies": [], "error": str(e)}


@router.post("/strategies")
async def create_strategy(req: StrategyCreate):
    from src.utils.db import DatabaseManager
    db = DatabaseManager()
    db.connect()
    db.execute(
        "INSERT INTO strategy (name, type, status, config, description) VALUES (:name, :type, 'draft', :config, :desc)",
        {"name": req.name, "type": req.type, "config": str(req.config), "desc": req.description}
    )
    return {"status": "created", "name": req.name}


@router.post("/backtest")
async def run_backtest(req: BacktestRequest):
    from src.backtest.backtest_engine import BacktestEngine
    engine = BacktestEngine()
    result = engine.run_backtest(
        strategy_config={"id": req.strategy_id},
        start_date=req.start_date, end_date=req.end_date,
        benchmark=req.benchmark, init_cash=req.init_cash,
    )
    return result


@router.get("/backtest/results/{strategy_id}")
async def get_backtest_results(strategy_id: int):
    from src.utils.db import DatabaseManager
    db = DatabaseManager()
    db.connect()
    df = db.query_df(
        "SELECT * FROM backtest_result WHERE strategy_id = :sid ORDER BY created_at DESC",
        {"sid": strategy_id}
    )
    return {"results": df.to_dict(orient="records")}


@router.get("/factors")
async def list_factors(category: Optional[str] = None):
    from src.utils.db import DatabaseManager
    try:
        db = DatabaseManager()
        db.connect()
        if category:
            df = db.query_df("SELECT * FROM factor WHERE category = :cat ORDER BY created_at DESC", {"cat": category})
        else:
            df = db.query_df("SELECT * FROM factor ORDER BY created_at DESC")
        return {"factors": df.to_dict(orient="records")}
    except Exception as e:
        return {"factors": [], "error": str(e)}


@router.get("/positions/{strategy_id}")
async def get_positions(strategy_id: int):
    from src.utils.db import DatabaseManager
    db = DatabaseManager()
    db.connect()
    df = db.query_df("SELECT * FROM position WHERE strategy_id = :sid", {"sid": strategy_id})
    return {"positions": df.to_dict(orient="records")}


@router.get("/data/status")
async def get_data_status():
    ensure_qlib(QLIB_DIR)
    instruments = _list_instruments()
    return {"total_instruments": len(instruments), "sample": instruments[:20]}


@router.get("/data/update-log")
async def get_update_log(limit: int = 10):
    from src.utils.db import DatabaseManager
    db = DatabaseManager()
    db.connect()
    df = db.query_df("SELECT * FROM data_update_log ORDER BY created_at DESC LIMIT :limit", {"limit": limit})
    return {"logs": df.to_dict(orient="records")}
