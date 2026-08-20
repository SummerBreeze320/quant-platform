"""Enhanced live trading API router.

Endpoints for:
- Qlib model-based signal generation
- Full trading pipeline with unified risk control
- Real-time position and PnL tracking
- P4: Daily trading engine status + schedule info + review history
"""
import logging
from typing import Optional, Dict, Any, List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/trading/live", tags=["trading-live"])


class QlibSignalRequest(BaseModel):
    model_path: Optional[str] = Field(None, description="模型文件路径")
    handler_type: str = Field("alpha158", description="数据处理器")
    instruments: str = Field("all", description="股票池")
    topk: int = Field(5, description="选股数量")
    max_position: float = Field(0.15, description="单票最大权重")
    objective: str = Field("equal_weight", description="权重优化目标")


class FullTradingRequest(BaseModel):
    model_name: str = Field("lightgbm", description="模型名称")
    handler_type: str = Field("alpha158", description="数据处理器")
    topk: int = Field(5, description="选股数量")
    init_capital: float = Field(1_000_000, description="初始资金")
    benchmark: str = Field("SH000300", description="基准")
    risk_check: bool = Field(True, description="是否执行统一风控")
    trade_date: Optional[str] = Field(None, description="交易日期(YYYY-MM-DD)")


@router.post("/qlib-signal")
async def generate_qlib_signal(req: QlibSignalRequest):
    """使用Qlib模型生成实盘信号"""
    try:
        from src.execution import QlibSignalGenerator

        gen = QlibSignalGenerator(
            model_path=req.model_path,
            handler_type=req.handler_type,
            topk=req.topk,
            max_position=req.max_position,
        )

        signal = gen.generate(
            instruments=req.instruments,
            objective=req.objective,
        )

        if "error" in signal:
            raise HTTPException(status_code=400, detail=signal["error"])

        return signal
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Qlib signal generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/full-pipeline")
async def run_full_trading_pipeline(req: FullTradingRequest):
    """运行完整交易管线: 信号→风控→调度→网关→执行"""
    try:
        from src.research import create_model, get_handler, create_dataset
        from src.execution import (
            UnifiedRiskController, PortfolioScheduler,
            SimulatedBroker, create_algo,
        )
        from src.core import ensure_qlib

        ensure_qlib()

        # 1. Train model (quick)
        handler = get_handler(
            handler_type=req.handler_type,
            instruments="all",
            start_time="2020-01-01",
            end_time="2024-12-31",
        )
        segments = {
            "train": ("2020-01-01", "2022-12-31"),
            "valid": ("2023-01-01", "2023-12-31"),
            "test": ("2024-01-01", "2024-12-31"),
        }
        dataset = create_dataset(handler, segments)
        model = create_model(req.model_name)
        train_data = dataset.prepare("train", col_set=["feature", "label"])
        model.fit(*train_data)

        # 2. Generate predictions
        pred = model.predict(dataset=dataset, segment="test")
        import pandas as pd
        if isinstance(pred, pd.DataFrame):
            pred = pred.iloc[:, 0]

        latest_date = pred.index.get_level_values("datetime").max()
        latest_scores = pred.xs(latest_date, level="datetime")
        top_stocks = latest_scores.nlargest(req.topk)
        target_weights = {
            str(k): float(1.0 / len(top_stocks))
            for k in top_stocks.index
        }

        # 3. Risk check
        risk = UnifiedRiskController()
        risk_passed, risk_msg = risk.signal_check(target_weights)

        if req.risk_check and not risk_passed:
            return {
                "status": "risk_rejected",
                "risk_message": risk_msg,
                "signal": {"target_weights": target_weights},
            }

        # 4. Portfolio scheduling
        scheduler = PortfolioScheduler(init_capital=req.init_capital)
        current_positions = {}
        rebalance = scheduler.generate_rebalance(
            current_positions, target_weights, req.init_capital
        )

        # 5. Execute through simulated broker
        broker = SimulatedBroker()
        broker.connect()

        trades = []
        for order in rebalance.get("orders", []):
            code = order.get("code")
            side = order.get("side", "buy")
            price = order.get("price", 10.0)
            volume = order.get("volume", 100)

            # Order-level risk check
            ok, msg = risk.order_check(
                code, side, price, volume,
                current_positions, req.init_capital,
            )

            if ok:
                trade = broker.submit_order(
                    code=code, side=side, price=price, volume=volume,
                )
                if trade:
                    trades.append(trade)
            else:
                logger.warning(f"Order rejected by risk: {msg}")

        # 6. Execution check
        nav = req.init_capital
        for t in trades:
            if t.get("side") == "buy":
                nav -= t.get("amount", 0)
            else:
                nav += t.get("amount", 0)

        exec_ok, exec_msg = risk.execution_check(
            trades, current_positions, nav, req.init_capital,
        )

        return {
            "status": "completed" if exec_ok else "risk_alert",
            "signal_date": latest_date.strftime("%Y-%m-%d"),
            "model": req.model_name,
            "handler": req.handler_type,
            "target_weights": target_weights,
            "n_orders": len(rebalance.get("orders", [])),
            "n_executed": len(trades),
            "nav": nav,
            "risk_alerts": risk.get_alerts(),
            "trades": [
                {
                    "code": t.get("code", ""),
                    "side": t.get("side", ""),
                    "price": t.get("price", 0),
                    "volume": t.get("volume", 0),
                    "amount": t.get("amount", 0),
                }
                for t in trades
            ],
        }

    except Exception as e:
        logger.error(f"Full trading pipeline failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/risk-config")
async def get_risk_config():
    """获取当前风控配置"""
    from src.execution import UnifiedRiskConfig
    config = UnifiedRiskConfig()
    return {
        "signal": {
            "max_single_weight": config.max_single_weight,
            "min_diversification": config.min_diversification,
            "max_score_outlier": config.max_score_outlier,
        },
        "schedule": {
            "max_turnover": config.max_turnover,
            "max_sector_exposure": config.max_sector_exposure,
            "max_total_exposure": config.max_total_exposure,
        },
        "gateway": {
            "max_price_deviation": config.max_price_deviation,
            "max_order_frequency": config.max_order_frequency,
            "min_order_value": config.min_order_value,
        },
        "execution": {
            "max_slippage": config.max_slippage,
            "daily_stop_loss": config.daily_stop_loss,
            "max_drawdown": config.max_drawdown,
            "drawdown_alert": config.drawdown_alert,
        },
    }


# ======== P4: Daily Trading Engine Endpoints ========

@router.get("/engine/status")
async def get_engine_status():
    """获取交易引擎状态"""
    try:
        from src.execution.trading_engine import TradingEngine
        engine = TradingEngine()
        return engine.get_status()
    except Exception as e:
        logger.error(f"Engine status failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/engine/history")
async def get_engine_history(n_days: int = 30):
    """获取交易引擎历史"""
    try:
        from src.execution.trading_engine import TradingEngine
        engine = TradingEngine()
        return {"history": engine.get_history(n_days)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/engine/schedule")
async def get_schedule_info():
    """获取调度器信息"""
    try:
        from src.execution.scheduler import DailyScheduler
        scheduler = DailyScheduler()
        return scheduler.get_schedule_info()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/engine/run")
async def run_trading_day(req: FullTradingRequest):
    """手动触发一个交易日"""
    try:
        from src.execution.trading_engine import TradingEngine

        engine = TradingEngine(
            init_capital=req.init_capital,
            benchmark=req.benchmark,
            topk=req.topk,
            handler_type=req.handler_type,
        )
        engine.initialize()

        trade_date = req.trade_date
        if not trade_date:
            from datetime import datetime
            trade_date = datetime.now().strftime("%Y-%m-%d")
        record = engine.run_daily(trade_date)

        return {
            "trade_date": record.trade_date,
            "phase": record.phase.value,
            "nav": record.nav,
            "pnl_pct": record.pnl_pct,
            "n_orders_generated": len(record.orders),
            "n_orders_executed": len(record.executed_orders),
            "n_pending": len(record.pending_orders),
            "executed_orders": record.executed_orders,
            "pending_orders": record.pending_orders,
            "signal": record.signal,
            "risk_assessment": record.risk_assessment,
            "error": record.error,
            "review": record.review,
        }
    except Exception as e:
        logger.error(f"Trading day failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ======== P6: T+1 Pending Orders Endpoints ========

@router.get("/engine/pending-orders")
async def get_pending_orders():
    """获取待执行订单 — T日生成, T+1日开盘执行"""
    try:
        from src.execution.trading_engine import TradingEngine
        engine = TradingEngine()
        pending = engine.get_pending_orders()
        return {
            "n_pending": len(pending),
            "orders": pending,
        }
    except Exception as e:
        logger.error(f"Pending orders query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/engine/execute-pending")
async def execute_pending_orders(trade_date: Optional[str] = None):
    """手动执行待执行订单 — T+1开盘执行"""
    try:
        from src.execution.trading_engine import TradingEngine
        from datetime import datetime

        engine = TradingEngine()
        engine.initialize()

        if not trade_date:
            trade_date = datetime.now().strftime("%Y-%m-%d")

        executed = engine._execute_pending_orders(trade_date)
        return {
            "trade_date": trade_date,
            "n_executed": len(executed),
            "n_remaining_pending": len(engine.pending_orders),
            "executed_orders": executed,
        }
    except Exception as e:
        logger.error(f"Pending order execution failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/engine/positions")
async def get_positions():
    """获取当前持仓明细"""
    try:
        from src.execution.trading_engine import TradingEngine
        engine = TradingEngine()
        positions = engine.get_positions()
        return {
            "n_positions": len(positions),
            "nav": engine.current_nav,
            "cash": engine.current_cash,
            "positions": positions,
        }
    except Exception as e:
        logger.error(f"Positions query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/reviews")
async def get_reviews(n_days: int = 30):
    """获取盘后复盘记录"""
    try:
        from src.execution.review import PerformanceReview
        review = PerformanceReview()
        summary = review.get_summary(n_days)
        return {
            "summary": summary,
            "history": review._history[-n_days:],
        }
    except Exception as e:
        logger.error(f"Reviews failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
