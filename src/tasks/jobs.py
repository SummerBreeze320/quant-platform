from datetime import datetime
from typing import Optional, Dict, Any, List
from sqlalchemy.orm import Session
from src.common.db import SessionLocal
from src.common.redis_client import get_redis
from src.common.config import get_settings
from src.common.logger import logger
from src.models.sync_log import SyncLog
from src.models.model_registry import ModelRegistry
from src.data_pipeline.collector import WindDataCollector
from src.data_pipeline.qlib_dumper import QlibDumper
from src.agent_research.rdagent import is_rdagent_available, run_rdagent_factor_loop
from src.qlib_engine.data_handler import DataHandler
from src.qlib_engine.model_trainer import BaseModelTrainer

def job_daily_data_sync(trade_date: Optional[str] = None, db: Optional[Session] = None) -> Dict[str, Any]:
    """
    Daily post-market data synchronization:
    1. Acquires distributed lock to avoid concurrent sync.
    2. Pulls day's market quotes for universe from WindPy.
    3. Appends/updates Qlib binary features and calendar.
    4. Logs execution result to PostgreSQL.
    """
    target_date = trade_date or datetime.now().strftime("%Y-%m-%d")
    settings = get_settings()
    redis_client = get_redis()
    
    sess = db or SessionLocal()
    should_close = db is None

    try:
        with redis_client.distributed_lock(f"sync_{target_date}", expire_seconds=300, timeout=5):
            logger.info(f"Executing daily data sync for {target_date}...")
            collector = WindDataCollector()
            dumper = QlibDumper(settings.QLIB_DATA_DIR)

            # Fetch CSI 300 constituents or active universe
            try:
                symbols = collector.get_sector_constituents(sector_id="000300.SH", date=target_date)
            except Exception as e:
                logger.warning(f"Could not fetch constituents, using default sample: {e}")
                symbols = ["000001.SZ", "600000.SH"]

            quotes_df = collector.get_daily_quotes(symbols=symbols, start_date=target_date, end_date=target_date)
            
            if not quotes_df.empty:
                # Append calendar and dump features
                current_cal = dumper.load_calendar()
                if target_date not in current_cal:
                    dumper.dump_calendars(current_cal + [target_date])
                dumper.dump_features(quotes_df)

            # Record success log
            log_entry = SyncLog(
                sync_date=target_date,
                sync_type="daily",
                status="SUCCESS",
                symbols_count=len(symbols),
                message=f"Synced {len(quotes_df)} records for date {target_date}"
            )
            sess.add(log_entry)
            sess.commit()
            
            logger.info(f"Daily data sync finished successfully for {target_date}.")
            return {"status": "SUCCESS", "date": target_date, "symbols_count": len(symbols)}

    except Exception as e:
        logger.error(f"Daily sync failed for {target_date}: {e}")
        try:
            log_entry = SyncLog(
                sync_date=target_date,
                sync_type="daily",
                status="FAILED",
                symbols_count=0,
                message=str(e)
            )
            sess.add(log_entry)
            sess.commit()
        except Exception:
            pass
        return {"status": "FAILED", "date": target_date, "error": str(e)}
    finally:
        if should_close:
            sess.close()

def job_daily_model_predict(trade_date: Optional[str] = None, db: Optional[Session] = None) -> Dict[str, Any]:
    """
    Daily model inference job:
    1. Loads the active registered model.
    2. Runs inference for next-day ranking.
    3. Caches Top-K recommendation in Redis.
    """
    target_date = trade_date or datetime.now().strftime("%Y-%m-%d")
    redis_client = get_redis()
    sess = db or SessionLocal()
    should_close = db is None

    try:
        active_model = sess.query(ModelRegistry).filter_by(is_active=True).first()
        if not active_model:
            logger.info("No active model registered for daily prediction.")
            return {"status": "SKIPPED", "reason": "No active model"}

        metrics = active_model.metrics or {}
        if metrics.get("status") != "TRAINED":
            logger.info(f"Active model '{active_model.model_name}' status is '{metrics.get('status')}', skipping prediction.")
            return {"status": "SKIPPED", "reason": f"Model status: {metrics.get('status')}"}

        logger.info(f"Running daily prediction with model '{active_model.model_name}' for {target_date}...")

        trainer = BaseModelTrainer.load(active_model.model_path)
        feature_names = trainer.feature_names

        pred_df = DataHandler.prepare_prediction_data(feature_names, target_date)
        X_pred = pred_df[[c for c in feature_names if c in pred_df.columns]]
        scores = trainer.predict(X_pred)

        pred_df["score"] = scores
        pred_df = pred_df.sort_values("score", ascending=False).reset_index(drop=True)
        pred_df["rank"] = range(1, len(pred_df) + 1)

        predictions = [
            {"symbol": row["symbol"], "score": round(float(row["score"]), 6), "rank": int(row["rank"])}
            for _, row in pred_df.iterrows()
        ]

        payload = {"date": target_date, "model_name": active_model.model_name, "predictions": predictions}
        redis_client.set_json(f"predictions:{target_date}", payload, ex=86400 * 3)
        redis_client.set_json("predictions:latest", payload, ex=86400 * 3)

        logger.info(f"Daily prediction complete: {len(predictions)} symbols cached for {target_date}.")
        return {"status": "SUCCESS", "date": target_date, "predictions_count": len(predictions)}
    except Exception as e:
        logger.error(f"Daily prediction failed: {e}")
        return {"status": "FAILED", "error": str(e)}
    finally:
        if should_close:
            sess.close()

def job_weekend_factor_mining(theme: str = "reversal", rounds: int = 3, db: Optional[Session] = None) -> Dict[str, Any]:
    """
    Weekend offline automated factor research using Microsoft RD-Agent:
    Executes multiple rounds of LLM-driven hypothesis generation, Co-STEER coding, and sandbox evaluation.
    """
    sess = db or SessionLocal()
    should_close = db is None
    try:
        if not is_rdagent_available():
            logger.warning("RD-Agent not available (LLM API key not configured). Skipping factor mining.")
            return {"status": "SKIPPED", "reason": "LLM API key not configured"}

        logger.info(f"Starting weekend RD-Agent factor mining loop (rounds={rounds}, theme={theme})...")
        results = run_rdagent_factor_loop(rounds=rounds, theme=theme, db=sess)
        passed_count = sum(1 for r in results if r.get("success"))
        logger.info(f"Weekend factor mining complete: {passed_count}/{rounds} factors passed quality gate.")
        return {"status": "SUCCESS", "rounds": rounds, "passed_count": passed_count, "results": results}
    except Exception as e:
        logger.error(f"Weekend factor mining failed: {e}")
        return {"status": "FAILED", "error": str(e)}
    finally:
        if should_close:
            sess.close()


def job_premarket_rebalance(
    strategy_id: str = "hft_stream_01",
    trade_date: Optional[str] = None,
    algo_type: str = "DIRECT",
    execution_mode: str = "SYNC",
    interval_seconds: float = 0.0,
    runtime: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    09:15 盘前根据最新 Alpha 分数与凸优化求解器生成日内调仓计划并执行
    """
    if runtime is None:
        try:
            from src.service.app import app
            runtime = getattr(app.state, "runtime", None)
        except Exception:
            runtime = None
    from src.tasks.pipeline import PremarketRebalancePipeline
    return PremarketRebalancePipeline.run(
        strategy_id=strategy_id,
        trade_date=trade_date,
        algo_type=algo_type,
        execution_mode=execution_mode,
        interval_seconds=interval_seconds,
        runtime=runtime,
    )


def job_daily_settlement(
    account_id: Optional[str] = None,
    trade_date: Optional[str] = None,
    runtime: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    15:30 盘后日终清算交收与 T+1 可用股份解冻
    """
    if runtime is None:
        try:
            from src.service.app import app
            runtime = getattr(app.state, "runtime", None)
        except Exception:
            runtime = None
    from src.tasks.pipeline import DailySettlementPipeline
    return DailySettlementPipeline.run(
        account_id=account_id,
        trade_date=trade_date,
        runtime=runtime,
    )


def job_start_market_feed(
    source: str = "PUBLIC",
    symbols: Optional[List[str]] = None,
    runtime: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    09:25 开盘前启动盘中行情流驱动器
    """
    if runtime is None:
        try:
            from src.service.app import app
            runtime = getattr(app.state, "runtime", None)
        except Exception:
            runtime = None
    if runtime and hasattr(runtime, "market"):
        syms = symbols or ["600000.SH", "000001.SZ", "600036.SH", "000858.SZ", "601318.SH"]
        return runtime.market.live_feed.start(source=source, symbols=syms)
    return {"status": "SKIPPED", "detail": "Runtime or market not initialized"}


def job_stop_market_feed(runtime: Optional[Any] = None) -> Dict[str, Any]:
    """
    15:05 收盘后停止盘中行情流驱动器以节省资源
    """
    if runtime is None:
        try:
            from src.service.app import app
            runtime = getattr(app.state, "runtime", None)
        except Exception:
            runtime = None
    if runtime and hasattr(runtime, "market"):
        return runtime.market.live_feed.stop()
    return {"status": "SKIPPED", "detail": "Runtime or market not initialized"}
