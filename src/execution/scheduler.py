"""Daily Scheduler — 定时调度器 (T+1执行)

Manages the daily trading schedule using APScheduler:

  Day T 09:00  pre_market_data   — Data refresh + factor computation
  Day T 15:00  trading_dispatch   — Feature → Model → Portfolio → Generate orders (pending)
  Day T 15:30  post_close_review  — Performance review
  Day T+1 09:30 open_execute      — Execute pending BUY/SELL at open

T+1 flow:
- Orders generated at T-day 15:00 close are stored as pending
- Pending orders execute at T+1 09:30 open price
- This matches A-share T+1 settlement rules

Features:
- Trading calendar check (skip weekends and holidays)
- Configurable schedule times
- Manual trigger support
- State persistence
"""
import logging
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass
from datetime import datetime, date
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ScheduleConfig:
    """Trading schedule configuration."""
    pre_market_time: str = "09:00"     # Data refresh
    open_execute_time: str = "09:30"  # T+1 pending order execution
    trading_time: str = "15:00"       # Signal + order generation
    review_time: str = "15:30"        # Post-close review
    timezone: str = "Asia/Shanghai"
    skip_weekends: bool = True
    skip_holidays: bool = True


class DailyScheduler:
    """Daily trading schedule manager.

    Uses APScheduler for cron-based scheduling.
    Falls back to manual trigger if APScheduler is unavailable.

    Usage:
        scheduler = DailyScheduler()
        scheduler.setup(engine)
        scheduler.start()
        # ... runs daily automatically
        scheduler.stop()

        # Or trigger manually
        scheduler.run_now()
    """

    def __init__(self, config: Optional[ScheduleConfig] = None):
        self.config = config or ScheduleConfig()
        self._scheduler = None
        self._engine = None
        self._running = False

    def setup(self, engine):
        """Configure the scheduler with a trading engine."""
        self._engine = engine
        logger.info(
            f"Scheduler setup: "
            f"pre_market={self.config.pre_market_time} "
            f"open_execute={self.config.open_execute_time} "
            f"trading={self.config.trading_time} "
            f"review={self.config.review_time}"
        )

    def start(self):
        """Start the automated daily schedule."""
        try:
            from apscheduler.schedulers.background import BackgroundScheduler
            from apscheduler.triggers.cron import CronTrigger

            self._scheduler = BackgroundScheduler(
                timezone=self.config.timezone
            )

            # 09:00 Pre-market data refresh
            hh, mm = self.config.pre_market_time.split(":")
            self._scheduler.add_job(
                func=self._pre_market_job,
                trigger=CronTrigger(hour=int(hh), minute=int(mm)),
                id="pre_market",
                name="Pre-market data refresh",
            )

            # 09:30 Open — execute pending orders (T+1)
            hh, mm = self.config.open_execute_time.split(":")
            self._scheduler.add_job(
                func=self._open_execute_job,
                trigger=CronTrigger(hour=int(hh), minute=int(mm)),
                id="open_execute",
                name="T+1 pending order execution",
            )

            # 15:00 Trading dispatch — signal + order generation
            hh, mm = self.config.trading_time.split(":")
            self._scheduler.add_job(
                func=self._trading_job,
                trigger=CronTrigger(hour=int(hh), minute=int(mm)),
                id="trading_dispatch",
                name="Trading dispatch (signal + orders)",
            )

            # 15:30 Post-close review
            hh, mm = self.config.review_time.split(":")
            self._scheduler.add_job(
                func=self._review_job,
                trigger=CronTrigger(hour=int(hh), minute=int(mm)),
                id="post_close",
                name="Post-close review",
            )

            self._scheduler.start()
            self._running = True
            logger.info("Scheduler started — 4 daily jobs running in background")

        except ImportError:
            logger.warning(
                "APScheduler not installed. "
                "Install: pip install apscheduler "
                "Or use run_now() for manual triggering."
            )
        except Exception as e:
            logger.error(f"Scheduler start failed: {e}")

    def stop(self):
        """Stop the scheduler."""
        if self._scheduler and self._running:
            self._scheduler.shutdown(wait=False)
            self._running = False
            logger.info("Scheduler stopped")

    def run_now(self) -> Dict[str, Any]:
        """Manually trigger a full trading day now."""
        if self._engine is None:
            return {"error": "No engine configured. Call setup() first."}

        trade_date = datetime.now().strftime("%Y-%m-%d")
        logger.info(f"Manual trigger: running full trading day for {trade_date}")

        record = self._engine.run_daily(trade_date)
        return {
            "trade_date": record.trade_date,
            "phase": record.phase.value,
            "nav": record.nav,
            "pnl_pct": record.pnl_pct,
            "n_orders_generated": len(record.orders),
            "n_orders_executed": len(record.executed_orders),
            "n_pending": len(record.pending_orders),
            "error": record.error,
        }

    def execute_pending_now(self) -> Dict[str, Any]:
        """Manually trigger pending order execution only."""
        if self._engine is None:
            return {"error": "No engine configured."}

        trade_date = datetime.now().strftime("%Y-%m-%d")
        executed = self._engine._execute_pending_orders(trade_date)
        return {
            "trade_date": trade_date,
            "n_executed": len(executed),
            "executed_orders": executed,
            "n_remaining_pending": len(self._engine.pending_orders),
        }

    def is_trading_day(self, check_date: Optional[date] = None) -> bool:
        """Check if a date is a trading day (not weekend/holiday)."""
        if check_date is None:
            check_date = date.today()

        if self.config.skip_weekends and check_date.weekday() >= 5:
            return False

        if self.config.skip_holidays:
            holidays = self._get_holidays()
            if check_date.isoformat() in holidays:
                return False

        return True

    def _get_holidays(self) -> set:
        """Get A-share market holidays."""
        try:
            from src.core import get_calendar
            cal = get_calendar()
            if cal:
                all_dates = pd.date_range("2020-01-01", "2026-12-31", freq="D")
                trading_days = set(pd.Timestamp(d).strftime("%Y-%m-%d") for d in cal)
                holidays = set(
                    d.strftime("%Y-%m-%d") for d in all_dates
                    if d.strftime("%Y-%m-%d") not in trading_days
                )
                return holidays
        except Exception:
            pass

        return set()

    def _pre_market_job(self):
        """09:00 Scheduled job: pre-market data refresh."""
        today = date.today()
        if not self.is_trading_day(today):
            logger.info(f"Skipping pre-market: {today} is not a trading day")
            return

        logger.info(f"[Scheduled 09:00] Pre-market data refresh for {today}")

    def _open_execute_job(self):
        """09:30 Scheduled job: execute pending orders at open."""
        today = date.today()
        if not self.is_trading_day(today):
            logger.info(f"Skipping open execution: {today} is not a trading day")
            return

        if self._engine is None:
            logger.warning("No engine configured for open execution")
            return

        logger.info(f"[Scheduled 09:30] Executing pending orders for {today}")
        executed = self._engine._execute_pending_orders(today.isoformat())
        logger.info(f"[09:30] Executed {len(executed)} pending orders")

    def _trading_job(self):
        """15:00 Scheduled job: trading dispatch."""
        today = date.today()
        if not self.is_trading_day(today):
            logger.info(f"Skipping trading: {today} is not a trading day")
            return

        if self._engine is None:
            logger.warning("No engine configured")
            return

        trade_date = today.strftime("%Y-%m-%d")
        logger.info(f"[Scheduled 15:00] Trading dispatch for {trade_date}")
        self._engine.run_daily(trade_date)

    def _review_job(self):
        """15:30 Scheduled job: post-close review."""
        today = date.today()
        if not self.is_trading_day(today):
            return

        logger.info(f"[Scheduled 15:30] Post-close review for {today}")

    def get_schedule_info(self) -> Dict[str, Any]:
        """Get schedule information."""
        return {
            "pre_market_time": self.config.pre_market_time,
            "open_execute_time": self.config.open_execute_time,
            "trading_time": self.config.trading_time,
            "review_time": self.config.review_time,
            "timezone": self.config.timezone,
            "running": self._running,
            "is_trading_day": self.is_trading_day(),
            "next_run": self._next_run_time(),
            "n_pending_orders": len(self._engine.pending_orders) if self._engine else 0,
        }

    def _next_run_time(self) -> Optional[str]:
        """Get next scheduled run time."""
        if not self._scheduler or not self._running:
            return None

        try:
            jobs = self._scheduler.get_jobs()
            if jobs:
                next_run = min(job.next_run_time for job in jobs if job.next_run_time)
                return next_run.isoformat() if next_run else None
        except Exception:
            pass
        return None
