from typing import Optional, List, Any
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from src.tasks.jobs import job_daily_data_sync, job_daily_model_predict, job_weekend_factor_mining
from src.common.logger import logger

class QuantScheduler:
    """Manages scheduled background automation for market data and AI quant research."""

    def __init__(self):
        self.scheduler = BackgroundScheduler()
        self._is_configured = False

    def configure_jobs(self):
        """Registers default cron triggers for quantitative daily and weekend pipeline."""
        if self._is_configured:
            return

        # 1. 16:30 Mon-Fri: Post-market daily data sync
        self.scheduler.add_job(
            job_daily_data_sync,
            trigger=CronTrigger(day_of_week="mon-fri", hour=16, minute=30),
            id="daily_data_sync",
            name="Daily Post-Market Data Sync",
            replace_existing=True
        )

        # 2. 17:15 Mon-Fri: Daily model prediction & score caching
        self.scheduler.add_job(
            job_daily_model_predict,
            trigger=CronTrigger(day_of_week="mon-fri", hour=17, minute=15),
            id="daily_model_predict",
            name="Daily Model Prediction",
            replace_existing=True
        )

        # 3. 10:00 Sunday: Weekend RD-Agent factor mining loop
        self.scheduler.add_job(
            job_weekend_factor_mining,
            trigger=CronTrigger(day_of_week="sun", hour=10, minute=0),
            id="weekend_factor_mining",
            name="Weekend RD-Agent Factor Mining",
            replace_existing=True
        )

        self._is_configured = True
        logger.info("Configured default quantitative scheduled jobs.")

    def start(self):
        """Starts the scheduler in background thread."""
        if not self._is_configured:
            self.configure_jobs()
        if not self.scheduler.running:
            self.scheduler.start()
            logger.info("QuantScheduler started successfully.")

    def shutdown(self):
        """Gracefully terminates background scheduler."""
        if self.scheduler.running:
            self.scheduler.shutdown(wait=False)
            logger.info("QuantScheduler shut down.")

    def get_jobs(self) -> List[Any]:
        """Returns list of active scheduled jobs."""
        return self.scheduler.get_jobs()

_scheduler_instance = QuantScheduler()

def get_scheduler() -> QuantScheduler:
    return _scheduler_instance
