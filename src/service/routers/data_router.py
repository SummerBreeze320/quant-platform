from fastapi import APIRouter, Depends, BackgroundTasks
from sqlalchemy.orm import Session
from src.common.db import get_db
from src.common.config import get_settings
from src.qlib_engine.initializer import is_qlib_initialized
from src.data_pipeline.qlib_dumper import QlibDumper
from src.tasks.jobs import job_daily_data_sync
from src.service.schemas.data_schema import DataSyncRequest, DataSyncResponse, DataStatusResponse

router = APIRouter(prefix="/data", tags=["Data Pipeline"])

@router.post("/sync/daily", response_model=DataSyncResponse)
def trigger_daily_sync(
    request: DataSyncRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Triggers day-end market data sync and Qlib binary update."""
    res = job_daily_data_sync(trade_date=request.trade_date, db=db)
    return res

@router.get("/status", response_model=DataStatusResponse)
def get_data_status():
    """Returns local Qlib storage status and trading calendar coverage."""
    settings = get_settings()
    dumper = QlibDumper(settings.QLIB_DATA_DIR)
    calendar = dumper.load_calendar()
    
    return {
        "qlib_initialized": is_qlib_initialized(),
        "qlib_data_dir": settings.QLIB_DATA_DIR,
        "calendar_days_count": len(calendar),
        "latest_calendar_date": calendar[-1] if calendar else None
    }
