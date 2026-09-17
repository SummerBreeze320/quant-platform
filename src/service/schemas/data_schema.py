from typing import Optional, List
from pydantic import BaseModel

class DataSyncRequest(BaseModel):
    trade_date: Optional[str] = None
    sync_type: str = "daily"

class DataSyncResponse(BaseModel):
    status: str
    date: Optional[str] = None
    symbols_count: Optional[int] = None
    error: Optional[str] = None

class DataStatusResponse(BaseModel):
    qlib_initialized: bool
    qlib_data_dir: str
    calendar_days_count: int
    latest_calendar_date: Optional[str] = None
