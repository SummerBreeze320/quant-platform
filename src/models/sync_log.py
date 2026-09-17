from typing import Optional
from sqlalchemy import String, Integer, Text
from sqlalchemy.orm import Mapped, mapped_column
from src.models.base import Base, TimestampMixin

class SyncLog(Base, TimestampMixin):
    __tablename__ = "sync_logs"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    sync_date: Mapped[str] = mapped_column(String(10), index=True, nullable=False)  # 'YYYY-MM-DD'
    sync_type: Mapped[str] = mapped_column(String(16), default="daily")  # 'daily', 'full', 'calendar'
    status: Mapped[str] = mapped_column(String(16), default="PENDING")  # 'SUCCESS', 'FAILED', 'RUNNING'
    symbols_count: Mapped[int] = mapped_column(Integer, default=0)
    message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
