from typing import Optional, Dict, Any
from sqlalchemy import String, Float, JSON, Integer, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column
from src.models.base import Base, TimestampMixin

class BacktestRecord(Base, TimestampMixin):
    __tablename__ = "backtest_records"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    model_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("model_registry.id"), nullable=True)
    strategy_name: Mapped[str] = mapped_column(String(64), nullable=False)
    start_date: Mapped[str] = mapped_column(String(10), nullable=False)
    end_date: Mapped[str] = mapped_column(String(10), nullable=False)
    benchmark: Mapped[str] = mapped_column(String(16), default="000300.SH")
    
    # Performance metrics
    annualized_return: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    sharpe_ratio: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    max_drawdown: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    win_rate: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    turnover_rate: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    # Detailed series (e.g. cumulative return series, drawdown curve)
    report_details: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
