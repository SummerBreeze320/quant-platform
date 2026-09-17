from typing import Optional, Dict, Any
from sqlalchemy import String, Text, Float, Boolean, JSON
from sqlalchemy.orm import Mapped, mapped_column
from src.models.base import Base, TimestampMixin

class FactorMetadata(Base, TimestampMixin):
    __tablename__ = "factor_metadata"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    expression: Mapped[str] = mapped_column(Text, nullable=False)  # Qlib expression or python code
    category: Mapped[str] = mapped_column(String(32), default="custom", index=True)  # reversal, momentum, volatility, fundamental
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Statistical validation metrics
    ic_mean: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    icir: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    t_stat: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    extra_metrics: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    
    # Provenance
    created_by: Mapped[str] = mapped_column(String(32), default="manual")  # 'manual', 'RD-Agent'
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
