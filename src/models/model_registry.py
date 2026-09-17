from typing import Optional, List, Dict, Any
from sqlalchemy import String, JSON, Boolean
from sqlalchemy.orm import Mapped, mapped_column
from src.models.base import Base, TimestampMixin

class ModelRegistry(Base, TimestampMixin):
    __tablename__ = "model_registry"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    model_name: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    model_type: Mapped[str] = mapped_column(String(32), default="LightGBM")
    feature_set: Mapped[List[str]] = mapped_column(JSON, default=list)
    train_start: Mapped[str] = mapped_column(String(10))
    train_end: Mapped[str] = mapped_column(String(10))
    model_path: Mapped[str] = mapped_column(String(256), nullable=False)
    metrics: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=False)
