from datetime import datetime
from typing import Optional
from sqlalchemy import String, Float, Integer, ForeignKey, UniqueConstraint, DateTime, func
from sqlalchemy.orm import Mapped, mapped_column, relationship
from src.models.base import Base, TimestampMixin

class TradingAccountModel(Base, TimestampMixin):
    __tablename__ = "trading_accounts"

    account_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    strategy_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    total_equity: Mapped[float] = mapped_column(Float, default=0.0)
    available_cash: Mapped[float] = mapped_column(Float, default=0.0)

    positions: Mapped[list["TradingPositionModel"]] = relationship(
        "TradingPositionModel", back_populates="account", cascade="all, delete-orphan"
    )

class TradingPositionModel(Base, TimestampMixin):
    __tablename__ = "trading_positions"
    __table_args__ = (UniqueConstraint("account_id", "symbol", name="uq_account_symbol"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    account_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("trading_accounts.account_id", ondelete="CASCADE"), index=True
    )
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    total_volume: Mapped[int] = mapped_column(Integer, default=0)
    available_volume: Mapped[int] = mapped_column(Integer, default=0)
    frozen_volume: Mapped[int] = mapped_column(Integer, default=0)
    avg_cost: Mapped[float] = mapped_column(Float, default=0.0)
    market_value: Mapped[float] = mapped_column(Float, default=0.0)
    last_price: Mapped[float] = mapped_column(Float, default=0.0)
    unrealized_pnl: Mapped[float] = mapped_column(Float, default=0.0)


    account: Mapped["TradingAccountModel"] = relationship("TradingAccountModel", back_populates="positions")

class OrderRecordModel(Base, TimestampMixin):
    __tablename__ = "trading_orders"

    order_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    account_id: Mapped[str] = mapped_column(String(64), index=True)
    strategy_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    direction: Mapped[str] = mapped_column(String(16))
    order_type: Mapped[str] = mapped_column(String(16), default="LIMIT")
    price: Mapped[float] = mapped_column(Float)
    volume: Mapped[int] = mapped_column(Integer)
    filled_volume: Mapped[int] = mapped_column(Integer, default=0)
    filled_amount: Mapped[float] = mapped_column(Float, default=0.0)
    status: Mapped[str] = mapped_column(String(32), default="PENDING")
    algo_type: Mapped[str] = mapped_column(String(32), default="DIRECT")
    parent_order_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    reject_reason: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)

class TradeRecordModel(Base, TimestampMixin):
    __tablename__ = "trading_trades"

    trade_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    order_id: Mapped[str] = mapped_column(String(64), index=True)
    account_id: Mapped[str] = mapped_column(String(64), index=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    direction: Mapped[str] = mapped_column(String(16))
    price: Mapped[float] = mapped_column(Float)
    volume: Mapped[int] = mapped_column(Integer)
    amount: Mapped[float] = mapped_column(Float)
    commission: Mapped[float] = mapped_column(Float, default=0.0)
    stamp_tax: Mapped[float] = mapped_column(Float, default=0.0)
    executed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=func.now())

class PMSTransferRecordModel(Base, TimestampMixin):
    __tablename__ = "pms_transfers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    master_id: Mapped[str] = mapped_column(String(64), default="master_default", index=True)
    strategy_id: Mapped[str] = mapped_column(String(64), index=True)
    transfer_amount: Mapped[float] = mapped_column(Float)
    reason: Mapped[str] = mapped_column(String(256), default="")
