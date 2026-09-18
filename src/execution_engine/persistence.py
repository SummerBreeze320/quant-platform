from datetime import datetime
from typing import Optional, List, Dict, Any, Callable
from contextlib import contextmanager
from sqlalchemy.orm import Session
from sqlalchemy import select, desc
from src.common.logger import logger
from src.execution_engine.models import AccountState, Position, Order, Trade
from src.pms.models import CashTransfer
from src.models.trading import (
    TradingAccountModel,
    TradingPositionModel,
    OrderRecordModel,
    TradeRecordModel,
    PMSTransferRecordModel,
)


class TradingStorage:
    """仓储持久化层：处理账户、持仓、委托订单、成交明细与资金划转在关系数据库中的落盘与对账。"""

    def __init__(self, session_factory: Optional[Callable[[], Session]] = None):
        self.session_factory = session_factory
        self._enabled: bool = session_factory is not None
        if self._enabled:
            try:
                with session_factory() as session:
                    session.execute(select(1))
            except Exception as e:
                logger.info(f"Database probe offline; TradingStorage fallback to in-memory: {e}")
                self._enabled = False

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    @contextmanager
    def _session(self):
        if not self._enabled or self.session_factory is None:
            yield None
            return
        session: Optional[Session] = None
        try:
            session = self.session_factory()
            yield session
            session.commit()
        except Exception as e:
            if session:
                session.rollback()
            logger.warning(f"TradingStorage session error: {e}")
            raise
        finally:
            if session:
                session.close()


    def save_account(self, account: AccountState) -> bool:
        """持久化或更新单个账户基本信息"""
        try:
            with self._session() as session:
                if session is None:
                    return False
                record = session.get(TradingAccountModel, account.account_id)
                if not record:
                    record = TradingAccountModel(
                        account_id=account.account_id,
                        strategy_id=account.strategy_id,
                        available_cash=account.available_cash,
                        total_equity=account.total_equity,
                    )
                    session.add(record)
                else:
                    record.strategy_id = account.strategy_id
                    record.available_cash = account.available_cash
                    record.total_equity = account.total_equity

                # 同步各持仓
                for sym, pos in account.positions.items():
                    pos_rec = session.execute(
                        select(TradingPositionModel).where(
                            TradingPositionModel.account_id == account.account_id,
                            TradingPositionModel.symbol == sym,
                        )
                    ).scalar_one_or_none()
                    if not pos_rec:
                        pos_rec = TradingPositionModel(
                            account_id=account.account_id,
                            symbol=sym,
                            total_volume=pos.total_volume,
                            available_volume=pos.available_volume,
                            frozen_volume=pos.frozen_volume,
                            avg_cost=pos.avg_cost,
                            market_value=pos.market_value,
                        )
                        session.add(pos_rec)
                    else:
                        pos_rec.total_volume = pos.total_volume
                        pos_rec.available_volume = pos.available_volume
                        pos_rec.frozen_volume = pos.frozen_volume
                        pos_rec.avg_cost = pos.avg_cost
                        pos_rec.market_value = pos.market_value
            return True
        except Exception as e:
            logger.warning(f"Failed to save account {account.account_id}: {e}")
            return False

    def save_position(self, account_id: str, pos: Position) -> bool:
        """更新单个持仓明细"""
        try:
            with self._session() as session:
                if session is None:
                    return False
                pos_rec = session.execute(
                    select(TradingPositionModel).where(
                        TradingPositionModel.account_id == account_id,
                        TradingPositionModel.symbol == pos.symbol,
                    )
                ).scalar_one_or_none()
                if not pos_rec:
                    pos_rec = TradingPositionModel(
                        account_id=account_id,
                        symbol=pos.symbol,
                        total_volume=pos.total_volume,
                        available_volume=pos.available_volume,
                        frozen_volume=pos.frozen_volume,
                        avg_cost=pos.avg_cost,
                        market_value=pos.market_value,
                    )
                    session.add(pos_rec)
                else:
                    pos_rec.total_volume = pos.total_volume
                    pos_rec.available_volume = pos.available_volume
                    pos_rec.frozen_volume = pos.frozen_volume
                    pos_rec.avg_cost = pos.avg_cost
                    pos_rec.market_value = pos.market_value
            return True
        except Exception as e:
            logger.warning(f"Failed to save position {pos.symbol} for account {account_id}: {e}")
            return False

    def delete_position(self, account_id: str, symbol: str) -> bool:
        """平仓出清后删除持仓记录"""
        try:
            with self._session() as session:
                if session is None:
                    return False
                pos_rec = session.execute(
                    select(TradingPositionModel).where(
                        TradingPositionModel.account_id == account_id,
                        TradingPositionModel.symbol == symbol,
                    )
                ).scalar_one_or_none()
                if pos_rec:
                    session.delete(pos_rec)
            return True
        except Exception as e:
            logger.warning(f"Failed to delete position {symbol} for account {account_id}: {e}")
            return False

    def save_order(self, order: Order) -> bool:
        """持久化委托订单"""
        try:
            with self._session() as session:
                if session is None:
                    return False
                record = session.get(OrderRecordModel, order.order_id)
                if not record:
                    record = OrderRecordModel(
                        order_id=order.order_id,
                        account_id=order.account_id,
                        strategy_id=order.strategy_id,
                        symbol=order.symbol,
                        direction=order.direction.value if hasattr(order.direction, "value") else str(order.direction),
                        order_type=order.order_type.value if hasattr(order.order_type, "value") else str(order.order_type),
                        price=order.price,
                        volume=order.volume,
                        filled_volume=order.filled_volume,
                        filled_amount=order.filled_amount,
                        status=order.status.value if hasattr(order.status, "value") else str(order.status),
                        algo_type=order.algo_type.value if hasattr(order.algo_type, "value") else str(order.algo_type),
                        parent_order_id=order.parent_order_id,
                        reject_reason=order.reject_reason,
                    )
                    session.add(record)
                else:
                    record.filled_volume = order.filled_volume
                    record.filled_amount = order.filled_amount
                    record.status = order.status.value if hasattr(order.status, "value") else str(order.status)
                    record.reject_reason = order.reject_reason
            return True
        except Exception as e:
            logger.warning(f"Failed to save order {order.order_id}: {e}")
            return False

    def save_trade(self, trade: Trade) -> bool:
        """持久化成交记录"""
        try:
            with self._session() as session:
                if session is None:
                    return False
                record = session.get(TradeRecordModel, trade.trade_id)
                if not record:
                    record = TradeRecordModel(
                        trade_id=trade.trade_id,
                        order_id=trade.order_id,
                        account_id=trade.account_id,
                        symbol=trade.symbol,
                        direction=trade.direction.value if hasattr(trade.direction, "value") else str(trade.direction),
                        price=trade.price,
                        volume=trade.volume,
                        amount=trade.amount,
                        commission=trade.commission,
                        stamp_tax=trade.stamp_tax,
                    )
                    session.add(record)
            return True
        except Exception as e:
            logger.warning(f"Failed to save trade {trade.trade_id}: {e}")
            return False

    def save_transfer(self, transfer: CashTransfer, master_id: str = "master_default") -> bool:
        """持久化资金划转记录"""
        try:
            with self._session() as session:
                if session is None:
                    return False
                record = PMSTransferRecordModel(
                    master_id=master_id,
                    strategy_id=transfer.strategy_id,
                    transfer_amount=transfer.transfer_amount,
                    reason=transfer.reason,
                )
                session.add(record)
            return True
        except Exception as e:
            logger.warning(f"Failed to save cash transfer for strategy {transfer.strategy_id}: {e}")
            return False

    def restore_broker_state(self, broker) -> int:
        """从数据库全量反序列化恢复所有账户资金与持仓"""
        restored = 0
        try:
            with self._session() as session:
                if session is None:
                    return 0
                accounts = session.scalars(select(TradingAccountModel)).all()
                for acc_rec in accounts:
                    acc = AccountState(
                        account_id=acc_rec.account_id,
                        strategy_id=acc_rec.strategy_id,
                        available_cash=acc_rec.available_cash,
                        total_equity=acc_rec.total_equity,
                        positions={},
                    )
                    positions = session.scalars(
                        select(TradingPositionModel).where(
                            TradingPositionModel.account_id == acc_rec.account_id
                        )
                    ).all()
                    for p in positions:
                        acc.positions[p.symbol] = Position(
                            symbol=p.symbol,
                            total_volume=p.total_volume,
                            available_volume=p.available_volume,
                            frozen_volume=p.frozen_volume,
                            avg_cost=p.avg_cost,
                            market_value=p.market_value,
                        )
                    broker.accounts[acc.account_id] = acc
                    restored += 1
                logger.info(f"Restored {restored} trading accounts and their positions from database.")
        except Exception as e:
            logger.warning(f"Could not restore broker state from database: {e}")
        return restored

    def get_historical_trades(self, account_id: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """查询历史成交明细"""
        try:
            with self._session() as session:
                if session is None:
                    return []
                stmt = select(TradeRecordModel).order_by(desc(TradeRecordModel.executed_at)).limit(limit)
                if account_id:
                    stmt = stmt.where(TradeRecordModel.account_id == account_id)
                records = session.scalars(stmt).all()
                return [
                    {
                        "trade_id": r.trade_id,
                        "order_id": r.order_id,
                        "account_id": r.account_id,
                        "symbol": r.symbol,
                        "direction": r.direction,
                        "price": r.price,
                        "volume": r.volume,
                        "amount": r.amount,
                        "commission": r.commission,
                        "stamp_tax": r.stamp_tax,
                        "executed_at": r.executed_at.isoformat() if r.executed_at else None,
                    }
                    for r in records
                ]
        except Exception as e:
            logger.warning(f"Failed to query historical trades: {e}")
            return []

    def get_historical_orders(self, account_id: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """查询历史订单明细"""
        try:
            with self._session() as session:
                if session is None:
                    return []
                stmt = select(OrderRecordModel).order_by(desc(OrderRecordModel.created_at)).limit(limit)
                if account_id:
                    stmt = stmt.where(OrderRecordModel.account_id == account_id)
                records = session.scalars(stmt).all()
                return [
                    {
                        "order_id": r.order_id,
                        "account_id": r.account_id,
                        "strategy_id": r.strategy_id,
                        "symbol": r.symbol,
                        "direction": r.direction,
                        "order_type": r.order_type,
                        "price": r.price,
                        "volume": r.volume,
                        "filled_volume": r.filled_volume,
                        "filled_amount": r.filled_amount,
                        "status": r.status,
                        "algo_type": r.algo_type,
                        "parent_order_id": r.parent_order_id,
                        "reject_reason": r.reject_reason,
                        "created_at": r.created_at.isoformat() if r.created_at else None,
                    }
                    for r in records
                ]
        except Exception as e:
            logger.warning(f"Failed to query historical orders: {e}")
            return []

    def get_historical_transfers(self, strategy_id: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """查询资金划转明细"""
        try:
            with self._session() as session:
                if session is None:
                    return []
                stmt = select(PMSTransferRecordModel).order_by(desc(PMSTransferRecordModel.created_at)).limit(limit)
                if strategy_id:
                    stmt = stmt.where(PMSTransferRecordModel.strategy_id == strategy_id)
                records = session.scalars(stmt).all()
                return [
                    {
                        "id": r.id,
                        "master_id": r.master_id,
                        "strategy_id": r.strategy_id,
                        "transfer_amount": r.transfer_amount,
                        "reason": r.reason,
                        "created_at": r.created_at.isoformat() if r.created_at else None,
                    }
                    for r in records
                ]
        except Exception as e:
            logger.warning(f"Failed to query historical transfers: {e}")
            return []
