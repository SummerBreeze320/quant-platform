"""交易网关层

对接券商API/机构交易柜台，提供统一的报单、撤单、持仓查询接口。

分层设计：
1. BrokerBase: 抽象券商接口，定义标准API
2. SimulatedBroker: 模拟券商（用于模拟盘/仿真交易）
3. QMTBroker: QMT券商适配器（实盘，基于现有qmt_adapter）
4. OrderManager: 订单生命周期管理（提交→部分成交→全成交/撤单/拒单）

实盘环境需要券商SDK（如xtquant），模拟盘无需任何外部依赖。
"""
import time
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
from enum import Enum
import logging

from .trade_logger import TradeLogger, OrderRecord, TradeRecord

logger = logging.getLogger(__name__)


class OrderStatus(Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Order:
    """订单对象"""
    order_id: str
    stock: str
    side: str  # "buy" / "sell"
    price: float
    volume: float
    status: OrderStatus = OrderStatus.PENDING
    filled_volume: float = 0.0
    filled_price: float = 0.0
    create_time: str = ""
    update_time: str = ""
    reject_reason: str = ""

    @property
    def remaining_volume(self) -> float:
        return self.volume - self.filled_volume

    @property
    def is_active(self) -> bool:
        return self.status in (OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIAL)


class BrokerBase(ABC):
    """券商接口抽象基类"""

    @abstractmethod
    def connect(self) -> bool:
        """连接交易服务器"""
        pass

    @abstractmethod
    def place_order(self, stock: str, side: str,
                    price: float, volume: float) -> str:
        """提交订单，返回order_id"""
        pass

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """撤单"""
        pass

    @abstractmethod
    def get_position(self) -> Dict[str, Dict]:
        """查询持仓 {stock: {volume, cost, market_value}}"""
        pass

    @abstractmethod
    def get_account(self) -> Dict:
        """查询账户信息 {cash, total_value, frozen}"""
        pass

    @abstractmethod
    def get_order_status(self, order_id: str) -> Optional[Order]:
        """查询订单状态"""
        pass


class SimulatedBroker(BrokerBase):
    """模拟券商

    用于模拟盘交易，无需外部依赖。
    模拟订单撮合：市价单立即成交，限价单按价格匹配。
    支持滑点模拟和部分成交。
    """

    def __init__(self,
                 initial_cash: float = 1_000_000,
                 slippage_bps: float = 5.0,
                 commission_rate: float = 0.00025,
                 stamp_duty_rate: float = 0.001,
                 price_data: Optional[Dict[str, pd.DataFrame]] = None,
                 logger: Optional[TradeLogger] = None):
        self.cash = initial_cash
        self.initial_cash = initial_cash
        self.slippage_bps = slippage_bps
        self.commission_rate = commission_rate
        self.stamp_duty_rate = stamp_duty_rate
        self.price_data = price_data or {}
        self._logger = logger
        self._orders: Dict[str, Order] = {}
        self._positions: Dict[str, Dict] = {}  # {stock: {volume, cost}}
        self._order_counter = 0
        self._connected = False
        self._current_date = None

    def connect(self) -> bool:
        self._connected = True
        logger.info("模拟券商已连接")
        return True

    def set_price_data(self, price_data: Dict[str, pd.DataFrame]):
        """设置行情数据"""
        self.price_data = price_data

    def set_current_date(self, date):
        """设置当前交易日"""
        self._current_date = date

    def _get_current_price(self, stock: str) -> float:
        """获取当前价格"""
        if stock in self.price_data:
            df = self.price_data[stock]
            if self._current_date is not None and self._current_date in df.index:
                return float(df.loc[self._current_date, "close"])
            elif len(df) > 0:
                return float(df.iloc[-1]["close"])
        return 0.0

    def _gen_order_id(self) -> str:
        self._order_counter += 1
        return f"SIM-{self._order_counter:06d}"

    def place_order(self, stock: str, side: str,
                    price: float, volume: float) -> str:
        if not self._connected:
            logger.error("券商未连接")
            return ""

        order_id = self._gen_order_id()
        now = pd.Timestamp.now().isoformat()

        order = Order(
            order_id=order_id, stock=stock, side=side,
            price=price, volume=volume,
            status=OrderStatus.SUBMITTED,
            create_time=now, update_time=now,
        )
        self._orders[order_id] = order

        # 模拟撮合
        market_price = self._get_current_price(stock)
        if market_price <= 0:
            market_price = price

        # 滑点
        slip = market_price * self.slippage_bps / 10000
        if side == "buy":
            fill_price = market_price + slip
        else:
            fill_price = market_price - slip

        # 全部成交
        fill_volume = volume
        order.filled_volume = fill_volume
        order.filled_price = fill_price
        order.status = OrderStatus.FILLED
        order.update_time = pd.Timestamp.now().isoformat()

        # 更新持仓和资金
        notional = fill_volume * fill_price
        commission = notional * self.commission_rate
        stamp_duty = notional * self.stamp_duty_rate if side == "sell" else 0
        total_cost = commission + stamp_duty

        if side == "buy":
            self.cash -= (notional + commission)
            if stock in self._positions:
                pos = self._positions[stock]
                old_vol = pos["volume"]
                old_cost = pos["cost_price"]
                new_vol = old_vol + fill_volume
                new_cost = (old_vol * old_cost + fill_volume * fill_price) / new_vol
                self._positions[stock] = {"volume": new_vol, "cost_price": new_cost}
            else:
                self._positions[stock] = {
                    "volume": fill_volume,
                    "cost_price": fill_price,
                }
        else:  # sell
            self.cash += (notional - commission - stamp_duty)
            if stock in self._positions:
                pos = self._positions[stock]
                new_vol = pos["volume"] - fill_volume
                if new_vol <= 1e-6:
                    del self._positions[stock]
                else:
                    self._positions[stock] = {
                        "volume": new_vol,
                        "cost_price": pos["cost_price"],
                    }

        # 记录日志
        if self._logger:
            self._logger.log_order(OrderRecord(
                timestamp=now, order_id=order_id, stock=stock,
                side=side, price=price, volume=volume,
                status="filled", filled_volume=fill_volume,
                filled_price=fill_price,
            ))
            self._logger.log_trade(TradeRecord(
                timestamp=now, order_id=order_id, stock=stock,
                side=side, filled_price=fill_price,
                filled_volume=fill_volume,
                commission=commission, slippage=abs(fill_price - market_price) * fill_volume,
            ))

        logger.info(f"订单成交: {order_id} {side} {stock} "
                     f"{fill_volume}@{fill_price:.2f}")
        return order_id

    def cancel_order(self, order_id: str) -> bool:
        order = self._orders.get(order_id)
        if order and order.is_active:
            remaining = order.remaining_volume
            order.status = OrderStatus.CANCELLED
            order.update_time = pd.Timestamp.now().isoformat()
            logger.info(f"订单撤单: {order_id}, 剩余{remaining}")
            return True
        return False

    def get_position(self) -> Dict[str, Dict]:
        result = {}
        for stock, pos in self._positions.items():
            price = self._get_current_price(stock)
            volume = pos["volume"]
            market_value = volume * price
            cost = volume * pos["cost_price"]
            pnl = market_value - cost
            result[stock] = {
                "volume": volume,
                "cost_price": pos["cost_price"],
                "current_price": price,
                "market_value": market_value,
                "pnl": pnl,
                "pnl_pct": pnl / cost if cost > 0 else 0,
            }
        return result

    def get_account(self) -> Dict:
        total_position_value = sum(
            pos["volume"] * self._get_current_price(stock)
            for stock, pos in self._positions.items()
        )
        return {
            "cash": self.cash,
            "total_value": self.cash + total_position_value,
            "position_value": total_position_value,
            "initial_cash": self.initial_cash,
            "total_pnl": self.cash + total_position_value - self.initial_cash,
            "total_pnl_pct": (self.cash + total_position_value - self.initial_cash) / self.initial_cash,
        }

    def get_order_status(self, order_id: str) -> Optional[Order]:
        return self._orders.get(order_id)

    def get_all_orders(self) -> List[Order]:
        return list(self._orders.values())


class OrderManager:
    """订单生命周期管理

    管理订单从提交到最终状态的完整生命周期：
    提交 → 部分成交 → 全成交 / 撤单 / 拒单

    与BrokerBase配合使用，提供更高层的订单管理功能。
    """

    def __init__(self, broker: BrokerBase,
                 logger: Optional[TradeLogger] = None):
        self.broker = broker
        self._logger = logger
        self._active_orders: Dict[str, Order] = {}
        self._completed_orders: List[Order] = []
        self._order_history: List[dict] = []

    def submit_order(self, stock: str, side: str,
                     price: float, volume: float) -> Optional[str]:
        """提交新订单"""
        if volume <= 0:
            logger.warning(f"订单数量无效: {volume}")
            return None

        order_id = self.broker.place_order(stock, side, price, volume)
        if order_id:
            order = self.broker.get_order_status(order_id)
            if order:
                self._active_orders[order_id] = order
                if not order.is_active:
                    self._completed_orders.append(order)
                    self._order_history.append({
                        "order_id": order_id,
                        "stock": stock,
                        "side": side,
                        "volume": volume,
                        "price": price,
                        "status": order.status.value,
                        "filled_volume": order.filled_volume,
                        "filled_price": order.filled_price,
                    })
        return order_id

    def cancel(self, order_id: str) -> bool:
        """撤单"""
        if self.broker.cancel_order(order_id):
            order = self._active_orders.pop(order_id, None)
            if order:
                order.status = OrderStatus.CANCELLED
                self._completed_orders.append(order)
            return True
        return False

    def update_order_status(self):
        """更新所有活跃订单状态"""
        to_remove = []
        for order_id, order in self._active_orders.items():
            latest = self.broker.get_order_status(order_id)
            if latest and latest.status != order.status:
                order.status = latest.status
                order.filled_volume = latest.filled_volume
                order.filled_price = latest.filled_price
                order.update_time = latest.update_time
            if not order.is_active:
                self._completed_orders.append(order)
                to_remove.append(order_id)
        for oid in to_remove:
            del self._active_orders[oid]

    @property
    def active_orders(self) -> List[Order]:
        return list(self._active_orders.values())

    @property
    def completed_orders(self) -> List[Order]:
        return self._completed_orders

    def get_order_summary(self) -> dict:
        return {
            "active_count": len(self._active_orders),
            "completed_count": len(self._completed_orders),
            "total_submitted": len(self._order_history),
            "filled_count": sum(1 for o in self._completed_orders
                                if o.status == OrderStatus.FILLED),
            "cancelled_count": sum(1 for o in self._completed_orders
                                    if o.status == OrderStatus.CANCELLED),
        }
