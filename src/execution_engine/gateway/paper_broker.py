import uuid
from typing import Optional, Dict, List
from src.execution_engine.models import (
    Order, Trade, Position, AccountState, OrderDirection, OrderStatus
)
from src.execution_engine.gateway.base import BaseBrokerGateway
from src.common.logger import logger

class PaperBroker(BaseBrokerGateway):
    def __init__(
        self,
        commission_rate: float = 0.0002,
        stamp_tax_rate: float = 0.0005,
        slippage_rate: float = 0.0005
    ):
        self.commission_rate = commission_rate
        self.stamp_tax_rate = stamp_tax_rate
        self.slippage_rate = slippage_rate
        self.accounts: Dict[str, AccountState] = {}
        self.trades: List[Trade] = []

    def create_account(self, account_id: str, initial_cash: float = 1_000_000.0) -> AccountState:
        acc = AccountState(
            account_id=account_id,
            total_equity=initial_cash,
            available_cash=initial_cash,
            positions={}
        )
        self.accounts[account_id] = acc
        return acc

    def get_account(self, account_id: str) -> AccountState:
        if account_id not in self.accounts:
            self.create_account(account_id)
        acc = self.accounts[account_id]
        # Re-calc market value and total equity
        mv_total = sum(p.market_value for p in acc.positions.values())
        acc.total_equity = acc.available_cash + mv_total
        return acc

    def cancel_order(self, order_id: str) -> bool:
        return True

    def submit_order(self, order: Order) -> Optional[Trade]:
        acc = self.get_account(order.account_id)
        sym = order.symbol
        vol = order.volume
        price = order.price

        if order.direction == OrderDirection.BUY:
            fill_price = round(price * (1.0 + self.slippage_rate), 4)
            gross_amount = fill_price * vol
            commission = gross_amount * self.commission_rate
            total_cost = gross_amount + commission

            if total_cost > acc.available_cash:
                order.status = OrderStatus.REJECTED
                order.reject_reason = f"现金不足: 所需 {total_cost:.2f} 大于可用 {acc.available_cash:.2f}"
                return None

            # Execute Buy
            acc.available_cash -= total_cost
            pos = acc.positions.get(sym, Position(symbol=sym))
            new_total = pos.total_volume + vol
            new_cost = (pos.total_volume * pos.avg_cost + gross_amount) / new_total if new_total > 0 else fill_price
            pos.total_volume = new_total
            pos.frozen_volume += vol  # T+1 freeze
            pos.avg_cost = new_cost
            pos.market_value = new_total * fill_price
            acc.positions[sym] = pos

            order.status = OrderStatus.FILLED
            order.filled_volume = vol
            order.filled_amount = gross_amount

            trade = Trade(
                trade_id=f"trd_{uuid.uuid4().hex[:8]}",
                order_id=order.order_id,
                account_id=order.account_id,
                symbol=sym,
                direction=OrderDirection.BUY,
                price=fill_price,
                volume=vol,
                amount=gross_amount,
                commission=commission,
                stamp_tax=0.0
            )
            self.trades.append(trade)
            self.get_account(order.account_id)
            return trade

        elif order.direction == OrderDirection.SELL:
            pos = acc.positions.get(sym)
            if not pos or pos.available_volume < vol:
                avail = pos.available_volume if pos else 0
                order.status = OrderStatus.REJECTED
                order.reject_reason = f"可用持仓不足(T+1受限): 申报 {vol} 可用 {avail}"
                return None

            fill_price = round(price * (1.0 - self.slippage_rate), 4)
            gross_amount = fill_price * vol
            commission = gross_amount * self.commission_rate
            stamp_tax = gross_amount * self.stamp_tax_rate
            net_proceeds = gross_amount - commission - stamp_tax

            # Execute Sell
            acc.available_cash += net_proceeds
            pos.total_volume -= vol
            pos.available_volume -= vol
            pos.market_value = pos.total_volume * fill_price
            if pos.total_volume == 0:
                del acc.positions[sym]

            order.status = OrderStatus.FILLED
            order.filled_volume = vol
            order.filled_amount = gross_amount

            trade = Trade(
                trade_id=f"trd_{uuid.uuid4().hex[:8]}",
                order_id=order.order_id,
                account_id=order.account_id,
                symbol=sym,
                direction=OrderDirection.SELL,
                price=fill_price,
                volume=vol,
                amount=gross_amount,
                commission=commission,
                stamp_tax=stamp_tax
            )
            self.trades.append(trade)
            self.get_account(order.account_id)
            return trade

        return None

    def settle_overnight(self, account_id: Optional[str] = None) -> None:
        target_accounts = [self.get_account(account_id)] if account_id else list(self.accounts.values())
        for acc in target_accounts:
            for pos in acc.positions.values():
                pos.available_volume += pos.frozen_volume
                pos.frozen_volume = 0
            logger.info(f"Overnight settlement completed for account {acc.account_id} (T+1 shares unlocked)")
