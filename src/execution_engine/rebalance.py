import uuid
from typing import Dict, Optional
from src.execution_engine.models import (
    AccountState, Order, OrderDirection, OrderType, AlgoType, RebalancePlan
)

class RebalanceOrderGenerator:
    @staticmethod
    def generate_plan(
        account: AccountState,
        target_weights: Dict[str, float],
        current_prices: Dict[str, float],
        algo_type: AlgoType = AlgoType.DIRECT
    ) -> RebalancePlan:
        equity = account.total_equity
        sell_orders = []
        buy_orders = []

        all_symbols = set(target_weights.keys()).union(account.positions.keys())

        for sym in sorted(list(all_symbols)):
            price = current_prices.get(sym, 0.0)
            if price <= 0:
                continue

            target_weight = target_weights.get(sym, 0.0)
            target_mv = target_weight * equity
            target_shares = int((target_mv / price) // 100) * 100

            current_pos = account.positions.get(sym)
            current_shares = current_pos.total_volume if current_pos else 0
            available_shares = current_pos.available_volume if current_pos else 0

            delta = target_shares - current_shares

            if delta < 0:
                # Sell order (capped at available_volume for T+1 safety)
                sell_vol = min(abs(delta), available_shares)
                if sell_vol > 0:
                    sell_orders.append(Order(
                        order_id=f"ord_{uuid.uuid4().hex[:8]}",
                        account_id=account.account_id,
                        symbol=sym,
                        direction=OrderDirection.SELL,
                        order_type=OrderType.LIMIT,
                        price=price,
                        volume=sell_vol,
                        algo_type=algo_type
                    ))
            elif delta > 0:
                # Buy order (must be multiples of 100)
                buy_vol = int(delta // 100) * 100
                if buy_vol > 0:
                    buy_orders.append(Order(
                        order_id=f"ord_{uuid.uuid4().hex[:8]}",
                        account_id=account.account_id,
                        symbol=sym,
                        direction=OrderDirection.BUY,
                        order_type=OrderType.LIMIT,
                        price=price,
                        volume=buy_vol,
                        algo_type=algo_type
                    ))

        return RebalancePlan(
            account_id=account.account_id,
            target_weights=target_weights,
            sell_orders=sell_orders,
            buy_orders=buy_orders
        )
