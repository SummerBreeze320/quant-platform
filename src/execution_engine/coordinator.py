from typing import Optional, Dict, Any, List
from src.execution_engine.models import (
    Order, Trade, AccountState, OrderDirection, OrderStatus, AlgoType, RebalancePlan
)
from src.execution_engine.rebalance import RebalanceOrderGenerator
from src.execution_engine.algos import BaseExecutionAlgo, DirectAlgo, TwapAlgo, VwapAlgo
from src.execution_engine.gateway.base import BaseBrokerGateway
from src.execution_engine.gateway.paper_broker import PaperBroker
from src.risk_engine.pre_trade import PreTradeRiskChecker
from src.risk_engine.circuit_breaker import CircuitBreakerManager
from src.risk_engine.models import OrderRiskRequest, CircuitBreakerLevel
from src.common.logger import logger

class ExecutionCoordinator:
    def __init__(
        self,
        gateway: Optional[BaseBrokerGateway] = None,
        risk_checker: Optional[PreTradeRiskChecker] = None,
        circuit_breaker: Optional[CircuitBreakerManager] = None
    ):
        self.gateway = gateway or PaperBroker()
        if risk_checker is not None:
            if circuit_breaker is not None and risk_checker.cb_manager is not circuit_breaker:
                raise ValueError("risk_checker and coordinator must share the same circuit breaker")
            self.circuit_breaker = risk_checker.cb_manager
            self.risk_checker = risk_checker
        else:
            self.circuit_breaker = circuit_breaker or CircuitBreakerManager()
            self.risk_checker = PreTradeRiskChecker(cb_manager=self.circuit_breaker)
        self.algo_registry: Dict[AlgoType, BaseExecutionAlgo] = {
            AlgoType.DIRECT: DirectAlgo(),
            AlgoType.TWAP: TwapAlgo(),
            AlgoType.VWAP: VwapAlgo()
        }

    def execute_rebalance(
        self,
        account_id: str,
        target_weights: Dict[str, float],
        current_prices: Dict[str, float],
        algo_type: AlgoType = AlgoType.DIRECT
    ) -> Dict[str, Any]:
        acc = self.gateway.get_account(account_id)
        plan = RebalanceOrderGenerator.generate_plan(
            account=acc,
            target_weights=target_weights,
            current_prices=current_prices,
            algo_type=algo_type
        )

        algo = self.algo_registry.get(algo_type, DirectAlgo())
        executed_trades: List[Trade] = []
        rejected_orders: List[Order] = []

        # Sequence: SELL orders first, then BUY orders
        order_queue = plan.sell_orders + plan.buy_orders

        for parent_order in order_queue:
            # Check Circuit Breaker before processing
            cb_state = self.circuit_breaker.get_state(account_id)
            if cb_state.level == CircuitBreakerLevel.RED_HALT:
                parent_order.status = OrderStatus.REJECTED
                parent_order.reject_reason = "RED_HALT 熔断激活，全局终止交易"
                rejected_orders.append(parent_order)
                break
            elif cb_state.level == CircuitBreakerLevel.ORANGE_RESTRICT_BUY and parent_order.direction == OrderDirection.BUY:
                parent_order.status = OrderStatus.REJECTED
                parent_order.reject_reason = "ORANGE 熔断激活，禁止开新仓/加仓"
                rejected_orders.append(parent_order)
                continue

            # Slice order
            sub_orders = algo.slice_order(parent_order)

            for sub_order in sub_orders:
                # 1. Pre-trade Risk Check
                acc_now = self.gateway.get_account(account_id)
                curr_pos_vol = acc_now.positions.get(sub_order.symbol).total_volume if acc_now.positions.get(sub_order.symbol) else 0

                risk_req = OrderRiskRequest(
                    account_id=account_id,
                    order_id=sub_order.order_id,
                    symbol=sub_order.symbol,
                    direction=sub_order.direction.value,
                    price=sub_order.price,
                    volume=sub_order.volume,
                    current_position=curr_pos_vol,
                    total_equity=acc_now.total_equity,
                    available_cash=acc_now.available_cash
                )
                risk_res = self.risk_checker.check_order(risk_req)

                if not risk_res.passed:
                    sub_order.status = OrderStatus.REJECTED
                    sub_order.reject_reason = "; ".join(risk_res.violations)
                    rejected_orders.append(sub_order)
                    continue

                # 2. Submit to Gateway
                trade = self.gateway.submit_order(sub_order)
                if trade:
                    executed_trades.append(trade)
                    # 3. Post-trade Equity Update to Circuit Breaker
                    updated_acc = self.gateway.get_account(account_id)
                    self.circuit_breaker.update_equity(account_id, updated_acc.total_equity)
                else:
                    rejected_orders.append(sub_order)

        final_acc = self.gateway.get_account(account_id)
        return {
            "account_id": account_id,
            "executed_trades": executed_trades,
            "rejected_orders": rejected_orders,
            "final_equity": final_acc.total_equity,
            "final_cash": final_acc.available_cash,
            "circuit_breaker_level": self.circuit_breaker.get_state(account_id).level
        }
