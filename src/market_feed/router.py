import uuid
from typing import Dict, List, Optional
from src.common.logger import logger
from src.market_feed.models import SignalEvent, SignalDirection
from src.execution_engine.models import Order, OrderDirection, OrderType
from src.execution_engine.gateway.paper_broker import PaperBroker
from src.risk_engine.models import OrderRiskRequest
from src.risk_engine.pre_trade import PreTradeRiskChecker

class SignalRouter:
    """高频交易信号路由协调器"""

    def __init__(
        self,
        broker: Optional[PaperBroker] = None,
        risk_checker: Optional[PreTradeRiskChecker] = None,
    ):
        self.broker = broker
        self.risk_checker = risk_checker
        self.strategy_configs: Dict[str, dict] = {}
        self.signals_history: List[dict] = []

    def register_strategy(
        self,
        strategy_id: str,
        auto_execute: bool = False,
        default_volume: int = 100
    ) -> None:
        self.strategy_configs[strategy_id] = {
            "auto_execute": auto_execute,
            "default_volume": default_volume,
        }

    def route_signal(self, signal: SignalEvent) -> dict:
        config = self.strategy_configs.get(signal.strategy_id, {"auto_execute": False, "default_volume": 100})
        auto_execute = config.get("auto_execute", False)

        record = {
            "signal_id": signal.signal_id,
            "strategy_id": signal.strategy_id,
            "symbol": signal.symbol,
            "direction": signal.direction.value,
            "price": signal.price,
            "volume": signal.volume,
            "reason": signal.reason,
            "status": "PENDING",
            "trade_id": None,
            "detail": ""
        }

        if not auto_execute:
            record["status"] = "BROADCASTED"
            record["detail"] = "策略为手动观察模式，已推送至告警与流广播"
            self.signals_history.append(record)
            logger.info(f"Signal broadcasted: [{signal.strategy_id}] {signal.symbol} {signal.direction} @ {signal.price}")
            return record

        if self.broker is None:
            record["status"] = "ERROR"
            record["detail"] = "未配置交易柜台"
            self.signals_history.append(record)
            return record

        # 构造执行订单
        order_dir = OrderDirection.BUY if signal.direction == SignalDirection.BUY else OrderDirection.SELL
        order = Order(
            order_id=f"ord_sig_{uuid.uuid4().hex[:8]}",
            account_id=signal.strategy_id,
            symbol=signal.symbol,
            direction=order_dir,
            price=signal.price,
            volume=signal.volume,
            order_type=OrderType.LIMIT,
        )

        acc = self.broker.get_account(signal.strategy_id)
        curr_pos = acc.positions.get(signal.symbol)
        curr_vol = curr_pos.total_volume if curr_pos else 0

        # 事前风控检查
        if self.risk_checker is not None:
            risk_req = OrderRiskRequest(
                account_id=signal.strategy_id,
                order_id=order.order_id,
                symbol=signal.symbol,
                direction=order_dir.value,
                price=signal.price,
                volume=signal.volume,
                current_position=curr_vol,
                total_equity=acc.total_equity,
                available_cash=acc.available_cash
            )
            risk_decision = self.risk_checker.check_order(risk_req)
            if not risk_decision.passed:
                record["status"] = "REJECTED"
                record["detail"] = f"事前风控拦截: {'; '.join(risk_decision.violations)}"
                self.signals_history.append(record)
                logger.warning(f"Signal rejected by risk: {risk_decision.violations}")
                return record

        # 柜台撮合成交
        trade = self.broker.submit_order(order)
        if trade:
            if self.risk_checker is not None:
                updated_account = self.broker.get_account(signal.strategy_id)
                self.risk_checker.cb_manager.update_equity(signal.strategy_id, updated_account.total_equity)
            record["status"] = "EXECUTED"
            record["trade_id"] = trade.trade_id
            record["detail"] = f"成交成功: 成交价 {trade.price}, 股数 {trade.volume}"
            logger.info(f"Signal executed: Trade {trade.trade_id} on {signal.symbol}")
        else:
            record["status"] = "REJECTED"
            record["detail"] = order.reject_reason or "柜台拒绝"

        self.signals_history.append(record)
        return record

    def get_recent_signals(self, limit: int = 50) -> List[dict]:
        return self.signals_history[-limit:]
