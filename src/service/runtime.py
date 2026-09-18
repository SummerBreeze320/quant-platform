"""Application-scoped composition root for the in-memory trading services."""
from threading import RLock

from fastapi import Request

from src.execution_engine.coordinator import ExecutionCoordinator
from src.execution_engine.gateway.paper_broker import PaperBroker
from src.pms.aggregator import PortfolioAggregator
from src.pms.manager import PortfolioManager
from src.risk_engine.alert import RiskAlertManager
from src.risk_engine.circuit_breaker import CircuitBreakerManager
from src.risk_engine.post_trade import PostTradeRiskMonitor
from src.risk_engine.pre_trade import PreTradeRiskChecker
from src.service.market_runtime import MarketRuntime


class ServiceRuntime:
    """One account ledger and risk policy per app, shared by all API routes.

    Callers hold lock while reading or mutating domain state. This deliberately
    remains a single-process simulation; separate workers do not share state.
    """

    def __init__(self):
        self.lock = RLock()
        self.broker = PaperBroker()
        self.circuit_breaker = CircuitBreakerManager()
        self.alert_manager = RiskAlertManager()
        self.risk_checker = PreTradeRiskChecker(
            cb_manager=self.circuit_breaker, alert_manager=self.alert_manager,
        )
        self.risk_monitor = PostTradeRiskMonitor(
            cb_manager=self.circuit_breaker, alert_manager=self.alert_manager,
        )
        self.coordinator = ExecutionCoordinator(
            gateway=self.broker, risk_checker=self.risk_checker,
            circuit_breaker=self.circuit_breaker,
        )
        self.pms_manager = PortfolioManager(broker=self.broker, circuit_breaker=self.circuit_breaker)
        self.aggregator = PortfolioAggregator(circuit_breaker=self.circuit_breaker)
        self.pms_manager.register_strategy(
            "hft_stream_01", "Market streaming signals", initial_budget=5_000_000,
        )
        self._market = None

    @property
    def market(self) -> MarketRuntime:
        # Redis connection probing belongs to first market use, not module import.
        with self.lock:
            if self._market is None:
                self._market = MarketRuntime(self.broker, self.risk_checker, self.lock)
            return self._market


def get_runtime(request: Request) -> ServiceRuntime:
    return request.app.state.runtime
