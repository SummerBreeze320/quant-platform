from typing import Optional, Dict, Any, List
import pandas as pd
import numpy as np
from src.risk_engine.models import (
    PortfolioRiskRequest, PortfolioRiskResult, CircuitBreakerLevel, AlertLevel
)
from src.risk_engine.circuit_breaker import CircuitBreakerManager
from src.risk_engine.alert import RiskAlertManager
from src.qlib_engine.risk_model import BarraRiskModel

class PostTradeRiskMonitor:
    def __init__(
        self,
        cb_manager: Optional[CircuitBreakerManager] = None,
        alert_manager: Optional[RiskAlertManager] = None,
        max_concentration_threshold: float = 0.20,
        max_top5_threshold: float = 0.60
    ):
        self.cb_manager = cb_manager or CircuitBreakerManager()
        self.alert_manager = alert_manager or RiskAlertManager()
        self.max_concentration_threshold = max_concentration_threshold
        self.max_top5_threshold = max_top5_threshold
        self.barra_model = BarraRiskModel()

    def analyze_portfolio(self, request: PortfolioRiskRequest) -> PortfolioRiskResult:
        equity = request.total_equity
        positions = request.positions
        alerts: List[str] = []

        # 1. Update circuit breaker state
        cb_state = self.cb_manager.update_equity(
            account_id=request.account_id,
            equity=equity,
            high_watermark=request.high_watermark
        )
        if cb_state.level > CircuitBreakerLevel.NORMAL:
            msg = f"账户熔断级别更新为 {cb_state.level.name}: {cb_state.trigger_reason}"
            alerts.append(msg)
            self.alert_manager.record_alert(
                level=AlertLevel.CRITICAL if cb_state.level == CircuitBreakerLevel.RED_HALT else AlertLevel.WARN,
                account_id=request.account_id,
                rule_name="CircuitBreaker",
                message=msg
            )

        # 2. Portfolio concentration metrics
        total_mv = sum(positions.values()) if positions else 0.0
        leverage_ratio = total_mv / equity if equity > 0 else 0.0

        weights = [mv / equity for mv in positions.values()] if equity > 0 else []
        max_stock_weight = max(weights) if weights else 0.0
        sorted_weights = sorted(weights, reverse=True)
        top5_weight = sum(sorted_weights[:5])

        if max_stock_weight > self.max_concentration_threshold:
            msg = f"单一持仓集中度 {max_stock_weight:.2%} 超过警示阈值 {self.max_concentration_threshold:.2%}"
            alerts.append(msg)
            self.alert_manager.record_alert(
                level=AlertLevel.WARN,
                account_id=request.account_id,
                rule_name="ConcentrationRisk",
                message=msg
            )

        if top5_weight > self.max_top5_threshold:
            msg = f"Top-5 持仓集中度 {top5_weight:.2%} 超过警示阈值 {self.max_top5_threshold:.2%}"
            alerts.append(msg)

        # 3. Optional Barra active risk calculation
        active_risk_metrics = None
        if request.benchmark_weights and request.factor_exposures and positions:
            active_risk_metrics = self._calc_barra_risk(request)

        return PortfolioRiskResult(
            account_id=request.account_id,
            current_equity=equity,
            high_watermark=cb_state.high_watermark,
            drawdown=(cb_state.high_watermark - equity) / cb_state.high_watermark if cb_state.high_watermark > 0 else 0.0,
            circuit_breaker_level=cb_state.level,
            max_stock_weight=max_stock_weight,
            top5_weight=top5_weight,
            leverage_ratio=leverage_ratio,
            active_risk=active_risk_metrics,
            alerts=alerts
        )

    def _calc_barra_risk(self, request: PortfolioRiskRequest) -> Optional[Dict[str, float]]:
        try:
            symbols = list(request.positions.keys())
            if not symbols:
                return None
            w = np.array([request.positions[s] / request.total_equity for s in symbols])
            bw = np.array([request.benchmark_weights.get(s, 0.0) for s in symbols])
            # Build mock/passed factor matrix
            factor_names = list(next(iter(request.factor_exposures.values())).keys())
            X = np.array([[request.factor_exposures[s].get(f, 0.0) for f in factor_names] for s in symbols])
            F = np.eye(len(factor_names)) * 0.0001
            delta = np.ones(len(symbols)) * 0.0004
            return self.barra_model.predict_portfolio_risk(
                weights=w,
                exposures=X,
                factor_cov=F,
                specific_var=delta,
                benchmark_weights=bw
            )
        except Exception:
            return None
