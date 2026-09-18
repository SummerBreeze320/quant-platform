import json
from datetime import datetime
from typing import Dict, Optional, Any
from src.risk_engine.models import CircuitBreakerLevel, CircuitBreakerState
from src.common.logger import logger

class CircuitBreakerManager:
    def __init__(
        self,
        warn_drawdown: float = 0.015,
        restrict_buy_drawdown: float = 0.025,
        halt_drawdown: float = 0.035,
        redis_client: Optional[Any] = None
    ):
        self.warn_drawdown = warn_drawdown
        self.restrict_buy_drawdown = restrict_buy_drawdown
        self.halt_drawdown = halt_drawdown
        self.redis_client = redis_client
        self._states: Dict[str, CircuitBreakerState] = {}

    def get_state(self, account_id: str) -> CircuitBreakerState:
        if account_id not in self._states:
            self._states[account_id] = CircuitBreakerState(
                account_id=account_id,
                level=CircuitBreakerLevel.NORMAL,
                high_watermark=0.0,
                current_equity=0.0,
                max_drawdown=0.0
            )
        return self._states[account_id]

    def update_equity(
        self,
        account_id: str,
        equity: float,
        high_watermark: Optional[float] = None
    ) -> CircuitBreakerState:
        state = self.get_state(account_id)
        if high_watermark is not None and high_watermark > 0:
            hw = max(high_watermark, equity)
        else:
            hw = max(state.high_watermark, equity)

        drawdown = (hw - equity) / hw if hw > 0 else 0.0
        max_dd = max(state.max_drawdown, drawdown)

        # Calculate tier
        if drawdown >= self.halt_drawdown:
            target_level = CircuitBreakerLevel.RED_HALT
            reason = f"日内回撤达到 {drawdown:.2%} 触碰全局硬熔断线 {self.halt_drawdown:.2%}"
        elif drawdown >= self.restrict_buy_drawdown:
            target_level = CircuitBreakerLevel.ORANGE_RESTRICT_BUY
            reason = f"日内回撤达到 {drawdown:.2%} 触碰禁买限制线 {self.restrict_buy_drawdown:.2%}"
        elif drawdown >= self.warn_drawdown:
            target_level = CircuitBreakerLevel.YELLOW_WARN
            reason = f"日内回撤达到 {drawdown:.2%} 触碰预警线 {self.warn_drawdown:.2%}"
        else:
            target_level = CircuitBreakerLevel.NORMAL
            reason = None

        # One-way ratchet: cannot automatically downgrade without manual reset
        new_level = max(state.level, target_level)
        triggered_at = datetime.now().isoformat() if new_level > state.level else state.last_triggered_at
        active_reason = reason if new_level > state.level else state.trigger_reason

        updated_state = CircuitBreakerState(
            account_id=account_id,
            level=new_level,
            high_watermark=hw,
            current_equity=equity,
            max_drawdown=max_dd,
            last_triggered_at=triggered_at,
            trigger_reason=active_reason
        )
        self._states[account_id] = updated_state
        self._sync_redis(account_id, updated_state)
        return updated_state

    def adjust_cash_flow(
        self, account_id: str, equity_before: float, equity_after: float
    ) -> CircuitBreakerState:
        """Rebase the watermark for external capital, without clearing risk history.

        Scaling both equity and its watermark preserves the current drawdown
        ratio. A zero-balance account starts a new capital baseline on funding,
        but retains its existing halt and maximum drawdown until manual reset.
        """
        if equity_before < 0 or equity_after < 0:
            raise ValueError("Cash flow equities must be non-negative")
        state = self.update_equity(account_id, equity_before)
        watermark = (
            state.high_watermark * equity_after / equity_before
            if equity_before > 0 else equity_after
        )
        adjusted = state.model_copy(update={
            "current_equity": equity_after,
            "high_watermark": watermark,
        })
        self._states[account_id] = adjusted
        self._sync_redis(account_id, adjusted)
        return adjusted

    def reset(self, account_id: str, reset_watermark: bool = True) -> CircuitBreakerState:
        state = self.get_state(account_id)
        hw = state.current_equity if (reset_watermark and state.current_equity > 0) else state.high_watermark
        new_state = CircuitBreakerState(
            account_id=account_id,
            level=CircuitBreakerLevel.NORMAL,
            high_watermark=hw,
            current_equity=state.current_equity,
            max_drawdown=0.0,
            last_triggered_at=None,
            trigger_reason="Manual reset"
        )
        self._states[account_id] = new_state
        self._sync_redis(account_id, new_state)
        logger.info(f"Circuit breaker reset for account {account_id}")
        return new_state

    def daily_reset(self) -> None:
        for acc in list(self._states.keys()):
            self.reset(acc, reset_watermark=True)

    def _sync_redis(self, account_id: str, state: CircuitBreakerState) -> None:
        if not self.redis_client:
            return
        try:
            client = self.redis_client.get_client() if hasattr(self.redis_client, "get_client") else self.redis_client
            if client:
                client.set(f"quant:risk:cb:{account_id}", state.model_dump_json(), ex=86400)
        except Exception as e:
            logger.debug(f"Redis cb sync skipped: {e}")
