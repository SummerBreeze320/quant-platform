from src.risk_engine.circuit_breaker import CircuitBreakerManager
from src.risk_engine.models import CircuitBreakerLevel

def test_circuit_breaker_transitions():
    cb = CircuitBreakerManager(warn_drawdown=0.015, restrict_buy_drawdown=0.025, halt_drawdown=0.035)
    acc = "acc_test"

    # Initial equity 1,000,000 (drawdown = 0)
    state = cb.update_equity(acc, 1000000.0)
    assert state.level == CircuitBreakerLevel.NORMAL
    assert state.max_drawdown == 0.0

    # Drawdown 1.8% -> Yellow Warn
    state = cb.update_equity(acc, 982000.0)
    assert state.level == CircuitBreakerLevel.YELLOW_WARN

    # Drawdown 2.8% -> Orange Restrict Buy
    state = cb.update_equity(acc, 972000.0)
    assert state.level == CircuitBreakerLevel.ORANGE_RESTRICT_BUY

    # Drawdown 3.8% -> Red Halt
    state = cb.update_equity(acc, 962000.0)
    assert state.level == CircuitBreakerLevel.RED_HALT

    # Intraday bounce (equity recovers to 980,000, dd 2.0%)
    # One-way protection: MUST remain RED_HALT until manual reset
    state = cb.update_equity(acc, 980000.0)
    assert state.level == CircuitBreakerLevel.RED_HALT

    # Manual Reset
    reset_state = cb.reset(acc, reset_watermark=True)
    assert reset_state.level == CircuitBreakerLevel.NORMAL
    assert reset_state.high_watermark == 980000.0
