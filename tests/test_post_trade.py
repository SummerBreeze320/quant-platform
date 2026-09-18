from src.risk_engine.post_trade import PostTradeRiskMonitor
from src.risk_engine.circuit_breaker import CircuitBreakerManager
from src.risk_engine.models import PortfolioRiskRequest, CircuitBreakerLevel

def test_post_trade_monitor():
    cb = CircuitBreakerManager(warn_drawdown=0.015, restrict_buy_drawdown=0.025, halt_drawdown=0.035)
    monitor = PostTradeRiskMonitor(cb_manager=cb)

    # 1. Total equity 1,000,000, positions sum to 800,000
    req = PortfolioRiskRequest(
        account_id="acc_01",
        total_equity=1000000.0,
        high_watermark=1000000.0,
        positions={"600519.SH": 200000.0, "000858.SZ": 150000.0, "601318.SH": 100000.0}
    )
    res = monitor.analyze_portfolio(req)
    assert res.drawdown == 0.0
    assert res.circuit_breaker_level == CircuitBreakerLevel.NORMAL
    assert res.max_stock_weight == 0.20 # 200k / 1M
    assert res.leverage_ratio == 0.45   # 450k / 1M

    # 2. Equity drops to 960,000 (dd = 4.0% > 3.5% -> RED_HALT)
    req_drawdown = PortfolioRiskRequest(
        account_id="acc_01",
        total_equity=960000.0,
        high_watermark=1000000.0,
        positions={"600519.SH": 180000.0}
    )
    res_dd = monitor.analyze_portfolio(req_drawdown)
    assert res_dd.drawdown == 0.04
    assert res_dd.circuit_breaker_level == CircuitBreakerLevel.RED_HALT
    assert len(res_dd.alerts) > 0
