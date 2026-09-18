import pytest
from src.pms.models import StrategyType, MasterAccount, StrategyAccount
from src.execution_engine.models import Position
from src.pms.aggregator import PortfolioAggregator
from src.risk_engine.circuit_breaker import CircuitBreakerManager

def test_consolidated_lookthrough_aggregation():
    # Setup Master with 2 strategies both holding 600519.SH and different stocks
    pos_a1 = Position(symbol="600519.SH", total_volume=500, avg_cost=1800.0, market_value=900_000.0)
    pos_a2 = Position(symbol="000858.SZ", total_volume=1000, avg_cost=150.0, market_value=150_000.0)
    strat_a = StrategyAccount(
        strategy_id="strat_A",
        name="Strategy A",
        current_cash=950_000.0,
        total_equity=2_000_000.0,
        positions={"600519.SH": pos_a1, "000858.SZ": pos_a2}
    )

    pos_b1 = Position(symbol="600519.SH", total_volume=1000, avg_cost=1800.0, market_value=1_800_000.0)
    pos_b2 = Position(symbol="601318.SH", total_volume=2000, avg_cost=50.0, market_value=100_000.0)
    strat_b = StrategyAccount(
        strategy_id="strat_B",
        name="Strategy B",
        current_cash=1_100_000.0,
        total_equity=3_000_000.0,
        positions={"600519.SH": pos_b1, "601318.SH": pos_b2}
    )

    master = MasterAccount(
        master_id="master_fund",
        reserve_cash=5_000_000.0,
        total_equity=10_000_000.0,
        strategies={"strat_A": strat_a, "strat_B": strat_b}
    )

    aggregator = PortfolioAggregator(max_stock_concentration=0.15, max_leverage_ratio=1.0)
    portfolio = aggregator.aggregate(master)

    # 1. Look-through positions
    assert len(portfolio.positions) == 3
    c_moutai = portfolio.positions["600519.SH"]
    assert c_moutai.total_volume == 1500
    assert c_moutai.total_market_value == 2_700_000.0
    assert set(c_moutai.contributing_strategies) == {"strat_A", "strat_B"}

    # 2. Total equity and cash
    assert portfolio.total_cash == 7_050_000.0 # 5M reserve + 0.95M A + 1.1M B
    assert portfolio.total_market_value == 2_950_000.0 # 2.7M + 0.15M + 0.1M
    assert portfolio.total_equity == 10_000_000.0

    # 3. Moutai concentration = 2,700,000 / 10,000,000 = 27%
    assert pytest.approx(c_moutai.weight_in_master, 0.001) == 0.27
    assert pytest.approx(portfolio.max_stock_concentration, 0.001) == 0.27

    # 4. Macro alert should trigger because 27% > 15% limit
    assert len(portfolio.macro_alerts) >= 1
    assert any("600519.SH" in alert for alert in portfolio.macro_alerts)

def test_master_circuit_breaker_link():
    master = MasterAccount(master_id="master_fund", reserve_cash=10_000_000.0, total_equity=10_000_000.0)
    cb = CircuitBreakerManager(warn_drawdown=0.015, restrict_buy_drawdown=0.025, halt_drawdown=0.035)

    aggregator = PortfolioAggregator(circuit_breaker=cb)
    # First update sets high watermark at 10,000,000
    res1 = aggregator.aggregate(master)
    assert res1.circuit_breaker_level == 0

    # Drop master equity to 9,600,000 (4% drawdown > 3.5% halt threshold)
    master.reserve_cash = 9_600_000.0
    master.total_equity = 9_600_000.0
    res2 = aggregator.aggregate(master)
    assert res2.circuit_breaker_level == 3 # RED_HALT
    assert any("全局熔断" in alert for alert in res2.macro_alerts)
