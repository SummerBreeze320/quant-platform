from src.pms.models import (
    StrategyType,
    AllocationMethod,
    StrategyAccount,
    MasterAccount,
    CashTransfer,
    CapitalAllocationPlan,
    ConsolidatedPosition,
    ConsolidatedPortfolio,
    StrategyPerformance,
)
from src.execution_engine.models import Position

def test_pms_enums():
    assert StrategyType.ALPHA == "ALPHA"
    assert StrategyType.INDEX_ENHANCEMENT == "INDEX_ENHANCEMENT"
    assert StrategyType.SECTOR_ROTATION == "SECTOR_ROTATION"
    assert StrategyType.MOMENTUM == "MOMENTUM"

    assert AllocationMethod.EQUAL_WEIGHT == "EQUAL_WEIGHT"
    assert AllocationMethod.FIXED_BUDGET == "FIXED_BUDGET"
    assert AllocationMethod.RISK_PARITY == "RISK_PARITY"
    assert AllocationMethod.SHARPE_WEIGHTED == "SHARPE_WEIGHTED"

def test_strategy_account_model():
    pos = Position(
        symbol="600519.SH",
        total_volume=500,
        available_volume=500,
        avg_cost=1800.0,
        market_value=900000.0,
    )
    strategy = StrategyAccount(
        strategy_id="strat_alpha_01",
        name="沪深300指增01号",
        strategy_type=StrategyType.INDEX_ENHANCEMENT,
        allocated_budget=2000000.0,
        current_cash=1100000.0,
        total_equity=2000000.0,
        positions={"600519.SH": pos},
    )
    assert strategy.strategy_id == "strat_alpha_01"
    assert strategy.is_active is True
    assert strategy.positions["600519.SH"].total_volume == 500

def test_master_account_model():
    master = MasterAccount(
        master_id="master_fund_01",
        reserve_cash=5000000.0,
        total_equity=5000000.0,
    )
    assert master.master_id == "master_fund_01"
    assert master.reserve_cash == 5000000.0
    assert len(master.strategies) == 0

def test_allocation_plan_model():
    transfer = CashTransfer(
        strategy_id="strat_01",
        transfer_amount=500000.0,
        reason="定期夏普比率加权调资",
    )
    plan = CapitalAllocationPlan(
        method=AllocationMethod.SHARPE_WEIGHTED,
        target_ratios={"strat_01": 0.6, "strat_02": 0.4},
        transfers=[transfer],
    )
    assert plan.method == AllocationMethod.SHARPE_WEIGHTED
    assert plan.transfers[0].transfer_amount == 500000.0

def test_consolidated_portfolio_model():
    cpos = ConsolidatedPosition(
        symbol="600519.SH",
        total_volume=1500,
        total_market_value=2700000.0,
        weight_in_master=0.27,
        contributing_strategies=["strat_01", "strat_02"],
    )
    portfolio = ConsolidatedPortfolio(
        master_id="master_fund_01",
        total_equity=10000000.0,
        total_cash=7300000.0,
        total_market_value=2700000.0,
        leverage_ratio=0.27,
        positions={"600519.SH": cpos},
        max_stock_concentration=0.27,
        top5_concentration=0.27,
        circuit_breaker_level=0,
        macro_alerts=["CRITICAL: 标的 600519.SH 穿透持仓占比 27.00% 超过宏观红线 15.00%"],
    )
    assert portfolio.max_stock_concentration == 0.27
    assert len(portfolio.macro_alerts) == 1
