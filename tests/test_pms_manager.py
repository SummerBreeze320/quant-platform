import pytest
from src.pms.models import StrategyType, AllocationMethod
from src.pms.manager import PortfolioManager
from src.execution_engine.gateway.paper_broker import PaperBroker

def test_portfolio_manager_initialization():
    mgr = PortfolioManager(master_id="master_fund", initial_reserve=10_000_000.0)
    assert mgr.master.master_id == "master_fund"
    assert mgr.master.reserve_cash == 10_000_000.0
    assert mgr.master.total_equity == 10_000_000.0

def test_register_strategy():
    broker = PaperBroker()
    mgr = PortfolioManager(master_id="master_fund", initial_reserve=10_000_000.0, broker=broker)
    strat = mgr.register_strategy(
        strategy_id="strat_01",
        name="沪深300指增",
        strategy_type=StrategyType.INDEX_ENHANCEMENT,
        initial_budget=3_000_000.0
    )
    assert strat.strategy_id == "strat_01"
    assert strat.current_cash == 3_000_000.0
    assert strat.total_equity == 3_000_000.0
    assert mgr.master.reserve_cash == 7_000_000.0
    assert mgr.master.total_equity == 10_000_000.0
    # Broker account should also be created
    acc = broker.get_account("strat_01")
    assert acc.available_cash == 3_000_000.0

def test_register_strategy_insufficient_reserve():
    mgr = PortfolioManager(master_id="master_fund", initial_reserve=1_000_000.0)
    with pytest.raises(ValueError, match="预备金不足"):
        mgr.register_strategy(
            strategy_id="strat_fail",
            name="超额预算策略",
            strategy_type=StrategyType.ALPHA,
            initial_budget=2_000_000.0
        )

def test_transfer_cash_deposit_and_withdraw():
    broker = PaperBroker()
    mgr = PortfolioManager(master_id="master_fund", initial_reserve=10_000_000.0, broker=broker)
    mgr.register_strategy("strat_01", "Alpha", StrategyType.ALPHA, 2_000_000.0)
    # Master reserve is now 8,000,000, strat_01 cash is 2,000,000

    # 1. Deposit additional 1,000,000 to strategy
    transfer = mgr.transfer_cash("strat_01", 1_000_000.0, reason="追加注资")
    assert transfer.transfer_amount == 1_000_000.0
    assert mgr.master.reserve_cash == 7_000_000.0
    assert mgr.master.strategies["strat_01"].current_cash == 3_000_000.0
    assert broker.get_account("strat_01").available_cash == 3_000_000.0

    # 2. Withdraw 500,000 from strategy back to master
    transfer2 = mgr.transfer_cash("strat_01", -500_000.0, reason="回笼利润")
    assert transfer2.transfer_amount == -500_000.0
    assert mgr.master.reserve_cash == 7_500_000.0
    assert mgr.master.strategies["strat_01"].current_cash == 2_500_000.0
    assert broker.get_account("strat_01").available_cash == 2_500_000.0

    # 3. Withdraw exceeding cash raises ValueError
    with pytest.raises(ValueError, match="策略现金不足"):
        mgr.transfer_cash("strat_01", -3_000_000.0)

def test_apply_allocation_plan():
    broker = PaperBroker()
    mgr = PortfolioManager(master_id="master_fund", initial_reserve=10_000_000.0, broker=broker)
    mgr.register_strategy("strat_A", "Strategy A", StrategyType.ALPHA, 3_000_000.0)
    mgr.register_strategy("strat_B", "Strategy B", StrategyType.MOMENTUM, 2_000_000.0)
    # Reserve is 5,000,000

    plan = mgr.allocator.generate_plan(
        master_equity=mgr.master.total_equity,
        current_equities=mgr.get_strategy_equities(),
        method=AllocationMethod.EQUAL_WEIGHT,
        investable_ratio=0.80 # 8,000,000 total -> 4,000,000 each
    )
    transfers = mgr.apply_allocation_plan(plan)
    assert len(transfers) == 2
    assert mgr.master.strategies["strat_A"].current_cash == 4_000_000.0
    assert mgr.master.strategies["strat_B"].current_cash == 4_000_000.0
    assert mgr.master.reserve_cash == 2_000_000.0

def test_strategy_performance_summary():
    mgr = PortfolioManager(master_id="master_fund", initial_reserve=10_000_000.0)
    strat = mgr.register_strategy("strat_A", "Strategy A", StrategyType.ALPHA, 2_000_000.0)
    strat.total_equity = 2_200_000.0 # +10% return
    mgr.refresh_total_equity()

    perfs = mgr.get_performance_summary()
    assert len(perfs) == 1
    p = perfs[0]
    assert p.strategy_id == "strat_A"
    assert pytest.approx(p.cumulative_return, 0.001) == 0.10
    assert pytest.approx(p.contribution_to_master, 0.001) == 2_200_000.0 / 10_200_000.0
