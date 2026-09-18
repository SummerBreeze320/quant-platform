import pytest
from src.pms.models import AllocationMethod
from src.pms.allocator import CapitalAllocator

def test_equal_weight_allocation():
    allocator = CapitalAllocator()
    ratios = allocator.allocate_equal_weight(["strat_A", "strat_B", "strat_C", "strat_D"])
    assert len(ratios) == 4
    for r in ratios.values():
        assert pytest.approx(r, 0.001) == 0.25

def test_fixed_budget_allocation():
    allocator = CapitalAllocator()
    base = {"strat_A": 0.5, "strat_B": 0.3, "strat_C": 0.2}
    ratios = allocator.allocate_fixed_budget(["strat_A", "strat_B", "strat_C"], base)
    assert ratios["strat_A"] == 0.5
    assert ratios["strat_B"] == 0.3
    assert ratios["strat_C"] == 0.2

def test_risk_parity_allocation():
    allocator = CapitalAllocator()
    # Volatilities: strat_A is low risk (0.10), strat_B is medium (0.20), strat_C is high (0.40)
    # Inv vols: 10, 5, 2.5 -> sum = 17.5
    # strat_A ratio = 10 / 17.5 = 0.5714
    vols = {"strat_A": 0.10, "strat_B": 0.20, "strat_C": 0.40}
    ratios = allocator.allocate_risk_parity(vols)
    assert ratios["strat_A"] > ratios["strat_B"] > ratios["strat_C"]
    assert pytest.approx(sum(ratios.values()), 0.0001) == 1.0
    assert pytest.approx(ratios["strat_A"], 0.001) == 10 / 17.5

def test_sharpe_weighted_allocation():
    allocator = CapitalAllocator()
    # Sharpes: strat_A = 2.0, strat_B = 1.0, strat_C = -0.5 (floored to 0.01)
    # sum = 2.0 + 1.0 + 0.01 = 3.01
    sharpes = {"strat_A": 2.0, "strat_B": 1.0, "strat_C": -0.5}
    ratios = allocator.allocate_sharpe_weighted(sharpes)
    assert ratios["strat_A"] > ratios["strat_B"] > ratios["strat_C"]
    assert pytest.approx(sum(ratios.values()), 0.0001) == 1.0
    assert pytest.approx(ratios["strat_A"], 0.001) == 2.0 / 3.01
    assert ratios["strat_C"] > 0.0

def test_generate_plan():
    allocator = CapitalAllocator()
    master_equity = 10_000_000.0
    current_equities = {
        "strat_A": 3_000_000.0,
        "strat_B": 2_000_000.0,
    }
    # Investable ratio 0.95 -> 9,500,000 to allocate. Equal weight -> 4,750,000 each.
    # strat_A delta = 4,750,000 - 3,000,000 = +1,750,000 (deposit)
    # strat_B delta = 4,750,000 - 2,000,000 = +2,750,000 (deposit)
    plan = allocator.generate_plan(
        master_equity=master_equity,
        current_equities=current_equities,
        method=AllocationMethod.EQUAL_WEIGHT,
        investable_ratio=0.95,
    )
    assert plan.method == AllocationMethod.EQUAL_WEIGHT
    assert plan.target_ratios["strat_A"] == 0.5
    assert plan.target_ratios["strat_B"] == 0.5
    transfer_map = {t.strategy_id: t.transfer_amount for t in plan.transfers}
    assert transfer_map["strat_A"] == 1_750_000.0
    assert transfer_map["strat_B"] == 2_750_000.0
