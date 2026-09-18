def test_pms_package_exports():
    import src.pms as pms
    assert hasattr(pms, "StrategyType")
    assert hasattr(pms, "AllocationMethod")
    assert hasattr(pms, "StrategyAccount")
    assert hasattr(pms, "MasterAccount")
    assert hasattr(pms, "CashTransfer")
    assert hasattr(pms, "CapitalAllocationPlan")
    assert hasattr(pms, "ConsolidatedPosition")
    assert hasattr(pms, "ConsolidatedPortfolio")
    assert hasattr(pms, "StrategyPerformance")
    assert hasattr(pms, "CapitalAllocator")
    assert hasattr(pms, "PortfolioManager")
    assert hasattr(pms, "PortfolioAggregator")
