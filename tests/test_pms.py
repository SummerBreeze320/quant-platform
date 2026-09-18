import pytest
from src.pms.models import StrategyType, AllocationMethod
from src.pms.manager import PortfolioManager
from src.pms.allocator import CapitalAllocator
from src.pms.aggregator import PortfolioAggregator
from src.execution_engine.gateway.paper_broker import PaperBroker
from src.execution_engine.models import Order, OrderDirection, OrderType
from src.risk_engine.circuit_breaker import CircuitBreakerManager

def test_full_pms_lifecycle_and_lookthrough_risk():
    """
    全生命周期综合场景验证:
    1. 母账户注资 10,000,000 预备金;
    2. 分配注册两支策略:
       - CSI300 Alpha: 初始预算 3,000,000;
       - Sector Rotation: 初始预算 2,000,000;
    3. 运行夏普比率动态配资生成划转计划并执行;
    4. 两个策略同时买入 600519.SH (贵州茅台);
    5. 全局穿透聚合器合并持仓市值并计算权重;
    6. 验证集中度超标 (15%) 报警触发;
    7. 模拟大幅回撤触发母级全局硬熔断 (RED_HALT)。
    """
    broker = PaperBroker()
    cb = CircuitBreakerManager(warn_drawdown=0.02, restrict_buy_drawdown=0.03, halt_drawdown=0.04)
    mgr = PortfolioManager(master_id="inst_fund_01", initial_reserve=10_000_000.0, broker=broker)
    aggregator = PortfolioAggregator(circuit_breaker=cb, max_stock_concentration=0.15, max_leverage_ratio=1.0)

    # 1. 注册策略
    s1 = mgr.register_strategy("csi300_alpha", "沪深300Alpha", StrategyType.ALPHA, 3_000_000.0)
    s2 = mgr.register_strategy("sector_rot", "行业轮动策略", StrategyType.SECTOR_ROTATION, 2_000_000.0)
    assert mgr.master.reserve_cash == 5_000_000.0
    assert mgr.master.total_equity == 10_000_000.0

    # 2. 动态调资 (夏普倾斜): csi300_alpha 表现更优 (Sharpe 2.0 vs 1.0)
    # Target ratios: 2.0/3.0 = 66.7%, 1.0/3.0 = 33.3%
    plan = mgr.allocator.generate_plan(
        master_equity=mgr.master.total_equity,
        current_equities=mgr.get_strategy_equities(),
        method=AllocationMethod.SHARPE_WEIGHTED,
        investable_ratio=0.80, # 8,000,000
        sharpes={"csi300_alpha": 2.0, "sector_rot": 1.0}
    )
    mgr.apply_allocation_plan(plan)
    assert mgr.master.strategies["csi300_alpha"].current_cash > 4_000_000.0

    # 3. 策略交易买入重叠股票 600519.SH
    # csi300_alpha 买入 800 股 @ 1800 -> 约 1,440,000
    order1 = Order(
        order_id="ord_pms_1",
        account_id="csi300_alpha",
        symbol="600519.SH",
        direction=OrderDirection.BUY,
        price=1800.0,
        volume=800
    )
    t1 = broker.submit_order(order1)
    assert t1 is not None

    # sector_rot 买入 600 股 @ 1800 -> 约 1,080,000
    order2 = Order(
        order_id="ord_pms_2",
        account_id="sector_rot",
        symbol="600519.SH",
        direction=OrderDirection.BUY,
        price=1800.0,
        volume=600
    )
    t2 = broker.submit_order(order2)
    assert t2 is not None

    # 同步柜台持仓
    mgr.sync_from_broker()

    # 4. 穿透合并持仓检查
    c_portfolio = aggregator.aggregate(mgr.master)
    assert "600519.SH" in c_portfolio.positions
    pos_combined = c_portfolio.positions["600519.SH"]
    assert pos_combined.total_volume == 1400
    assert set(pos_combined.contributing_strategies) == {"csi300_alpha", "sector_rot"}

    # 5. 集中度超标检测: 1400 * 1800 = 2,520,000，占 10M 母账户约 25.2% > 15%
    assert pos_combined.weight_in_master > 0.15
    assert any("超过宏观红线 15.00%" in alert for alert in c_portfolio.macro_alerts)

    # 6. 模拟母账户总资产回撤触发全局熔断
    # 将母账户储备现金扣减 600,000 (总资产回撤 > 5% 突破 4% halt_drawdown)
    mgr.master.reserve_cash -= 600_000.0
    mgr.refresh_total_equity()
    c_portfolio2 = aggregator.aggregate(mgr.master)
    assert c_portfolio2.circuit_breaker_level == 3
    assert any("全局熔断" in alert for alert in c_portfolio2.macro_alerts)
