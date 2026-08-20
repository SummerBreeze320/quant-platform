"""P5验证脚本 — RiskAgent + BacktestAgent + ExperimentAgent 三大Agent增强验证

验证项:
1. contracts.py 数据契约完整性
2. RiskAgent 恐慌指标 + 直接输入模式 + 仓位决策依据
3. BacktestAgent 交易成本 + 综合评分(胜率+IC+成本惩罚)
4. ExperimentAgent 代码生成 + 实验对比表 + best_config
5. 三Agent联动 — 从风险评估到回测到实验对比的完整链路
"""
import sys
import os

# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

PASS = 0
FAIL = 0
RESULTS = []


def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        RESULTS.append(f"  [PASS] {name}")
    else:
        FAIL += 1
        RESULTS.append(f"  [FAIL] {name} — {detail}")


def test_contracts():
    """1. 验证数据契约完整性"""
    print("\n=== 1. Contracts 数据契约验证 ===")

    from src.agents.contracts import (
        RiskAssessmentRequest,
        RiskAssessment,
        BacktestMetrics,
        BacktestResult,
        ExperimentPlan,
        ExperimentResult,
        ExperimentComparison,
        MarketState,
    )

    # RiskAssessmentRequest 有直接输入字段
    req = RiskAssessmentRequest(
        volatility=0.25,
        trend_signal=-0.03,
        panic_index=0.6,
        volume_ratio=1.8,
    )
    check("RiskAssessmentRequest 有 volatility 字段", req.volatility == 0.25)
    check("RiskAssessmentRequest 有 trend_signal 字段", req.trend_signal == -0.03)
    check("RiskAssessmentRequest 有 panic_index 字段", req.panic_index == 0.6)
    check("RiskAssessmentRequest 有 volume_ratio 字段", req.volume_ratio == 1.8)

    # RiskAssessment 有恐慌指标 + 仓位依据
    ra = RiskAssessment(
        market_state=MarketState.BEAR,
        position_ratio=0.2,
        panic_index=0.7,
        position_reason="熊市环境 → 仓位20%",
    )
    check("RiskAssessment 有 panic_index 字段", ra.panic_index == 0.7)
    check("RiskAssessment 有 position_reason 字段", len(ra.position_reason) > 0)

    # BacktestMetrics 有全部7个核心指标 + 交易成本
    m = BacktestMetrics(
        annual_return=0.15,
        sharpe_ratio=1.2,
        max_drawdown=-0.08,
        win_rate=0.55,
        turnover=0.3,
        ic_mean=0.05,
        transaction_cost=0.002,
        net_return_after_cost=0.148,
    )
    check("BacktestMetrics 有 annual_return", m.annual_return == 0.15)
    check("BacktestMetrics 有 sharpe_ratio", m.sharpe_ratio == 1.2)
    check("BacktestMetrics 有 max_drawdown", m.max_drawdown == -0.08)
    check("BacktestMetrics 有 win_rate", m.win_rate == 0.55)
    check("BacktestMetrics 有 turnover", m.turnover == 0.3)
    check("BacktestMetrics 有 ic_mean", m.ic_mean == 0.05)
    check("BacktestMetrics 有 transaction_cost", m.transaction_cost == 0.002)
    check("BacktestMetrics 有 net_return_after_cost", m.net_return_after_cost == 0.148)
    check("BacktestMetrics 有 intraday 指标", hasattr(m, "intraday_entry_accuracy"))

    # ExperimentComparison 有 best_config
    comp = ExperimentComparison(
        comparisons=[{"experiment_id": "exp1", "score": 0.3}],
        best_experiment_id="exp1",
        best_score=0.3,
        best_config={"model": "lightgbm", "score": 0.3},
        ranking=["exp1"],
    )
    check("ExperimentComparison 有 best_config", comp.best_config is not None)
    check("ExperimentComparison 有 ranking", len(comp.ranking) == 1)


def test_risk_agent():
    """2. 验证 RiskAgent 增强"""
    print("\n=== 2. RiskAgent 恐慌指标 + 直接输入验证 ===")

    from src.agents.risk_agent import RiskAgent, POSITION_RATIO_MAP
    from src.agents.contracts import RiskAssessmentRequest, MarketState

    agent = RiskAgent()

    # 2a. 直接输入模式 — 牛市 (低波动 + 上升趋势)
    bull_req = RiskAssessmentRequest(
        volatility=0.10,
        trend_signal=0.05,
        volume_ratio=1.0,
    )
    bull_result = agent.execute(bull_req)
    check("RiskAgent 牛市识别 (低波动+上升)", bull_result.market_state == MarketState.BULL)
    check("RiskAgent 牛市仓位=100%", bull_result.position_ratio == 1.0)
    check("RiskAgent 牛市有仓位依据", len(bull_result.position_reason) > 0)

    # 2b. 直接输入模式 — 熊市 (高波动 + 下跌趋势 + 高恐慌)
    bear_req = RiskAssessmentRequest(
        volatility=0.40,
        trend_signal=-0.08,
        volume_ratio=2.0,
    )
    bear_result = agent.execute(bear_req)
    check("RiskAgent 熊市识别 (高波动+下跌)", bear_result.market_state in (MarketState.BEAR, MarketState.HIGH_VOL))
    check("RiskAgent 熊市仓位=20%", bear_result.position_ratio == 0.2)

    # 2c. 直接输入模式 — 震荡市
    sideways_req = RiskAssessmentRequest(
        volatility=0.18,
        trend_signal=0.0,
        volume_ratio=1.0,
    )
    sideways_result = agent.execute(sideways_req)
    check("RiskAgent 震荡市仓位=50%", sideways_result.position_ratio == 0.5)

    # 2d. 恐慌指标计算
    panic = agent._calc_panic_index({
        "volatility": 0.35,
        "trend": -0.06,
        "volume_ratio": 2.0,
    })
    check("RiskAgent 恐慌指标 > 0.3 (高波动+下跌)", panic > 0.3)

    panic_low = agent._calc_panic_index({
        "volatility": 0.10,
        "trend": 0.03,
        "volume_ratio": 0.8,
    })
    check("RiskAgent 恐慌指标 < 0.1 (低波动+上升)", panic_low < 0.1)

    # 2e. live_risk_check 接口
    check_result = agent.live_risk_check(
        target_weights={"SH600519": 0.3, "SH601318": 0.2},
        direct_inputs={"volatility": 0.15, "trend_signal": 0.02},
    )
    check("RiskAgent live_risk_check 返回 approved", "approved" in check_result)
    check("RiskAgent live_risk_check 返回 position_ratio", "position_ratio" in check_result)
    check("RiskAgent live_risk_check 返回 panic_index", "panic_index" in check_result)
    check("RiskAgent live_risk_check 返回 adjusted_weights", "adjusted_weights" in check_result)
    check("RiskAgent live_risk_check 返回 position_reason", "position_reason" in check_result)


def test_backtest_agent():
    """3. 验证 BacktestAgent 增强"""
    print("\n=== 3. BacktestAgent 交易成本 + 综合评分验证 ===")

    from src.agents.backtest_agent import (
        BacktestAgent,
        SCORING_WEIGHTS,
        COMMISSION_RATE,
        STAMP_TAX_RATE,
        SLIPPAGE_RATE,
    )
    from src.agents.contracts import BacktestMetrics

    agent = BacktestAgent()

    # 3a. 评分权重包含胜率 + IC + 交易成本
    check("评分权重含 annual_return", "annual_return" in SCORING_WEIGHTS)
    check("评分权重含 sharpe_ratio", "sharpe_ratio" in SCORING_WEIGHTS)
    check("评分权重含 max_drawdown", "max_drawdown" in SCORING_WEIGHTS)
    check("评分权重含 win_rate", "win_rate" in SCORING_WEIGHTS)
    check("评分权重含 ic_mean", "ic_mean" in SCORING_WEIGHTS)
    check("评分权重含 transaction_cost", "transaction_cost" in SCORING_WEIGHTS)
    check("评分权重含 turnover (惩罚)", "turnover" in SCORING_WEIGHTS)

    # 3b. 交易成本参数
    check("佣金率=万三", COMMISSION_RATE == 0.0003)
    check("印花税率=千一", STAMP_TAX_RATE == 0.001)
    check("滑点率=万五", SLIPPAGE_RATE == 0.0005)

    # 3c. 交易成本计算
    metrics = BacktestMetrics(
        annual_return=0.10,
        turnover=2.0,
    )
    metrics = agent._add_transaction_cost(metrics, init_cash=1_000_000)
    check("交易成本 > 0", metrics.transaction_cost > 0)
    check("净收益 < 年化收益 (扣成本)", metrics.net_return_after_cost < metrics.annual_return)
    check("净收益 = 年化 - 成本", abs(metrics.net_return_after_cost - (0.10 - metrics.transaction_cost)) < 0.001)

    # 3d. 综合评分 — 胜率和IC纳入
    good_metrics = BacktestMetrics(
        annual_return=0.15,
        sharpe_ratio=1.5,
        max_drawdown=-0.08,
        turnover=0.3,
        win_rate=0.55,
        ic_mean=0.06,
        transaction_cost=0.002,
    )
    good_score = agent._compute_score(good_metrics)

    bad_metrics = BacktestMetrics(
        annual_return=0.15,
        sharpe_ratio=1.5,
        max_drawdown=-0.08,
        turnover=0.3,
        win_rate=0.35,
        ic_mean=0.01,
        transaction_cost=0.002,
    )
    bad_score = agent._compute_score(bad_metrics)

    check("高胜率+高IC策略评分更高", good_score > bad_score)

    # 3e. 交易成本惩罚
    high_cost_metrics = BacktestMetrics(
        annual_return=0.15,
        sharpe_ratio=1.5,
        max_drawdown=-0.08,
        turnover=0.3,
        win_rate=0.55,
        ic_mean=0.06,
        transaction_cost=0.01,
    )
    high_cost_score = agent._compute_score(high_cost_metrics)
    check("高交易成本导致评分降低", high_cost_score < good_score)


def test_experiment_agent():
    """4. 验证 ExperimentAgent 增强"""
    print("\n=== 4. ExperimentAgent 代码生成 + 实验对比验证 ===")

    from src.agents.experiment_agent import (
        ExperimentAgent,
        EXPERIMENT_TEMPLATES,
    )
    from src.agents.contracts import (
        ExperimentPlan,
        FactorSet,
        ModelConfig,
        PortfolioConfig,
        InvestmentObjective,
    )

    agent = ExperimentAgent()

    # 4a. 实验模板存在
    check("实验模板含 momentum_5d", "momentum_5d" in EXPERIMENT_TEMPLATES)
    check("实验模板含 momentum_10d", "momentum_10d" in EXPERIMENT_TEMPLATES)
    check("实验模板含 add_news_factor", "add_news_factor" in EXPERIMENT_TEMPLATES)
    check("实验模板含 xgboost_model", "xgboost_model" in EXPERIMENT_TEMPLATES)
    check("实验模板含 transformer_model", "transformer_model" in EXPERIMENT_TEMPLATES)

    # 4b. 自动生成实验提案
    plans = agent.generate_experiments(
        base_factors=["RESI5", "STD5", "RSQR5"],
        model_type="lightgbm",
        n_variants=3,
    )
    check("生成3组实验提案", len(plans) == 3)
    check("实验ID格式正确", all(p.experiment_id.startswith("exp_") for p in plans))
    check("每组实验有因子集", all(p.factor_set is not None for p in plans))
    check("每组实验有模型配置", all(p.model_setting is not None for p in plans))

    # 4c. 代码生成 — 模板模式
    plan = ExperimentPlan(
        experiment_id="test_code_gen",
        factor_set=FactorSet(factors=["RESI5", "STD5", "RSQR5"]),
        model_setting=ModelConfig(model_type="lightgbm", params={"num_leaves": 64}),
        portfolio_config=PortfolioConfig(),
        objective=InvestmentObjective(),
    )
    code = agent.generate_code(plan)
    check("代码生成非空", len(code) > 100)
    check("代码含 ensure_qlib", "ensure_qlib" in code)
    check("代码含 create_model", "create_model" in code)
    check("代码含 backtest", "backtest" in code)

    # 4d. 实验对比表构建
    from src.agents.contracts import (
        ExperimentResult,
        ExperimentStatus,
        BacktestResult,
        BacktestMetrics,
        ReviewResult,
    )

    mock_results = [
        ExperimentResult(
            experiment_id="exp_001",
            status=ExperimentStatus.APPROVED,
            score=0.35,
            factor_set=FactorSet(factors=["momentum_5d"]),
            model_setting=ModelConfig(model_type="lightgbm", params={"n_estimators": 100}),
            backtest_result=BacktestResult(
                experiment_id="exp_001",
                metrics=BacktestMetrics(
                    annual_return=0.12, sharpe_ratio=1.1, max_drawdown=-0.07,
                    win_rate=0.52, turnover=0.4, ic_mean=0.04,
                    transaction_cost=0.003, net_return_after_cost=0.117,
                ),
                score=0.35,
                status="completed",
            ),
            review_result=ReviewResult(experiment_id="exp_001", approved=True),
        ),
        ExperimentResult(
            experiment_id="exp_002",
            status=ExperimentStatus.REJECTED,
            score=0.20,
            factor_set=FactorSet(factors=["momentum_10d"]),
            model_setting=ModelConfig(model_type="lightgbm", params={"n_estimators": 200}),
            backtest_result=BacktestResult(
                experiment_id="exp_002",
                metrics=BacktestMetrics(
                    annual_return=0.08, sharpe_ratio=0.6, max_drawdown=-0.12,
                    win_rate=0.42, turnover=0.6, ic_mean=0.01,
                    transaction_cost=0.005, net_return_after_cost=0.075,
                ),
                score=0.20,
                status="completed",
            ),
            review_result=ReviewResult(experiment_id="exp_002", approved=False),
        ),
    ]

    comp = agent._build_comparison(mock_results)
    check("对比表含2组实验", len(comp.comparisons) == 2)
    check("最佳实验ID正确", comp.best_experiment_id == "exp_001")
    check("最佳评分正确", comp.best_score == 0.35)
    check("best_config非空", comp.best_config is not None)
    check("best_config含model", comp.best_config.get("model") == "lightgbm")
    check("best_config含score", comp.best_config.get("score") == 0.35)
    check("ranking含2个实验", len(comp.ranking) == 2)
    check("ranking第一名是最佳", comp.ranking[0] == "exp_001")

    # 4e. 对比表指标字段完整
    first_comp = comp.comparisons[0]
    check("对比表含 annual_return", "annual_return" in first_comp)
    check("对比表含 sharpe", "sharpe" in first_comp)
    check("对比表含 max_drawdown", "max_drawdown" in first_comp)
    check("对比表含 win_rate", "win_rate" in first_comp)
    check("对比表含 turnover", "turnover" in first_comp)
    check("对比表含 ic_mean", "ic_mean" in first_comp)
    check("对比表含 transaction_cost", "transaction_cost" in first_comp)
    check("对比表含 net_return", "net_return" in first_comp)
    check("对比表含 approved", "approved" in first_comp)


def test_integration():
    """5. 验证三Agent联动"""
    print("\n=== 5. 三Agent联动验证 ===")

    from src.agents.risk_agent import RiskAgent
    from src.agents.backtest_agent import BacktestAgent, SCORING_WEIGHTS
    from src.agents.experiment_agent import ExperimentAgent, EXPERIMENT_TEMPLATES
    from src.agents.contracts import (
        RiskAssessmentRequest,
        BacktestMetrics,
    )

    # 联动流程: RiskAgent评估市场 → 影响仓位 → BacktestAgent计算指标 → ExperimentAgent对比

    # Step 1: RiskAgent 判断今天应该多少仓位
    risk_agent = RiskAgent()
    risk_result = risk_agent.execute(RiskAssessmentRequest(
        volatility=0.22,
        trend_signal=-0.02,
        volume_ratio=1.3,
    ))
    print(f"  RiskAgent: 市场状态={risk_result.market_state.value}, "
          f"仓位={risk_result.position_ratio:.0%}, "
          f"恐慌={risk_result.panic_index:.2f}")

    # Step 2: BacktestAgent 评分 (模拟指标)
    bt_agent = BacktestAgent()
    metrics = BacktestMetrics(
        annual_return=0.10,
        sharpe_ratio=0.8,
        max_drawdown=-0.10,
        turnover=0.5,
        win_rate=0.50,
        ic_mean=0.03,
    )
    metrics = bt_agent._add_transaction_cost(metrics)
    score = bt_agent._compute_score(metrics)
    print(f"  BacktestAgent: 年化={metrics.annual_return:.1%}, "
          f"夏普={metrics.sharpe_ratio:.2f}, "
          f"成本={metrics.transaction_cost:.4%}, "
          f"净收益={metrics.net_return_after_cost:.2%}, "
          f"评分={score:.4f}")

    # Step 3: ExperimentAgent 生成实验方案
    exp_agent = ExperimentAgent()
    plans = exp_agent.generate_experiments(
        base_factors=["RESI5", "STD5", "RSQR5"],
        n_variants=3,
    )
    print(f"  ExperimentAgent: 生成{len(plans)}组实验方案")
    for p in plans:
        code = exp_agent.generate_code(p)
        print(f"    {p.experiment_id}: {len(code)} chars code generated")

    check("三Agent联动 — RiskAgent输出仓位", risk_result.position_ratio > 0)
    check("三Agent联动 — BacktestAgent输出评分", score != 0)
    check("三Agent联动 — ExperimentAgent生成方案", len(plans) > 0)
    check("三Agent联动 — 代码生成成功", len(exp_agent.generate_code(plans[0])) > 100)


def main():
    print("=" * 60)
    print("P5 Agent增强验证 — RiskAgent + BacktestAgent + ExperimentAgent")
    print("=" * 60)

    tests = [
        ("Contracts", test_contracts),
        ("RiskAgent", test_risk_agent),
        ("BacktestAgent", test_backtest_agent),
        ("ExperimentAgent", test_experiment_agent),
        ("Integration", test_integration),
    ]

    for name, func in tests:
        try:
            func()
        except Exception as e:
            global FAIL
            FAIL += 1
            RESULTS.append(f"  [ERROR] {name} 异常: {e}")

    print("\n" + "=" * 60)
    print("验证结果:")
    for r in RESULTS:
        print(r)
    print(f"\n总计: {PASS} PASS / {FAIL} FAIL")
    print("=" * 60)

    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
