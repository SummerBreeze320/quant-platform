"""P1 Agent骨架验证脚本

验证所有Agent模块可正常导入、实例化、获取状态信息。
运行: python scripts/verify_agents.py
"""
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

PASS = 0
FAIL = 0


def check(name: str, func):
    global PASS, FAIL
    try:
        func()
        print(f"  [PASS] {name}")
        PASS += 1
    except Exception as e:
        print(f"  [FAIL] {name}: {e}")
        FAIL += 1


def test_imports():
    """1. 包级导入"""
    from src.agents import (
        BaseAgent, AgentState,
        ResearchAgent, FactorAgent, ModelAgent, PortfolioAgent,
        RiskAgent, BacktestAgent, CriticAgent, ExperimentAgent,
        MarketStateRecognizer,
        InvestmentObjective, FactorSet, ModelConfig, PortfolioConfig,
        RiskAssessment, ExperimentResult, MarketState,
    )
    assert BaseAgent is not None
    assert ResearchAgent is not None


def test_contracts():
    """2. Contracts 实例化"""
    from src.agents import (
        InvestmentObjective, FactorSet, ModelConfig,
        PortfolioConfig, RiskAssessment, ExperimentResult, MarketState,
    )
    obj = InvestmentObjective(goal="max_return")
    assert obj.universe == "A_share"

    fs = FactorSet(factors=["MA5", "RSI20"])
    assert len(fs.factors) == 2

    ra = RiskAssessment(market_state=MarketState.BULL, position_ratio=1.0)
    assert ra.risk_score == 0.5


def test_agent_instantiation():
    """3. Agent实例化 + 状态"""
    from src.agents import (
        ResearchAgent, FactorAgent, ModelAgent, PortfolioAgent,
        RiskAgent, BacktestAgent, CriticAgent, ExperimentAgent,
    )
    agents = [
        ("ResearchAgent", ResearchAgent),
        ("FactorAgent", FactorAgent),
        ("ModelAgent", ModelAgent),
        ("PortfolioAgent", PortfolioAgent),
        ("RiskAgent", RiskAgent),
        ("BacktestAgent", BacktestAgent),
        ("CriticAgent", CriticAgent),
        ("ExperimentAgent", ExperimentAgent),
    ]
    for name, cls in agents:
        a = cls()
        info = a.get_info()
        assert info["name"] == name
        assert info["status"] == "idle"
        assert info["run_count"] == 0


def test_risk_agent_fallback():
    """4. RiskAgent fallback (无Qlib数据时)"""
    from src.agents import RiskAgent, RiskAssessmentRequest, RiskAssessment
    agent = RiskAgent()
    agent.initialize()
    req = RiskAssessmentRequest()
    result = agent.execute(req)
    assert isinstance(result, RiskAssessment)
    assert 0 < result.position_ratio <= 1.0
    assert 0 <= result.risk_score <= 1.0


def test_critic_agent():
    """5. CriticAgent 评审逻辑"""
    from src.agents import CriticAgent, ReviewRequest, BacktestMetrics
    agent = CriticAgent()
    req = ReviewRequest(
        experiment_id="test_001",
        metrics=BacktestMetrics(
            annual_return=0.25,
            sharpe_ratio=2.5,
            max_drawdown=0.15,
            turnover=0.40,
        ),
        factor_names=["MA5", "RSI20", "VOL20"],
        backtest_period="2020-01-01 to 2024-12-31",
    )
    result = agent.execute(req)
    assert result.experiment_id == "test_001"
    assert 0 <= result.overfitting_score <= 1
    assert 0 <= result.generalization_score <= 1
    assert isinstance(result.approved, bool)
    assert isinstance(result.reasons, list)


def test_unified_risk_integration():
    """6. UnifiedRiskController + RiskAssessment 集成"""
    from src.execution.unified_risk import UnifiedRiskController
    from src.agents import RiskAssessment, MarketState

    ctrl = UnifiedRiskController()
    original_exposure = ctrl.config.max_total_exposure

    assessment = RiskAssessment(
        market_state=MarketState.BEAR,
        position_ratio=0.2,
        risk_score=0.8,
        warnings=["熊市环境: 建议仓位降至20%"],
    )
    adjustments = ctrl.apply_risk_assessment(assessment)
    assert len(adjustments) > 0
    assert ctrl.config.max_total_exposure <= original_exposure
    assert ctrl.config.max_single_weight <= 0.08
    alerts = ctrl.get_alerts()
    assert len(alerts) > 0


def test_research_agent_workflow():
    """7. ResearchAgent 初始化 + sub-agent注册"""
    from src.agents import ResearchAgent
    agent = ResearchAgent()
    agent.initialize()
    assert agent._factor_agent is not None
    assert agent._model_agent is not None
    assert agent._portfolio_agent is not None
    assert agent._risk_agent is not None
    assert agent._experiment_agent is not None


def main():
    print("=" * 60)
    print("P1 Agent骨架验证")
    print("=" * 60)

    tests = [
        ("包级导入", test_imports),
        ("Contracts实例化", test_contracts),
        ("Agent实例化+状态", test_agent_instantiation),
        ("RiskAgent fallback", test_risk_agent_fallback),
        ("CriticAgent评审", test_critic_agent),
        ("UnifiedRisk集成", test_unified_risk_integration),
        ("ResearchAgent工作流", test_research_agent_workflow),
    ]

    print()
    for name, func in tests:
        check(name, func)

    print()
    print("=" * 60)
    print(f"结果: {PASS} passed, {FAIL} failed, total {PASS + FAIL}")
    print("=" * 60)
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
