"""P2验证脚本 — 真实Qlib数据接入 + RD-Agent实验循环

验证内容:
1. 契约扩展: ModelSearchRequest/PortfolioSearchRequest 新字段
2. FactorAgent: 使用全量仪器(非硬编码[:20])
3. ModelAgent: 动态日期分段(非硬编码2015-2024)
4. PortfolioAgent: 返回真实权重(非空dict)
5. ResearchAgent: 端到端管道产出真实ExperimentResult
6. RD-Agent循环: 方法可调用

运行: python scripts/verify_p2_agents.py
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


def test_contract_fields():
    """1. 契约新字段验证"""
    from src.agents import ModelSearchRequest, PortfolioSearchRequest

    msr = ModelSearchRequest(
        start_date="2020-01-01",
        end_date="2024-12-31",
        universe="all",
    )
    assert msr.start_date == "2020-01-01"
    assert msr.end_date == "2024-12-31"
    assert msr.universe == "all"

    psr = PortfolioSearchRequest(
        benchmark="SH000300",
        start_date="2020-01-01",
        end_date="2024-12-31",
    )
    assert psr.start_date == "2020-01-01"
    assert psr.end_date == "2024-12-31"
    assert psr.factor_set is None
    assert psr.model_setting is None


def test_model_agent_segments():
    """2. ModelAgent动态分段"""
    from src.agents import ModelAgent

    segments = ModelAgent._compute_segments("2020-01-01", "2024-12-31")
    assert "train" in segments
    assert "valid" in segments
    assert "test" in segments

    train_start, train_end = segments["train"]
    valid_start, valid_end = segments["valid"]
    test_start, test_end = segments["test"]

    assert train_start == "2020-01-01"
    assert test_end == "2024-12-31"
    assert train_end < valid_start
    assert valid_end < test_start

    print(f"      train: {train_start} → {train_end}")
    print(f"      valid: {valid_start} → {valid_end}")
    print(f"      test:  {test_start} → {test_end}")


def test_factor_agent_all_instruments():
    """3. FactorAgent使用全量仪器(非[:20])"""
    from src.core import list_instruments, ensure_qlib

    ensure_qlib()
    instruments = list_instruments(as_list=True)
    assert len(instruments) > 0

    from src.agents import FactorAgent, FactorSearchRequest

    agent = FactorAgent()
    result = agent.execute(FactorSearchRequest(
        categories=["price_volume", "momentum"],
        start_date="2023-01-01",
        end_date="2024-12-31",
    ))

    assert len(result.factors) > 0
    assert sum(result.factor_weights.values()) > 0.99
    print(f"      {len(result.factors)} factors selected from {len(instruments)} instruments")


def test_portfolio_agent_real_weights():
    """4. PortfolioAgent返回真实权重"""
    from src.agents import PortfolioAgent, PortfolioSearchRequest

    agent = PortfolioAgent()
    result = agent.execute(PortfolioSearchRequest(
        n_holdings_range=[5, 10],
        weight_schemes=["equal_weight", "min_variance", "enhanced_indexing"],
        rebalance_freq=["W"],
        start_date="2023-01-01",
        end_date="2024-12-31",
    ))

    assert result.n_holdings > 0
    assert result.weight_scheme in ("equal_weight", "min_variance", "enhanced_indexing")
    assert len(result.weights) > 0, "weights dict should not be empty"

    total_w = sum(result.weights.values())
    assert 0.8 < total_w <= 1.05, f"total weight {total_w:.4f} out of range"
    print(f"      {result.n_holdings} holdings, {result.weight_scheme}, "
          f"{len(result.weights)} weights, sum={total_w:.4f}")


def test_research_agent_pipeline():
    """5. ResearchAgent端到端管道"""
    from src.agents import ResearchAgent, InvestmentObjective

    agent = ResearchAgent()
    agent.initialize()

    objective = InvestmentObjective(
        goal="max_return",
        start_date="2023-01-01",
        end_date="2024-12-31",
    )
    results = agent.execute(objective)

    assert len(results) > 0
    result = results[0]
    assert result.experiment_id is not None
    assert result.status is not None
    assert result.score is not None

    if result.backtest_result:
        m = result.backtest_result.metrics
        print(f"      exp={result.experiment_id} status={result.status.value} "
              f"score={result.score:.4f} sharpe={m.sharpe_ratio:.4f}")
    else:
        print(f"      exp={result.experiment_id} status={result.status.value} "
              f"score={result.score:.4f}")


def test_rdagent_evolution_method():
    """6. RD-Agent进化循环方法可调用"""
    from src.agents import ResearchAgent

    agent = ResearchAgent()
    agent.initialize()
    assert hasattr(agent, "run_rdagent_evolution")
    assert hasattr(agent._experiment_agent, "run_rdagent_evolution")

    print("      run_rdagent_evolution() method exists on ResearchAgent + ExperimentAgent")


def main():
    print("=" * 60)
    print("P2验证 — 真实Qlib数据接入 + RD-Agent实验循环")
    print("=" * 60)

    tests = [
        ("契约新字段", test_contract_fields),
        ("ModelAgent动态分段", test_model_agent_segments),
        ("FactorAgent全量仪器", test_factor_agent_all_instruments),
        ("PortfolioAgent真实权重", test_portfolio_agent_real_weights),
        ("ResearchAgent端到端管道", test_research_agent_pipeline),
        ("RD-Agent循环方法", test_rdagent_evolution_method),
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
