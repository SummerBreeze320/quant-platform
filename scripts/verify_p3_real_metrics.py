"""P3验证脚本 — 真实回测指标产出

依赖: lightgbm + xgboost 已安装 (torch可选)
运行: python scripts/verify_p3_real_metrics.py
"""
import os
os.environ['MLFLOW_ALLOW_FILE_STORE'] = 'true'
os.environ['GIT_PYTHON_REFRESH'] = 'quiet'

import sys
import time
from pathlib import Path
from datetime import datetime

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


def test_model_agent_real():
    """1. ModelAgent真实模型训练(LightGBM)"""
    from src.agents import ModelAgent, ModelSearchRequest, FactorSet

    agent = ModelAgent()
    fs = FactorSet(factors=["kmid", "rocp_5", "rocp_20", "ma_5", "std_20", "rsi_14"])
    result = agent.execute(ModelSearchRequest(
        model_types=["lightgbm"],
        factor_set=fs,
        start_date="2020-01-01",
        end_date="2024-12-31",
        universe="all",
    ))

    assert result.model_type == "lightgbm"
    assert result.cv_score >= 0
    print(f"      model={result.model_type} cv_score={result.cv_score:.4f} "
          f"ic_score={result.ic_score:.4f}")


def test_backtest_agent_real():
    """2. BacktestAgent真实回测"""
    from src.agents import BacktestAgent, BacktestRequest

    agent = BacktestAgent()
    result = agent.execute(BacktestRequest(
        experiment_id="p3_backtest",
        factor_names=["kmid", "rocp_5", "rocp_20", "ma_5", "std_20"],
        model_type="lightgbm",
        n_holdings=10,
        weight_scheme="enhanced_indexing",
        start_date="2020-01-01",
        end_date="2024-12-31",
        benchmark="SH000300",
    ))

    assert result.status in ("completed", "error")
    if result.status == "completed":
        m = result.metrics
        print(f"      annual_return={m.annual_return:.4f} "
              f"sharpe={m.sharpe_ratio:.4f} "
              f"max_dd={m.max_drawdown:.4f} "
              f"score={result.score:.4f}")
        assert m.annual_return != 0 or m.sharpe_ratio != 0, "metrics all zero"
    else:
        print(f"      backtest error: {result.error}")
        raise AssertionError(f"Backtest failed: {result.error}")


def test_research_agent_full():
    """3. ResearchAgent完整管道真实结果"""
    from src.agents import ResearchAgent, InvestmentObjective

    agent = ResearchAgent()
    agent.initialize()

    objective = InvestmentObjective(
        goal="max_return",
        start_date="2020-01-01",
        end_date="2024-12-31",
        init_cash=1000000,
    )
    results = agent.execute(objective)

    assert len(results) > 0
    r = results[0]
    print(f"      exp_id={r.experiment_id}")
    print(f"      status={r.status.value} score={r.score:.4f}")

    if r.factor_set:
        print(f"      factors: {len(r.factor_set.factors)}")
    if r.model_setting:
        print(f"      model: {r.model_setting.model_type} "
              f"(cv={r.model_setting.cv_score:.4f})")
    if r.portfolio_config:
        print(f"      portfolio: {r.portfolio_config.n_holdings} holdings, "
              f"{len(r.portfolio_config.weights)} weights")
    if r.risk_assessment:
        print(f"      risk: state={r.risk_assessment.market_state.value} "
              f"position={r.risk_assessment.position_ratio:.0%}")
    if r.backtest_result:
        m = r.backtest_result.metrics
        print(f"      backtest: annual={m.annual_return:.4f} "
              f"sharpe={m.sharpe_ratio:.4f} "
              f"max_dd={m.max_drawdown:.4f}")
    if r.review_result:
        print(f"      review: approved={r.review_result.approved} "
              f"overfit={r.review_result.overfitting_score:.2f}")

    assert r.experiment_id is not None


def test_critic_agent_real():
    """4. CriticAgent真实评审"""
    from src.agents import CriticAgent, ReviewRequest, BacktestMetrics

    agent = CriticAgent()
    result = agent.execute(ReviewRequest(
        experiment_id="p3_critic",
        metrics=BacktestMetrics(
            annual_return=0.15,
            sharpe_ratio=1.8,
            max_drawdown=0.12,
            turnover=0.45,
            ic_mean=0.05,
            icir=0.8,
        ),
        factor_names=["kmid", "rocp_5", "ma_5", "std_20", "rsi_14"],
        backtest_period="2020-01-01 to 2024-12-31",
    ))

    assert result.experiment_id == "p3_critic"
    assert isinstance(result.approved, bool)
    assert 0 <= result.overfitting_score <= 1
    print(f"      approved={result.approved} "
          f"overfit={result.overfitting_score:.2f} "
          f"general={result.generalization_score:.2f}")


def main():
    print("=" * 60)
    print("P3验证 — 真实回测指标产出 (torch+xgboost已安装)")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    tests = [
        ("ModelAgent真实训练", test_model_agent_real),
        ("BacktestAgent真实回测", test_backtest_agent_real),
        ("ResearchAgent完整管道", test_research_agent_full),
        ("CriticAgent真实评审", test_critic_agent_real),
    ]

    print()
    for name, func in tests:
        t0 = time.time()
        check(name, func)
        print(f"      (耗时 {time.time()-t0:.1f}s)")

    print()
    print("=" * 60)
    print(f"结果: {PASS} passed, {FAIL} failed, total {PASS + FAIL}")
    print("=" * 60)
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
