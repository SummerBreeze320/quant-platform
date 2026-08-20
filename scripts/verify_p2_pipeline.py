"""P2: 离线投研闭环流水线 — 验证脚本

Tests all P2 components:
1. WalkForward fold splitting + validation
2. StrategyRegistry register + list + retrieve
3. StrategyEnsemble select + allocate + rebalance
4. ScoringFunction compute + rank + explain
5. SearchSpace sample + cardinality + reduced
6. ResearchAgent.run_orchestrated method
7. CriticAgent walk-forward integration
8. Workflow score_result method
"""
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"


def test_1_walk_forward():
    """Test 1: Walk-Forward fold splitting"""
    print("\n=== Test 1: WalkForward fold splitting ===")
    from src.research.walk_forward import WalkForwardValidator

    wf = WalkForwardValidator(n_folds=5)
    folds = wf.split("2020-01-01", "2024-12-31")

    assert len(folds) == 5, f"Expected 5 folds, got {len(folds)}"

    for f in folds:
        assert f.train_end < f.test_start, (
            f"Fold {f.fold_id}: train_end ({f.train_end}) "
            f"must be before test_start ({f.test_start})"
        )

    print(f"  {len(folds)} folds generated")
    for f in folds[:2]:
        print(f"  Fold {f.fold_id}: train→{f.train_end} valid→{f.valid_end} test→{f.test_end}")

    print(f"{PASS} WalkForward fold splitting ({len(folds)} folds, no look-ahead)")
    return True


def test_2_strategy_registry():
    """Test 2: StrategyRegistry"""
    print("\n=== Test 2: StrategyRegistry ===")
    from src.research.strategy_registry import StrategyRegistry
    import tempfile
    from pathlib import Path

    tmp = Path(tempfile.mktemp(suffix=".json"))
    reg = StrategyRegistry(registry_path=tmp)

    sid = reg.register(
        name="lgbm_ew",
        factor_set=["RESI5", "STD5", "RSQR5"],
        model_type="lightgbm",
        model_params={"n_estimators": 100},
        n_holdings=10,
        weight_scheme="equal_weight",
        rebalance_freq="W",
        market_states=["bull", "sideways"],
        annual_return=0.12,
        sharpe_ratio=0.8,
        max_drawdown=-0.15,
        score=0.35,
        approved=True,
        overfitting_score=0.25,
    )

    entry = reg.get(sid)
    assert entry is not None, "Strategy not found after register"
    assert entry.name == "lgbm_ew"
    assert entry.version == "v1"

    approved = reg.list_approved()
    assert len(approved) == 1

    by_market = reg.list_by_market_state("bull")
    assert len(by_market) == 1

    best = reg.best_for_market("bull")
    assert best.strategy_id == sid

    summary = reg.summary()
    assert summary["total"] == 1
    assert summary["approved"] == 1

    tmp.unlink(missing_ok=True)
    print(f"  Registered: {sid} (v1)")
    print(f"  Approved: {summary['approved']}, Best score: {summary['best_score']}")
    print(f"{PASS} StrategyRegistry (register/list/best_for_market/summary)")
    return True


def test_3_strategy_ensemble():
    """Test 3: StrategyEnsemble"""
    print("\n=== Test 3: StrategyEnsemble ===")
    from src.research.strategy_ensemble import StrategyEnsemble
    from src.research.strategy_registry import StrategyRegistry
    from src.agents.contracts import MarketState
    import tempfile
    from pathlib import Path

    tmp = Path(tempfile.mktemp(suffix=".json"))
    reg = StrategyRegistry(registry_path=tmp)

    for i in range(3):
        reg.register(
            name=f"strat_{i}",
            factor_set=["f1", "f2"],
            model_type="lightgbm",
            model_params={},
            n_holdings=10 + i * 5,
            weight_scheme="equal_weight",
            rebalance_freq="W",
            market_states=["bull", "sideways"],
            annual_return=0.10 + i * 0.02,
            sharpe_ratio=0.5 + i * 0.2,
            max_drawdown=-0.15,
            score=0.30 + i * 0.05,
            approved=True,
            overfitting_score=0.25,
        )

    ensemble = StrategyEnsemble(registry=reg)
    allocation = ensemble.rebalance(MarketState.BULL)

    assert len(allocation.strategy_weights) > 0, "No allocation"
    total_w = sum(allocation.strategy_weights.values())
    assert abs(total_w - 1.0) < 0.01, f"Weights sum to {total_w}, not 1.0"

    summary = ensemble.performance_summary()
    assert summary["total_rebalances"] == 1

    tmp.unlink(missing_ok=True)
    print(f"  {allocation.total_positions} strategies selected for {allocation.market_state}")
    print(f"  Weights: {list(allocation.strategy_weights.values())}")
    print(f"  Signal: {allocation.signal}")
    print(f"{PASS} StrategyEnsemble (select/allocate/rebalance)")
    return True


def test_4_scoring():
    """Test 4: ScoringFunction"""
    print("\n=== Test 4: ScoringFunction ===")
    from src.research.scoring import ScoringFunction, ScoringWeights
    from src.agents.contracts import BacktestMetrics, InvestmentObjective

    scorer = ScoringFunction()
    metrics = BacktestMetrics(
        annual_return=0.15,
        sharpe_ratio=0.8,
        max_drawdown=-0.12,
        turnover=0.3,
    )

    score = scorer.compute(metrics)
    assert -1.0 <= score <= 1.0, f"Score {score} out of range"

    explanation = scorer.explain(metrics)
    assert "components" in explanation
    assert "total_score" in explanation

    obj = InvestmentObjective(goal="max_return")
    score2 = scorer.compute(metrics, obj)

    results = {"exp1": metrics, "exp2": BacktestMetrics(annual_return=0.05)}
    ranked = scorer.rank(results, obj)
    assert ranked[0][0] == "exp1", "Best experiment should be ranked first"

    print(f"  Score (balanced): {score:.4f}")
    print(f"  Score (max_return): {score2:.4f}")
    print(f"  Rank: {ranked}")
    print(f"{PASS} ScoringFunction (compute/rank/explain)")
    return True


def test_5_search_space():
    """Test 5: SearchSpace"""
    print("\n=== Test 5: SearchSpace ===")
    from src.research.search_space import SearchSpace

    space = SearchSpace()
    config = space.sample(start_date="2020-01-01", end_date="2024-12-31")

    assert config.factor.max_factors > 0
    assert len(config.model.model_types) > 0
    assert len(config.portfolio.n_holdings_range) > 0
    assert len(config.market_states) > 0

    card = space.cardinality()
    assert card > 0

    reduced = space.reduced_space(max_configs=10)
    reduced_card = reduced.cardinality()

    print(f"  Full space cardinality: {card}")
    print(f"  Reduced space cardinality: {reduced_card}")
    print(f"  Sample: max_factors={config.factor.max_factors} models={config.model.model_types}")
    print(f"{PASS} SearchSpace (sample/cardinality/reduced)")
    return True


def test_6_research_agent_methods():
    """Test 6: ResearchAgent new methods exist"""
    print("\n=== Test 6: ResearchAgent new methods ===")
    from src.agents.research_agent import ResearchAgent

    agent = ResearchAgent()
    assert hasattr(agent, "run_orchestrated"), "Missing run_orchestrated"
    assert hasattr(agent, "run_closed_loop"), "Missing run_closed_loop"
    assert hasattr(agent, "run_rdagent_evolution"), "Missing run_rdagent_evolution"

    import inspect
    sig = inspect.signature(agent.run_orchestrated)
    assert "n_configs" in sig.parameters

    sig2 = inspect.signature(agent.run_closed_loop)
    assert "n_iterations" in sig2.parameters

    print(f"  run_orchestrated(n_configs={sig.parameters['n_configs'].default})")
    print(f"  run_closed_loop(n_iterations={sig2.parameters['n_iterations'].default})")
    print(f"{PASS} ResearchAgent methods (run_orchestrated/run_closed_loop)")
    return True


def test_7_critic_walk_forward():
    """Test 7: CriticAgent walk-forward integration"""
    print("\n=== Test 7: CriticAgent walk-forward integration ===")
    from src.agents.critic_agent import CriticAgent
    from src.agents.contracts import ReviewRequest, BacktestMetrics

    critic = CriticAgent()

    metrics = BacktestMetrics(
        annual_return=0.15,
        sharpe_ratio=1.5,
        max_drawdown=-0.10,
        ic_mean=0.03,
        icir=0.4,
        information_ratio=0.8,
    )

    req = ReviewRequest(
        experiment_id="test_wf",
        metrics=metrics,
        factor_names=["RESI5", "STD5"],
        backtest_period="2020-2024",
    )

    # Without walk-forward
    result1 = critic.execute(req)
    print(f"  Without WF: approved={result1.approved}")

    # With passing walk-forward
    from src.research.walk_forward import WalkForwardResult
    wf_pass = WalkForwardResult(
        n_folds=5,
        avg_sharpe=0.6,
        avg_max_drawdown=-0.08,
        return_consistency=0.8,
        passed=True,
    )
    result2 = critic.execute(req, walk_forward_result=wf_pass)
    wf_reasons = [r for r in result2.reasons if "Walk-Forward" in r]
    assert len(wf_reasons) > 0, "Walk-forward reason not in results"

    # With failing walk-forward
    wf_fail = WalkForwardResult(
        n_folds=5,
        avg_sharpe=-0.1,
        return_consistency=0.2,
        passed=False,
    )
    result3 = critic.execute(req, walk_forward_result=wf_fail)
    assert not result3.approved, "Should be rejected with failed walk-forward"

    print(f"  With WF pass: approved={result2.approved}")
    print(f"  With WF fail: approved={result3.approved}")
    print(f"{PASS} CriticAgent walk-forward integration (pass/fail)")
    return True


def test_8_workflow_scoring():
    """Test 8: Workflow score_result method"""
    print("\n=== Test 8: Workflow score_result ===")
    from src.research.workflow import ResearchWorkflow, WorkflowConfig, WorkflowResult

    wf = ResearchWorkflow(config=WorkflowConfig())
    assert hasattr(wf, "run_with_walk_forward"), "Missing run_with_walk_forward"
    assert hasattr(wf, "score_result"), "Missing score_result"

    # Create a mock result
    mock_result = WorkflowResult(config=wf.config)
    mock_result.report = {
        "backtest": {
            "annual_return": 0.12,
            "sharpe": 0.8,
            "max_drawdown": -0.15,
            "ic": 0.02,
            "icir": 0.35,
        }
    }
    mock_result.success = True

    score = wf.score_result(mock_result)
    assert -1.0 <= score <= 1.0

    print(f"  Mock result score: {score:.4f}")
    print(f"{PASS} Workflow score_result method")
    return True


def test_9_orchestrator_import():
    """Test 9: Orchestrator import + status"""
    print("\n=== Test 9: Orchestrator import + status ===")
    from src.agents.orchestrator import ResearchOrchestrator
    from src.research.search_space import SearchSpace

    space = SearchSpace().reduced_space(max_configs=10)
    orchestrator = ResearchOrchestrator(
        search_space=space,
        n_configs=1,
    )

    assert hasattr(orchestrator, "run"), "Missing run"
    assert hasattr(orchestrator, "evolve"), "Missing evolve"
    assert hasattr(orchestrator, "status"), "Missing status"

    status = orchestrator.status()
    assert "search_space_cardinality" in status
    assert "registry" in status
    assert "ensemble" in status

    print(f"  Cardinality: {status['search_space_cardinality']}")
    print(f"  Registry: {status['registry']}")
    print(f"  Ensemble: {status['ensemble']}")
    print(f"{PASS} Orchestrator (import/status/ready)")
    return True


def main():
    print("=" * 60)
    print("P2: 离线投研闭环流水线 — 验证")
    print("=" * 60)

    tests = [
        ("WalkForward fold splitting", test_1_walk_forward),
        ("StrategyRegistry", test_2_strategy_registry),
        ("StrategyEnsemble", test_3_strategy_ensemble),
        ("ScoringFunction", test_4_scoring),
        ("SearchSpace", test_5_search_space),
        ("ResearchAgent methods", test_6_research_agent_methods),
        ("CriticAgent walk-forward", test_7_critic_walk_forward),
        ("Workflow score_result", test_8_workflow_scoring),
        ("Orchestrator import", test_9_orchestrator_import),
    ]

    passed = 0
    failed = 0

    for name, fn in tests:
        try:
            if fn():
                passed += 1
            else:
                failed += 1
                print(f"{FAIL} {name}")
        except Exception as e:
            failed += 1
            import traceback
            print(f"{FAIL} {name}")
            print(f"  Error: {e}")
            traceback.print_exc()

    print("\n" + "=" * 60)
    print(f"Results: {passed}/{passed + failed} passed")
    print("=" * 60)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
