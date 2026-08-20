"""P4: 日频交易生产链路 — 验证脚本

Tests:
1. TradingEngine import + status
2. DailyScheduler import + schedule info + trading day check
3. PerformanceReview review_day + decay detection
4. PortfolioScheduler generate_rebalance (direct mode)
5. RiskAgent live_risk_check
6. TradingLive router new endpoints
7. run_trading_engine.py script exists
"""
import sys
import os
import tempfile
import shutil
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"


def test_1_trading_engine():
    """Test 1: TradingEngine import + status"""
    print("\n=== Test 1: TradingEngine import ===")
    from src.execution.trading_engine import TradingEngine, TradingPhase, TradingDayRecord

    engine = TradingEngine(init_capital=1_000_000, topk=5)
    assert engine.init_capital == 1_000_000
    assert engine.current_nav == 1_000_000
    assert engine.phase == TradingPhase.IDLE

    status = engine.get_status()
    assert status["phase"] == "idle"
    assert status["nav"] == 1_000_000
    assert status["n_trading_days"] == 0

    history = engine.get_history()
    assert len(history) == 0

    print(f"  Phase: {status['phase']}")
    print(f"  NAV: {status['nav']:.0f}")
    print(f"  PnL: {status['pnl_pct']:.2%}")
    print(f"{PASS} TradingEngine import + status")
    return True


def test_2_scheduler():
    """Test 2: DailyScheduler import + schedule info"""
    print("\n=== Test 2: DailyScheduler ===")
    from src.execution.scheduler import DailyScheduler, ScheduleConfig

    config = ScheduleConfig()
    assert config.pre_market_time == "09:00"
    assert config.trading_time == "15:00"
    assert config.review_time == "15:30"
    assert config.timezone == "Asia/Shanghai"

    scheduler = DailyScheduler(config)
    assert scheduler.config.trading_time == "15:00"

    info = scheduler.get_schedule_info()
    assert info["pre_market_time"] == "09:00"
    assert info["trading_time"] == "15:00"
    assert info["review_time"] == "15:30"
    assert info["running"] == False

    # Trading day check
    import datetime
    weekend = datetime.date(2024, 6, 1)  # Saturday
    weekday = datetime.date(2024, 6, 3)  # Monday
    assert not scheduler.is_trading_day(weekend)
    assert scheduler.is_trading_day(weekday)

    # Manual trigger without engine
    result = scheduler.run_now()
    assert "error" in result

    print(f"  Schedule: pre_market={info['pre_market_time']} trading={info['trading_time']} review={info['review_time']}")
    print(f"  Weekend is trading day: {scheduler.is_trading_day(weekend)}")
    print(f"{PASS} DailyScheduler import + schedule info + trading day check")
    return True


def test_3_performance_review():
    """Test 3: PerformanceReview review_day + decay detection"""
    print("\n=== Test 3: PerformanceReview ===")
    from src.execution.review import PerformanceReview

    tmp = Path(tempfile.mkdtemp()) / "reviews"
    review = PerformanceReview(review_dir=tmp)

    # Review a single day
    result = review.review_day(
        trade_date="2024-06-03",
        nav=1_050_000,
        init_capital=1_000_000,
        positions={"SH600000": {"weight": 0.5, "price": 10.0}},
        orders=[
            {"side": "buy", "delta_weight": 0.5, "stock": "SH600000"},
        ],
        signal={"signal_date": "2024-06-03", "n_selected": 5},
        risk={"market_state": "bull", "position_ratio": 1.0, "warnings": []},
    )

    assert result["trade_date"] == "2024-06-03"
    assert result["nav"] == 1_050_000
    assert result["pnl_pct"] == 0.05
    assert result["n_positions"] == 1
    assert result["n_orders"] == 1
    assert result["decay_detected"] == False

    # Summary
    summary = review.get_summary()
    assert summary["total_days"] == 1
    assert summary["recent_days"] == 1
    assert summary["avg_pnl"] == 0.05

    # Persist and reload
    review2 = PerformanceReview(review_dir=tmp)
    assert len(review2._history) == 1

    shutil.rmtree(tmp)
    print(f"  PnL: {result['pnl_pct']:.2%}")
    print(f"  Decay: {result['decay_detected']}")
    print(f"  Summary: {summary['total_days']} day(s)")
    print(f"{PASS} PerformanceReview review_day + persistence")
    return True


def test_4_portfolio_scheduler_direct():
    """Test 4: PortfolioScheduler generate_rebalance (direct mode)"""
    print("\n=== Test 4: PortfolioScheduler direct mode ===")
    from src.execution.portfolio_scheduler import PortfolioScheduler

    # Create without broker (direct mode)
    scheduler = PortfolioScheduler(init_capital=1_000_000)
    assert scheduler.broker is None
    assert scheduler.init_capital == 1_000_000

    current = {
        "SH600000": {"weight": 0.3, "price": 10.0},
        "SH600001": {"weight": 0.2, "price": 20.0},
    }
    target = {
        "SH600000": 0.4,
        "SH600001": 0.1,
        "SH600002": 0.5,
    }

    result = scheduler.generate_rebalance(current, target, 1_000_000)
    assert "orders" in result
    assert "summary" in result
    assert len(result["orders"]) == 3

    # Check order details
    orders = result["orders"]
    buy_orders = [o for o in orders if o["side"] == "buy"]
    sell_orders = [o for o in orders if o["side"] == "sell"]
    assert len(buy_orders) == 2  # SH600000 (0.3→0.4) and SH600002 (0→0.5)
    assert len(sell_orders) == 1  # SH600001 (0.2→0.1)

    print(f"  Orders: {result['summary']['n_orders']}")
    print(f"  Buy: {result['summary']['n_buy']}, Sell: {result['summary']['n_sell']}")
    print(f"  Turnover: {result['summary']['total_turnover']:.4f}")
    print(f"{PASS} PortfolioScheduler generate_rebalance (direct mode)")
    return True


def test_5_risk_agent_live_check():
    """Test 5: RiskAgent live_risk_check"""
    print("\n=== Test 5: RiskAgent live_risk_check ===")
    from src.agents.risk_agent import RiskAgent

    agent = RiskAgent()
    agent.initialize()

    target_weights = {"SH600000": 0.3, "SH600001": 0.3, "SH600002": 0.4}

    result = agent.live_risk_check(target_weights)

    assert "approved" in result
    assert "position_ratio" in result
    assert "market_state" in result
    assert "risk_score" in result
    assert "warnings" in result
    assert "adjusted_weights" in result

    # Adjusted weights should sum to position_ratio
    total_w = sum(result["adjusted_weights"].values())
    assert abs(total_w - result["position_ratio"]) < 0.01 or total_w == 0

    print(f"  Approved: {result['approved']}")
    print(f"  Market: {result['market_state']}")
    print(f"  Position: {result['position_ratio']:.0%}")
    print(f"  Adjusted weights: {result['adjusted_weights']}")
    print(f"{PASS} RiskAgent live_risk_check")
    return True


def test_6_trading_live_endpoints():
    """Test 6: TradingLive router new endpoints"""
    print("\n=== Test 6: TradingLive endpoints ===")
    from src.dashboard.api.routers.trading_live import router

    routes = [r.path for r in router.routes]
    paths = [r for r in routes]

    assert "/api/trading/live/engine/status" in paths, f"Missing engine/status in {paths}"
    assert "/api/trading/live/engine/history" in paths, "Missing engine/history"
    assert "/api/trading/live/engine/schedule" in paths, "Missing engine/schedule"
    assert "/api/trading/live/engine/run" in paths, "Missing engine/run"
    assert "/api/trading/live/reviews" in paths, "Missing reviews"

    print(f"  Endpoints: {len(routes)}")
    for p in paths:
        if "engine" in p or "review" in p:
            print(f"    {p}")
    print(f"{PASS} TradingLive P4 endpoints (5 new)")
    return True


def test_7_run_script():
    """Test 7: run_trading_engine.py script exists"""
    print("\n=== Test 7: run_trading_engine.py ===")
    script_path = Path("scripts/run_trading_engine.py")
    assert script_path.exists(), f"Script not found: {script_path}"

    content = script_path.read_text(encoding="utf-8")
    assert "--mode" in content
    assert "auto" in content
    assert "manual" in content
    assert "TradingEngine" in content
    assert "DailyScheduler" in content

    print(f"  Script: {script_path}")
    print(f"  Modes: auto, manual, status")
    print(f"{PASS} run_trading_engine.py script exists")
    return True


def main():
    print("=" * 60)
    print("P4: 日频交易生产链路 — 验证")
    print("=" * 60)

    tests = [
        ("TradingEngine import", test_1_trading_engine),
        ("DailyScheduler", test_2_scheduler),
        ("PerformanceReview", test_3_performance_review),
        ("PortfolioScheduler direct mode", test_4_portfolio_scheduler_direct),
        ("RiskAgent live_risk_check", test_5_risk_agent_live_check),
        ("TradingLive endpoints", test_6_trading_live_endpoints),
        ("run_trading_engine.py", test_7_run_script),
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
