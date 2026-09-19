from datetime import datetime
import pytest

from src.market_feed.models import MarketTick, SignalDirection
from src.market_feed.strategies import (
    DynamicGridStrategy,
    IntradayMomentumBreakoutStrategy,
    RealtimeStrategyCoordinator,
)
from src.market_feed.router import SignalRouter
from src.execution_engine.gateway.paper_broker import PaperBroker
from src.risk_engine.pre_trade import PreTradeRiskChecker
from src.risk_engine.circuit_breaker import CircuitBreakerManager


def _make_tick(symbol: str, price: float, volume: int = 1000, bids=None, asks=None) -> MarketTick:
    b_vols = bids or [500, 400, 300, 200, 100]
    a_vols = asks or [100, 200, 300, 400, 500]
    return MarketTick(
        symbol=symbol,
        last_price=price,
        volume=volume,
        turnover=price * volume,
        bid_prices=[round(price - 0.01 * (i + 1), 2) for i in range(5)],
        bid_volumes=b_vols,
        ask_prices=[round(price + 0.01 * (i + 1), 2) for i in range(5)],
        ask_volumes=a_vols,
        timestamp=datetime.now().isoformat(),
    )


def test_dynamic_grid_strategy_down_and_up():
    strat = DynamicGridStrategy(
        strategy_id="grid_test_01",
        symbol="600000.SH",
        base_price=10.0,
        grid_num_levels=3,
        grid_step_pct=0.01,
        order_volume_per_grid=100,
        max_position_volume=500,
    )

    # 1. Price drops to 9.89 (Level 1 buy at 9.90)
    tick1 = _make_tick("600000.SH", 9.89)
    sig1 = strat.on_tick(tick1)
    assert sig1 is not None
    assert sig1.direction == SignalDirection.BUY
    assert sig1.volume == 100
    assert strat.current_position == 100
    assert strat.buy_levels_filled[0] is True

    # 2. Price drops to 9.79 (Level 2 buy at 9.80)
    tick2 = _make_tick("600000.SH", 9.79)
    sig2 = strat.on_tick(tick2)
    assert sig2 is not None
    assert sig2.direction == SignalDirection.BUY
    assert strat.current_position == 200
    assert strat.buy_levels_filled[1] is True

    # 3. Price rebounds to 9.91 (Rebound above level 2 target 9.90 -> sell level 2)
    tick3 = _make_tick("600000.SH", 9.91)
    sig3 = strat.on_tick(tick3)
    assert sig3 is not None
    assert sig3.direction == SignalDirection.SELL
    assert sig3.volume == 100
    assert strat.current_position == 100
    assert strat.buy_levels_filled[1] is False


def test_dynamic_grid_max_position_limit():
    strat = DynamicGridStrategy(
        strategy_id="grid_test_cap",
        symbol="000001.SZ",
        base_price=10.0,
        grid_num_levels=3,
        grid_step_pct=0.01,
        order_volume_per_grid=100,
        max_position_volume=100,  # Max allowed is only 1 grid lot
    )

    # First buy fills 100 shares
    t1 = _make_tick("000001.SZ", 9.89)
    s1 = strat.on_tick(t1)
    assert s1 is not None
    assert strat.current_position == 100

    # Second drop hits level 2 (9.80), but max position 100 reached -> rejected
    t2 = _make_tick("000001.SZ", 9.79)
    s2 = strat.on_tick(t2)
    assert s2 is None
    assert strat.current_position == 100


def test_dynamic_grid_stop_loss():
    strat = DynamicGridStrategy(
        strategy_id="grid_test_sl",
        symbol="600036.SH",
        base_price=10.0,
        grid_num_levels=3,
        grid_step_pct=0.01,
        order_volume_per_grid=100,
        stop_loss_pct=0.05,  # Stop loss at 9.50
    )

    # Buy at level 1
    strat.on_tick(_make_tick("600036.SH", 9.89))
    assert strat.current_position == 100

    # Gap down to 9.40 (below 9.50)
    tick_sl = _make_tick("600036.SH", 9.40)
    sig_sl = strat.on_tick(tick_sl)
    assert sig_sl is not None
    assert sig_sl.direction == SignalDirection.SELL
    assert sig_sl.volume == 100
    assert "止损" in sig_sl.reason
    assert strat.current_position == 0
    assert strat.stop_loss_triggered is True


def test_momentum_breakout_strategy_entry_and_trailing_stop():
    strat = IntradayMomentumBreakoutStrategy(
        strategy_id="mom_strat_01",
        symbol="600519.SH",
        ofi_threshold=0.30,
        trailing_stop_pct=0.01,  # 1% trailing stop
        take_profit_pct=0.05,
        order_volume=100,
    )

    # Warm up indicators with baseline prices
    for i in range(12):
        strat.on_tick(_make_tick("600519.SH", 100.0, volume=1000 + i * 10))

    # Breakout tick: price surges to 103.0 with heavy buy pressure (OFI > 0.3)
    breakout_tick = _make_tick("600519.SH", 103.0, bids=[10000, 8000, 5000, 3000, 1000], asks=[100, 100, 100, 100, 100])
    sig_entry = strat.on_tick(breakout_tick)

    assert sig_entry is not None
    assert sig_entry.direction == SignalDirection.BUY
    assert strat.in_position is True
    assert strat.peak_price == 103.0

    # Price rises further to 105.0 -> peak updates
    strat.on_tick(_make_tick("600519.SH", 105.0))
    assert strat.peak_price == 105.0

    # Price pulls back to 103.5 (< 105.0 * 0.99 = 103.95) -> Trailing stop triggered
    sig_exit = strat.on_tick(_make_tick("600519.SH", 103.5))
    assert sig_exit is not None
    assert sig_exit.direction == SignalDirection.SELL
    assert "追踪止损" in sig_exit.reason
    assert strat.in_position is False


def test_realtime_strategy_coordinator_routing():
    broker = PaperBroker()
    broker.create_account("grid_exec_01", initial_cash=1_000_000.0)

    cb = CircuitBreakerManager()
    risk_checker = PreTradeRiskChecker(cb_manager=cb, max_order_amount=200_000.0)
    router = SignalRouter(broker=broker, risk_checker=risk_checker)

    # Register strategy config in router for auto-execution
    router.register_strategy(strategy_id="grid_exec_01", auto_execute=True, default_volume=100)

    coordinator = RealtimeStrategyCoordinator(router=router)
    grid_strat = DynamicGridStrategy(
        strategy_id="grid_exec_01",
        symbol="601318.SH",
        base_price=50.0,
        grid_num_levels=3,
        grid_step_pct=0.01,
    )
    coordinator.register_strategy(grid_strat)

    # Send tick triggering level 1 BUY
    tick = _make_tick("601318.SH", 49.40)
    signals = coordinator.on_tick(tick)

    assert len(signals) == 1
    assert signals[0].direction == SignalDirection.BUY

    # Verify order was routed and filled by broker
    acc = broker.get_account("grid_exec_01")
    assert "601318.SH" in acc.positions
    assert acc.positions["601318.SH"].total_volume == 100
