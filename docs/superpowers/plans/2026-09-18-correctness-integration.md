# Correctness Integration Implementation Plan

> **For agentic workers:** Use executing-plans for inline execution. Steps use checkbox syntax for tracking.

**Goal:** Preserve Qlib history and make risk, execution, PMS and market APIs operate on shared application state.

**Architecture:** App-scoped dependency injection for existing services; merge-and-replace for Qlib features; stored indicator snapshots for API reads.

**Tech Stack:** Python 3.11, FastAPI, Pydantic, NumPy, pandas, pytest.

**Spec:** docs/superpowers/specs/2026-09-18-correctness-integration-design.md

## Global Constraints

- Preserve REST paths and request formats.
- Preserve the existing uncommitted market API work.
- Keep state process-local; no claim of durability or multi-worker consistency.
- Atomicity applies to each feature file, not the entire dataset.
- Existing feature calendars can only be extended at the end.

## Task 1: Qlib history

Files: src/data_pipeline/qlib_dumper.py; tests/test_qlib_incremental.py.

- [x] Add failing cases for append, correction, gaps, duplicate dates, absent fields, invalid dates, calendar rewrite and interrupted replacement. Read raw float32 binaries to independently check expected values.
- [x] Run the new tests with the Python 3.11 interpreter.
- [x] Merge existing feature ranges and incoming date indices; atomically replace files. Validate the calendar and existing binary before mutation.
- [x] Run both incremental tests and the existing Qlib reader test.

Expected merge example:

```python
# calendar = [day1, day2, day3]
# old = [start_index=0, 10, 11]
# incoming = [(day2, 12), (day3, 13)]
np.testing.assert_array_equal(actual, [0, 10, 12, 13])
```

## Task 2: Shared application services

Files: src/service/runtime.py (new); src/service/app.py; risk_router.py; execution_router.py; pms_router.py; market_router.py; src/execution_engine/coordinator.py; src/market_feed/router.py; tests/test_service_integration.py (new).

Interfaces: ServiceRuntime owns broker, circuit_breaker, risk_checker, alert_manager, risk_monitor, coordinator, pms_manager, aggregator and market state. get_runtime(request: Request) -> ServiceRuntime reads request.app.state.runtime. create_app() creates an independent runtime.

- [x] Write cross-API tests using actual PaperBroker and risk rules. Include app isolation and a default coordinator whose risk checker observes its circuit breaker.
- [x] Run tests against existing code and verify observable mismatches.
- [x] Build shared dependencies with explicit constructor injection and a common threading.RLock for route mutations; replace route globals.
- [x] Register the default market strategy through PMS, preserving its 5 million budget within the 10 million mother account.
- [x] Test API blacklist / circuit breaker rejection and successful shared-account trades; run existing risk, execution, PMS and market tests.

Example assertion:

```python
client.post('/api/v1/pms/strategies', json={
    'strategy_id': 'shared', 'name': 'shared', 'initial_budget': 2000000,
})
account = client.get('/api/v1/execution/account', params={'account_id': 'shared'}).json()
assert account['available_cash'] == 2000000
```

## Task 3: Market API semantics

Files: src/market_feed/signal_engine.py; src/service/market_runtime.py (new); src/service/routers/market_router.py; tests/test_market_router.py; tests/test_service_integration.py.

Interfaces: SignalEngine.get_indicator_snapshot(symbol: str) -> dict exposes the last completed calculation. MarketRuntime owns bus, signal engine/router, replay engine and WebSocket connection manager, using the shared broker/checker.

- [x] Add failing numerical API tests (prices 10 and 12 imply mean 11), unchanged snapshots on repeat GET, independent app state, and replay interval validation.
- [x] Store indicator results before signal cooldown handling and expose read-only serialized snapshots.
- [x] Use TickReplayEngine for replay, validate speed >= 0, and schedule WebSocket sends on each connection's event loop.
- [x] Run market tests including the existing WebSocket scenario.

## Task 4: Verification

- [x] Run the full pytest suite in Python 3.11, using a unique writable temporary directory.
- [x] Inspect diff for state duplication, unintended changes and stale comments.
- [x] Record test results, remaining warnings and scope limitations in the final report.

## Verification Report

**Date:** 2026-09-18
**Python:** 3.11.13 (conda env: python311)
**Pytest:** 9.1.1

### Test Results

| Suite | Tests | Result |
|-------|-------|--------|
| test_qlib_incremental.py | 8 | All passed |
| test_qlib_dumper.py | 1 | Passed |
| test_service_integration.py | 17 | All passed |
| test_market_router.py | 1 | Passed |
| **Full suite** | **129** | **All passed** |

### Remaining Warnings

None. Both previously reported warnings have been fixed:

1. **CVXPY solution accuracy** — Added OSQP solver options (`max_iter=20000`, `eps_abs=1e-6`, `eps_rel=1e-6`, `polishing=True`) and wrapped solve calls in `warnings.catch_warnings` since the code already handles `optimal_inaccurate` status.
2. **LightGBM `eval_set` deprecation** — Migrated to the LightGBM 4.x API: `eval_X`/`eval_y` instead of `eval_set`.

### Scope Limitations

- State is process-local; no durability or multi-worker consistency.
- Atomicity applies per feature file, not across the full dataset.
- Real training/prediction/backtest pipeline, QMT, distributed consumer recovery, strategy inheritance, real-time revaluation, persistence and authentication are out of scope for this phase.
