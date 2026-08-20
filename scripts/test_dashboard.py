"""Test dashboard API endpoints."""
import urllib.request
import json
import sys

BASE = "http://127.0.0.1:8001"

def get(path):
    r = urllib.request.urlopen(f"{BASE}{path}")
    return json.loads(r.read().decode())

def post(path, data):
    req = urllib.request.Request(
        f"{BASE}{path}",
        data=json.dumps(data).encode(),
        headers={"Content-Type": "application/json"},
    )
    r = urllib.request.urlopen(req)
    return json.loads(r.read().decode())

# 1. Health
health = get("/api/qlib/health")
print(f"[OK] Health: {health['total_instruments']} instruments, {health['calendar_days']} calendar days")

# 2. Instruments with names
instruments = get("/api/qlib/instruments")
first = instruments["instruments"][0]
print(f"[OK] Instruments: {instruments['total']}, first={first['code']} ({first['name']})")

# 3. Strategies
strategies = get("/api/qlib/strategies")
print(f"[OK] Strategies: {len(strategies['strategies'])} available")

# 4. Backtest with benchmark + drawdown
bt = post("/api/qlib/backtest", {
    "strategy_type": "ma_trend",
    "code": "SH510300",
    "params": {"short_window": 5, "long_window": 20},
    "init_cash": 1000000,
})
print(f"[OK] Backtest: {bt['strategy']} on {bt['code_name']}")
print(f"     Annual: {bt['metrics']['annual_return']}, Sharpe: {bt['metrics']['sharpe_ratio']}")
print(f"     Benchmark return: {bt['metrics']['benchmark_return']}")
print(f"     Excess return: {bt['metrics']['excess_return']}")
print(f"     Benchmark points: {len(bt.get('benchmark_curve', []))}")
print(f"     Drawdown points: {len(bt.get('drawdown_curve', []))}")

# 5. Factor evaluation
fe = post("/api/qlib/factors/evaluate", {
    "code": "SH600000",
    "forward_period": 10,
})
print(f"[OK] Factors: {len(fe['factors'])} evaluated for {fe['code']}")
for f in fe["factors"]:
    print(f"     {f['name']:<15} IC={f['ic']:+.4f}  RankIC={f['rank_ic']:+.4f}")

# 6. OHLCV
ohlcv = get("/api/qlib/ohlcv/SH510300")
print(f"[OK] OHLCV: {len(ohlcv['data'])} bars for {ohlcv['code']}")

print("\n=== All API tests passed ===")
