import pytest
from starlette.testclient import TestClient
from src.service.app import app
from src.risk_engine import (
    PreTradeRiskChecker,
    PostTradeRiskMonitor,
    CircuitBreakerManager,
    RiskAlertManager,
    BaseRiskRule,
    BlacklistRule,
    MaxOrderAmountRule,
    MaxStockWeightRule,
    AdvLiquidityRule,
    CashSufficiencyRule,
    CircuitBreakerRule,
    FeishuWebhookAdapter,
    DingTalkWebhookAdapter,
    OrderRiskRequest,
    OrderRiskResult,
    PortfolioRiskRequest,
    PortfolioRiskResult,
    CircuitBreakerState,
    RiskAlertEvent,
    RiskAction,
    CircuitBreakerLevel,
    AlertLevel,
)

@pytest.fixture
def client():
    return TestClient(app)

def test_models_and_enums():
    assert RiskAction.PASS == "PASS"
    assert CircuitBreakerLevel.RED_HALT == 3
    assert AlertLevel.CRITICAL == "CRITICAL"

    req = OrderRiskRequest(
        account_id="acc_01",
        symbol="600519.SH",
        direction="BUY",
        price=1800.0,
        volume=100,
        total_equity=1000000.0,
        available_cash=500000.0
    )
    assert req.symbol == "600519.SH"

def test_rules_suite():
    # 1. BlacklistRule
    bl_rule = BlacklistRule(blacklist={"000001.SZ"})
    req_pass = OrderRiskRequest(symbol="600519.SH", direction="BUY", price=100.0, volume=100, total_equity=1e6, available_cash=1e5)
    action, _ = bl_rule.check(req_pass, {})
    assert action == RiskAction.PASS

    req_reject = OrderRiskRequest(symbol="000001.SZ", direction="BUY", price=100.0, volume=100, total_equity=1e6, available_cash=1e5)
    action, msg = bl_rule.check(req_reject, {})
    assert action == RiskAction.REJECT
    assert "黑名单" in msg

    # 2. MaxOrderAmountRule
    amount_rule = MaxOrderAmountRule(max_order_amount=500000.0)
    req_pass_amount = OrderRiskRequest(symbol="600519.SH", direction="BUY", price=1000.0, volume=100, total_equity=1e6, available_cash=1e5)
    action, _ = amount_rule.check(req_pass_amount, {})
    assert action == RiskAction.PASS

    req_reject_amount = OrderRiskRequest(symbol="600519.SH", direction="BUY", price=1000.0, volume=600, total_equity=1e6, available_cash=1e6)
    action, msg = amount_rule.check(req_reject_amount, {})
    assert action == RiskAction.REJECT
    assert "单笔限额" in msg

    # 3. MaxStockWeightRule
    weight_rule = MaxStockWeightRule(max_stock_weight=0.10)
    req_reject_weight = OrderRiskRequest(symbol="600519.SH", direction="BUY", price=100.0, volume=600, current_position=500, total_equity=1e6, available_cash=1e5)
    action, msg = weight_rule.check(req_reject_weight, {})
    assert action == RiskAction.REJECT
    assert "集中度限制" in msg

    # 4. AdvLiquidityRule
    adv_rule = AdvLiquidityRule(max_adv_ratio=0.10)
    req_adv = OrderRiskRequest(symbol="600519.SH", direction="BUY", price=100.0, volume=1500, total_equity=1e6, available_cash=1e5, adv_20=10000.0)
    action, msg = adv_rule.check(req_adv, {})
    assert action == RiskAction.REJECT
    assert "冲击成本上限" in msg

    # 5. CashSufficiencyRule
    cash_rule = CashSufficiencyRule(fee_buffer_rate=0.002)
    req_cash = OrderRiskRequest(symbol="600519.SH", direction="BUY", price=100.0, volume=1000, total_equity=1e6, available_cash=90000.0)
    action, msg = cash_rule.check(req_cash, {})
    assert action == RiskAction.REJECT
    assert "可用现金不足" in msg

def test_circuit_breaker_transitions_and_enforcement():
    cb = CircuitBreakerManager(warn_drawdown=0.015, restrict_buy_drawdown=0.025, halt_drawdown=0.035)
    acc = "acc_cb_test"

    # Initial
    s0 = cb.update_equity(acc, 1000000.0)
    assert s0.level == CircuitBreakerLevel.NORMAL

    # Yellow
    s1 = cb.update_equity(acc, 982000.0)
    assert s1.level == CircuitBreakerLevel.YELLOW_WARN

    # Orange
    s2 = cb.update_equity(acc, 972000.0)
    assert s2.level == CircuitBreakerLevel.ORANGE_RESTRICT_BUY

    # Enforcement at Orange: BUY is rejected, SELL is allowed
    cb_rule = CircuitBreakerRule(cb_manager=cb)
    req_buy = OrderRiskRequest(account_id=acc, symbol="600519.SH", direction="BUY", price=100.0, volume=10, total_equity=1e6, available_cash=1e5)
    action_buy, _ = cb_rule.check(req_buy, {})
    assert action_buy == RiskAction.REJECT

    req_sell = OrderRiskRequest(account_id=acc, symbol="600519.SH", direction="SELL", price=100.0, volume=10, total_equity=1e6, available_cash=1e5)
    action_sell, _ = cb_rule.check(req_sell, {})
    assert action_sell == RiskAction.PASS

    # Red
    s3 = cb.update_equity(acc, 960000.0)
    assert s3.level == CircuitBreakerLevel.RED_HALT

    # Enforcement at Red: Both BUY and SELL rejected
    action_red_sell, _ = cb_rule.check(req_sell, {})
    assert action_red_sell == RiskAction.REJECT

    # Bounce check: One-way ratchet
    s_bounce = cb.update_equity(acc, 990000.0)
    assert s_bounce.level == CircuitBreakerLevel.RED_HALT

    # Manual reset
    s_reset = cb.reset(acc, reset_watermark=True)
    assert s_reset.level == CircuitBreakerLevel.NORMAL
    assert s_reset.high_watermark == 990000.0

def test_pre_trade_checker():
    cb = CircuitBreakerManager()
    checker = PreTradeRiskChecker(cb_manager=cb)
    checker.add_blacklist("600000.SH")

    req_bl = OrderRiskRequest(
        account_id="acc_01",
        symbol="600000.SH",
        direction="BUY",
        price=10.0,
        volume=100,
        total_equity=1e6,
        available_cash=1e5
    )
    res = checker.check_order(req_bl)
    assert res.passed is False
    assert res.action == RiskAction.REJECT
    assert any("黑名单" in v for v in res.violations)

    checker.remove_blacklist("600000.SH")
    res_after = checker.check_order(req_bl)
    assert res_after.passed is True

def test_post_trade_monitor():
    cb = CircuitBreakerManager()
    monitor = PostTradeRiskMonitor(cb_manager=cb)

    req = PortfolioRiskRequest(
        account_id="acc_post_01",
        total_equity=1000000.0,
        high_watermark=1000000.0,
        positions={"600519.SH": 250000.0, "000858.SZ": 200000.0}
    )
    res = monitor.analyze_portfolio(req)
    assert res.max_stock_weight == 0.25
    assert any("单一持仓集中度" in a for a in res.alerts)

def test_risk_alert_manager():
    mgr = RiskAlertManager(maxlen=5)
    for i in range(10):
        mgr.record_alert(
            event_id=f"e_{i}",
            level=AlertLevel.WARN if i % 2 == 0 else AlertLevel.CRITICAL,
            account_id="acc_01",
            rule_name="TestRule",
            message=f"Alert {i}"
        )
    assert len(mgr.get_recent_alerts(limit=20)) == 5
    crit = mgr.get_recent_alerts(level=AlertLevel.CRITICAL)
    assert all(c.level == AlertLevel.CRITICAL for c in crit)

    # Webhook adapters
    evt = RiskAlertEvent(
        event_id="e_99",
        timestamp="2026-09-18T12:00:00",
        level=AlertLevel.CRITICAL,
        account_id="acc_01",
        rule_name="CircuitBreaker",
        message="Emergency stop"
    )
    feishu = FeishuWebhookAdapter.format(evt)
    assert feishu["msg_type"] == "interactive"
    ding = DingTalkWebhookAdapter.format(evt)
    assert ding["msgtype"] == "markdown"

def test_fastapi_risk_router_endpoints(client):
    # 1. /api/v1/risk/check_order
    order_payload = {
        "account_id": "api_test_acc",
        "symbol": "600519.SH",
        "direction": "BUY",
        "price": 800.0,
        "volume": 100,
        "total_equity": 1000000.0,
        "available_cash": 500000.0
    }
    resp = client.post("/api/v1/risk/check_order", json=order_payload)
    assert resp.status_code == 200
    assert resp.json()["passed"] is True

    # 2. /api/v1/risk/check_portfolio
    portfolio_payload = {
        "account_id": "api_test_acc",
        "total_equity": 1000000.0,
        "positions": {"600519.SH": 100000.0}
    }
    resp_pf = client.post("/api/v1/risk/check_portfolio", json=portfolio_payload)
    assert resp_pf.status_code == 200
    assert resp_pf.json()["drawdown"] == 0.0

    # 3. /api/v1/risk/circuit_breaker/status & reset
    resp_cb = client.get("/api/v1/risk/circuit_breaker/status?account_id=api_test_acc")
    assert resp_cb.status_code == 200
    assert resp_cb.json()["account_id"] == "api_test_acc"

    resp_reset = client.post("/api/v1/risk/circuit_breaker/reset", json={"account_id": "api_test_acc"})
    assert resp_reset.status_code == 200
    assert resp_reset.json()["level"] == 0

    # 4. /api/v1/risk/blacklist
    resp_bl_add = client.post("/api/v1/risk/blacklist", json={"action": "add", "symbols": ["002594.SZ"]})
    assert resp_bl_add.status_code == 200
    assert "002594.SZ" in resp_bl_add.json()["blacklist"]

    resp_bl_del = client.post("/api/v1/risk/blacklist", json={"action": "remove", "symbols": ["002594.SZ"]})
    assert resp_bl_del.status_code == 200
    assert "002594.SZ" not in resp_bl_del.json()["blacklist"]

    # 5. /api/v1/risk/alerts
    resp_alerts = client.get("/api/v1/risk/alerts?limit=5")
    assert resp_alerts.status_code == 200
    assert isinstance(resp_alerts.json(), list)
