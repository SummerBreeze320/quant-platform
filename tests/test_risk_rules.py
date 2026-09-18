from src.risk_engine.models import OrderRiskRequest, RiskAction, CircuitBreakerLevel
from src.risk_engine.rules import (
    BlacklistRule, MaxOrderAmountRule, MaxStockWeightRule,
    AdvLiquidityRule, CashSufficiencyRule, CircuitBreakerRule
)

def test_blacklist_rule():
    rule = BlacklistRule(blacklist={"000001.SZ", "600000.SH"})
    req_pass = OrderRiskRequest(symbol="600519.SH", direction="BUY", price=100.0, volume=100, total_equity=1e6, available_cash=1e5)
    action, msg = rule.check(req_pass, {})
    assert action == RiskAction.PASS
    assert msg is None

    req_reject = OrderRiskRequest(symbol="000001.SZ", direction="BUY", price=100.0, volume=100, total_equity=1e6, available_cash=1e5)
    action, msg = rule.check(req_reject, {})
    assert action == RiskAction.REJECT
    assert "黑名单" in msg

def test_max_order_amount_rule():
    rule = MaxOrderAmountRule(max_order_amount=500000.0)
    req_pass = OrderRiskRequest(symbol="600519.SH", direction="BUY", price=1000.0, volume=100, total_equity=1e6, available_cash=1e5) # 100k
    action, _ = rule.check(req_pass, {})
    assert action == RiskAction.PASS

    req_reject = OrderRiskRequest(symbol="600519.SH", direction="BUY", price=1000.0, volume=600, total_equity=1e6, available_cash=1e6) # 600k
    action, msg = rule.check(req_reject, {})
    assert action == RiskAction.REJECT
    assert "单笔限额" in msg

def test_max_stock_weight_rule():
    rule = MaxStockWeightRule(max_stock_weight=0.10) # max 10%
    # total equity 1,000,000. current pos 500 shares * 100 = 50k. buy 600 shares * 100 = 60k. post = 110k (11% > 10%)
    req_reject = OrderRiskRequest(symbol="600519.SH", direction="BUY", price=100.0, volume=600, current_position=500, total_equity=1e6, available_cash=1e5)
    action, msg = rule.check(req_reject, {})
    assert action == RiskAction.REJECT
    assert "集中度限制" in msg

def test_adv_liquidity_rule():
    rule = AdvLiquidityRule(max_adv_ratio=0.10)
    req = OrderRiskRequest(symbol="600519.SH", direction="BUY", price=100.0, volume=1500, total_equity=1e6, available_cash=1e5, adv_20=10000.0)
    action, msg = rule.check(req, {})
    assert action == RiskAction.REJECT
    assert "冲击成本上限" in msg

def test_cash_sufficiency_rule():
    rule = CashSufficiencyRule(fee_buffer_rate=0.002)
    # price 100 * volume 1000 = 100,000. available cash 90,000
    req = OrderRiskRequest(symbol="600519.SH", direction="BUY", price=100.0, volume=1000, total_equity=1e6, available_cash=90000.0)
    action, msg = rule.check(req, {})
    assert action == RiskAction.REJECT
    assert "可用现金不足" in msg
