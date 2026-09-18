from src.risk_engine.alert import RiskAlertManager, FeishuWebhookAdapter, DingTalkWebhookAdapter
from src.risk_engine.models import AlertLevel, RiskAlertEvent

def test_alert_manager_ring_buffer():
    mgr = RiskAlertManager(maxlen=5)
    for i in range(10):
        mgr.record_alert(
            event_id=f"evt_{i}",
            level=AlertLevel.WARN if i % 2 == 0 else AlertLevel.CRITICAL,
            account_id="acc_01",
            rule_name="TestRule",
            message=f"Alert message {i}"
        )

    alerts = mgr.get_recent_alerts(limit=10)
    assert len(alerts) == 5  # Ring buffer capped at 5
    assert alerts[-1].event_id == "evt_9"

    # Test filtering
    critical_alerts = mgr.get_recent_alerts(level=AlertLevel.CRITICAL)
    assert all(a.level == AlertLevel.CRITICAL for a in critical_alerts)

def test_webhook_adapters():
    evt = RiskAlertEvent(
        event_id="evt_01",
        timestamp="2026-09-18T10:00:00",
        level=AlertLevel.CRITICAL,
        account_id="acc_01",
        rule_name="CircuitBreakerRule",
        message="RED_HALT triggered"
    )
    feishu_payload = FeishuWebhookAdapter.format(evt)
    assert feishu_payload["msg_type"] == "interactive"

    ding_payload = DingTalkWebhookAdapter.format(evt)
    assert ding_payload["msgtype"] == "markdown"
