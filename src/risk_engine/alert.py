import json
import uuid
from collections import deque
from datetime import datetime
from typing import Optional, List, Dict, Any
from src.risk_engine.models import AlertLevel, RiskAlertEvent
from src.common.logger import logger

class BaseWebhookAdapter:
    @staticmethod
    def format(event: RiskAlertEvent) -> Dict[str, Any]:
        return event.model_dump()

class FeishuWebhookAdapter(BaseWebhookAdapter):
    @staticmethod
    def format(event: RiskAlertEvent) -> Dict[str, Any]:
        color = "red" if event.level == AlertLevel.CRITICAL else "orange" if event.level == AlertLevel.WARN else "blue"
        return {
            "msg_type": "interactive",
            "card": {
                "header": {
                    "title": {"tag": "plain_text", "content": f"【风控告警】{event.rule_name}"},
                    "template": color
                },
                "elements": [
                    {
                        "tag": "div",
                        "text": {"tag": "lark_md", "content": f"**级别**: {event.level.value}\n**账户**: {event.account_id}\n**时间**: {event.timestamp}\n**详情**: {event.message}"}
                    }
                ]
            }
        }

class DingTalkWebhookAdapter(BaseWebhookAdapter):
    @staticmethod
    def format(event: RiskAlertEvent) -> Dict[str, Any]:
        return {
            "msgtype": "markdown",
            "markdown": {
                "title": f"风控告警: {event.rule_name}",
                "text": f"### 风控告警 - {event.level.value}\n- **账户**: {event.account_id}\n- **规则**: {event.rule_name}\n- **详情**: {event.message}\n- **时间**: {event.timestamp}"
            }
        }

class RiskAlertManager:
    def __init__(self, maxlen: int = 200, webhook_url: Optional[str] = None):
        self._history: deque = deque(maxlen=maxlen)
        self.webhook_url = webhook_url

    def record_alert(
        self,
        event_id: Optional[str] = None,
        level: AlertLevel = AlertLevel.WARN,
        account_id: str = "default",
        rule_name: str = "GeneralRisk",
        message: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ) -> RiskAlertEvent:
        evt = RiskAlertEvent(
            event_id=event_id or f"evt_{uuid.uuid4().hex[:8]}",
            timestamp=datetime.now().isoformat(),
            level=level,
            account_id=account_id,
            rule_name=rule_name,
            message=message,
            metadata=metadata or {}
        )
        self._history.append(evt)
        if level in (AlertLevel.CRITICAL, AlertLevel.ERROR):
            logger.error(f"[RISK-{level.value}] {rule_name}: {message} (Account: {account_id})")
        else:
            logger.warning(f"[RISK-{level.value}] {rule_name}: {message} (Account: {account_id})")

        self._dispatch_webhook(evt)
        return evt

    def get_recent_alerts(
        self,
        limit: int = 50,
        level: Optional[AlertLevel] = None,
        account_id: Optional[str] = None,
    ) -> List[RiskAlertEvent]:
        events = list(self._history)
        if account_id:
            events = [e for e in events if e.account_id == account_id]
        if level:
            events = [e for e in events if e.level == level]
        return events[-limit:]

    def _dispatch_webhook(self, event: RiskAlertEvent) -> None:
        if not self.webhook_url:
            return
        try:
            import urllib.request
            data = json.dumps(BaseWebhookAdapter.format(event)).encode("utf-8")
            req = urllib.request.Request(self.webhook_url, data=data, headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=2)
        except Exception as e:
            logger.debug(f"Webhook push failed (silent ignore): {e}")
