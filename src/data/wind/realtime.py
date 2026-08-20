"""实时行情订阅"""
import logging
import json
from typing import Callable, Optional
from .wind_client import WindClient

logger = logging.getLogger(__name__)


class RealtimeSubscriber:
    """Wind实时行情订阅"""

    SNAPSHOT_FIELDS = "rt_last,rt_last_qty,rt_bid,rt_ask,rt_bid_qty,rt_ask_qty,rt_high,rt_low,rt_open,rt_pre_close,rt_pct_chg,rt_vwap"

    def __init__(self, client: Optional[WindClient] = None):
        self.client = client or WindClient()
        self._subscriptions = {}
        self._callbacks: list[Callable] = []

    def _on_data(self, indata):
        """Wind wsq回调"""
        if not isinstance(indata, list):
            return

        for item in indata:
            if item.Fields and item.Data:
                snapshot = dict(zip(item.Fields, item.Data))
                snapshot["code"] = item.Codes[0] if item.Codes else None
                for callback in self._callbacks:
                    callback(snapshot)

    def subscribe(
        self,
        codes: list[str],
        callback: Callable[[dict], None],
        fields: str = None,
    ):
        """订阅实时行情

        Args:
            codes: 证券代码列表
            callback: 数据回调函数
            fields: 订阅字段（默认使用SNAPSHOT_FIELDS）
        """
        code_str = ",".join(codes)
        fields = fields or self.SNAPSHOT_FIELDS

        if callback not in self._callbacks:
            self._callbacks.append(callback)

        self._subscriptions[code_str] = codes
        self.client.wsq(code_str, fields, self._on_data)
        logger.info(f"Subscribed to {len(codes)} instruments")

    def unsubscribe(self, codes: list[str] = None):
        """取消订阅"""
        if codes:
            code_str = ",".join(codes)
            if code_str in self._subscriptions:
                del self._subscriptions[code_str]
        else:
            self._subscriptions.clear()
        logger.info(f"Unsubscribed from {len(codes) if codes else 'all'} instruments")
