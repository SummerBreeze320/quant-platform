"""日志持久层

记录所有交易系统事件，用于事后复盘和审计：
1. 信号日志：模型输出的目标持仓、因子得分
2. 订单日志：提交、修改、撤单记录
3. 成交日志：实际成交回报
4. 行情日志：关键行情快照
5. 系统日志：风控告警、异常事件

所有日志同时写入文件（JSON Lines格式）和内存缓存，支持实时查询。
"""
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field, asdict
import logging
import threading

logger = logging.getLogger(__name__)


@dataclass
class SignalRecord:
    """信号记录"""
    timestamp: str
    signal_type: str  # "target_position" / "factor_score"
    strategy: str
    target_weights: Dict[str, float] = field(default_factory=dict)
    factor_scores: Dict[str, float] = field(default_factory=dict)
    metadata: Dict = field(default_factory=dict)


@dataclass
class OrderRecord:
    """订单记录"""
    timestamp: str
    order_id: str
    stock: str
    side: str  # "buy" / "sell"
    price: float
    volume: float
    status: str  # "submitted" / "partial" / "filled" / "cancelled" / "rejected"
    filled_volume: float = 0.0
    filled_price: float = 0.0
    reject_reason: str = ""


@dataclass
class TradeRecord:
    """成交记录"""
    timestamp: str
    order_id: str
    stock: str
    side: str
    filled_price: float
    filled_volume: float
    commission: float = 0.0
    slippage: float = 0.0


@dataclass
class AlertRecord:
    """告警记录"""
    timestamp: str
    alert_type: str  # "position_deviation" / "risk_breach" / "market_data" / "system"
    severity: str  # "info" / "warning" / "critical"
    message: str
    details: Dict = field(default_factory=dict)


class TradeLogger:
    """交易日志持久化

    线程安全，支持多模块同时写入。
    日志文件按日期分片，格式为JSON Lines。
    """

    def __init__(self, log_dir: str = "data/trade_logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._memory_cache: List[Dict] = []
        self._max_cache = 10000

    def _write(self, record_type: str, data: dict):
        """写入日志文件和内存缓存"""
        data["record_type"] = record_type
        data["logged_at"] = datetime.now().isoformat()

        with self._lock:
            # 写入文件（JSON Lines）
            date_str = datetime.now().strftime("%Y%m%d")
            log_file = self.log_dir / f"trade_{date_str}.jsonl"
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(data, ensure_ascii=False, default=str) + "\n")

            # 内存缓存
            self._memory_cache.append(data)
            if len(self._memory_cache) > self._max_cache:
                self._memory_cache = self._memory_cache[-self._max_cache:]

    def log_signal(self, record: SignalRecord):
        """记录信号"""
        self._write("signal", asdict(record))
        logger.info(f"信号记录: {record.signal_type} - {record.strategy}")

    def log_order(self, record: OrderRecord):
        """记录订单"""
        self._write("order", asdict(record))
        logger.info(f"订单记录: {record.order_id} {record.side} {record.stock} "
                     f"{record.volume}@{record.price} -> {record.status}")

    def log_trade(self, record: TradeRecord):
        """记录成交"""
        self._write("trade", asdict(record))
        logger.info(f"成交记录: {record.order_id} {record.stock} "
                     f"{record.filled_volume}@{record.filled_price}")

    def log_alert(self, record: AlertRecord):
        """记录告警"""
        self._write("alert", asdict(record))
        if record.severity == "critical":
            logger.error(f"严重告警: {record.alert_type} - {record.message}")
        elif record.severity == "warning":
            logger.warning(f"告警: {record.alert_type} - {record.message}")
        else:
            logger.info(f"信息: {record.alert_type} - {record.message}")

    def log_system(self, event: str, details: Dict = None):
        """记录系统事件"""
        self._write("system", {
            "event": event,
            "details": details or {},
        })

    def query(self, record_type: Optional[str] = None,
              start_time: Optional[str] = None,
              end_time: Optional[str] = None,
              limit: int = 100) -> List[Dict]:
        """查询日志

        Args:
            record_type: 记录类型过滤
            start_time: 起始时间(ISO格式)
            end_time: 结束时间
            limit: 最大返回数量
        """
        with self._lock:
            results = list(self._memory_cache)

        if record_type:
            results = [r for r in results if r.get("record_type") == record_type]
        if start_time:
            results = [r for r in results if r.get("logged_at", "") >= start_time]
        if end_time:
            results = [r for r in results if r.get("logged_at", "") <= end_time]

        return results[-limit:]

    def get_recent_alerts(self, limit: int = 20) -> List[Dict]:
        """获取最近告警"""
        return self.query(record_type="alert", limit=limit)

    def get_order_history(self, stock: Optional[str] = None,
                          limit: int = 50) -> List[Dict]:
        """获取订单历史"""
        orders = self.query(record_type="order", limit=limit * 5)
        if stock:
            orders = [o for o in orders if o.get("stock") == stock]
        return orders[-limit:]

    def get_trade_history(self, limit: int = 50) -> List[Dict]:
        """获取成交历史"""
        return self.query(record_type="trade", limit=limit)

    def get_daily_summary(self, date: Optional[str] = None) -> dict:
        """获取日度汇总"""
        if date is None:
            date = datetime.now().strftime("%Y%m%d")

        log_file = self.log_dir / f"trade_{date}.jsonl"
        if not log_file.exists():
            return {"date": date, "total_records": 0}

        orders = 0
        trades = 0
        signals = 0
        alerts = 0
        total_commission = 0.0
        total_slippage = 0.0

        with open(log_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    rt = rec.get("record_type", "")
                    if rt == "order":
                        orders += 1
                    elif rt == "trade":
                        trades += 1
                        total_commission += rec.get("commission", 0)
                        total_slippage += rec.get("slippage", 0)
                    elif rt == "signal":
                        signals += 1
                    elif rt == "alert":
                        alerts += 1
                except json.JSONDecodeError:
                    continue

        return {
            "date": date,
            "total_orders": orders,
            "total_trades": trades,
            "total_signals": signals,
            "total_alerts": alerts,
            "total_commission": round(total_commission, 2),
            "total_slippage": round(total_slippage, 2),
        }
