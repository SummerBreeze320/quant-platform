"""监控告警层

实时监控交易系统状态，触发告警和止损：

1. 持仓偏离监控：实际持仓 vs 目标持仓偏差
2. 风控超限监控：净值回撤、单日亏损、VaR超限
3. 行情中断监控：行情数据延迟/中断检测
4. 订单状态监控：订单超时未成交、异常拒绝
5. 系统健康监控：连接状态、内存/CPU使用

告警级别：
- info: 信息级（订单成交通知等）
- warning: 警告级（偏离超标、回撤告警）
- critical: 严重级（止损触发、连接断开）
"""
import time
import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import threading

from .trade_logger import TradeLogger, AlertRecord
from .risk_manager import IntraTradeRiskMonitor, RiskLimits

logger = logging.getLogger(__name__)


@dataclass
class AlertRule:
    """告警规则"""
    name: str
    check_interval: int = 60  # 检查间隔(秒)
    enabled: bool = True


class TradingMonitor:
    """交易系统监控器

    后台线程定期检查各项指标，触发告警。
    """

    def __init__(self,
                 trade_logger: Optional[TradeLogger] = None,
                 risk_limits: Optional[RiskLimits] = None,
                 check_interval: int = 30):
        self.trade_logger = trade_logger
        self.risk_monitor = IntraTradeRiskMonitor(risk_limits or RiskLimits())
        self.check_interval = check_interval

        self._target_weights: Dict[str, float] = {}
        self._last_nav: float = 0.0
        self._last_data_time: Optional[datetime] = None
        self._alerts: List[dict] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._callbacks: List[Callable] = []

        # 告警规则
        self.rules = {
            "position_deviation": AlertRule("持仓偏离", 60),
            "drawdown": AlertRule("回撤监控", 30),
            "daily_loss": AlertRule("单日亏损", 30),
            "market_data": AlertRule("行情中断", 60),
            "connection": AlertRule("连接状态", 30),
        }

    def add_callback(self, callback: Callable):
        """添加告警回调函数"""
        self._callbacks.append(callback)

    def update_state(self, nav: float = None,
                     target_weights: Dict = None,
                     current_weights: Dict = None,
                     data_time: datetime = None,
                     broker_connected: bool = True):
        """更新监控状态"""
        if nav is not None:
            self._last_nav = nav
            self.risk_monitor.update(nav)
        if target_weights is not None:
            self._target_weights = target_weights
        if data_time is not None:
            self._last_data_time = data_time

        # 执行检查
        if nav is not None:
            self._check_drawdown(nav)
        if current_weights is not None and target_weights is not None:
            self._check_position_deviation(current_weights, target_weights)
        if data_time is not None:
            self._check_market_data(data_time)
        if not broker_connected:
            self._alert("connection", "critical", "券商连接断开")

    def _check_drawdown(self, nav: float):
        """检查回撤"""
        passed, msg = self.risk_monitor.check_drawdown(nav)
        if not passed:
            self._alert("drawdown", "critical", f"回撤止损触发: {msg}")
        elif "告警" in msg:
            self._alert("drawdown", "warning", msg)

    def _check_position_deviation(self, current: Dict, target: Dict):
        """检查持仓偏离"""
        all_stocks = set(current.keys()) | set(target.keys())
        max_deviation = 0
        deviating_stocks = []

        for stock in all_stocks:
            cw = current.get(stock, 0)
            tw = target.get(stock, 0)
            dev = abs(cw - tw)
            if dev > max_deviation:
                max_deviation = dev
            if dev > 0.02:  # 偏离>2%
                deviating_stocks.append({
                    "stock": stock,
                    "current": cw,
                    "target": tw,
                    "deviation": dev,
                })

        if max_deviation > 0.05:  # 最大偏离>5%
            self._alert("position_deviation", "warning",
                        f"持仓偏离超限: 最大偏离{max_deviation:.2%}",
                        {"deviating_stocks": deviating_stocks})

    def _check_market_data(self, data_time: datetime):
        """检查行情中断"""
        if self._last_data_time is None:
            return
        lag = (datetime.now() - data_time).total_seconds()
        if lag > 300:  # 5分钟无更新
            self._alert("market_data", "critical",
                        f"行情中断: 最后数据时间{data_time}, 延迟{lag:.0f}秒")
        elif lag > 60:  # 1分钟无更新
            self._alert("market_data", "warning",
                        f"行情延迟: {lag:.0f}秒")

    def _alert(self, alert_type: str, severity: str,
               message: str, details: Dict = None):
        """发送告警"""
        alert = {
            "timestamp": datetime.now().isoformat(),
            "alert_type": alert_type,
            "severity": severity,
            "message": message,
            "details": details or {},
        }

        self._alerts.append(alert)
        if len(self._alerts) > 1000:
            self._alerts = self._alerts[-500:]

        # 写入日志
        if self.trade_logger:
            self.trade_logger.log_alert(AlertRecord(
                timestamp=alert["timestamp"],
                alert_type=alert_type,
                severity=severity,
                message=message,
                details=details or {},
            ))

        # 回调通知
        for cb in self._callbacks:
            try:
                cb(alert)
            except Exception as e:
                logger.error(f"告警回调异常: {e}")

    def check_daily_loss(self, daily_return: float):
        """检查单日亏损"""
        passed, msg = self.risk_monitor.check_daily_loss(daily_return)
        if not passed:
            self._alert("daily_loss", "critical", f"单日止损触发: {msg}")

    def get_alerts(self, limit: int = 50,
                   severity: Optional[str] = None) -> List[dict]:
        """获取告警历史"""
        alerts = list(self._alerts)
        if severity:
            alerts = [a for a in alerts if a["severity"] == severity]
        return alerts[-limit:]

    def get_status(self) -> dict:
        """获取监控状态"""
        current_dd = self.risk_monitor.get_current_drawdown(self._last_nav) if self._last_nav > 0 else 0
        return {
            "nav": self._last_nav,
            "current_drawdown": current_dd,
            "peak_nav": self.risk_monitor.peak_nav,
            "n_alerts": len(self._alerts),
            "n_critical": sum(1 for a in self._alerts if a["severity"] == "critical"),
            "n_warnings": sum(1 for a in self._alerts if a["severity"] == "warning"),
            "last_data_time": self._last_data_time.isoformat() if self._last_data_time else None,
            "rules": {k: {"enabled": v.enabled, "interval": v.check_interval}
                      for k, v in self.rules.items()},
        }
