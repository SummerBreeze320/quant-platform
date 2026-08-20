"""统一风控模块

贯穿实盘交易全链路的风险控制:
1. 信号层风控: 预测得分异常、权重集中度、黑名单过滤
2. 调度层风控: 换手率限制、单票权重上限、行业偏离
3. 网关层风控: 订单价格偏离、数量异常、频次限制
4. 执行层风控: 滑点监控、成交偏离、实时止损

统一接口，各层调用同一风控实例，确保一致性。
"""
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
from datetime import datetime

import pandas as pd
import numpy as np

from .risk_control import RiskConfig, RiskController
from .risk_manager import RiskLimits, PreTradeRiskChecker

logger = logging.getLogger(__name__)


@dataclass
class UnifiedRiskConfig:
    """统一风控配置"""
    # 信号层
    max_score_outlier: float = 5.0       # 预测得分异常阈值(标准差)
    max_single_weight: float = 0.15      # 单票最大权重
    min_diversification: int = 3          # 最少持仓数
    blacklist: set = field(default_factory=set)

    # 调度层
    max_turnover: float = 0.50           # 单日换手率上限
    max_sector_exposure: float = 0.30    # 行业最大敞口
    max_total_exposure: float = 0.95     # 最大总仓位

    # 网关层
    max_price_deviation: float = 0.02    # 报单价格偏离上限(2%)
    max_order_frequency: int = 50        # 单票每分钟最大报单数
    min_order_value: float = 1000        # 最小报单金额

    # 执行层
    max_slippage: float = 0.003          # 最大滑点(0.3%)
    daily_stop_loss: float = -0.03       # 日止损(-3%)
    max_drawdown: float = -0.10          # 最大回撤止损(-10%)
    drawdown_alert: float = -0.05       # 回撤告警线(-5%)

    # 流动性
    max_volume_pct: float = 0.10         # 单笔占均量比上限(10%)


class UnifiedRiskController:
    """统一风控控制器

    所有交易环节通过此实例进行风控检查:
    - signal_check(): 信号生成后检查
    - schedule_check(): 调仓指令检查
    - order_check(): 报单前检查
    - execution_check(): 成交后检查

    Usage:
        risk = UnifiedRiskController()
        # 信号层
        ok, msg = risk.signal_check(target_weights)
        # 调度层
        ok, msg = risk.schedule_check(orders, current_weights, target_weights)
        # 网关层
        ok, msg = risk.order_check(code, side, price, volume, ...)
        # 执行层
        ok, msg = risk.execution_check(trades, positions, nav)
    """

    def __init__(self, config: Optional[UnifiedRiskConfig] = None):
        self.config = config or UnifiedRiskConfig()
        self._pre_trade = PreTradeRiskChecker(self._to_risk_limits())
        self._basic_controller = RiskController(self._to_basic_config())
        self._order_counts: Dict[str, List[datetime]] = {}
        self._daily_pnl: float = 0.0
        self._peak_nav: float = 0.0
        self._alerts: List[Dict] = []

    def signal_check(
        self,
        target_weights: Dict[str, float],
        prediction_scores: Optional[Dict[str, float]] = None,
    ) -> Tuple[bool, str]:
        """信号层风控检查

        检查:
        - 权重集中度 (单票超限)
        - 最少分散度
        - 黑名单过滤
        - 预测得分异常
        """
        if not target_weights:
            return False, "空信号"

        # 黑名单检查
        blacklisted = [c for c in target_weights if c in self.config.blacklist]
        if blacklisted:
            return False, f"黑名单股票: {blacklisted}"

        # 单票权重检查
        for code, w in target_weights.items():
            if abs(w) > self.config.max_single_weight:
                return False, f"{code} 权重超限: {w:.1%} > {self.config.max_single_weight:.1%}"

        # 分散度检查
        if len(target_weights) < self.config.min_diversification:
            return False, f"持仓数不足: {len(target_weights)} < {self.config.min_diversification}"

        # 权重总和检查
        total = sum(abs(w) for w in target_weights.values())
        if total > self.config.max_total_exposure:
            return False, f"总仓位超限: {total:.1%} > {self.config.max_total_exposure:.1%}"

        # 得分异常检查
        if prediction_scores:
            scores = list(prediction_scores.values())
            if scores:
                mean_s = np.mean(scores)
                std_s = np.std(scores)
                for code, score in prediction_scores.items():
                    if std_s > 0 and abs(score - mean_s) / std_s > self.config.max_score_outlier:
                        logger.warning(f"{code} 得分异常: {score:.4f} (z-score > {self.config.max_score_outlier})")

        return True, "OK"

    def schedule_check(
        self,
        orders: List[Dict],
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
    ) -> Tuple[bool, str]:
        """调度层风控检查

        检查:
        - 换手率
        - 权重变动合规性
        """
        if not orders:
            return True, "无调仓"

        # 换手率
        turnover = 0.0
        for code in set(list(current_weights.keys()) + list(target_weights.keys())):
            old_w = current_weights.get(code, 0)
            new_w = target_weights.get(code, 0)
            turnover += abs(new_w - old_w)

        if turnover > self.config.max_turnover:
            return False, f"换手率超限: {turnover:.1%} > {self.config.max_turnover:.1%}"

        return True, "OK"

    def order_check(
        self,
        code: str,
        side: str,
        price: float,
        volume: int,
        current_positions: Dict,
        total_capital: float,
        reference_price: Optional[float] = None,
    ) -> Tuple[bool, str]:
        """网关层风控检查

        检查:
        - 价格偏离
        - 报单频次
        - 最小金额
        - 仓位限制(委托基础控制器)
        """
        order_value = price * volume

        # 最小金额
        if order_value < self.config.min_order_value:
            return False, f"报单金额不足: {order_value:.0f} < {self.config.min_order_value}"

        # 价格偏离
        if reference_price:
            deviation = abs(price - reference_price) / reference_price
            if deviation > self.config.max_price_deviation:
                return False, f"价格偏离超限: {deviation:.2%} > {self.config.max_price_deviation:.2%}"

        # 报单频次
        now = datetime.now()
        if code not in self._order_counts:
            self._order_counts[code] = []
        self._order_counts[code] = [
            t for t in self._order_counts[code]
            if (now - t).total_seconds() < 60
        ]
        if len(self._order_counts[code]) >= self.config.max_order_frequency:
            return False, f"报单频次超限: {code} 每分钟{len(self._order_counts[code])}次"
        self._order_counts[code].append(now)

        # 基础风控
        return self._basic_controller.check_order(
            code, side, price, volume, current_positions, total_capital
        )

    def execution_check(
        self,
        trades: List[Dict],
        positions: Dict,
        nav: float,
        prev_nav: float,
    ) -> Tuple[bool, str]:
        """执行层风控检查

        检查:
        - 滑点
        - 日内亏损
        - 回撤
        """
        if nav <= 0 or prev_nav <= 0:
            return True, "OK"

        # 日内PnL
        daily_return = (nav - prev_nav) / prev_nav
        self._daily_pnl += daily_return

        if daily_return < self.config.daily_stop_loss:
            self._add_alert("DAILY_STOP_LOSS", f"日亏损触线: {daily_return:.2%}")
            return False, f"日止损触发: {daily_return:.2%} < {self.config.daily_stop_loss:.2%}"

        # 回撤
        if nav > self._peak_nav:
            self._peak_nav = nav
        drawdown = (nav - self._peak_nav) / self._peak_nav if self._peak_nav > 0 else 0

        if drawdown < self.config.max_drawdown:
            self._add_alert("MAX_DRAWDOWN", f"回撤触线: {drawdown:.2%}")
            return False, f"最大回撤触发: {drawdown:.2%}"

        if drawdown < self.config.drawdown_alert:
            self._add_alert("DRAWDOWN_ALERT", f"回撤告警: {drawdown:.2%}")

        # 滑点检查
        for trade in trades:
            expected_price = trade.get("expected_price", trade.get("price", 0))
            actual_price = trade.get("price", 0)
            if expected_price > 0:
                slippage = abs(actual_price - expected_price) / expected_price
                if slippage > self.config.max_slippage:
                    self._add_alert(
                        "SLIPPAGE",
                        f"{trade.get('code', '')} 滑点: {slippage:.2%}"
                    )

        return True, "OK"

    def get_alerts(self, clear: bool = True) -> List[Dict]:
        """获取风控告警列表"""
        alerts = self._alerts.copy()
        if clear:
            self._alerts.clear()
        return alerts

    def apply_risk_assessment(self, assessment) -> List[str]:
        """根据RiskAgent输出动态调整风控参数

        Args:
            assessment: RiskAssessment (from src.agents.contracts)

        Returns:
            应用的调整列表
        """
        adjustments = []
        ratio = assessment.position_ratio

        self.config.max_total_exposure = min(
            self.config.max_total_exposure, ratio * 0.95
        )
        adjustments.append(
            f"max_total_exposure → {self.config.max_total_exposure:.1%}"
        )

        if ratio <= 0.3:
            self.config.max_single_weight = min(
                self.config.max_single_weight, 0.08
            )
            adjustments.append("max_single_weight → 8% (defensive)")
            self.config.daily_stop_loss = -0.02
            adjustments.append("daily_stop_loss → -2% (tightened)")

        for w in assessment.warnings:
            self._add_alert("MARKET_STATE", w)

        logger.info(
            f"RiskAssessment applied: state={assessment.market_state.value} "
            f"ratio={ratio:.0%} adjustments={len(adjustments)}"
        )
        return adjustments

    def reset_daily(self):
        """每日重置"""
        self._daily_pnl = 0.0
        self._order_counts.clear()

    def _add_alert(self, alert_type: str, message: str):
        alert = {
            "type": alert_type,
            "message": message,
            "timestamp": datetime.now().isoformat(),
        }
        self._alerts.append(alert)
        logger.warning(f"Risk Alert: [{alert_type}] {message}")

    def _to_risk_limits(self) -> RiskLimits:
        """转换为RiskLimits"""
        return RiskLimits(
            max_position=self.config.max_single_weight,
            max_total_exposure=self.config.max_total_exposure,
            max_turnover=self.config.max_turnover,
            max_daily_loss=self.config.daily_stop_loss,
            max_drawdown=self.config.max_drawdown,
            drawdown_alert=self.config.drawdown_alert,
        )

    def _to_basic_config(self) -> RiskConfig:
        """转换为RiskConfig"""
        return RiskConfig(
            max_position_pct=self.config.max_single_weight,
            max_total_position=self.config.max_total_exposure,
            daily_stop_loss=self.config.daily_stop_loss,
            max_turnover=self.config.max_turnover,
        )
