"""三层风控体系

高级量化的风控生命线，分为：
1. 事前风控（Pre-Trade）：下单前校验权重/敞口/黑名单
2. 事中风控（Intra-Trade）：盘中实时监控净值回撤/亏损/漂移
3. 事后风控（Post-Trade）：盘后风险归因/压力测试/绩效衰减

事前：防患于未然，阻止违规订单
事中：实时监控，触发止损/告警
事后：复盘分析，持续改进
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, List, Dict
import logging

logger = logging.getLogger(__name__)


@dataclass
class RiskLimits:
    """风控阈值配置"""
    # 事前
    max_position: float = 0.05  # 单票最大权重
    max_industry: float = 0.30  # 行业最大敞口
    max_total_exposure: float = 0.95  # 最大总仓位
    max_turnover: float = 0.50  # 最大换手率
    blacklist_stocks: set = field(default_factory=set)  # 黑名单(ST/停牌)

    # 事中
    max_daily_loss: float = -0.03  # 单日最大亏损(-3%)
    max_drawdown: float = -0.10  # 最大回撤止损线(-10%)
    max_var_95: float = 0.02  # 95%VaR上限(2%)
    drawdown_alert: float = -0.05  # 回撤告警线(-5%)

    # 事后
    stress_test_scenarios: list = field(default_factory=lambda: [
        {"name": "暴跌", "shock": -0.05},
        {"name": "流动性枯竭", "shock": -0.03, "volume_drop": 0.5},
        {"name": "极端波动", "shock": 0.0, "vol_multiplier": 3.0},
    ])


class PreTradeRiskChecker:
    """事前风控：下单前校验"""

    def __init__(self, limits: RiskLimits):
        self.limits = limits

    def check_order(self, order: dict, current_positions: dict) -> tuple:
        """检查单笔订单

        Returns:
            (passed, message)
        """
        stock = order.get("stock", "")

        # 黑名单检查
        if stock in self.limits.blacklist_stocks:
            return False, f"{stock}在黑名单中"

        # 计算下单后权重
        total_value = sum(current_positions.values())
        order_value = order.get("value", 0)
        new_weight = order_value / (total_value + order_value + 1e-10)

        # 单票权重检查
        current_weight = current_positions.get(stock, 0) / (total_value + 1e-10)
        post_weight = current_weight + new_weight
        if post_weight > self.limits.max_position:
            return False, f"单票权重{post_weight:.2%}超限({self.limits.max_position:.2%})"

        # 总仓位检查
        new_total = (total_value + order_value) / (total_value + order_value + 1e-10)
        if new_total > self.limits.max_total_exposure:
            return False, f"总仓位{new_total:.2%}超限"

        return True, "OK"

    def check_portfolio(self, target_weights: pd.Series,
                        industry_map: Optional[pd.Series] = None) -> dict:
        """检查目标组合权重

        Returns:
            {all_pass: bool, violations: list}
        """
        violations = []

        # 单票权重
        over_position = target_weights[target_weights > self.limits.max_position + 1e-6]
        if len(over_position) > 0:
            violations.append(f"单票超限: {over_position.to_dict()}")

        # 行业敞口
        if industry_map is not None:
            common = target_weights.index.intersection(industry_map.index)
            ind_weights = target_weights.loc[common].groupby(industry_map.loc[common]).sum()
            over_industry = ind_weights[ind_weights > self.limits.max_industry + 1e-6]
            if len(over_industry) > 0:
                violations.append(f"行业超限: {over_industry.to_dict()}")

        # 黑名单
        in_blacklist = [s for s in target_weights.index if s in self.limits.blacklist_stocks]
        if in_blacklist:
            violations.append(f"持仓黑名单: {in_blacklist}")

        return {
            "all_pass": len(violations) == 0,
            "violations": violations,
        }


class IntraTradeRiskMonitor:
    """事中风控：盘中实时监控"""

    def __init__(self, limits: RiskLimits):
        self.limits = limits
        self.peak_nav = 0.0
        self.daily_pnl = 0.0
        self.nav_history = []

    def update(self, current_nav: float):
        """更新净值状态"""
        self.nav_history.append(current_nav)
        self.peak_nav = max(self.peak_nav, current_nav)

    def check_drawdown(self, current_nav: float) -> tuple:
        """检查回撤"""
        if self.peak_nav == 0:
            return True, "OK"
        drawdown = (current_nav - self.peak_nav) / self.peak_nav

        if drawdown <= self.limits.max_drawdown:
            return False, f"回撤止损: {drawdown:.2%} <= {self.limits.max_drawdown:.2%}"

        if drawdown <= self.limits.drawdown_alert:
            logger.warning(f"回撤告警: {drawdown:.2%}")

        return True, f"回撤{drawdown:.2%}"

    def check_daily_loss(self, daily_return: float) -> tuple:
        """检查单日亏损"""
        if daily_return <= self.limits.max_daily_loss:
            return False, f"单日止损: {daily_return:.2%} <= {self.limits.max_daily_loss:.2%}"
        return True, "OK"

    def check_var(self, returns: pd.Series, confidence: float = 0.95) -> tuple:
        """检查VaR"""
        from src.backtest.risk_metrics import calc_var
        var = calc_var(returns, confidence)
        if var < -self.limits.max_var_95:
            return False, f"VaR超限: {var:.2%} < {-self.limits.max_var_95:.2%}"
        return True, f"VaR={var:.2%}"

    def get_current_drawdown(self, current_nav: float) -> float:
        """获取当前回撤"""
        if self.peak_nav == 0:
            return 0.0
        return (current_nav - self.peak_nav) / self.peak_nav


class PostTradeRiskAnalyzer:
    """事后风控：盘后分析"""

    def __init__(self, limits: RiskLimits):
        self.limits = limits

    def analyze(self, backtest_result: dict) -> dict:
        """全面风险分析

        Args:
            backtest_result: 回测结果dict
        Returns:
            风险分析报告
        """
        report = {}

        # 1. 基本指标
        if "metrics" in backtest_result:
            report["performance"] = backtest_result["metrics"]

        # 2. 回撤分析
        if "daily_returns" in backtest_result:
            returns = backtest_result["daily_returns"]
            cum = (1 + returns).cumprod()
            peak = cum.expanding().max()
            drawdown = (cum - peak) / peak
            report["drawdown"] = {
                "max_drawdown": float(drawdown.min()),
                "current_drawdown": float(drawdown.iloc[-1]),
                "avg_drawdown": float(drawdown.mean()),
                "drawdown_periods": self._count_dd_periods(drawdown),
            }

        # 3. 交易成本分析
        if "cost_log" in backtest_result:
            cost_log = backtest_result["cost_log"]
            total_cost = sum(c["cost"] for c in cost_log)
            avg_cost = total_cost / len(cost_log) if cost_log else 0
            report["cost_analysis"] = {
                "total_cost": total_cost,
                "average_cost_per_rebalance": avg_cost,
                "n_rebalances": len(cost_log),
            }

        # 4. 压力测试
        report["stress_test"] = self._stress_test(backtest_result)

        # 5. 换手率分析
        if "trades" in backtest_result:
            trades = backtest_result["trades"]
            total_turnover = sum(
                sum(abs(t_["value"]) for t_ in t["trades"])
                for t in trades
            ) / 2  # 买卖双向
            report["turnover_analysis"] = {
                "total_turnover": total_turnover,
                "avg_turnover": total_turnover / len(trades) if trades else 0,
            }

        return report

    def _count_dd_periods(self, drawdown: pd.Series) -> int:
        """计算回撤期间数量"""
        in_dd = drawdown < -1e-6
        transitions = (in_dd != in_dd.shift()).sum()
        return transitions // 2

    def _stress_test(self, backtest_result: dict) -> list:
        """压力测试：模拟极端行情"""
        results = []
        returns = backtest_result.get("daily_returns")
        if returns is None or len(returns) == 0:
            return results

        for scenario in self.limits.stress_test_scenarios:
            shock = scenario.get("shock", 0)
            vol_mult = scenario.get("vol_multiplier", 1.0)

            stressed = returns * vol_mult + shock
            cum = (1 + stressed).cumprod()
            max_dd = float(((cum - cum.expanding().max()) / cum.expanding().max()).min())
            total_ret = float(cum.iloc[-1] - 1)

            results.append({
                "name": scenario["name"],
                "stressed_return": round(total_ret, 4),
                "stressed_max_drawdown": round(max_dd, 4),
            })

        return results

    def compliance_report(self, target_weights: pd.Series,
                          trades: list,
                          industry_map: Optional[pd.Series] = None) -> dict:
        """合规检查报告"""
        checker = PreTradeRiskChecker(self.limits)
        portfolio_check = checker.check_portfolio(target_weights, industry_map)

        # 交易合规检查
        trade_violations = []
        for trade in trades:
            stock = trade.get("stock", "")
            if stock in self.limits.blacklist_stocks:
                trade_violations.append(f"交易黑名单股票: {stock}")

        return {
            "portfolio_check": portfolio_check,
            "trade_violations": trade_violations,
            "compliance_pass": portfolio_check["all_pass"] and len(trade_violations) == 0,
        }
