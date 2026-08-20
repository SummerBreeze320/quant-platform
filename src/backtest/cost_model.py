"""交易成本模型

高级回测必须建模真实的交易成本，否则回测收益虚高。

成本构成：
1. 佣金（Commission）：固定费率，A股约万分之2.5
2. 印花税（Stamp Duty）：卖出单边收取，千分之1
3. 滑点（Slippage）：固定基点或线性模型
4. 冲击成本（Market Impact）：交易量占ADV的比例，平方根模型

流动性约束：单笔交易不超过日成交量的一定比例
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional


@dataclass
class Trade:
    """交易记录"""
    date: str
    stock: str
    side: str  # "buy" or "sell"
    shares: float
    price: float
    volume: float = 0.0  # 当日成交量(用于冲击成本计算)


class CostModel:
    """交易成本计算模型"""

    def __init__(self,
                 commission_rate: float = 0.00025,
                 stamp_duty_rate: float = 0.001,
                 slippage_bps: float = 5.0,
                 impact_coefficient: float = 0.1,
                 max_volume_part: float = 0.05):
        """
        Args:
            commission_rate: 佣金费率（双向），默认万分之2.5
            stamp_duty_rate: 印花税率（卖出单边），千分之1
            slippage_bps: 固定滑点（基点），5bp = 0.05%
            impact_coefficient: 冲击成本系数，用于平方根模型
            max_volume_part: 单笔交易占日成交量上限，默认5%
        """
        self.commission_rate = commission_rate
        self.stamp_duty_rate = stamp_duty_rate
        self.slippage_bps = slippage_bps
        self.impact_coefficient = impact_coefficient
        self.max_volume_part = max_volume_part

    def calculate_trade_cost(self, trade: Trade) -> dict:
        """计算单笔交易成本

        Returns:
            {commission, stamp_duty, slippage, impact, total, cost_rate}
        """
        notional = trade.shares * trade.price

        # 1. 佣金（双向）
        commission = notional * self.commission_rate

        # 2. 印花税（仅卖出）
        stamp_duty = notional * self.stamp_duty_rate if trade.side == "sell" else 0.0

        # 3. 固定滑点
        slippage = notional * self.slippage_bps / 10000

        # 4. 冲击成本（平方根模型: Almgren-Chriss style）
        # impact = coefficient * sqrt(trade_volume / ADV) * price
        if trade.volume > 0:
            volume_ratio = trade.shares / trade.volume
            volume_ratio = min(volume_ratio, 1.0)
            impact = notional * self.impact_coefficient * np.sqrt(volume_ratio)
        else:
            impact = 0.0

        total = commission + stamp_duty + slippage + impact
        cost_rate = total / notional if notional > 0 else 0.0

        return {
            "commission": commission,
            "stamp_duty": stamp_duty,
            "slippage": slippage,
            "impact": impact,
            "total": total,
            "cost_rate": cost_rate,
        }

    def check_liquidity(self, trade: Trade) -> tuple:
        """检查流动性约束

        Returns:
            (passed, message)
        """
        if trade.volume <= 0:
            return True, "无成交量数据，跳过检查"
        ratio = trade.shares / trade.volume
        if ratio > self.max_volume_part:
            return False, f"流动性超限: {ratio:.2%} > {self.max_volume_part:.2%}"
        return True, "OK"

    def adjust_shares_for_liquidity(self, target_shares: float,
                                    daily_volume: float) -> float:
        """根据流动性约束调整目标持仓量"""
        if daily_volume <= 0:
            return target_shares
        max_shares = daily_volume * self.max_volume_part
        return min(target_shares, max_shares)

    def calculate_rebalance_cost(self,
                                  old_weights: pd.Series,
                                  new_weights: pd.Series,
                                  portfolio_value: float,
                                  prices: pd.Series,
                                  volumes: Optional[pd.Series] = None) -> dict:
        """计算调仓成本

        Args:
            old_weights: 旧权重
            new_weights: 新权重
            portfolio_value: 当前组合总值
            prices: 股票价格
            volumes: 当日成交量
        Returns:
            {total_cost, cost_breakdown, trades, cost_rate}
        """
        trades = []
        total_cost = 0.0
        breakdown = {"commission": 0, "stamp_duty": 0, "slippage": 0, "impact": 0}

        common_stocks = old_weights.index.intersection(new_weights.index)
        weight_change = new_weights - old_weights

        for stock in common_stocks:
            wc = weight_change.loc[stock]
            if abs(wc) < 1e-6:
                continue

            side = "buy" if wc > 0 else "sell"
            trade_value = abs(wc) * portfolio_value
            price = prices.get(stock, 0)
            if price <= 0:
                continue

            shares = trade_value / price
            vol = volumes.get(stock, 0) if volumes is not None else 0

            trade = Trade(
                date="", stock=stock, side=side,
                shares=shares, price=price, volume=vol
            )

            # 流动性检查
            passed, msg = self.check_liquidity(trade)
            if not passed:
                shares = self.adjust_shares_for_liquidity(shares, vol)
                trade = Trade(
                    date="", stock=stock, side=side,
                    shares=shares, price=price, volume=vol
                )

            cost = self.calculate_trade_cost(trade)
            total_cost += cost["total"]
            for k in breakdown:
                breakdown[k] += cost[k]

            trades.append({
                "stock": stock,
                "side": side,
                "shares": shares,
                "price": price,
                "value": shares * price,
                **cost,
            })

        cost_rate = total_cost / portfolio_value if portfolio_value > 0 else 0
        return {
            "total_cost": total_cost,
            "cost_breakdown": breakdown,
            "trades": trades,
            "cost_rate": cost_rate,
        }


def create_default_cost_model() -> CostModel:
    """创建A股默认成本模型"""
    return CostModel(
        commission_rate=0.00025,
        stamp_duty_rate=0.001,
        slippage_bps=5.0,
        impact_coefficient=0.1,
        max_volume_part=0.05,
    )
