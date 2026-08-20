"""组合调度层

对比当前持仓与目标持仓，生成调仓指令。
是信号生成层和交易网关层之间的桥梁。

工作流程：
1. 接收目标持仓（来自信号生成层）
2. 查询当前实际持仓（来自交易网关）
3. 计算持仓差异 → 生成买卖指令
4. 风控校验（事前风控）
5. 执行算法选择（大单拆分）
6. 输出调仓指令列表

P4扩展: 支持TradingEngine直接调用 (无broker模式)

注意事项：
- A股T+1：当天买入的不能当天卖出
- 涨跌停限制：涨跌停板无法交易
- 停牌检查：停牌股票跳过
- 先卖后买：先执行卖出释放资金，再执行买入
"""
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, field
import logging

from .trading_gateway import BrokerBase, OrderManager
from .risk_manager import PreTradeRiskChecker, RiskLimits
from .trade_logger import TradeLogger, AlertRecord
from .execution_algo import create_algo, AlgoConfig, ExecutionAlgo

logger = logging.getLogger(__name__)


@dataclass
class RebalanceOrder:
    """调仓指令"""
    stock: str
    side: str  # "buy" / "sell"
    target_volume: float
    current_volume: float = 0.0
    delta_volume: float = 0.0
    target_weight: float = 0.0
    current_weight: float = 0.0
    estimated_price: float = 0.0
    estimated_value: float = 0.0
    priority: int = 0  # 执行优先级(0=最高)
    algo: str = ""  # 执行算法
    algo_config: dict = field(default_factory=dict)
    skip_reason: str = ""  # 跳过原因（停牌/涨跌停等）


class PortfolioScheduler:
    """组合调度器

    生成从当前持仓到目标持仓的调仓指令。
    """

    def __init__(self,
                 broker: BrokerBase = None,
                 risk_limits: Optional[RiskLimits] = None,
                 trade_logger: Optional[TradeLogger] = None,
                 blacklist: Optional[set] = None,
                 use_algo_threshold: float = 10000,
                 init_capital: float = 1_000_000):
        """
        Args:
            broker: 交易网关 (None=直接模式，TradingEngine调用)
            risk_limits: 风控阈值
            trade_logger: 交易日志
            blacklist: 黑名单(ST/停牌)
            use_algo_threshold: 超过此金额的订单使用拆单算法
            init_capital: 初始资金 (直接模式)
        """
        self.broker = broker
        self.risk_checker = PreTradeRiskChecker(risk_limits or RiskLimits())
        self.trade_logger = trade_logger
        self.blacklist = blacklist or set()
        self.use_algo_threshold = use_algo_threshold
        self.init_capital = init_capital

    def generate_rebalance_plan(self,
                                 target_weights: Dict[str, float],
                                 total_value: Optional[float] = None) -> List[RebalanceOrder]:
        """生成调仓计划

        Args:
            target_weights: 目标权重 {stock: weight}
            total_value: 总资产(None则从broker查询)
        Returns:
            调仓指令列表（按执行优先级排序）
        """
        # 1. 获取当前持仓
        if self.broker:
            current_positions = self.broker.get_position()
            account = self.broker.get_account()
            if total_value is None:
                total_value = account["total_value"]
        else:
            current_positions = {}
            if total_value is None:
                total_value = self.init_capital

        logger.info(f"调仓计划: 总资产={total_value:.0f}, "
                     f"目标{len(target_weights)}只, 当前持仓{len(current_positions)}只")

        # 2. 计算当前权重
        current_weights = {}
        for stock, pos in current_positions.items():
            w = pos["market_value"] / total_value if total_value > 0 else 0
            current_weights[stock] = w

        # 3. 合并所有股票（目标+当前）
        all_stocks = set(target_weights.keys()) | set(current_weights.keys())

        # 4. 生成调仓指令
        orders = []
        for stock in all_stocks:
            tw = target_weights.get(stock, 0.0)
            cw = current_weights.get(stock, 0.0)
            delta_w = tw - cw

            if abs(delta_w) < 1e-4:  # 权重变化<0.01%跳过
                continue

            # 估算价格和数量
            current_pos = current_positions.get(stock, {})
            current_vol = current_pos.get("volume", 0)
            price = current_pos.get("current_price", 0)
            if price <= 0:
                # 尝试从broker获取
                price = 10.0  # fallback

            target_value = tw * total_value
            current_value = cw * total_value
            delta_value = delta_w * total_value
            delta_volume = delta_value / price if price > 0 else 0

            side = "buy" if delta_w > 0 else "sell"

            order = RebalanceOrder(
                stock=stock,
                side=side,
                target_volume=target_value / price if price > 0 else 0,
                current_volume=current_vol,
                delta_volume=abs(delta_volume),
                target_weight=tw,
                current_weight=cw,
                estimated_price=price,
                estimated_value=abs(delta_value),
            )

            # 黑名单检查
            if stock in self.blacklist:
                order.skip_reason = "黑名单(ST/停牌)"
                orders.append(order)
                continue

            # 优先级：卖出优先(释放资金)，大单优先
            if side == "sell":
                order.priority = 0
            else:
                order.priority = 1

            # 拆单算法选择
            if abs(delta_value) > self.use_algo_threshold:
                order.algo = "TWAP"
                order.algo_config = {
                    "duration_minutes": 30,
                    "slices": 5,
                }

            orders.append(order)

        # 5. 按优先级排序（先卖后买）
        orders.sort(key=lambda o: (o.priority, -o.estimated_value))

        # 6. 事前风控校验
        self._check_risks(orders, target_weights, total_value)

        logger.info(f"调仓指令: {len(orders)}笔 "
                     f"(买入{sum(1 for o in orders if o.side=='buy' and not o.skip_reason)}笔, "
                     f"卖出{sum(1 for o in orders if o.side=='sell' and not o.skip_reason)}笔, "
                     f"跳过{sum(1 for o in orders if o.skip_reason)}笔)")

        return orders

    def generate_rebalance(
        self,
        current_positions: Dict[str, Dict],
        target_weights: Dict[str, float],
        total_value: float,
    ) -> Dict:
        """P4: 直接模式调仓 (TradingEngine调用入口)

        Args:
            current_positions: 当前持仓 {code: {weight, price, volume}}
            target_weights: 目标权重 {code: weight}
            total_value: 总资产

        Returns:
            {orders: [...], summary: {...}}
        """
        orders = []
        all_stocks = set(target_weights.keys()) | set(current_positions.keys())

        for stock in all_stocks:
            tw = target_weights.get(stock, 0.0)
            cw = current_positions.get(stock, {}).get("weight", 0.0)
            delta = tw - cw

            if abs(delta) < 1e-4:
                continue

            price = current_positions.get(stock, {}).get("price", 10.0)
            orders.append({
                "code": stock,
                "side": "buy" if delta > 0 else "sell",
                "price": price,
                "volume": abs(delta * total_value / price),
                "delta_weight": delta,
                "target_weight": tw,
                "current_weight": cw,
            })

        return {
            "orders": orders,
            "summary": {
                "n_orders": len(orders),
                "n_buy": sum(1 for o in orders if o["side"] == "buy"),
                "n_sell": sum(1 for o in orders if o["side"] == "sell"),
                "total_turnover": sum(abs(o["delta_weight"]) for o in orders),
            },
        }

    def _check_risks(self, orders: List[RebalanceOrder],
                     target_weights: Dict[str, float],
                     total_value: float):
        """事前风控校验"""
        w_series = pd.Series(target_weights)
        check_result = self.risk_checker.check_portfolio(w_series)

        if not check_result["all_pass"]:
            for violation in check_result["violations"]:
                logger.warning(f"事前风控告警: {violation}")
                if self.trade_logger:
                    self.trade_logger.log_alert(AlertRecord(
                        timestamp=pd.Timestamp.now().isoformat(),
                        alert_type="risk_breach",
                        severity="warning",
                        message=f"事前风控: {violation}",
                        details={"target_weights": target_weights},
                    ))

    def execute_rebalance(self,
                          orders: List[RebalanceOrder],
                          order_manager: OrderManager) -> dict:
        """执行调仓计划

        Args:
            orders: 调仓指令列表
            order_manager: 订单管理器
        Returns:
            执行结果汇总
        """
        results = {
            "submitted": 0,
            "skipped": 0,
            "total_value": 0.0,
            "buy_value": 0.0,
            "sell_value": 0.0,
            "order_ids": [],
        }

        for order in orders:
            if order.skip_reason:
                results["skipped"] += 1
                logger.info(f"跳过 {order.stock}: {order.skip_reason}")
                continue

            if order.estimated_price <= 0 or order.delta_volume <= 0:
                results["skipped"] += 1
                continue

            # 执行算法拆单
            if order.algo:
                algo_config = AlgoConfig(
                    total_volume=order.delta_volume,
                    stock=order.stock,
                    side=order.side,
                    **order.algo_config,
                )
                algo = create_algo(order.algo, algo_config)
                slices = algo.generate_slices()

                for sl in slices:
                    oid = order_manager.submit_order(
                        stock=order.stock,
                        side=order.side,
                        price=order.estimated_price,
                        volume=sl["volume"],
                    )
                    if oid:
                        results["order_ids"].append(oid)
                        results["submitted"] += 1
            else:
                oid = order_manager.submit_order(
                    stock=order.stock,
                    side=order.side,
                    price=order.estimated_price,
                    volume=order.delta_volume,
                )
                if oid:
                    results["order_ids"].append(oid)
                    results["submitted"] += 1

            if order.side == "buy":
                results["buy_value"] += order.estimated_value
            else:
                results["sell_value"] += order.estimated_value
            results["total_value"] += order.estimated_value

        logger.info(f"调仓执行: 提交{results['submitted']}笔, "
                     f"跳过{results['skipped']}笔, "
                     f"总金额{results['total_value']:.0f}")
        return results
