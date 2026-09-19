import math
import uuid
from typing import List, Optional, Dict, Any
from src.execution_engine.models import Order, AlgoType, OrderDirection
from src.execution_engine.algos.base import BaseExecutionAlgo


class ImplementationShortfallAlgo(BaseExecutionAlgo):
    """
    实施亏损最小化算法 (Implementation Shortfall - IS Algo):
    基于 Almgren-Chriss 最优执行模型框架，在市场冲击成本 (Market Impact) 与价格漂移风险 (Price Risk) 之间动态寻优。
    
    核心机制：
    1. 紧迫度衰减曲线 (Urgency Decay Profile)：
       根据设定的紧迫度系数 kappa 计算前期加速出清权重 w_k ~ exp(-kappa * t_k)；
    2. 行情漂移反馈机制 (Price Drift Adaptive Response)：
       传入 arrival_price (决策基准价) 与 current_price (当前市价)，若出现不利漂移 (如买入时持续上涨) 则动态自适应提高 kappa，
       加快订单出清节奏以锁定滑点与风险；若漂移平稳则降低冲击平缓执行。
    3. 整百股圆整：确保每笔子订单满足整百股且总量严格守恒。
    """

    def __init__(
        self,
        num_slices: int = 5,
        default_urgency: float = 0.5,
    ):
        super().__init__(name="IS")
        self.num_slices = max(1, num_slices)
        self.default_urgency = max(0.0, default_urgency)

    def slice_order(self, parent_order: Order, ctx: Optional[Dict[str, Any]] = None) -> List[Order]:
        total_vol = parent_order.volume
        ctx = ctx or {}
        n = max(1, ctx.get("num_slices", self.num_slices))

        if n <= 1 or total_vol <= 100:
            sub = parent_order.model_copy()
            sub.algo_type = AlgoType.IS
            return [sub]

        base_urgency = float(ctx.get("urgency", self.default_urgency))
        arrival_price = ctx.get("arrival_price", parent_order.price)
        current_price = ctx.get("current_price", arrival_price)

        # 动态偏离度反馈：检测价格向不利方向漂移
        eff_urgency = base_urgency
        if arrival_price and arrival_price > 0 and current_price and current_price > 0:
            if parent_order.direction == OrderDirection.BUY:
                # 买入时价格上涨为不利漂移
                drift = (current_price - arrival_price) / arrival_price
            else:
                # 卖出时价格下跌为不利漂移
                drift = (arrival_price - current_price) / arrival_price

            if drift > 0:
                # 价格不利漂移越大，执行紧迫度越高（自适应系数 1.0 + 5.0 * drift）
                eff_urgency = base_urgency * (1.0 + 5.0 * drift)

        # 构建指数衰减权重序列
        if eff_urgency < 1e-4:
            weights = [1.0 / n] * n
        else:
            raw_weights = [math.exp(-eff_urgency * (i / (n - 1))) for i in range(n)]
            sum_w = sum(raw_weights)
            weights = [w / sum_w for w in raw_weights]

        slices: List[Order] = []
        allocated = 0

        for i in range(n - 1):
            w = weights[i]
            vol = int((total_vol * w) // 100) * 100
            allocated += vol
            slices.append(
                Order(
                    order_id=f"sub_is_{uuid.uuid4().hex[:8]}",
                    account_id=parent_order.account_id,
                    symbol=parent_order.symbol,
                    direction=parent_order.direction,
                    order_type=parent_order.order_type,
                    price=parent_order.price,
                    volume=vol,
                    algo_type=AlgoType.IS,
                    parent_order_id=parent_order.order_id,
                )
            )

        last_vol = total_vol - allocated
        slices.append(
            Order(
                order_id=f"sub_is_{uuid.uuid4().hex[:8]}",
                account_id=parent_order.account_id,
                symbol=parent_order.symbol,
                direction=parent_order.direction,
                order_type=parent_order.order_type,
                price=parent_order.price,
                volume=last_vol,
                algo_type=AlgoType.IS,
                parent_order_id=parent_order.order_id,
            )
        )

        return [s for s in slices if s.volume > 0]
