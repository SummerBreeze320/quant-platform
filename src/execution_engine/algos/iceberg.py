import uuid
import random
from typing import List, Optional, Dict, Any
from src.execution_engine.models import Order, AlgoType
from src.execution_engine.algos.base import BaseExecutionAlgo


class IcebergAlgo(BaseExecutionAlgo):
    """
    冰山委托算法 (Iceberg Algorithm):
    将大额母单拆分为多个连续小额切片依次对外展示，避免在盘口过早暴露真实深度引发市场逆向冲击。
    
    核心特性：
    1. display_volume / display_ratio: 支持固定手数或占母单总量的比例拆分；
    2. randomize: 支持对展示委托量进行反侦测随机扰动 (±variance_pct)；
    3. 整百股合规：严格遵循 A 股整百股 (100 shares / 1 lot) 交易要求，尾差并入终单。
    """

    def __init__(
        self,
        display_volume: Optional[int] = None,
        display_ratio: float = 0.15,
        randomize: bool = True,
        variance_pct: float = 0.2,
        min_slice_volume: int = 100,
    ):
        super().__init__(name="ICEBERG")
        self.display_volume = display_volume
        self.display_ratio = max(0.01, min(1.0, display_ratio))
        self.randomize = randomize
        self.variance_pct = max(0.0, min(0.5, variance_pct))
        self.min_slice_volume = max(100, (min_slice_volume // 100) * 100)

    def slice_order(self, parent_order: Order, ctx: Optional[Dict[str, Any]] = None) -> List[Order]:
        total_vol = parent_order.volume
        if total_vol <= self.min_slice_volume:
            sub = parent_order.model_copy()
            sub.algo_type = AlgoType.ICEBERG
            return [sub]

        ctx = ctx or {}
        display_vol = ctx.get("display_volume", self.display_volume)
        display_ratio = ctx.get("display_ratio", self.display_ratio)
        randomize = ctx.get("randomize", self.randomize)
        variance_pct = ctx.get("variance_pct", self.variance_pct)

        if display_vol is not None and display_vol > 0:
            target_slice_base = int((display_vol // 100) * 100)
        else:
            target_slice_base = int(((total_vol * display_ratio) // 100) * 100)

        target_slice_base = max(self.min_slice_volume, target_slice_base)

        slices: List[Order] = []
        allocated = 0

        while allocated < total_vol:
            remaining = total_vol - allocated
            if remaining <= target_slice_base:
                slice_vol = remaining
            else:
                if randomize and variance_pct > 0:
                    factor = 1.0 + random.uniform(-variance_pct, variance_pct)
                    slice_vol = int((round(target_slice_base * factor) // 100) * 100)
                    slice_vol = max(self.min_slice_volume, slice_vol)
                    slice_vol = min(slice_vol, remaining - 100)
                else:
                    slice_vol = min(target_slice_base, remaining)

            allocated += slice_vol
            slices.append(
                Order(
                    order_id=f"sub_iceberg_{uuid.uuid4().hex[:8]}",
                    account_id=parent_order.account_id,
                    symbol=parent_order.symbol,
                    direction=parent_order.direction,
                    order_type=parent_order.order_type,
                    price=parent_order.price,
                    volume=slice_vol,
                    algo_type=AlgoType.ICEBERG,
                    parent_order_id=parent_order.order_id,
                )
            )

        return [s for s in slices if s.volume > 0]
