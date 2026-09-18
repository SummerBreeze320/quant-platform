import uuid
from typing import List, Optional, Dict, Any
from src.execution_engine.models import Order, AlgoType
from src.execution_engine.algos.base import BaseExecutionAlgo

class VwapAlgo(BaseExecutionAlgo):
    DEFAULT_PROFILE = [0.30, 0.15, 0.10, 0.15, 0.30]

    def __init__(self, volume_profile: Optional[List[float]] = None):
        super().__init__(name="VWAP")
        raw_profile = volume_profile or self.DEFAULT_PROFILE
        total_p = sum(raw_profile)
        self.profile = [p / total_p for p in raw_profile] if total_p > 0 else [1.0]

    def slice_order(self, parent_order: Order, ctx: Optional[Dict[str, Any]] = None) -> List[Order]:
        total_vol = parent_order.volume
        n = len(self.profile)
        if n <= 1 or total_vol <= 100:
            sub = parent_order.model_copy()
            sub.algo_type = AlgoType.VWAP
            return [sub]

        slices = []
        allocated = 0

        for i in range(n - 1):
            ratio = self.profile[i]
            vol = int((total_vol * ratio) // 100) * 100
            allocated += vol
            slices.append(Order(
                order_id=f"sub_vwap_{uuid.uuid4().hex[:8]}",
                account_id=parent_order.account_id,
                symbol=parent_order.symbol,
                direction=parent_order.direction,
                order_type=parent_order.order_type,
                price=parent_order.price,
                volume=vol,
                algo_type=AlgoType.VWAP,
                parent_order_id=parent_order.order_id
            ))

        last_vol = total_vol - allocated
        slices.append(Order(
            order_id=f"sub_vwap_{uuid.uuid4().hex[:8]}",
            account_id=parent_order.account_id,
            symbol=parent_order.symbol,
            direction=parent_order.direction,
            order_type=parent_order.order_type,
            price=parent_order.price,
            volume=last_vol,
            algo_type=AlgoType.VWAP,
            parent_order_id=parent_order.order_id
        ))

        return [s for s in slices if s.volume > 0]
