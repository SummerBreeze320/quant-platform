import uuid
from typing import List, Optional, Dict, Any
from src.execution_engine.models import Order, AlgoType
from src.execution_engine.algos.base import BaseExecutionAlgo

class TwapAlgo(BaseExecutionAlgo):
    def __init__(self, num_slices: int = 5):
        super().__init__(name="TWAP")
        self.num_slices = max(1, num_slices)

    def slice_order(self, parent_order: Order, ctx: Optional[Dict[str, Any]] = None) -> List[Order]:
        total_vol = parent_order.volume
        if self.num_slices <= 1 or total_vol <= 100:
            sub = parent_order.model_copy()
            sub.algo_type = AlgoType.TWAP
            return [sub]

        base_vol = int((total_vol / self.num_slices) // 100) * 100
        slices = []
        allocated = 0

        for i in range(self.num_slices - 1):
            if allocated + base_vol <= total_vol:
                vol = base_vol
            else:
                vol = 0
            allocated += vol
            slices.append(Order(
                order_id=f"sub_twap_{uuid.uuid4().hex[:8]}",
                account_id=parent_order.account_id,
                symbol=parent_order.symbol,
                direction=parent_order.direction,
                order_type=parent_order.order_type,
                price=parent_order.price,
                volume=vol,
                algo_type=AlgoType.TWAP,
                parent_order_id=parent_order.order_id
            ))

        # Remainder in last slice
        last_vol = total_vol - allocated
        slices.append(Order(
            order_id=f"sub_twap_{uuid.uuid4().hex[:8]}",
            account_id=parent_order.account_id,
            symbol=parent_order.symbol,
            direction=parent_order.direction,
            order_type=parent_order.order_type,
            price=parent_order.price,
            volume=last_vol,
            algo_type=AlgoType.TWAP,
            parent_order_id=parent_order.order_id
        ))

        return [s for s in slices if s.volume > 0]
