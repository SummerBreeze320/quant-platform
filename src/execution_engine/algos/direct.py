from typing import List, Optional, Dict, Any
from src.execution_engine.models import Order, AlgoType
from src.execution_engine.algos.base import BaseExecutionAlgo

class DirectAlgo(BaseExecutionAlgo):
    def __init__(self):
        super().__init__(name="DIRECT")

    def slice_order(self, parent_order: Order, ctx: Optional[Dict[str, Any]] = None) -> List[Order]:
        sub = parent_order.model_copy()
        sub.algo_type = AlgoType.DIRECT
        return [sub]
