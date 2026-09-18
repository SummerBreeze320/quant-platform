from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from src.execution_engine.models import Order

class BaseExecutionAlgo(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def slice_order(self, parent_order: Order, ctx: Optional[Dict[str, Any]] = None) -> List[Order]:
        pass
