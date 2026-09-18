from abc import ABC, abstractmethod
from typing import Optional, Dict
from src.execution_engine.models import Order, Trade, AccountState, Position

class BaseBrokerGateway(ABC):
    @abstractmethod
    def submit_order(self, order: Order) -> Optional[Trade]:
        pass

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        pass

    @abstractmethod
    def get_account(self, account_id: str) -> AccountState:
        pass
