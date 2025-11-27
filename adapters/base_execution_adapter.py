from abc import ABC, abstractmethod
from typing import List, Dict, Optional


class BaseExecutionAdapter(ABC):
    """Abstract interface for trading execution backends."""

    @abstractmethod
    def get_open_positions(self, symbol: str) -> List[Dict]:
        raise NotImplementedError

    @abstractmethod
    def place_order(
        self,
        symbol: str,
        side: str,
        size: float,
        order_type: str = "market",
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        leverage: Optional[float] = None,
    ) -> Dict:
        raise NotImplementedError

    @abstractmethod
    def cancel_all_orders(self, symbol: str) -> None:
        raise NotImplementedError
