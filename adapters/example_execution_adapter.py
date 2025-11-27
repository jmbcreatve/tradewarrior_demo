import os
from typing import List, Dict, Optional

from .base_execution_adapter import BaseExecutionAdapter


class ExampleExecutionAdapter(BaseExecutionAdapter):
    """Template for a real execution venue.

    Reads API keys from env, but does not implement real network calls.
    """

    def __init__(self):
        self.api_key = os.getenv("EXAMPLE_EXEC_API_KEY", None)
        self.api_secret = os.getenv("EXAMPLE_EXEC_API_SECRET", None)

    def get_open_positions(self, symbol: str) -> List[Dict]:
        if not self.api_key or not self.api_secret:
            raise RuntimeError("Missing EXAMPLE_EXEC_API_KEY/SECRET")
        raise NotImplementedError

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
        if not self.api_key or not self.api_secret:
            raise RuntimeError("Missing EXAMPLE_EXEC_API_KEY/SECRET")
        raise NotImplementedError

    def cancel_all_orders(self, symbol: str) -> None:
        if not self.api_key or not self.api_secret:
            raise RuntimeError("Missing EXAMPLE_EXEC_API_KEY/SECRET")
        raise NotImplementedError
