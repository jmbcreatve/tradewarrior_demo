import os
from typing import List, Dict, Optional

from .base_data_adapter import BaseDataAdapter


class ExampleDataAdapter(BaseDataAdapter):
    """Template for a real exchange or data provider adapter.

    This class does NOT implement a real API. It shows how you would:
    - Read API keys from environment variables.
    - Normalize external candles into the internal candle format.
    - Implement funding / OI / skew fetches.
    """

    def __init__(self):
        self.api_key = os.getenv("EXAMPLE_DATA_API_KEY", None)

    def fetch_recent_candles(self, symbol: str, timeframe: str, limit: int) -> List[Dict]:
        if not self.api_key:
            raise RuntimeError("EXAMPLE_DATA_API_KEY missing; cannot use ExampleDataAdapter.")
        # Placeholder: call real REST/WebSocket API here.
        raise NotImplementedError("ExampleDataAdapter is a template; implement your venue here.")

    def fetch_funding(self, symbol: str) -> Optional[float]:
        if not self.api_key:
            return None
        raise NotImplementedError

    def fetch_open_interest(self, symbol: str) -> Optional[float]:
        if not self.api_key:
            return None
        raise NotImplementedError

    def fetch_skew(self, symbol: str) -> Optional[float]:
        if not self.api_key:
            return None
        raise NotImplementedError

    def health_check(self) -> bool:
        return bool(self.api_key)
