import random
import time
from typing import List, Dict, Optional

from .base_data_adapter import BaseDataAdapter


class MockDataAdapter(BaseDataAdapter):
    """Synthetic data source for DEMO and testing.

    Generates a simple random walk for price and lightweight fake flow metrics.
    """

    def __init__(self, start_price: float = 30000.0):
        self._price = start_price

    def fetch_recent_candles(self, symbol: str, timeframe: str, limit: int) -> List[Dict]:
        now = int(time.time())
        candles: List[Dict] = []
        price = self._price
        for i in range(limit):
            # Simple random walk
            delta = random.uniform(-0.003, 0.003) * price
            open_price = price
            close_price = price + delta
            high_price = max(open_price, close_price) * (1 + random.uniform(0, 0.001))
            low_price = min(open_price, close_price) * (1 - random.uniform(0, 0.001))
            volume = random.uniform(1.0, 10.0)
            ts = now - (limit - i) * 60  # assume 1m candles
            candles.append(
                {
                    "timestamp": ts,
                    "open": open_price,
                    "high": high_price,
                    "low": low_price,
                    "close": close_price,
                    "volume": volume,
                }
            )
            price = close_price
        self._price = price
        return candles

    def fetch_funding(self, symbol: str) -> Optional[float]:
        return random.uniform(-0.01, 0.01)

    def fetch_open_interest(self, symbol: str) -> Optional[float]:
        return random.uniform(1000.0, 5000.0)

    def fetch_skew(self, symbol: str) -> Optional[float]:
        return random.uniform(-2.0, 2.0)

    def health_check(self) -> bool:
        return True
