from abc import ABC, abstractmethod
from typing import List, Dict, Optional


class BaseDataAdapter(ABC):
    """Abstract interface for market data providers."""

    @abstractmethod
    def fetch_recent_candles(self, symbol: str, timeframe: str, limit: int) -> List[Dict]:
        """Return a list of candle dicts with keys:
        - timestamp (unix seconds)
        - open, high, low, close
        - volume
        """
        raise NotImplementedError

    @abstractmethod
    def fetch_funding(self, symbol: str) -> Optional[float]:
        raise NotImplementedError

    @abstractmethod
    def fetch_open_interest(self, symbol: str) -> Optional[float]:
        raise NotImplementedError

    @abstractmethod
    def fetch_skew(self, symbol: str) -> Optional[float]:
        raise NotImplementedError

    @abstractmethod
    def health_check(self) -> bool:
        """Return True if the adapter is healthy and usable."""
        raise NotImplementedError
