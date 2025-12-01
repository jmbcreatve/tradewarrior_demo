from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from adapters.base_data_adapter import BaseDataAdapter
from logger_utils import get_logger

logger = get_logger(__name__)


class HyperliquidDataAdapter(BaseDataAdapter):
    """
    Placeholder Hyperliquid data adapter (testnet-focused).

    This class is wired into the adapter router but does not yet make
    real HTTP/WebSocket calls. For now:

      - health_check() always returns False
      - fetch_* methods log a warning and return empty/dummy values

    That means data_router will fall back to mock/example adapters
    until we implement real connectivity.
    """

    def __init__(self, base_url: Optional[str] = None) -> None:
        # Allow overriding from env; keep default as a clearly fake placeholder.
        env_url = os.getenv("HL_TESTNET_REST_URL")
        self.base_url = base_url or env_url or "https://testnet.hyperliquid.local"
        # Wallet/env hints for future use (we do NOT log secrets).
        self._wallet_address_env = "HL_TESTNET_MAIN_WALLET_ADDRESS"
        self._api_key_env = "HL_TESTNET_API_WALLET_PRIVATE_KEY"

    # ------------------------------------------------------------------ #
    # BaseDataAdapter interface                                          #
    # ------------------------------------------------------------------ #

    def fetch_recent_candles(
        self,
        symbol: str,
        timeframe: str,
        limit: int,
    ) -> List[Dict[str, Any]]:
        logger.warning(
            "HyperliquidDataAdapter.fetch_recent_candles called for %s %s "
            "but adapter is not implemented yet. Returning empty list.",
            symbol,
            timeframe,
        )
        return []

    def fetch_funding(self, symbol: str) -> Optional[float]:
        logger.warning(
            "HyperliquidDataAdapter.fetch_funding called for %s but adapter "
            "is not implemented yet. Returning None.",
            symbol,
        )
        return None

    def fetch_open_interest(self, symbol: str) -> Optional[float]:
        logger.warning(
            "HyperliquidDataAdapter.fetch_open_interest called for %s but adapter "
            "is not implemented yet. Returning None.",
            symbol,
        )
        return None

    def fetch_skew(self, symbol: str) -> Optional[float]:
        logger.warning(
            "HyperliquidDataAdapter.fetch_skew called for %s but adapter "
            "is not implemented yet. Returning None.",
            symbol,
        )
        return None

    def health_check(self) -> bool:
        """
        Return False so the router treats this adapter as unavailable until
        we actually implement real connectivity.
        """
        logger.info(
            "HyperliquidDataAdapter.health_check: returning False (not implemented)."
        )
        return False
