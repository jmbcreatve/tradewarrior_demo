from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from adapters.base_execution_adapter import BaseExecutionAdapter
from logger_utils import get_logger

logger = get_logger(__name__)


class HyperliquidExecutionAdapter(BaseExecutionAdapter):
    """
    Placeholder Hyperliquid execution adapter (testnet-focused).

    This class is wired into the adapter router but does NOT send real orders.
    For now:

      - place_order() logs and returns a 'not_implemented' result
      - get_open_positions() returns an empty list
      - cancel_all_orders() returns a 'not_implemented' result
      - health_check() returns False so the router prefers mock execution

    That allows us to develop wiring and risk logic without touching any
    real wallet or API yet.
    """

    def __init__(self, base_url: Optional[str] = None) -> None:
        env_url = os.getenv("HL_TESTNET_REST_URL")
        self.base_url = base_url or env_url or "https://testnet.hyperliquid.local"
        # Env var names only; we never log their contents.
        self._wallet_address_env = "HL_TESTNET_MAIN_WALLET_ADDRESS"
        self._api_key_env = "HL_TESTNET_API_WALLET_PRIVATE_KEY"

    # ------------------------------------------------------------------ #
    # BaseExecutionAdapter interface                                     #
    # ------------------------------------------------------------------ #

    def get_open_positions(self, symbol: str) -> List[Dict[str, Any]]:
        """
        Return an empty list for now; real implementation will query
        Hyperliquid testnet open positions.
        """
        logger.info(
            "HyperliquidExecutionAdapter.get_open_positions called for %s "
            "but adapter is not implemented yet. Returning [].",
            symbol,
        )
        return []

    def place_order(
        self,
        symbol: str,
        side: str,
        size: float,
        price: Optional[float] = None,
        reduce_only: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Placeholder order placement. Does NOT hit any API.

        Returns a dict compatible with what the engine expects from
        mock/example adapters, but clearly marked as not implemented.
        """
        logger.warning(
            "HyperliquidExecutionAdapter.place_order called for %s %s size=%.6f "
            "(price=%s, reduce_only=%s) but adapter is not implemented yet.",
            symbol,
            side,
            size,
            price,
            reduce_only,
        )
        return {
            "status": "not_implemented",
            "order_id": None,
            "symbol": symbol,
            "side": side,
            "size": size,
            "price": price,
            "reduce_only": reduce_only,
            "reason": "hyperliquid_execution_not_implemented",
        }

    def cancel_all_orders(self, symbol: str) -> Dict[str, Any]:
        """
        Placeholder cancel_all_orders. Does NOT hit any API.
        """
        logger.warning(
            "HyperliquidExecutionAdapter.cancel_all_orders called for %s "
            "but adapter is not implemented yet.",
            symbol,
        )
        return {
            "status": "not_implemented",
            "symbol": symbol,
            "reason": "hyperliquid_execution_not_implemented",
        }

    def health_check(self) -> bool:
        """
        Always returns False so data_router prefers mock execution adapters
        until we explicitly implement and enable Hyperliquid.
        """
        logger.info(
            "HyperliquidExecutionAdapter.health_check: returning False (not implemented)."
        )
        return False
