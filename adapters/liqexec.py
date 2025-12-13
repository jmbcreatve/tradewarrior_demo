from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from adapters.base_execution_adapter import BaseExecutionAdapter
from logger_utils import get_logger

logger = get_logger(__name__)

# Hyperliquid SDK imports
try:
    from hyperliquid.exchange import Exchange
    from hyperliquid.info import Info
    from hyperliquid.utils import constants
    from eth_account import Account
    HL_SDK_AVAILABLE = True
except ImportError:
    HL_SDK_AVAILABLE = False
    logger.warning("hyperliquid-python-sdk not installed. Run: pip install hyperliquid-python-sdk")

# Map common symbol formats to Hyperliquid format (same as data adapter)
SYMBOL_MAP = {
    "BTCUSDT": "BTC",
    "ETHUSDT": "ETH",
    "SOLUSDT": "SOL",
    "BTC-USD": "BTC",
    "ETH-USD": "ETH",
    "SOL-USD": "SOL",
    "BTC": "BTC",
    "ETH": "ETH",
    "SOL": "SOL",
}


def _normalize_symbol(symbol: str) -> str:
    """Convert various symbol formats to Hyperliquid format."""
    return SYMBOL_MAP.get(symbol.upper(), symbol.upper().replace("USDT", "").replace("-USD", ""))


def _extract_oid_from_response(resp: Dict[str, Any]) -> Optional[Any]:
    """
    Extract an order ID (oid) from common Hyperliquid responses.

    Supports responses shaped like:
      {"response": {"data": {"statuses": [{"filled": {"oid": ...}}]}}}
      {"response": {"data": {"statuses": [{"resting": {"oid": ...}}]}}}
    """
    try:
        statuses = resp.get("response", {}).get("data", {}).get("statuses", [])
        if not statuses:
            return None
        status = statuses[0]
        if "filled" in status:
            return status["filled"].get("oid")
        if "resting" in status:
            return status["resting"].get("oid")
        # Fallback: sometimes oid is top-level in status
        return status.get("oid")
    except Exception:
        return None


class HyperliquidTestnetExecutionAdapter(BaseExecutionAdapter):
    """
    Hyperliquid testnet execution adapter for placing real testnet orders.
    
    This adapter:
    - Only works when explicitly configured for testnet mode
    - Places market orders only
    - Respects RiskDecision sizing, leverage, and stop/take levels
    - On errors, logs clearly and returns "no_trade" result instead of throwing
    - Requires HL_TESTNET_PRIVATE_KEY env var
    """

    def __init__(self, use_testnet: bool = True, base_url: Optional[str] = None) -> None:
        """
        Initialize the Hyperliquid execution adapter.
        
        Args:
            use_testnet: If True, use testnet API. Default True for safety.
            base_url: Optional override for API URL (for testing)
        """
        self._exchange = None
        self._info = None
        self._account = None

        if not HL_SDK_AVAILABLE:
            logger.error("Hyperliquid SDK not available. Install with: pip install hyperliquid-python-sdk")
            self._use_testnet = use_testnet
            return
        
        # Safety: only allow testnet by default
        if not use_testnet:
            logger.error("HyperliquidTestnetExecutionAdapter: use_testnet=False is not allowed for safety. Use testnet only.")
            raise ValueError("HyperliquidTestnetExecutionAdapter only supports testnet mode")
        
        self._use_testnet = True
        
        # Get API URL
        if base_url:
            api_url = base_url
        else:
            api_url = constants.TESTNET_API_URL
        
        # Read private key from environment
        private_key = os.getenv("HL_TESTNET_PRIVATE_KEY")
        
        if not private_key:
            logger.error(
                "HyperliquidTestnetExecutionAdapter: Missing required env var. "
                "Need HL_TESTNET_PRIVATE_KEY"
            )
            self._exchange = None
            self._info = None
            self._account = None
            return
        
        # Initialize Wallet, Exchange, and Info clients
        try:
            # Create account from private key using eth_account
            self._account = Account.from_key(private_key)
            # Exchange uses account and base_url
            self._exchange = Exchange(self._account, base_url=api_url)
            self._info = Info(api_url, skip_ws=True)
            logger.info(f"HyperliquidTestnetExecutionAdapter initialized (testnet=True, url={api_url})")
        except Exception as e:
            logger.error(f"Failed to initialize Hyperliquid clients: {e}")
            self._exchange = None
            self._info = None
            self._account = None

    # ------------------------------------------------------------------ #
    # BaseExecutionAdapter interface                                     #
    # ------------------------------------------------------------------ #

    def get_open_positions(self, symbol: str) -> List[Dict[str, Any]]:
        """
        Query Hyperliquid testnet for open positions.
        
        Returns a list of position dicts with keys like:
        - symbol, side, size, entry_price, leverage, etc.
        """
        if self._info is None:
            logger.warning("HyperliquidTestnetExecutionAdapter: SDK not available, returning empty positions")
            return []
        
        try:
            hl_symbol = _normalize_symbol(symbol)
            account_address = getattr(self._account, "address", None)
            if not account_address:
                logger.warning("HyperliquidTestnetExecutionAdapter: Account not available, returning empty positions")
                return []
            user_state = self._info.user_state(account_address)
            
            if not user_state or "assetPositions" not in user_state:
                return []
            
            positions = []
            for pos in user_state.get("assetPositions", []):
                pos_info = pos.get("position", {})
                if pos_info.get("coin") == hl_symbol:
                    positions.append({
                        "symbol": symbol,
                        "side": "long" if float(pos_info.get("szi", 0)) > 0 else "short",
                        "size": abs(float(pos_info.get("szi", 0))),
                        "entry_price": float(pos_info.get("entryPx", 0)),
                        "leverage": float(pos_info.get("leverage", {}).get("value", 1)),
                        "unrealized_pnl": float(pos_info.get("unrealizedPnl", 0)),
                    })
            
            return positions
        except Exception as e:
            logger.error(f"Failed to fetch open positions for {symbol}: {e}", exc_info=True)
            return []

    def place_order(
        self,
        symbol: str,
        side: str,
        size: float,
        order_type: str = "market",
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        leverage: Optional[float] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Place a market order on Hyperliquid testnet.
        
        Args:
            symbol: Trading pair (e.g., "BTCUSDT", "BTC")
            side: "long" or "short"
            size: Position size in base asset units
            order_type: Must be "market" (only market orders supported)
            stop_loss: Optional stop loss price
            take_profit: Optional take profit price
            leverage: Optional leverage to set
            **kwargs: Additional parameters (ignored for market orders)
        
        Returns:
            Dict with status, fill_price, fee_paid, etc. On error, returns status="no_trade"
        """
        # Default result structure
        result: Dict[str, Any] = {
            "status": "no_trade",
            "symbol": symbol,
            "side": side,
            "size": size,
            "fill_price": 0.0,
            "fee_paid": 0.0,
            "realized_pnl": 0.0,
            "order_id": None,
        }
        
        # Safety checks
        if self._exchange is None or self._info is None:
            logger.error("HyperliquidTestnetExecutionAdapter: SDK not initialized, cannot place order")
            result["reason"] = "sdk_not_initialized"
            return result
        
        # Only market orders supported
        if order_type != "market":
            logger.warning(
                "HyperliquidTestnetExecutionAdapter: Only market orders supported, got order_type=%s. "
                "Refusing order.",
                order_type
            )
            result["reason"] = "unsupported_order_type"
            return result
        
        # Validate side
        side_norm = str(side).lower()
        if side_norm not in {"long", "short"}:
            logger.warning("HyperliquidTestnetExecutionAdapter: Invalid side %s, refusing order", side)
            result["reason"] = "invalid_side"
            return result
        
        # Validate size
        try:
            trade_size = float(size)
            if trade_size <= 0.0:
                logger.warning("HyperliquidTestnetExecutionAdapter: Non-positive size %s, refusing order", size)
                result["reason"] = "invalid_size"
                return result
        except (TypeError, ValueError) as e:
            logger.warning("HyperliquidTestnetExecutionAdapter: Invalid size %s: %s", size, e)
            result["reason"] = "invalid_size"
            return result
        
        # Normalize symbol
        hl_symbol = _normalize_symbol(symbol)
        
        try:
            # Set leverage if provided
            if leverage is not None and leverage > 0:
                try:
                    lev_result = self._exchange.update_leverage(int(leverage), hl_symbol)
                    if not lev_result.get("status") == "ok":
                        logger.warning(
                            "HyperliquidTestnetExecutionAdapter: Failed to set leverage %s: %s",
                            leverage, lev_result
                        )
                        # Continue anyway - leverage might already be set
                except Exception as e:
                    logger.warning("HyperliquidTestnetExecutionAdapter: Error setting leverage: %s", e)
                    # Continue anyway
            
            # Build order specification
            # Hyperliquid uses "B" for buy (long) and "A" for sell (short)
            is_buy = side_norm == "long"
            
            # Market order specification (aggressive limit IoC)
            order_result = self._exchange.market_open(
                name=hl_symbol,
                is_buy=is_buy,
                sz=trade_size,
            )
            
            # Check result
            if not order_result or order_result.get("status") != "ok":
                # Try to extract error message from response
                error_msg = "Unknown error"
                try:
                    response_data = order_result.get("response", {}).get("data", {})
                    statuses = response_data.get("statuses", [])
                    if statuses:
                        first_status = statuses[0]
                        if "resting" in first_status:
                            error_msg = first_status["resting"].get("message", "Order rejected")
                        elif "err" in first_status:
                            error_msg = first_status.get("err", "Order error")
                except Exception:
                    pass  # Use default error message
                
                logger.error(
                    "HyperliquidTestnetExecutionAdapter: Order failed for %s %s size=%.6f: %s",
                    symbol, side, trade_size, error_msg
                )
                result["reason"] = f"order_rejected: {error_msg}"
                return result
            
            # Extract order info
            order_data = order_result.get("response", {}).get("data", {})
            statuses = order_data.get("statuses", [])
            
            if not statuses:
                logger.error("HyperliquidTestnetExecutionAdapter: No status in order response")
                result["reason"] = "no_status_in_response"
                return result
            
            # Get fill information
            filled_status = statuses[0]
            if "filled" in filled_status:
                fill_info = filled_status["filled"]
                # Get average fill price
                # Hyperliquid returns totalPx (total price) and totalSz (total size)
                total_px = float(fill_info.get("totalPx", fill_info.get("totalSz", 0)))
                total_sz = float(fill_info.get("totalSz", 0))
                avg_fill_price = total_px / total_sz if total_sz > 0 else 0.0
                
                # Estimate fee (Hyperliquid charges maker/taker fees)
                # For simplicity, use a conservative estimate
                fee_rate = 0.0002  # 0.02% typical taker fee
                fee_paid = abs(trade_size * avg_fill_price * fee_rate)
                
                result.update({
                    "status": "filled",
                    "fill_price": avg_fill_price,
                    "avg_fill_price": avg_fill_price,
                    "fee_paid": fee_paid,
                    "order_id": fill_info.get("oid"),
                })
                
                logger.info(
                    "HyperliquidTestnetExecutionAdapter: Order filled %s %s size=%.6f @ %.2f",
                    symbol, side, trade_size, avg_fill_price
                )
                
                # Place protective orders (stop loss / take profit) as grouped triggers
                if stop_loss is not None or take_profit is not None:
                    try:
                        protective_resp = self._place_protective_orders(
                            hl_symbol=hl_symbol,
                            side_norm=side_norm,
                            trade_size=trade_size,
                            stop_loss=stop_loss,
                            take_profit=take_profit,
                        )
                        if protective_resp is not None:
                            result["protective_orders"] = protective_resp
                    except Exception as protect_err:
                        logger.error(
                            "HyperliquidTestnetExecutionAdapter: Protective order placement failed for %s %s: %s",
                            symbol,
                            side,
                            protect_err,
                            exc_info=True,
                        )
                        try:
                            flatten_resp = self._exchange.market_close(hl_symbol, sz=trade_size)
                        except Exception as close_err:
                            logger.critical(
                                "HyperliquidTestnetExecutionAdapter: Failed to flatten after protective order failure: %s",
                                close_err,
                                exc_info=True,
                            )
                            flatten_resp = {"status": "error", "error": str(close_err)}

                        result.update({
                            "status": "no_trade",
                            "reason": "protective_order_failure",
                            "protective_orders": {"status": "failed"},
                            "flatten_result": flatten_resp,
                        })
                        return result
                
            elif "resting" in filled_status:
                # Order is resting (limit order, not filled immediately)
                resting_info = filled_status["resting"]
                logger.warning(
                    "HyperliquidTestnetExecutionAdapter: Order is resting (not filled): %s",
                    resting_info
                )
                result.update({
                    "status": "resting",
                    "order_id": resting_info.get("oid"),
                    "reason": "order_resting",
                })
            else:
                logger.error("HyperliquidTestnetExecutionAdapter: Unexpected order status: %s", filled_status)
                result["reason"] = "unexpected_status"
                return result
            
            return result
            
        except Exception as e:
            logger.error(
                "HyperliquidTestnetExecutionAdapter: Exception placing order for %s %s size=%.6f: %s",
                symbol, side, size, e,
                exc_info=True
            )
            result["reason"] = f"exception: {str(e)}"
            return result

    def _place_protective_orders(
        self,
        hl_symbol: str,
        side_norm: str,
        trade_size: float,
        stop_loss: Optional[float],
        take_profit: Optional[float],
    ) -> Optional[Dict[str, Any]]:
        """
        Submit stop-loss / take-profit as grouped trigger orders (OCO style).
        """
        if stop_loss is None and take_profit is None:
            return None

        reduce_only = True
        close_is_buy = side_norm == "short"
        orders: List[Dict[str, Any]] = []

        if stop_loss is not None:
            sl_px = float(stop_loss)
            orders.append({
                "coin": hl_symbol,
                "is_buy": close_is_buy,
                "sz": trade_size,
                "limit_px": sl_px,
                "order_type": {"trigger": {"triggerPx": sl_px, "isMarket": True, "tpsl": "sl"}},
                "reduce_only": reduce_only,
            })

        if take_profit is not None:
            tp_px = float(take_profit)
            orders.append({
                "coin": hl_symbol,
                "is_buy": close_is_buy,
                "sz": trade_size,
                "limit_px": tp_px,
                "order_type": {"trigger": {"triggerPx": tp_px, "isMarket": True, "tpsl": "tp"}},
                "reduce_only": reduce_only,
            })

        try:
            resp = self._exchange.bulk_orders(orders, grouping="normalTpsl")
        except Exception as e:
            raise RuntimeError(f"protective_order_exception: {e}") from e

        if not resp or resp.get("status") != "ok":
            raise RuntimeError(f"protective_order_rejected: {resp}")

        return resp

    def cancel_all_orders(self, symbol: str) -> None:
        """
        Cancel all open orders for a symbol on Hyperliquid testnet.
        """
        if self._exchange is None or self._info is None:
            logger.warning("HyperliquidTestnetExecutionAdapter: SDK not initialized, cannot cancel orders")
            return
        
        try:
            hl_symbol = _normalize_symbol(symbol)
            account_address = getattr(self._account, "address", None)
            if not account_address:
                logger.warning("HyperliquidTestnetExecutionAdapter: Account not available, cannot cancel orders")
                return

            open_orders = self._info.frontend_open_orders(account_address) or []
            cancels = []
            for order in open_orders:
                if str(order.get("coin")) == hl_symbol and "oid" in order:
                    cancels.append({"coin": hl_symbol, "oid": order["oid"]})

            if not cancels:
                logger.info("HyperliquidTestnetExecutionAdapter: No open orders to cancel for %s", symbol)
                return

            result = self._exchange.bulk_cancel(cancels)
            
            if result.get("status") == "ok":
                logger.info("HyperliquidTestnetExecutionAdapter: Cancelled all orders for %s", symbol)
            else:
                logger.warning(
                    "HyperliquidTestnetExecutionAdapter: Failed to cancel orders for %s: %s",
                    symbol, result
                )
        except Exception as e:
            logger.error(
                "HyperliquidTestnetExecutionAdapter: Exception cancelling orders for %s: %s",
                symbol, e,
                exc_info=True
            )

    # ------------------------------------------------------------------ #
    # Extended helpers (TW-5 + execution utilities)                      #
    # ------------------------------------------------------------------ #

    def get_account_value_usd(self) -> float:
        """
        Return account value in USD from user_state(). Falls back to 0.0 on error.
        """
        if self._info is None or self._account is None:
            logger.warning("HyperliquidTestnetExecutionAdapter: SDK not initialized, account value unavailable")
            return 0.0
        try:
            user_state = self._info.user_state(self._account.address) or {}
            margin = user_state.get("marginSummary", {}) or {}
            acct_val = margin.get("accountValue", margin.get("accountValueCombined"))
            return float(acct_val) if acct_val is not None else 0.0
        except Exception as e:
            logger.error("HyperliquidTestnetExecutionAdapter: Failed to fetch account value: %s", e, exc_info=True)
            return 0.0

    def get_open_orders(self, symbol: str) -> List[Dict[str, Any]]:
        """
        Return open orders for a symbol (frontend_open_orders filtered by coin).
        """
        if self._info is None or self._account is None:
            logger.warning("HyperliquidTestnetExecutionAdapter: SDK not initialized, cannot fetch open orders")
            return []

        try:
            hl_symbol = _normalize_symbol(symbol)
            account_address = self._account.address
            orders = self._info.frontend_open_orders(account_address) or []
            return [o for o in orders if str(o.get("coin")) == hl_symbol]
        except Exception as e:
            logger.error("HyperliquidTestnetExecutionAdapter: Failed to fetch open orders for %s: %s", symbol, e)
            return []

    def cancel_orders(self, symbol: str, oids: List[int]) -> Dict[str, Any]:
        """
        Bulk-cancel specific order IDs for a symbol.
        """
        if self._exchange is None or self._info is None:
            logger.warning("HyperliquidTestnetExecutionAdapter: SDK not initialized, cannot cancel orders")
            return {"status": "error", "reason": "sdk_not_initialized"}

        hl_symbol = _normalize_symbol(symbol)
        cancels = [{"coin": hl_symbol, "oid": oid} for oid in oids or []]
        if not cancels:
            return {"status": "ok", "cancelled": 0}

        try:
            return self._exchange.bulk_cancel(cancels)
        except Exception as e:
            logger.error("HyperliquidTestnetExecutionAdapter: bulk_cancel failed: %s", e, exc_info=True)
            return {"status": "error", "reason": str(e)}

    def place_limit_order(
        self,
        symbol: str,
        is_buy: bool,
        sz: float,
        limit_px: float,
        reduce_only: bool = False,
        tif: str = "Gtc",
    ) -> Dict[str, Any]:
        """
        Place a limit order. Developer note: HL limit payload is order_type={"limit": {"tif": "<tif>"}}.
        """
        if self._exchange is None or self._info is None:
            logger.warning("HyperliquidTestnetExecutionAdapter: SDK not initialized, cannot place limit order")
            return {"status": "error", "reason": "sdk_not_initialized"}

        order = {
            "coin": _normalize_symbol(symbol),
            "is_buy": bool(is_buy),
            "sz": float(sz),
            "limit_px": float(limit_px),
            "reduce_only": bool(reduce_only),
            "order_type": {"limit": {"tif": tif}},
        }

        try:
            resp = self._exchange.place_order(order)
        except Exception as e:
            logger.error("HyperliquidTestnetExecutionAdapter: place_limit_order failed: %s", e, exc_info=True)
            return {"status": "error", "reason": str(e)}

        oid = _extract_oid_from_response(resp)
        if oid is not None:
            resp = dict(resp)
            resp["oid"] = oid
        return resp

    def place_trigger_order(
        self,
        symbol: str,
        is_buy: bool,
        sz: float,
        trigger_px: float,
        reduce_only: bool = True,
        tpsl: str = "sl",
        is_market: bool = True,
        limit_px: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Place a trigger order (stop/tp). Supports stop-market; attempts stop-limit if is_market=False and limit_px is set.
        """
        if self._exchange is None or self._info is None:
            logger.warning("HyperliquidTestnetExecutionAdapter: SDK not initialized, cannot place trigger order")
            return {"status": "error", "reason": "sdk_not_initialized"}

        if not is_market and limit_px is None:
            logger.info("HyperliquidTestnetExecutionAdapter: stop-limit requested without limit_px; falling back to stop-market")
            is_market = True

        limit_price = float(limit_px if limit_px is not None else trigger_px)
        order = {
            "coin": _normalize_symbol(symbol),
            "is_buy": bool(is_buy),
            "sz": float(sz),
            "limit_px": limit_price,
            "order_type": {"trigger": {"triggerPx": float(trigger_px), "isMarket": bool(is_market), "tpsl": tpsl}},
            "reduce_only": bool(reduce_only),
        }
        # Developer note: HL stop-limit uses isMarket=False + limit_px; some venues may ignore stop-limits.

        try:
            resp = self._exchange.place_order(order)
        except Exception as e:
            logger.error("HyperliquidTestnetExecutionAdapter: place_trigger_order failed: %s", e, exc_info=True)
            return {"status": "error", "reason": str(e)}

        oid = _extract_oid_from_response(resp)
        if oid is not None:
            resp = dict(resp)
            resp["oid"] = oid
        return resp

    def place_tw5_bracket(
        self,
        coin: str,
        side: str,
        total_size: float,
        stop_price: float,
        tp_levels: List[tuple],
    ) -> Dict[str, Any]:
        """
        Place a TW-5 style bracket: reduce-only stop + up to 3 reduce-only limit TPs.
        Returns a dict with stop_oid and list of tp_oids (missing if placement failed).
        """
        side_norm = (side or "").lower()
        if side_norm not in ("long", "short"):
            raise ValueError("side must be 'long' or 'short'")

        close_is_buy = side_norm == "short"

        result: Dict[str, Any] = {"stop_oid": None, "tp_oids": []}

        stop_resp = self.place_trigger_order(
            symbol=coin,
            is_buy=close_is_buy,
            sz=total_size,
            trigger_px=stop_price,
            reduce_only=True,
            tpsl="sl",
            is_market=True,
            limit_px=stop_price,
        )
        result["stop_oid"] = _extract_oid_from_response(stop_resp)

        for price, size in tp_levels:
            tp_resp = self.place_limit_order(
                symbol=coin,
                is_buy=close_is_buy,
                sz=size,
                limit_px=price,
                reduce_only=True,
                tif="Gtc",
            )
            result["tp_oids"].append(_extract_oid_from_response(tp_resp))

        return result

    def health_check(self) -> bool:
        """
        Check if the adapter can connect to Hyperliquid testnet.
        
        Returns True if:
        - SDK is available
        - Credentials are set
        - Can successfully query user state
        """
        if not HL_SDK_AVAILABLE:
            logger.error("HyperliquidTestnetExecutionAdapter health check failed: SDK not available")
            return False
        
        if self._exchange is None or self._info is None:
            logger.error(
                "HyperliquidTestnetExecutionAdapter health check failed: exchange/info not initialized "
                "(exchange=%s, info=%s)",
                bool(self._exchange),
                bool(self._info),
            )
            return False
        
        try:
            # Try to fetch user state as a health check
            if self._account is None:
                logger.error("HyperliquidTestnetExecutionAdapter health check failed: account not initialized")
                return False
            account_address = self._account.address
            user_state = self._info.user_state(account_address)
            return user_state is not None
        except Exception as e:
            logger.error(
                "HyperliquidTestnetExecutionAdapter health check exception for account %s: %s",
                getattr(self._account, "address", None),
                e,
                exc_info=True,
            )
            return False


# Backwards compatibility alias for legacy imports
HyperliquidExecutionAdapter = HyperliquidTestnetExecutionAdapter
