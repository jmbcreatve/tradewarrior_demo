"""Hyperliquid data adapter for real market data."""
from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

from adapters.base_data_adapter import BaseDataAdapter
from logger_utils import get_logger

logger = get_logger(__name__)

# Hyperliquid SDK imports
try:
    from hyperliquid.info import Info
    from hyperliquid.utils import constants
    HL_SDK_AVAILABLE = True
except ImportError:
    HL_SDK_AVAILABLE = False
    logger.warning("hyperliquid-python-sdk not installed. Run: pip install hyperliquid-python-sdk")


# Map common symbol formats to Hyperliquid format
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

# Map timeframe formats
TIMEFRAME_MAP = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "1h": "1h",
    "4h": "4h",
    "1d": "1d",
}


def _normalize_symbol(symbol: str) -> str:
    """Convert various symbol formats to Hyperliquid format."""
    return SYMBOL_MAP.get(symbol.upper(), symbol.upper().replace("USDT", "").replace("-USD", ""))


def _normalize_timeframe(tf: str) -> str:
    """Convert timeframe to Hyperliquid format."""
    return TIMEFRAME_MAP.get(tf.lower(), tf.lower())


class HyperliquidDataAdapter(BaseDataAdapter):
    """
    Hyperliquid data adapter for fetching real market data.
    
    Uses the official hyperliquid-python-sdk to fetch:
    - OHLCV candles
    - Funding rates
    - Open interest
    - Mark/oracle prices
    """

    def __init__(self, use_testnet: bool = False) -> None:
        """
        Initialize the Hyperliquid data adapter.
        
        Args:
            use_testnet: If True, use testnet API. Default is mainnet for real data.
        """
        if not HL_SDK_AVAILABLE:
            logger.error("Hyperliquid SDK not available. Install with: pip install hyperliquid-python-sdk")
            self._info = None
            self._asset_map: Dict[str, int] = {}
            return
            
        api_url = constants.TESTNET_API_URL if use_testnet else constants.MAINNET_API_URL
        self._info = Info(api_url, skip_ws=True)
        self._asset_map: Dict[str, int] = {}
        self._asset_ctxs_cache: Optional[List[Dict]] = None
        self._cache_time: float = 0.0
        self._cache_ttl: float = 5.0  # Cache asset contexts for 5 seconds
        
        # Build symbol -> index map
        self._refresh_asset_map()
        
        logger.info(f"HyperliquidDataAdapter initialized (testnet={use_testnet}, url={api_url})")

    def _refresh_asset_map(self) -> None:
        """Refresh the symbol -> index mapping from meta."""
        if self._info is None:
            return
        try:
            meta = self._info.meta()
            universe = meta.get("universe", [])
            self._asset_map = {u["name"]: i for i, u in enumerate(universe)}
            logger.debug(f"Loaded {len(self._asset_map)} assets from Hyperliquid")
        except Exception as e:
            logger.warning(f"Failed to refresh asset map: {e}")

    def _get_asset_ctxs(self) -> List[Dict[str, Any]]:
        """Get asset contexts with caching."""
        if self._info is None:
            return []
            
        now = time.time()
        if self._asset_ctxs_cache is not None and (now - self._cache_time) < self._cache_ttl:
            return self._asset_ctxs_cache
            
        try:
            result = self._info.meta_and_asset_ctxs()
            if len(result) > 1:
                self._asset_ctxs_cache = result[1]
                self._cache_time = now
                return self._asset_ctxs_cache
        except Exception as e:
            logger.warning(f"Failed to fetch asset contexts: {e}")
            
        return self._asset_ctxs_cache or []

    def _get_asset_ctx(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get asset context for a specific symbol."""
        hl_symbol = _normalize_symbol(symbol)
        idx = self._asset_map.get(hl_symbol)
        if idx is None:
            logger.warning(f"Symbol {symbol} (mapped to {hl_symbol}) not found in asset map")
            return None
            
        ctxs = self._get_asset_ctxs()
        if idx < len(ctxs):
            return ctxs[idx]
        return None

    def fetch_recent_candles(
        self,
        symbol: str,
        timeframe: str,
        limit: int,
    ) -> List[Dict[str, Any]]:
        """
        Fetch recent OHLCV candles from Hyperliquid.
        
        Args:
            symbol: Trading pair (e.g., "BTCUSDT", "BTC", "BTC-USD")
            timeframe: Candle interval (e.g., "1m", "5m", "1h")
            limit: Number of candles to fetch
            
        Returns:
            List of candle dicts with keys: timestamp, open, high, low, close, volume
        """
        if self._info is None:
            logger.warning("Hyperliquid SDK not available, returning empty candles")
            return []
            
        hl_symbol = _normalize_symbol(symbol)
        hl_tf = _normalize_timeframe(timeframe)
        
        # Calculate time range
        # Hyperliquid expects milliseconds
        end_time = int(time.time() * 1000)
        
        # Calculate start time based on timeframe and limit
        tf_seconds = {
            "1m": 60,
            "5m": 300,
            "15m": 900,
            "1h": 3600,
            "4h": 14400,
            "1d": 86400,
        }
        interval_secs = tf_seconds.get(hl_tf, 60)
        start_time = end_time - (limit * interval_secs * 1000)
        
        try:
            raw_candles = self._info.candles_snapshot(hl_symbol, hl_tf, start_time, end_time)
            
            # Convert to our format
            candles = []
            for c in raw_candles:
                candles.append({
                    "timestamp": int(c["t"]) // 1000,  # Convert ms to seconds
                    "open": float(c["o"]),
                    "high": float(c["h"]),
                    "low": float(c["l"]),
                    "close": float(c["c"]),
                    "volume": float(c["v"]),
                })
            
            # Sort by timestamp and take last `limit` candles
            candles.sort(key=lambda x: x["timestamp"])
            if len(candles) > limit:
                candles = candles[-limit:]
                
            logger.debug(f"Fetched {len(candles)} candles for {hl_symbol} {hl_tf}")
            return candles
            
        except Exception as e:
            logger.error(f"Failed to fetch candles for {symbol}: {e}")
            return []

    def fetch_funding(self, symbol: str) -> Optional[float]:
        """
        Fetch current funding rate for a symbol.
        
        Returns:
            Funding rate as a float (e.g., 0.0001 = 0.01%)
        """
        ctx = self._get_asset_ctx(symbol)
        if ctx is None:
            return None
            
        try:
            return float(ctx.get("funding", 0.0))
        except (TypeError, ValueError):
            return None

    def fetch_open_interest(self, symbol: str) -> Optional[float]:
        """
        Fetch current open interest for a symbol.
        
        Returns:
            Open interest in base asset units (e.g., BTC)
        """
        ctx = self._get_asset_ctx(symbol)
        if ctx is None:
            return None
            
        try:
            return float(ctx.get("openInterest", 0.0))
        except (TypeError, ValueError):
            return None

    def fetch_skew(self, symbol: str) -> Optional[float]:
        """
        Fetch market skew (premium/discount to oracle).
        
        Returns:
            Skew as a float (negative = discount, positive = premium)
        """
        ctx = self._get_asset_ctx(symbol)
        if ctx is None:
            return None
            
        try:
            # Premium is the skew from oracle
            return float(ctx.get("premium", 0.0))
        except (TypeError, ValueError):
            return None

    def fetch_mark_price(self, symbol: str) -> Optional[float]:
        """Fetch current mark price."""
        ctx = self._get_asset_ctx(symbol)
        if ctx is None:
            return None
            
        try:
            return float(ctx.get("markPx", 0.0))
        except (TypeError, ValueError):
            return None

    def fetch_oracle_price(self, symbol: str) -> Optional[float]:
        """Fetch current oracle price."""
        ctx = self._get_asset_ctx(symbol)
        if ctx is None:
            return None
            
        try:
            return float(ctx.get("oraclePx", 0.0))
        except (TypeError, ValueError):
            return None

    def health_check(self) -> bool:
        """
        Check if the adapter can connect to Hyperliquid.
        
        Returns:
            True if healthy and can fetch data.
        """
        if self._info is None:
            return False
            
        try:
            # Try to fetch meta as a health check
            meta = self._info.meta()
            return bool(meta and meta.get("universe"))
        except Exception as e:
            logger.warning(f"Hyperliquid health check failed: {e}")
            return False
