# tw5/snapshot_builder.py

"""
Build the tiny TW-5 snapshot panel from raw market data.

Design:
- Signature: build_tw5_snapshot(config, market_data, state) -> Tw5Snapshot
- Inputs:
    - config: Config (we use symbol, timeframe)
    - market_data: dict from data_router.get_market_data (uses ["candles"])
    - state: state dict from state_memory (for position info)
- Output:
    - Tw5Snapshot: small, human-like panel for GPT
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from logger_utils import get_logger
from .schemas import Tw5Snapshot

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------


def build_tw5_snapshot(
    config: Any,
    market_data: Dict[str, Any],
    state: Dict[str, Any],
) -> Tw5Snapshot:
    """
    Build a minimal TW-5 snapshot from config + market_data + state.

    This function:
    - Pulls candles from market_data["candles"]
    - Computes price, trends, range, swing, fibs, vol_mode, ATR-ish pct
    - Computes a crude last impulse
    - Reads position state from last_execution_result in state
    """

    candles = _get_candles_from_market_data(market_data)
    position_side, position_size, position_entry_price = _extract_position_state(state)

    if not candles:
        logger.warning("TW-5 snapshot_builder: no candles available; returning empty snapshot.")
        return _build_empty_snapshot(
            symbol=config.symbol,
            timeframe=config.timeframe,
            position_side=position_side,
            position_size=position_size,
            position_entry_price=position_entry_price,
        )

    # Price & timestamp
    price, timestamp = _get_price_and_timestamp(candles)

    # Trend windows based on timeframe
    tf_minutes = _timeframe_to_minutes(str(getattr(config, "timeframe", "1m")))
    bars_1h = max(2, int(round(60.0 / tf_minutes))) if tf_minutes > 0 else 2
    bars_4h = max(2, int(round(240.0 / tf_minutes))) if tf_minutes > 0 else 2

    trend_1h = _compute_trend_for_window(candles, bars_1h)
    trend_4h = _compute_trend_for_window(candles, bars_4h)

    # Range low/high over full window + position
    range_low_7d, range_high_7d, range_position_7d = _compute_range_and_position(candles, price)

    # Swing = full-window range for v1
    swing_low = range_low_7d
    swing_high = range_high_7d
    fib_0_382, fib_0_5, fib_0_618, fib_0_786 = _compute_fibs(swing_low, swing_high)

    # Volatility via average absolute close-to-close returns
    atr_pct, vol_mode = _compute_atr_pct_and_vol_mode(candles)

    # Simple impulse; sweeps off in v1
    last_impulse_direction, last_impulse_size_pct = _compute_last_impulse(candles, lookback_bars=5)
    swept_prev_high = False
    swept_prev_low = False

    snapshot = Tw5Snapshot(
        symbol=config.symbol,
        timeframe=str(config.timeframe),
        timestamp=timestamp,
        price=price,
        trend_1h=trend_1h,
        trend_4h=trend_4h,
        range_low_7d=range_low_7d,
        range_high_7d=range_high_7d,
        range_position_7d=range_position_7d,
        swing_low=swing_low,
        swing_high=swing_high,
        fib_0_382=fib_0_382,
        fib_0_5=fib_0_5,
        fib_0_618=fib_0_618,
        fib_0_786=fib_0_786,
        vol_mode=vol_mode,
        atr_pct=atr_pct,
        last_impulse_direction=last_impulse_direction,
        last_impulse_size_pct=last_impulse_size_pct,
        swept_prev_high=swept_prev_high,
        swept_prev_low=swept_prev_low,
        position_side=position_side,
        position_size=position_size,
        position_entry_price=position_entry_price,
    )
    return snapshot


# ---------------------------------------------------------------------------
# Helpers: candles & basic series
# ---------------------------------------------------------------------------


def _get_candles_from_market_data(market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    candles = market_data.get("candles") or []
    if not isinstance(candles, Sequence):
        return []
    out: List[Dict[str, Any]] = []
    for c in candles:
        if isinstance(c, dict):
            out.append(c)
    return out


def _get_price_and_timestamp(candles: List[Dict[str, Any]]) -> tuple[float, float]:
    if not candles:
        return 0.0, 0.0
    last = candles[-1]
    try:
        price = float(last.get("close", 0.0))
    except (TypeError, ValueError):
        price = 0.0
    try:
        ts = float(last.get("timestamp", 0.0))
    except (TypeError, ValueError):
        ts = 0.0
    return price, ts


def _get_closes(candles: List[Dict[str, Any]]) -> List[float]:
    closes: List[float] = []
    for c in candles:
        try:
            closes.append(float(c["close"]))
        except (KeyError, TypeError, ValueError):
            closes.append(closes[-1] if closes else 0.0)
    return closes


def _safe_div(num: float, denom: float, default: float = 0.0) -> float:
    try:
        if denom == 0:
            return default
        return num / denom
    except Exception:
        return default


def _timeframe_to_minutes(tf: str) -> float:
    tf = tf.strip().lower()
    if tf.endswith("m"):
        try:
            return float(tf[:-1])
        except ValueError:
            return 1.0
    if tf.endswith("h"):
        try:
            return float(tf[:-1]) * 60.0
        except ValueError:
            return 60.0
    if tf.endswith("s"):
        try:
            return float(tf[:-1]) / 60.0
        except ValueError:
            return 1.0
    # Fallback: assume 1m
    return 1.0


# ---------------------------------------------------------------------------
# Trend
# ---------------------------------------------------------------------------


def _compute_trend_for_window(candles: List[Dict[str, Any]], bars: int) -> str:
    """
    Very simple trend classifier based on start vs end close,
    using the last `bars` candles if available.
    """
    if len(candles) < 2 or bars < 2:
        return "unknown"

    window = candles[-bars:] if len(candles) >= bars else candles
    closes = _get_closes(window)
    if len(closes) < 2:
        return "unknown"

    first = closes[0]
    last = closes[-1]
    if first == 0:
        return "unknown"

    change = _safe_div(last - first, abs(first), default=0.0)

    if change > 0.005:
        return "up"
    if change < -0.005:
        return "down"
    return "sideways"


# ---------------------------------------------------------------------------
# Range & range position
# ---------------------------------------------------------------------------


def _compute_range_and_position(
    candles: List[Dict[str, Any]],
    price: float,
) -> tuple[float, float, str]:
    """
    Compute low/high of closes over full window and where current price sits.

    Range position thresholds mirror classic:
        pct <= 0.1  -> extreme_low
        pct <= 0.3  -> low
        pct >= 0.9  -> extreme_high
        pct >= 0.7  -> high
        else        -> mid
    """
    if not candles:
        return 0.0, 0.0, "mid"

    closes = _get_closes(candles)
    low = min(closes)
    high = max(closes)

    if high == low:
        return low, high, "mid"

    pct = _safe_div(price - low, high - low, default=0.0)

    if pct <= 0.1:
        rp = "extreme_low"
    elif pct <= 0.3:
        rp = "low"
    elif pct >= 0.9:
        rp = "extreme_high"
    elif pct >= 0.7:
        rp = "high"
    else:
        rp = "mid"

    return low, high, rp


# ---------------------------------------------------------------------------
# Fibs
# ---------------------------------------------------------------------------


def _compute_fibs(swing_low: float, swing_high: float) -> tuple[float, float, float, float]:
    if swing_high <= swing_low:
        # Degenerate swing; collapse fibs at low
        return swing_low, swing_low, swing_low, swing_low

    span = swing_high - swing_low
    fib_0_382 = swing_high - 0.382 * span
    fib_0_5 = swing_high - 0.5 * span
    fib_0_618 = swing_high - 0.618 * span
    fib_0_786 = swing_high - 0.786 * span
    return fib_0_382, fib_0_5, fib_0_618, fib_0_786


# ---------------------------------------------------------------------------
# Volatility & ATR-ish pct
# ---------------------------------------------------------------------------


def _compute_atr_pct_and_vol_mode(candles: List[Dict[str, Any]]) -> tuple[float, str]:
    """
    Approximate ATR% via average absolute close-to-close returns,
    and classify vol_mode with fixed thresholds.
    """
    if len(candles) < 2:
        return 0.0, "unknown"

    closes = _get_closes(candles)
    returns: List[float] = []
    for i in range(1, len(closes)):
        prev = closes[i - 1]
        cur = closes[i]
        if prev:
            returns.append(abs(cur - prev) / abs(prev))

    if not returns:
        return 0.0, "unknown"

    avg_ret = sum(returns) / len(returns)

    # Reuse classic volatility thresholds
    if avg_ret < 0.001:
        vol_mode = "low"
    elif avg_ret < 0.003:
        vol_mode = "normal"
    elif avg_ret < 0.007:
        vol_mode = "high"
    else:
        vol_mode = "explosive"

    atr_pct = avg_ret
    return atr_pct, vol_mode


# ---------------------------------------------------------------------------
# Impulse
# ---------------------------------------------------------------------------


def _compute_last_impulse(
    candles: List[Dict[str, Any]],
    lookback_bars: int = 5,
) -> tuple[str, float]:
    """
    Crude impulse over the last `lookback_bars` closes.

    Returns:
        (direction, size_pct)
        direction: "up" | "down" | "chop" | "unknown"
        size_pct: abs(change) as fraction of price
    """
    closes = _get_closes(candles)
    if len(closes) < 2:
        return "unknown", 0.0

    lookback = min(max(1, lookback_bars), len(closes) - 1)
    base = closes[-lookback - 1]
    last = closes[-1]
    if base == 0:
        return "unknown", 0.0

    change = _safe_div(last - base, abs(base), default=0.0)
    size_pct = abs(change)

    if change > 0.001:
        direction = "up"
    elif change < -0.001:
        direction = "down"
    else:
        direction = "chop"

    return direction, size_pct


# ---------------------------------------------------------------------------
# Position state
# ---------------------------------------------------------------------------


def _extract_position_state(state: Dict[str, Any]) -> tuple[str, float, Optional[float]]:
    """
    Pull a lightweight position view out of state.

    For v1 we only look at last_execution_result; adapters are responsible
    for keeping this in sync with actual open positions.
    """
    last_exec = state.get("last_execution_result") or {}

    side = str(last_exec.get("side", "flat") or "flat").lower()
    if side not in ("flat", "long", "short"):
        side = "flat"

    try:
        size = float(last_exec.get("position_size", 0.0) or 0.0)
    except (TypeError, ValueError):
        size = 0.0

    entry_price_raw = last_exec.get("fill_price")
    try:
        entry_price = float(entry_price_raw) if entry_price_raw is not None else None
    except (TypeError, ValueError):
        entry_price = None

    return side, size, entry_price


# ---------------------------------------------------------------------------
# Empty snapshot for missing data
# ---------------------------------------------------------------------------


def _build_empty_snapshot(
    symbol: str,
    timeframe: str,
    position_side: str,
    position_size: float,
    position_entry_price: Optional[float],
) -> Tw5Snapshot:
    return Tw5Snapshot(
        symbol=symbol,
        timeframe=str(timeframe),
        timestamp=0.0,
        price=0.0,
        trend_1h="unknown",
        trend_4h="unknown",
        range_low_7d=0.0,
        range_high_7d=0.0,
        range_position_7d="unknown",
        swing_low=0.0,
        swing_high=0.0,
        fib_0_382=0.0,
        fib_0_5=0.0,
        fib_0_618=0.0,
        fib_0_786=0.0,
        vol_mode="unknown",
        atr_pct=0.0,
        last_impulse_direction="unknown",
        last_impulse_size_pct=0.0,
        swept_prev_high=False,
        swept_prev_low=False,
        position_side=position_side,
        position_size=position_size,
        position_entry_price=position_entry_price,
    )
