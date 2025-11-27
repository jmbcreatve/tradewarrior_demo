from typing import Dict, Any, List

from config import Config
from enums import Trend, RangePosition, VolatilityMode, SkewBias, TimingState, enum_to_str
from schemas import MarketSnapshot, validate_snapshot_dict
from shapes_module import detect_shapes
from logger_utils import get_logger

logger = get_logger(__name__)


def _compute_trend(candles: List[Dict[str, float]]) -> str:
    if len(candles) < 2:
        return enum_to_str(Trend.UNKNOWN)
    first = candles[0]["close"]
    last = candles[-1]["close"]
    change = (last - first) / max(first, 1e-8)
    if change > 0.005:
        return enum_to_str(Trend.UP)
    if change < -0.005:
        return enum_to_str(Trend.DOWN)
    return enum_to_str(Trend.SIDEWAYS)


def _compute_range_position(candles: List[Dict[str, float]], price: float) -> str:
    closes = [c["close"] for c in candles]
    low = min(closes)
    high = max(closes)
    if high == low:
        return enum_to_str(RangePosition.UNKNOWN)
    pct = (price - low) / (high - low)
    if pct < 0.2:
        return enum_to_str(RangePosition.EXTREME_LOW)
    if pct < 0.4:
        return enum_to_str(RangePosition.LOW)
    if pct > 0.8:
        return enum_to_str(RangePosition.EXTREME_HIGH)
    if pct > 0.6:
        return enum_to_str(RangePosition.HIGH)
    return enum_to_str(RangePosition.MID)


def _compute_volatility_mode(candles: List[Dict[str, float]]) -> str:
    if len(candles) < 2:
        return enum_to_str(VolatilityMode.UNKNOWN)
    returns = []
    for i in range(1, len(candles)):
        prev = candles[i - 1]["close"]
        cur = candles[i]["close"]
        returns.append(abs(cur - prev) / max(prev, 1e-8))
    avg_ret = sum(returns) / len(returns)
    if avg_ret < 0.001:
        return enum_to_str(VolatilityMode.LOW)
    if avg_ret < 0.003:
        return enum_to_str(VolatilityMode.NORMAL)
    if avg_ret < 0.007:
        return enum_to_str(VolatilityMode.HIGH)
    return enum_to_str(VolatilityMode.EXPLOSIVE)


def _compute_recent_price_path(candles: List[Dict[str, float]]) -> Dict[str, Any]:
    """Summarise recent path so GPT sees a 'mini movie', not just the last tick."""
    n = len(candles)
    if n == 0:
        return {
            "lookback_bars": 0,
            "ret_1": 0.0,
            "ret_5": 0.0,
            "ret_15": 0.0,
            "impulse_state": "unknown",
        }

    closes = [c["close"] for c in candles]
    last = closes[-1]

    def ret_over(k: int) -> float:
        if n <= k:
            return 0.0
        base = closes[-1 - k]
        return (last - base) / max(base, 1e-8)

    ret_1 = ret_over(1)
    ret_5 = ret_over(5)
    ret_15 = ret_over(15)

    # Simple impulse classifier.
    big = 0.01
    med = 0.003

    if ret_5 > big or ret_15 > big:
        impulse = "ripping_up"
    elif ret_5 < -big or ret_15 < -big:
        impulse = "ripping_down"
    elif ret_5 > med and ret_15 > med:
        impulse = "grinding_up"
    elif ret_5 < -med and ret_15 < -med:
        impulse = "grinding_down"
    else:
        impulse = "chop"

    return {
        "lookback_bars": n,
        "ret_1": ret_1,
        "ret_5": ret_5,
        "ret_15": ret_15,
        "impulse_state": impulse,
    }


def _build_flow(market_data: Dict[str, Any]) -> Dict[str, Any]:
    funding = market_data.get("funding")
    oi = market_data.get("open_interest")
    skew = market_data.get("skew")

    return {
        "funding": funding,
        "funding_trend": "unknown",
        "open_interest": oi,
        "oi_trend": "unknown",
        "skew": skew,
        "skew_bias": enum_to_str(SkewBias.UNKNOWN),
    }


def _build_liquidity_context(market_data: Dict[str, Any]) -> Dict[str, Any]:
    # Placeholder: later we can plug in real liquidity maps.
    return {
        "liquidity_above": None,
        "liquidity_below": None,
    }


def _build_fib_context(market_data: Dict[str, Any]) -> Dict[str, Any]:
    # Placeholder: later we can plug in real fib/flush zones.
    return {
        "macro_zone": "unknown",
        "micro_zone": "unknown",
    }


def build_snapshot(config: Config, market_data: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
    """Combine market data + shapes + state into a rich snapshot dict.

    This is what we hand to GPT (after validate_snapshot_dict), and it should encode
    both *where* we are and *how we got here recently* plus account context.
    """
    candles = market_data.get("candles") or []
    if not candles:
        logger.warning("No candles in market_data; building minimal snapshot.")
        price = 0.0
        timestamp = 0.0
    else:
        price = candles[-1]["close"]
        timestamp = float(candles[-1].get("timestamp", 0.0))

    shapes = detect_shapes(candles)

    trend = _compute_trend(candles)
    range_position = _compute_range_position(candles, price)
    vol_mode = _compute_volatility_mode(candles)
    recent_path = _compute_recent_price_path(candles)

    flow = _build_flow(market_data)
    liquidity_context = _build_liquidity_context(market_data)
    fib_context = _build_fib_context(market_data)

    # Basic account / risk context from state.
    equity = float(state.get("equity", 0.0) or 0.0)
    max_dd = float(state.get("max_drawdown", 0.0) or 0.0)
    open_positions_summary = state.get("open_positions_summary") or []
    last_decision = state.get("last_gpt_decision") or {}
    last_action = str(last_decision.get("action", "flat"))
    try:
        last_conf = float(last_decision.get("confidence", 0.0) or 0.0)
    except (TypeError, ValueError):
        last_conf = 0.0

    risk_context = {
        "equity": equity,
        "max_drawdown": max_dd,
        "open_positions_summary": open_positions_summary,
        "last_action": last_action,
        "last_confidence": last_conf,
    }

    gpt_state_note = state.get("gpt_state_note")

    # Simple danger heuristic: treat explosive volatility as dangerous.
    danger_mode = vol_mode == enum_to_str(VolatilityMode.EXPLOSIVE)

    # For now, timing_state is unknown; later we can map real session timing.
    timing_state = enum_to_str(TimingState.UNKNOWN)

    snap = MarketSnapshot(
        symbol=state.get("symbol", config.symbol),
        timestamp=timestamp,
        price=price,
        trend=trend,
        range_position=range_position,
        volatility_mode=vol_mode,
        flow=flow,
        microstructure=shapes,
        liquidity_context=liquidity_context,
        fib_context=fib_context,
        danger_mode=danger_mode,
        timing_state=timing_state,
        recent_price_path=recent_path,
        risk_context=risk_context,
        gpt_state_note=gpt_state_note,
    )

    snap_dict = snap.to_dict()
    return validate_snapshot_dict(snap_dict)
