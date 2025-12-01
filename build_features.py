from __future__ import annotations

from typing import Dict, Any, List, Optional

from config import Config
from enums import (
    Trend,
    RangePosition,
    VolatilityMode,
    SkewBias,
    TimingState,
    coerce_enum,
    enum_to_str,
)
from schemas import MarketSnapshot, validate_snapshot_dict
from risk_envelope import compute_risk_envelope, RiskEnvelope
from shapes_module import detect_shapes
from logger_utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Small numeric helpers
# ---------------------------------------------------------------------------

def _safe_last_candle(candles: List[Dict[str, float]]) -> Optional[Dict[str, float]]:
    if not candles:
        return None
    return candles[-1]


def _get_closes(candles: List[Dict[str, float]]) -> List[float]:
    closes: List[float] = []
    for c in candles:
        try:
            closes.append(float(c["close"]))
        except (KeyError, TypeError, ValueError):
            # If we hit a bad candle, just reuse the last close or 0.0
            closes.append(closes[-1] if closes else 0.0)
    return closes


def _safe_div(num: float, denom: float) -> float:
    if denom == 0:
        return 0.0
    return num / denom


# ---------------------------------------------------------------------------
# Regime / structure computation
# ---------------------------------------------------------------------------

def _compute_trend(candles: List[Dict[str, float]]) -> str:
    """Very simple trend classifier based on start vs end close."""
    if len(candles) < 2:
        return enum_to_str(Trend.UNKNOWN)

    closes = _get_closes(candles)
    first = closes[0]
    last = closes[-1]
    change = _safe_div(last - first, abs(first)) if first else 0.0

    if change > 0.005:
        return enum_to_str(Trend.UP)
    if change < -0.005:
        return enum_to_str(Trend.DOWN)
    return enum_to_str(Trend.SIDEWAYS)


def _compute_range_position(candles: List[Dict[str, float]], price: float) -> str:
    """Where current price sits inside recent high/low range."""
    if not candles:
        return enum_to_str(RangePosition.MID)

    closes = _get_closes(candles)
    low = min(closes)
    high = max(closes)

    if high == low:
        return enum_to_str(RangePosition.MID)

    pct = _safe_div(price - low, high - low)

    if pct <= 0.1:
        return enum_to_str(RangePosition.EXTREME_LOW)
    if pct <= 0.3:
        return enum_to_str(RangePosition.LOW)
    if pct >= 0.9:
        return enum_to_str(RangePosition.EXTREME_HIGH)
    if pct >= 0.7:
        return enum_to_str(RangePosition.HIGH)
    return enum_to_str(RangePosition.MID)


def _compute_volatility_mode(candles: List[Dict[str, float]]) -> str:
    """Simple realized-vol regime based on absolute bar-to-bar returns."""
    if len(candles) < 2:
        return enum_to_str(VolatilityMode.UNKNOWN)

    closes = _get_closes(candles)
    returns: List[float] = []
    for i in range(1, len(closes)):
        prev = closes[i - 1]
        cur = closes[i]
        if prev:
            returns.append(abs(cur - prev) / abs(prev))

    if not returns:
        return enum_to_str(VolatilityMode.UNKNOWN)

    avg_ret = sum(returns) / len(returns)

    if avg_ret < 0.001:
        return enum_to_str(VolatilityMode.LOW)
    if avg_ret < 0.003:
        return enum_to_str(VolatilityMode.NORMAL)
    if avg_ret < 0.007:
        return enum_to_str(VolatilityMode.HIGH)
    return enum_to_str(VolatilityMode.EXPLOSIVE)


def _compute_recent_price_path(candles: List[Dict[str, float]]) -> Dict[str, Any]:
    """Short-horizon returns + crude impulse labeling."""
    result: Dict[str, Any] = {
        "lookback_bars": len(candles),
        "ret_1": 0.0,
        "ret_5": 0.0,
        "ret_15": 0.0,
        "impulse_state": "unknown",
    }
    if not candles:
        return result

    closes = _get_closes(candles)
    last = closes[-1]

    def _ret(n: int) -> float:
        if len(closes) <= n:
            return 0.0
        base = closes[-1 - n]
        if not base:
            return 0.0
        return _safe_div(last - base, abs(base))

    result["ret_1"] = _ret(1)
    result["ret_5"] = _ret(5)
    result["ret_15"] = _ret(15)

    # Toy impulse classifier based on 5-bar return
    r5 = result["ret_5"]
    if r5 > 0.01:
        result["impulse_state"] = "ripping_up"
    elif r5 < -0.01:
        result["impulse_state"] = "ripping_down"
    elif r5 > 0.003:
        result["impulse_state"] = "grinding_up"
    elif r5 < -0.003:
        result["impulse_state"] = "grinding_down"
    else:
        result["impulse_state"] = "chop"

    return result


# ---------------------------------------------------------------------------
# Context layers: flow, liquidity, fib, timing, risk
# ---------------------------------------------------------------------------

def _build_flow_layer(market_data: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize basic flow metrics into a single dict."""
    flow: Dict[str, Any] = {
        "funding": market_data.get("funding"),
        "open_interest": market_data.get("open_interest"),
        "skew": market_data.get("skew"),
    }

    # Optional: very rough skew bias classification here, but the canonical skew_bias
    # is still derived in validate_snapshot_dict via SkewBias.
    skew_val = flow["skew"]
    try:
        skew_f = float(skew_val) if skew_val is not None else 0.0
    except (TypeError, ValueError):
        skew_f = 0.0

    if skew_f > 0.5:
        flow["skew_bias_hint"] = enum_to_str(SkewBias.CALL_DOMINANT)
    elif skew_f < -0.5:
        flow["skew_bias_hint"] = enum_to_str(SkewBias.PUT_DOMINANT)
    else:
        flow["skew_bias_hint"] = enum_to_str(SkewBias.NEUTRAL)

    return flow


def _build_liquidity_context(candles: List[Dict[str, float]]) -> Dict[str, Any]:
    """Placeholder liquidity layer (will be replaced with real depth/liquidation logic)."""
    # For now, we just export empty hints; schemas.validate_snapshot_dict knows how to
    # deal with missing values and keep the schema stable.
    return {
        "liquidity_above": None,
        "liquidity_below": None,
    }


def _build_fib_context(candles: List[Dict[str, float]]) -> Dict[str, Any]:
    """Placeholder macro/micro fib zones.

    Later this will look at higher timeframe structure; for now, we just
    distinguish 'macro_discount' / 'macro_premium' around the mid of the range.
    """
    if not candles:
        return {
            "macro_zone": "unknown",
            "micro_zone": "unknown",
        }

    closes = _get_closes(candles)
    low = min(closes)
    high = max(closes)
    mid = (low + high) / 2 if (high != low) else low
    last = closes[-1]

    if last <= mid:
        macro = "macro_discount"
    else:
        macro = "macro_premium"

    # Micro is intentionally dumb for now.
    micro = "micro_discount" if last <= mid else "micro_premium"

    return {
        "macro_zone": macro,
        "micro_zone": micro,
    }


def _compute_danger_mode(
    vol_mode: str,
    recent_path: Dict[str, Any],
) -> bool:
    """Danger flag used by gatekeeper + risk.

    For now:
    - EXPLOSIVE vol or 'ripping' impulses set danger_mode True.
    """
    if vol_mode == enum_to_str(VolatilityMode.EXPLOSIVE):
        return True

    impulse = str(recent_path.get("impulse_state", "unknown"))
    if impulse in ("ripping_up", "ripping_down"):
        return True

    return False


def _compute_timing_state(timestamp: float) -> str:
    """Very coarse timing state.

    We don't know the user's timezone or exchange session here, so v1 is simply
    'normal' unless we explicitly wire a session model later.
    """
    # Placeholder: always NORMAL for now.
    return enum_to_str(TimingState.NORMAL)


def _build_risk_context(state: Dict[str, Any]) -> Dict[str, Any]:
    """Compress account/position state into a small dict."""
    risk: Dict[str, Any] = {}

    try:
        risk["equity"] = float(state.get("equity", 10_000.0) or 10_000.0)
    except (TypeError, ValueError):
        risk["equity"] = 10_000.0

    try:
        risk["max_drawdown"] = float(state.get("max_drawdown", 0.0) or 0.0)
    except (TypeError, ValueError):
        risk["max_drawdown"] = 0.0

    ops = state.get("open_positions_summary", [])
    risk["open_positions_summary"] = ops if isinstance(ops, list) else []

    last_decision = state.get("last_gpt_decision") or {}
    last_action = last_decision.get("action", state.get("last_action", "flat"))
    if not isinstance(last_action, str) or not last_action:
        last_action = "flat"
    risk["last_action"] = last_action

    try:
        last_conf = float(
            last_decision.get("confidence", state.get("last_confidence", 0.0)) or 0.0
        )
    except (TypeError, ValueError):
        last_conf = 0.0
    risk["last_confidence"] = last_conf

    return risk


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_snapshot(config: Config, market_data: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a snapshot dict from raw market data + persistent state.

    Contract:
    - Returns a JSON-safe dict that matches the shape described in brain.txt
      and schemas.validate_snapshot_dict.
    - The rest of the pipeline (gatekeeper, GPT, risk, execution) expects a
      plain dict and uses snapshot.get(...).
    """
    candles: List[Dict[str, float]] = list(market_data.get("candles") or [])
    last_candle = _safe_last_candle(candles)

    if last_candle is None:
        price = 0.0
        ts = 0.0
    else:
        try:
            price = float(last_candle["close"])
        except (KeyError, TypeError, ValueError):
            price = 0.0
        try:
            ts = float(last_candle.get("timestamp", 0.0) or 0.0)
        except (TypeError, ValueError):
            ts = 0.0

    trend = _compute_trend(candles)
    range_pos = _compute_range_position(candles, price)
    vol_mode = _compute_volatility_mode(candles)
    recent_path = _compute_recent_price_path(candles)

    flow = _build_flow_layer(market_data)
    shapes = detect_shapes(candles)
    liquidity_context = _build_liquidity_context(candles)
    fib_context = _build_fib_context(candles)

    danger_mode = _compute_danger_mode(vol_mode, recent_path)
    timing_state = _compute_timing_state(ts)

    risk_context = _build_risk_context(state)
    try:
        equity = float(risk_context.get("equity", 0.0) or 0.0)
    except (TypeError, ValueError):
        equity = 0.0

    vol_enum = coerce_enum(vol_mode, VolatilityMode, VolatilityMode.UNKNOWN)
    timing_enum = coerce_enum(timing_state, TimingState, TimingState.UNKNOWN)

    risk_envelope: RiskEnvelope = compute_risk_envelope(
        config=config,
        equity=equity,
        volatility_mode=vol_enum,
        danger_mode=danger_mode,
        timing_state=timing_enum,
    )

    gpt_state_note = state.get("gpt_state_note")

    snap = MarketSnapshot(
        symbol=str(state.get("symbol", config.symbol)),
        timestamp=ts,
        price=price,
        trend=trend,
        range_position=range_pos,
        volatility_mode=vol_mode,
        flow=flow,
        microstructure=shapes,
        liquidity_context=liquidity_context,
        fib_context=fib_context,
        danger_mode=danger_mode,
        timing_state=timing_state,
        recent_price_path=recent_path,
        risk_context=risk_context,
        risk_envelope=risk_envelope.to_dict(),
        gpt_state_note=gpt_state_note,
    )

    # Important: callers expect a dict, not the dataclass, so we normalise via
    # to_dict() + validate_snapshot_dict().
    snap_dict = snap.to_dict()
    snap_dict["risk_envelope"] = risk_envelope.to_dict()
    return validate_snapshot_dict(snap_dict)
