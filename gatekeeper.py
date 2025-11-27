from __future__ import annotations

import time
from typing import Dict, Any, Optional

from logger_utils import get_logger

logger = get_logger(__name__)

# --- Cost protection (NO fixed delay, just hourly cap) -----------------------

GPT_HOURLY_WINDOW_SECONDS = 60.0 * 60.0
MAX_GPT_CALLS_PER_HOUR = 40  # hard safety rail

# How different price must be from the last GPT call before we bother again.
MIN_PRICE_MOVE_PCT_FROM_LAST_CALL = 0.001  # 0.1% (was 0.3%)

# What we consider a “strong enough” microstructure impulse for DEMO data.
STRONG_SHAPE_SCORE = 0.3  # DEMO mostly uses 0.3 and 0.7; 0.3 is now allowed


def _get_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _extract_core_features(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Pull out the fields we care about for gating."""
    micro = snapshot.get("microstructure") or {}

    return {
        "timestamp": _get_float(snapshot.get("timestamp"), 0.0),
        "price": _get_float(snapshot.get("price"), 0.0),
        "trend": (snapshot.get("trend") or "unknown").lower(),
        "range_position": (snapshot.get("range_position") or "unknown").lower(),
        "volatility_mode": (snapshot.get("volatility_mode") or "unknown").lower(),
        "danger_mode": bool(snapshot.get("danger_mode")),
        "shape_bias": (micro.get("shape_bias") or "none").lower(),
        "shape_score": _get_float(micro.get("shape_score"), 0.0),
    }


def _update_call_timestamps(state: Dict[str, Any], now_wall: float) -> None:
    calls = [
        _get_float(t)
        for t in state.get("gpt_call_timestamps", [])
        if now_wall - _get_float(t) < GPT_HOURLY_WINDOW_SECONDS
    ]
    calls.append(now_wall)
    state["gpt_call_timestamps"] = calls
    state["last_gpt_call_walltime"] = now_wall


def _is_potential_setup(features: Dict[str, Any]) -> bool:
    """Heuristic: is this a *candidate* long/short window?

    This version is deliberately more permissive for DEMO:
    - allows shape_score >= 0.3
    - allows mids as well as extremes
    """

    if features["danger_mode"]:
        # Later we can add special "panic / hedge" logic; for now, avoid.
        return False

    vol = features["volatility_mode"]
    if vol == "unknown":
        return False  # we don't know enough

    shape_bias = features["shape_bias"]
    shape_score = features["shape_score"]

    if shape_bias not in ("bull", "bear"):
        return False
    if shape_score < STRONG_SHAPE_SCORE:
        # For demo, 0.3 is "acceptable", <0.3 is noise.
        return False

    range_pos = features["range_position"]
    trend = features["trend"]

    # Long-ish candidate: bull impulse from low/mid in up/sideways trend.
    long_like = (
        shape_bias == "bull"
        and range_pos in ("extreme_low", "low", "mid")
        and trend in ("up", "sideways", "unknown")
    )

    # Short-ish candidate: bear impulse from high/mid in down/sideways trend.
    short_like = (
        shape_bias == "bear"
        and range_pos in ("extreme_high", "high", "mid")
        and trend in ("down", "sideways", "unknown")
    )

    return long_like or short_like


def should_call_gpt(
    snapshot: Dict[str, Any],
    prev_snapshot: Optional[Dict[str, Any]],
    state: Dict[str, Any],
) -> bool:
    """Gatekeeper that opens based on market conditions, not a fixed timer.

    Rules:

    1) Snapshot timestamp must advance.
    2) Do not exceed MAX_GPT_CALLS_PER_HOUR.
    3) Require a "potential setup" (location + impulse + volatility).
    4) Avoid calling again if price hasn't moved enough since last GPT call.
    """
    cur_features = _extract_core_features(snapshot)
    cur_ts = cur_features["timestamp"]

    prev_ts = _get_float(prev_snapshot.get("timestamp"), 0.0) if prev_snapshot else None
    if prev_ts is not None and cur_ts <= prev_ts:
        logger.info(
            "Gatekeeper: snapshot timestamp not advanced (prev=%s, cur=%s); skipping GPT.",
            prev_ts,
            cur_ts,
        )
        return False

    now_wall = time.time()

    # --- Hourly cap (cost protection only, no fixed delay) -------------------
    calls = [
        _get_float(t)
        for t in state.get("gpt_call_timestamps", [])
        if now_wall - _get_float(t) < GPT_HOURLY_WINDOW_SECONDS
    ]
    if len(calls) >= MAX_GPT_CALLS_PER_HOUR:
        logger.info(
            "Gatekeeper: hourly GPT call cap hit (%s >= %s); skipping GPT.",
            len(calls),
            MAX_GPT_CALLS_PER_HOUR,
        )
        state["gpt_call_timestamps"] = calls
        return False

    # --- First-ever call: only if this looks like a setup --------------------
    last_gpt_snapshot = state.get("last_gpt_snapshot")
    if last_gpt_snapshot is None:
        if _is_potential_setup(cur_features):
            logger.info("Gatekeeper: first candidate setup detected; calling GPT.")
            state["last_gpt_snapshot"] = cur_features
            _update_call_timestamps(state, now_wall)
            return True
        logger.info("Gatekeeper: first snapshot but no setup yet; skipping GPT.")
        return False

    # --- From here on, decide based on setups + change vs last GPT snapshot ---

    if not _is_potential_setup(cur_features):
        logger.info("Gatekeeper: no candidate setup in current snapshot; skipping GPT.")
        return False

    # Compare to the last snapshot we *actually* called GPT on.
    prev_features = last_gpt_snapshot
    price = cur_features["price"]
    prev_price = _get_float(prev_features.get("price"), 0.0)

    price_move_pct = 1.0
    if prev_price > 0:
        price_move_pct = abs(price - prev_price) / prev_price

    same_direction = cur_features["shape_bias"] == prev_features.get("shape_bias")
    same_range_bucket = cur_features["range_position"] == prev_features.get("range_position")

    if price_move_pct < MIN_PRICE_MOVE_PCT_FROM_LAST_CALL and same_direction and same_range_bucket:
        logger.info(
            "Gatekeeper: setup looks similar to last GPT call; "
            "price_move_pct=%.4f < %.4f; skipping.",
            price_move_pct,
            MIN_PRICE_MOVE_PCT_FROM_LAST_CALL,
        )
        return False

    logger.info(
        "Gatekeeper: candidate setup detected; calling GPT "
        "(price_move_pct=%.4f, shape_bias=%s, range_position=%s, vol=%s, trend=%s).",
        price_move_pct,
        cur_features["shape_bias"],
        cur_features["range_position"],
        cur_features["volatility_mode"],
        cur_features["trend"],
    )

    state["last_gpt_snapshot"] = cur_features
    _update_call_timestamps(state, now_wall)
    return True
