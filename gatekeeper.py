from __future__ import annotations

import time
from typing import Any, Dict, Optional

from enums import VolatilityMode, RangePosition, TimingState, coerce_enum
from logger_utils import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Tuning knobs – safe, conservative defaults
# ---------------------------------------------------------------------------

MAX_GPT_CALLS_PER_HOUR = 12          # hard cap per rolling hour
MIN_SECONDS_BETWEEN_CALLS = 60       # min wall-clock spacing
MIN_PRICE_MOVE_PCT = 0.001           # 0.1% move since last GPT snapshot

# "Interesting setup" definition
MIN_SHAPE_SCORE = 0.3                # how strong microstructure must be
EXTREME_RANGE_WEIGHT = 0.1           # extra price-move tolerance at extremes


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_float(d: Dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(d.get(key, default) or default)
    except (TypeError, ValueError):
        return default


def _extract_core_features(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Pull out just what gatekeeper cares about from the snapshot dict."""
    micro = snapshot.get("microstructure") or {}
    recent = snapshot.get("recent_price_path") or {}

    price = _get_float(snapshot, "price", 0.0)
    vol_raw = snapshot.get("volatility_mode", "unknown")
    vol_enum = coerce_enum(str(vol_raw), VolatilityMode, VolatilityMode.UNKNOWN)

    range_raw = snapshot.get("range_position", "mid")
    range_enum = coerce_enum(str(range_raw), RangePosition, RangePosition.MID)

    timing_raw = snapshot.get("timing_state", "normal")
    timing_enum = coerce_enum(str(timing_raw), TimingState, TimingState.NORMAL)

    shape_bias = str(micro.get("shape_bias", "none"))
    try:
        shape_score = float(micro.get("shape_score", 0.0) or 0.0)
    except (TypeError, ValueError):
        shape_score = 0.0

    impulse_state = str(recent.get("impulse_state", "unknown"))
    danger_mode = bool(snapshot.get("danger_mode", False))

    return {
        "price": price,
        "vol_mode": vol_enum,
        "range_pos": range_enum,
        "timing_state": timing_enum,
        "shape_bias": shape_bias,
        "shape_score": shape_score,
        "impulse_state": impulse_state,
        "danger_mode": danger_mode,
    }


def _price_move_pct(cur_price: float, ref_price: float) -> float:
    if cur_price <= 0 or ref_price <= 0:
        return 0.0
    return abs(cur_price - ref_price) / ref_price


def _prune_old_timestamps(timestamps: Any, now_ts: float) -> list[float]:
    """Keep only timestamps in the last 3600s."""
    if not isinstance(timestamps, (list, tuple)):
        return []
    cutoff = now_ts - 3600.0
    cleaned: list[float] = []
    for t in timestamps:
        try:
            ft = float(t)
        except (TypeError, ValueError):
            continue
        if ft >= cutoff:
            cleaned.append(ft)
    return cleaned


def _is_candidate_setup(cur: Dict[str, Any], prev: Optional[Dict[str, Any]]) -> bool:
    """Decide if the microstructure / regime suggests a potential setup."""
    shape_score = cur["shape_score"]
    shape_bias = cur["shape_bias"]
    range_enum = cur["range_pos"]
    vol_enum = cur["vol_mode"]
    impulse_state = cur["impulse_state"]

    # Basic "interesting" shape: some directional bias with enough score.
    interesting_shape = shape_bias in {"bull", "bear"} and shape_score >= MIN_SHAPE_SCORE

    # Range extremes can be interesting even with weaker shapes.
    at_extreme = range_enum in {RangePosition.EXTREME_LOW, RangePosition.EXTREME_HIGH}

    # Big impulse moves are interesting.
    impulsive = impulse_state in {"ripping_up", "ripping_down"}

    if not (interesting_shape or at_extreme or impulsive):
        return False

    # If we have a previous snapshot, prefer to trigger on *changes*.
    if prev is not None:
        prev_core = _extract_core_features(prev)
        prev_bias = prev_core["shape_bias"]
        prev_score = prev_core["shape_score"]
        prev_range = prev_core["range_pos"]

        # New bias, or score crossed the threshold, or range regime changed.
        bias_changed = prev_bias != shape_bias
        score_crossed = prev_score < MIN_SHAPE_SCORE <= shape_score
        range_changed = prev_range != range_enum

        if bias_changed or score_crossed or range_changed or impulsive:
            return True
        # Otherwise it's just the same setup continuing; not enough to wake GPT.
        return False

    # No previous snapshot → treat as candidate if any of the raw conditions fire.
    return True


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------

def should_call_gpt(
    snapshot: Dict[str, Any],
    prev_snapshot: Optional[Dict[str, Any]],
    state: Dict[str, Any],
) -> bool:
    """
    Decide whether to call GPT on this tick.

    Uses:
      - Hard caps on calls/hour and min spacing.
      - Price move vs last GPT snapshot.
      - Presence of a "candidate setup" in microstructure / regime.
      - Timing / danger flags to avoid useless or dangerous calls.

    Mutates state in-place when it approves a call:
      - Appends to state["gpt_call_timestamps"].
      - Updates state["last_gpt_call_walltime"].
      - Sets state["last_gpt_snapshot"] to the current snapshot dict.
    """
    now_ts = float(snapshot.get("timestamp") or time.time())

    core = _extract_core_features(snapshot)
    price = core["price"]

    # If timing state explicitly says AVOID, don't even consider GPT.
    if core["timing_state"] == TimingState.AVOID:
        logger.info("Gatekeeper: timing_state=AVOID; skipping GPT.")
        return False

    # Danger mode does *not* automatically wake GPT; it's a risk concept.
    # We still require a candidate setup or sufficient price movement.

    # --- Call frequency & spacing -------------------------------------------
    call_timestamps = _prune_old_timestamps(state.get("gpt_call_timestamps"), now_ts)
    state["gpt_call_timestamps"] = call_timestamps

    if len(call_timestamps) >= MAX_GPT_CALLS_PER_HOUR:
        logger.info(
            "Gatekeeper: max GPT calls/hour reached (%d); skipping GPT.",
            MAX_GPT_CALLS_PER_HOUR,
        )
        return False

    last_call_ts = float(state.get("last_gpt_call_walltime") or 0.0)
    if last_call_ts and (now_ts - last_call_ts) < MIN_SECONDS_BETWEEN_CALLS:
        logger.info(
            "Gatekeeper: last GPT call %.1fs ago (< %ds); skipping GPT.",
            now_ts - last_call_ts,
            MIN_SECONDS_BETWEEN_CALLS,
        )
        return False

    # --- Price move vs last GPT snapshot ------------------------------------
    last_gpt_snapshot = state.get("last_gpt_snapshot") or {}
    ref_price = _get_float(last_gpt_snapshot, "price", 0.0)

    # At range extremes we relax the price-move requirement slightly.
    range_enum = core["range_pos"]
    move_threshold = MIN_PRICE_MOVE_PCT
    if range_enum in {RangePosition.EXTREME_LOW, RangePosition.EXTREME_HIGH}:
        move_threshold *= (1.0 - EXTREME_RANGE_WEIGHT)  # e.g. 0.9x threshold

    move_pct = _price_move_pct(price, ref_price) if ref_price > 0.0 else 1.0

    # --- Microstructure / regime setup --------------------------------------
    candidate = _is_candidate_setup(core, prev_snapshot)

    if not candidate and move_pct < move_threshold:
        logger.info(
            "Gatekeeper: no candidate setup and price move %.4f < %.4f; skipping GPT.",
            move_pct,
            move_threshold,
        )
        return False

    # --- Approve GPT call ----------------------------------------------------
    call_timestamps.append(now_ts)
    state["gpt_call_timestamps"] = call_timestamps
    state["last_gpt_call_walltime"] = now_ts
    state["last_gpt_snapshot"] = snapshot

    logger.info(
        "Gatekeeper: candidate setup detected (candidate=%s, move_pct=%.4f, "
        "range=%s, vol=%s, shape_bias=%s, shape_score=%.3f); calling GPT.",
        candidate,
        move_pct,
        range_enum,
        core["vol_mode"],
        core["shape_bias"],
        core["shape_score"],
    )
    return True
