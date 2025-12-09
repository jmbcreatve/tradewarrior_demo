# tw5/gatekeeper.py

"""
Simple gatekeeper for when TW-5 should call GPT.

It uses:
  - snapshot sanity (price, vol_mode, trend)
  - time since last GPT call
  - calls per hour cap
  - price move vs ATR
  - range extremes (extreme_low / extreme_high)

Engine is responsible for:
  - tracking gpt_calls_last_hour
  - passing in thresholds (min_seconds_between_calls, max_calls_per_hour, min_atr_move_mult)
"""

from __future__ import annotations

from typing import Tuple

from .schemas import Tw5Snapshot


def should_call_gpt_tw5(
    snapshot: Tw5Snapshot,
    last_snapshot: Tw5Snapshot | None,
    now_ts: float,
    last_gpt_ts: float | None,
    gpt_calls_last_hour: int,
    min_seconds_between_calls: float,
    max_calls_per_hour: int,
    min_atr_move_mult: float,
) -> Tuple[bool, str]:
    """
    Decide whether to call GPT on this tick.

    Returns:
        (should_call, reason)
        - should_call: True if GPT should be called.
        - reason: short string describing the decision.

    Logic:
      1) Snapshot sanity:
         - price must be > 0
         - vol_mode must not be "unknown"
         - at least one of trend_1h / trend_4h must be known
      2) Per-hour cap:
         - if gpt_calls_last_hour >= max_calls_per_hour -> deny
      3) Min seconds between calls:
         - if last_gpt_ts is None -> allow (first call)
         - elif now_ts - last_gpt_ts < min_seconds_between_calls -> deny
      4) Trigger conditions:
         - if last_snapshot is None -> allow (first call)
         - else:
             * compute price move in ATR units
             * allow if move >= min_atr_move_mult
             * OR if range_position_7d is at an extreme (extreme_low/extreme_high)
    """

    # 1) Snapshot sanity
    if snapshot.price <= 0.0:
        return False, "deny:price_invalid"

    if snapshot.vol_mode == "unknown":
        return False, "deny:vol_unknown"

    if snapshot.trend_1h == "unknown" and snapshot.trend_4h == "unknown":
        return False, "deny:trend_unknown"

    # 2) Per-hour cap
    if max_calls_per_hour > 0 and gpt_calls_last_hour >= max_calls_per_hour:
        return False, "deny:per_hour_cap"

    # 3) Min seconds between calls
    if last_gpt_ts is None:
        # First-ever call; snapshot already sane.
        return True, "allow:first_call"

    if min_seconds_between_calls > 0 and (now_ts - last_gpt_ts) < min_seconds_between_calls:
        return False, "deny:too_soon_since_last"

    # 4) Trigger conditions
    if last_snapshot is None:
        # We haven't seen a previous snapshot (but have a last_gpt_ts for some reason).
        return True, "allow:no_last_snapshot"

    # Compute price move in ATR units
    move_reason = _price_move_trigger(snapshot, last_snapshot, min_atr_move_mult)
    if move_reason is not None:
        return True, move_reason

    # Range extremes trigger even without big move
    if snapshot.range_position_7d in ("extreme_low", "extreme_high"):
        return True, "allow:range_extreme"

    # Otherwise, no need to wake GPT on this tick
    return False, "deny:no_trigger"


def _price_move_trigger(
    snapshot: Tw5Snapshot,
    last_snapshot: Tw5Snapshot,
    min_atr_move_mult: float,
) -> str | None:
    """
    Return a reason string if price move vs ATR is large enough, else None.

    move_atr = |price_now - price_prev| / (price_now * max(atr_pct, eps))

    If atr_pct is very small or zero, we fall back to a pure % move threshold.
    """
    if min_atr_move_mult <= 0.0:
        return None

    price_now = snapshot.price
    price_prev = last_snapshot.price
    if price_now <= 0.0 or price_prev <= 0.0:
        return None

    # Avoid division by zero if atr_pct is tiny
    eps = 1e-8
    atr = max(snapshot.atr_pct, eps)

    move_frac = abs(price_now - price_prev) / (price_now * atr)

    if move_frac >= min_atr_move_mult:
        return f"allow:price_move_atr_{move_frac:.2f}_>=_{min_atr_move_mult:.2f}"

    return None
