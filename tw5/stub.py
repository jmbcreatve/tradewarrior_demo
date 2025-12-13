# tw5/stub.py

"""
TW-5 stub policy: simple fib-based trend follower.

This consumes the same Tw5Snapshot and returns an OrderPlan, allowing us
to test wiring and replay without calling GPT.

Design goals:
- Deterministic (optionally seedable, but no RNG needed in v1).
- Very simple logic:
    - Follow 4h trend (up -> long, down -> short, else flat).
    - Use fib levels as pullback entries in trend direction.
    - Stops beyond swing extremes.
    - TPs at 1R and 2R.
- Let risk_clamp decide actual notional; stub only defines structure.
"""

from __future__ import annotations

from typing import Optional, List

from .schemas import Tw5Snapshot, OrderPlan, OrderLeg


def generate_tw5_stub_plan(snapshot: Tw5Snapshot, seed: Optional[int] = None) -> OrderPlan:
    """
    Return a deterministic, simple baseline OrderPlan based on the snapshot.

    This is intentionally dumb but structured:
    - If 4h trend is "up": propose a long ladder using fib pullbacks below price.
    - If 4h trend is "down": propose a short ladder using fib pullbacks above price.
    - Otherwise: flat (no trade).
    """
    # Basic sanity checks
    if snapshot.price <= 0.0:
        return OrderPlan.empty_flat("Stub: price <= 0, refuse to trade.")

    if snapshot.swing_high <= snapshot.swing_low:
        return OrderPlan.empty_flat("Stub: degenerate swing, refuse to trade.")

    trend_4h = (snapshot.trend_4h or "").lower()
    if trend_4h == "up":
        return _build_stub_long_plan(snapshot)
    if trend_4h == "down":
        return _build_stub_short_plan(snapshot)

    # No clear higher timeframe trend -> no trade
    return OrderPlan.empty_flat(f"Stub: no clear 4h trend ({snapshot.trend_4h}).")


# ---------------------------------------------------------------------------
# Long stub: fib ladder below price
# ---------------------------------------------------------------------------


def _build_stub_long_plan(snapshot: Tw5Snapshot) -> OrderPlan:
    price = snapshot.price
    swing_low = snapshot.swing_low
    swing_high = snapshot.swing_high

    # Candidate entry levels (pullbacks) below current price.
    candidate_entries: List[float] = [
        snapshot.fib_0_382,
        snapshot.fib_0_5,
        snapshot.fib_0_618,
    ]

    entries = [
        lvl
        for lvl in candidate_entries
        if swing_low < lvl < price  # proper pullback
    ]

    # If none of the fibs are below price, fall back to a single entry at price.
    if not entries:
        entries = [price]

    # Cap to at most 2 legs for simplicity.
    if len(entries) > 2:
        # Prefer deeper pullbacks (closer to swing_low)
        entries = sorted(entries)[:2]
    else:
        entries = sorted(entries)

    # Normalize size_fracs to sum to 1.0 (stub intends full aggression; clamp will adjust).
    size_fracs = _normalized_fracs(len(entries))

    legs: List[OrderLeg] = []
    for idx, (entry_price, size_frac) in enumerate(zip(entries, size_fracs), start=1):
        stop_loss = _make_stop_long(snapshot, entry_price)
        leg = OrderLeg(
            id=f"leg_long_{idx}",
            entry_type="limit",
            entry_price=entry_price,
            entry_tag=_long_entry_tag_for_level(snapshot, entry_price),
            size_frac=size_frac,
            stop_loss=stop_loss,
            take_profits=[],  # exits are standardized by executor (ladder + trailing)
        )
        legs.append(leg)

    rationale = (
        "Stub: follow 4h uptrend with fib-based pullback entries and swing-based stops. "
        "Exit ladder/trailing handled by executor."
    )

    return OrderPlan(
        mode="enter",
        side="long",
        legs=legs,
        max_total_size_frac=1.0,
        confidence=0.5,
        rationale=rationale,
    )


def _make_stop_long(snapshot: Tw5Snapshot, entry_price: float) -> float:
    """For a long, place stop below swing_low."""
    swing_low = snapshot.swing_low
    # Base risk distance: at least 0.5% of price, but also respect ATR.
    min_risk = 0.005 * snapshot.price
    atr_risk = snapshot.atr_pct * snapshot.price * 1.5
    risk_distance = max(min_risk, atr_risk)

    # Stop should not be above swing_low; bias towards swing_low region.
    raw_stop = entry_price - risk_distance
    stop_loss = min(raw_stop, swing_low * 0.999)  # a bit below swing_low

    # Make sure stop isn't negative.
    if stop_loss <= 0:
        stop_loss = max(0.0001, swing_low * 0.999)

    per_unit_risk = entry_price - stop_loss
    if per_unit_risk <= 0:
        # Degenerate; fall back to tiny risk
        per_unit_risk = max(0.001 * snapshot.price, 1.0)
    return stop_loss


def _long_entry_tag_for_level(snapshot: Tw5Snapshot, level: float) -> str:
    """
    Provide a simple tag describing which zone this long entry level belongs to.
    """
    # Naive nearest-fib tag
    fib_levels = {
        "fib_0.382": snapshot.fib_0_382,
        "fib_0.5": snapshot.fib_0_5,
        "fib_0.618": snapshot.fib_0_618,
        "price": snapshot.price,
    }
    closest_tag = min(fib_levels, key=lambda k: abs(fib_levels[k] - level))
    return closest_tag


# ---------------------------------------------------------------------------
# Short stub: fib ladder above price
# ---------------------------------------------------------------------------


def _build_stub_short_plan(snapshot: Tw5Snapshot) -> OrderPlan:
    price = snapshot.price
    swing_low = snapshot.swing_low
    swing_high = snapshot.swing_high

    candidate_entries: List[float] = [
        snapshot.fib_0_618,
        snapshot.fib_0_5,
        snapshot.fib_0_382,
    ]

    entries = [
        lvl
        for lvl in candidate_entries
        if price < lvl < swing_high  # proper pullback above price
    ]

    # If none of the fibs are above price, fall back to a single entry at price.
    if not entries:
        entries = [price]

    # Cap to at most 2 legs.
    if len(entries) > 2:
        # Prefer closer to swing_high (higher levels)
        entries = sorted(entries, reverse=True)[:2]
    else:
        entries = sorted(entries, reverse=True)

    size_fracs = _normalized_fracs(len(entries))

    legs: List[OrderLeg] = []
    for idx, (entry_price, size_frac) in enumerate(zip(entries, size_fracs), start=1):
        stop_loss = _make_stop_short(snapshot, entry_price)
        leg = OrderLeg(
            id=f"leg_short_{idx}",
            entry_type="limit",
            entry_price=entry_price,
            entry_tag=_short_entry_tag_for_level(snapshot, entry_price),
            size_frac=size_frac,
            stop_loss=stop_loss,
            take_profits=[],  # exits are standardized by executor (ladder + trailing)
        )
        legs.append(leg)

    rationale = (
        "Stub: follow 4h downtrend with fib-based pullback shorts and swing-based stops. "
        "Exit ladder/trailing handled by executor."
    )

    return OrderPlan(
        mode="enter",
        side="short",
        legs=legs,
        max_total_size_frac=1.0,
        confidence=0.5,
        rationale=rationale,
    )


def _make_stop_short(snapshot: Tw5Snapshot, entry_price: float) -> float:
    """For a short, place stop above swing_high."""
    swing_high = snapshot.swing_high
    min_risk = 0.005 * snapshot.price
    atr_risk = snapshot.atr_pct * snapshot.price * 1.5
    risk_distance = max(min_risk, atr_risk)

    raw_stop = entry_price + risk_distance
    stop_loss = max(raw_stop, swing_high * 1.001)  # a bit above swing_high

    per_unit_risk = stop_loss - entry_price
    if per_unit_risk <= 0:
        per_unit_risk = max(0.001 * snapshot.price, 1.0)
    return stop_loss


def _short_entry_tag_for_level(snapshot: Tw5Snapshot, level: float) -> str:
    fib_levels = {
        "fib_0.618": snapshot.fib_0_618,
        "fib_0.5": snapshot.fib_0_5,
        "fib_0.382": snapshot.fib_0_382,
        "price": snapshot.price,
    }
    closest_tag = min(fib_levels, key=lambda k: abs(fib_levels[k] - level))
    return closest_tag


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _normalized_fracs(n: int) -> List[float]:
    """
    Return n positive fractions that sum to 1.0.

    For now we just split evenly; risk_clamp will still apply global caps.
    """
    if n <= 0:
        return []
    frac = 1.0 / float(n)
    # Small numerical adjustment so sum is exactly 1.0 in common cases
    fracs = [frac for _ in range(n)]
    total = sum(fracs)
    if total != 0:
        fracs[-1] += (1.0 - total)
    return fracs
