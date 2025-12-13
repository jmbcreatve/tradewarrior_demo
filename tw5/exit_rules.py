"""Pure helpers for TW-5 exit logic (TP ladder + trailing stops)."""

from __future__ import annotations

import math
from typing import Iterable, List, Sequence, Tuple

from config import Config


def _normalize_side(side: str) -> str:
    normalized = (side or "").lower()
    if normalized not in ("long", "short"):
        raise ValueError(f"side must be 'long' or 'short', got {side}")
    return normalized


def _to_float(value: float | int) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("value must be numeric") from exc


def compute_tp_absolute_fracs_from_remaining(remaining_fracs: Sequence[float]) -> List[float]:
    """
    Convert human-friendly “close % of remaining” fractions into absolute fractions of the original size.

    Example:
        remaining_fracs [0.30, 0.30, 1.00] -> absolute [0.30, 0.21, 0.49]

    Validates bounds and ensures the absolute fractions sum to ~1.0.
    """
    if not isinstance(remaining_fracs, Iterable) or isinstance(remaining_fracs, (str, bytes)):
        raise ValueError("remaining_fracs must be an iterable of numbers")

    abs_fracs: List[float] = []
    remaining = 1.0
    for frac in remaining_fracs:
        f = _to_float(frac)
        if f < 0.0 or f > 1.0:
            raise ValueError("remaining fractions must be between 0 and 1")
        take = remaining * f
        abs_fracs.append(take)
        remaining -= take

    total = sum(abs_fracs)
    if total <= 0.0:
        raise ValueError("sum of absolute take-profit fractions must be positive")

    if math.isclose(total, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        # Snap tiny numeric drift to exactly 1.0 for downstream clamps.
        abs_fracs[-1] += 1.0 - total
        return abs_fracs

    raise ValueError(f"take-profit fractions must sum to ~1.0 (got {total:.6f})")


def compute_tp_levels(
    entry: float,
    stop: float,
    side: str,
    r_mults: Sequence[float],
    remaining_fracs: Sequence[float],
) -> List[Tuple[float, float, str]]:
    """
    Build TP levels expressed as absolute prices and absolute size fractions of the original leg.
    """
    entry_f = _to_float(entry)
    stop_f = _to_float(stop)
    R = abs(entry_f - stop_f)
    if R <= 0.0:
        raise ValueError("R distance must be positive")

    norm_side = _normalize_side(side)
    direction = 1.0 if norm_side == "long" else -1.0

    if not r_mults:
        raise ValueError("r_mults must be non-empty")

    abs_fracs = compute_tp_absolute_fracs_from_remaining(remaining_fracs)
    if len(abs_fracs) != len(r_mults):
        raise ValueError("r_mults and remaining_fracs must have the same length")

    levels: List[Tuple[float, float, str]] = []
    for r_mult, abs_frac in zip(r_mults, abs_fracs):
        rm = _to_float(r_mult)
        if rm <= 0.0:
            raise ValueError("r_multipliers must be positive")
        price = entry_f + direction * rm * R
        tag = f"{rm:g}R"
        levels.append((price, abs_frac, tag))

    return levels


def compute_trailing_stop(
    entry: float,
    initial_stop: float,
    side: str,
    R: float,
    high_water_price: float,
    tp1_hit: bool,
    tp2_hit: bool,
    cfg: Config,
) -> float:
    """
    Compute a monotonic (tightening-only) trailing stop based on TW-5 config.
    """
    entry_f = _to_float(entry)
    stop_f = _to_float(initial_stop)
    R_f = _to_float(R)
    high_water_f = _to_float(high_water_price)
    if R_f <= 0.0:
        raise ValueError("R must be positive")

    norm_side = _normalize_side(side)
    direction = 1.0 if norm_side == "long" else -1.0

    def tighten(current: float, proposed: float) -> float:
        """Move stop closer to entry, never farther."""
        if direction > 0:
            return max(current, proposed)
        return min(current, proposed)

    high_water_R = direction * (high_water_f - entry_f) / R_f

    stop_out = stop_f

    early_trigger = _to_float(getattr(cfg, "tw5_trail_early_trigger_r", 0.0))
    early_stop_r = _to_float(getattr(cfg, "tw5_trail_early_stop_r", 0.0))
    if high_water_R >= early_trigger:
        early_stop = entry_f + direction * early_stop_r * R_f
        stop_out = tighten(stop_out, early_stop)

    if tp1_hit:
        after_tp1_stop_r = _to_float(getattr(cfg, "tw5_trail_after_tp1_stop_r", 0.0))
        tp1_stop = entry_f + direction * after_tp1_stop_r * R_f
        stop_out = tighten(stop_out, tp1_stop)

    if tp2_hit:
        after_tp2_stop_r = _to_float(getattr(cfg, "tw5_trail_after_tp2_stop_r", 0.0))
        giveback_r = _to_float(getattr(cfg, "tw5_trail_runner_giveback_r", 0.0))
        runner_floor_r = max(after_tp2_stop_r, high_water_R - giveback_r)
        tp2_stop = entry_f + direction * runner_floor_r * R_f
        stop_out = tighten(stop_out, tp2_stop)

    return stop_out
