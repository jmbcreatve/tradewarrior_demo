from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from config import Config
from enums import TimingState, VolatilityMode


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


@dataclass
class RiskEnvelope:
    max_notional: float
    max_leverage: float
    max_risk_per_trade_pct: float
    min_stop_distance_pct: float
    max_stop_distance_pct: float
    max_daily_loss_pct: float
    note: str

    def to_dict(self) -> dict:
        return asdict(self)


def compute_risk_envelope(
    config: Config,
    equity: float,
    volatility_mode: VolatilityMode,
    danger_mode: bool,
    timing_state: TimingState,
) -> RiskEnvelope:
    """
    Build a conservative risk envelope for the current regime.

    The envelope never exceeds config caps and defaults to the safest posture
    when given unknown enums or danger conditions.
    """

    base_max_leverage = max(_safe_float(getattr(config, "max_leverage", 0.0)), 0.0)
    base_risk_pct = max(_safe_float(getattr(config, "risk_per_trade", 0.0)), 0.0)

    effective_leverage = base_max_leverage
    effective_risk_pct = base_risk_pct
    note_parts = []

    vol = volatility_mode if isinstance(volatility_mode, VolatilityMode) else VolatilityMode.UNKNOWN
    timing = timing_state if isinstance(timing_state, TimingState) else TimingState.UNKNOWN

    if danger_mode:
        effective_leverage = 0.0
        effective_risk_pct = 0.0
        note_parts.append("danger_mode")
    else:
        # Volatility trims risk; unknown treated as high vol to stay safe.
        if vol in {VolatilityMode.HIGH, VolatilityMode.EXPLOSIVE, VolatilityMode.UNKNOWN}:
            effective_risk_pct *= 0.5
            effective_leverage *= 0.7
            note_parts.append("trim_for_vol")
        else:
            note_parts.append("baseline_vol")

        # Timing: avoid kills risk; cautious trims; unknown trims modestly.
        if timing == TimingState.AVOID:
            effective_risk_pct = 0.0
            effective_leverage = 0.0
            note_parts.append("timing_avoid")
        elif timing == TimingState.CAUTIOUS:
            effective_risk_pct *= 0.5
            effective_leverage *= 0.8
            note_parts.append("timing_cautious")
        elif timing == TimingState.UNKNOWN:
            effective_risk_pct *= 0.7
            effective_leverage *= 0.8
            note_parts.append("timing_unknown")
        elif timing == TimingState.AGGRESSIVE:
            note_parts.append("timing_aggressive")
        else:
            note_parts.append("timing_normal")

    # Enforce caps and floors.
    effective_leverage = min(max(effective_leverage, 0.0), base_max_leverage)
    effective_risk_pct = min(max(effective_risk_pct, 0.0), base_risk_pct)

    equity_value = max(_safe_float(equity, 0.0), 0.0)
    max_notional = equity_value * effective_leverage

    # Base stop distances (fractions of price).
    min_stop_pct = 0.005  # 0.5%
    max_stop_pct = 0.03   # 3%

    stop_tighten_mult = 1.0
    if danger_mode or timing == TimingState.AVOID:
        stop_tighten_mult = 0.7
    elif vol in {VolatilityMode.HIGH, VolatilityMode.EXPLOSIVE, VolatilityMode.UNKNOWN}:
        stop_tighten_mult = 0.8
    elif timing == TimingState.CAUTIOUS or timing == TimingState.UNKNOWN:
        stop_tighten_mult = 0.85

    min_stop_pct = max(0.002, min_stop_pct * stop_tighten_mult)
    max_stop_pct = min(0.05, max_stop_pct * stop_tighten_mult)
    max_stop_pct = max(max_stop_pct, min_stop_pct * 2.0)

    max_daily_loss_pct = 0.03  # 3% daily loss cap

    if not note_parts:
        note_parts.append("baseline")
    note = ";".join(note_parts)

    return RiskEnvelope(
        max_notional=max_notional,
        max_leverage=effective_leverage,
        max_risk_per_trade_pct=effective_risk_pct,
        min_stop_distance_pct=min_stop_pct,
        max_stop_distance_pct=max_stop_pct,
        max_daily_loss_pct=max_daily_loss_pct,
        note=note,
    )
