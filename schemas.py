from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional

from enums import (
    Trend,
    RangePosition,
    VolatilityMode,
    SkewBias,
    TimingState,
    enum_to_str,
    coerce_enum,
)


@dataclass
class MarketSnapshot:
    """Structured representation of the full decision context we send to GPT.

    This is *not* required at runtime, but documents and normalises the snapshot shape.
    """
    symbol: str
    timestamp: float
    price: float
    trend: str
    range_position: str
    volatility_mode: str
    flow: Dict[str, Any] = field(default_factory=dict)
    microstructure: Dict[str, Any] = field(default_factory=dict)
    liquidity_context: Dict[str, Any] = field(default_factory=dict)
    fib_context: Dict[str, Any] = field(default_factory=dict)
    danger_mode: bool = False
    timing_state: str = field(default_factory=lambda: enum_to_str(TimingState.UNKNOWN))
    recent_price_path: Dict[str, Any] = field(default_factory=dict)
    risk_context: Dict[str, Any] = field(default_factory=dict)
    risk_envelope: Dict[str, Any] = field(default_factory=dict)
    gpt_state_note: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": str(self.symbol),
            "timestamp": float(self.timestamp),
            "price": float(self.price),
            "trend": str(self.trend),
            "range_position": str(self.range_position),
            "volatility_mode": str(self.volatility_mode),
            "flow": dict(self.flow or {}),
            "microstructure": dict(self.microstructure or {}),
            "liquidity_context": dict(self.liquidity_context or {}),
            "fib_context": dict(self.fib_context or {}),
            "danger_mode": bool(self.danger_mode),
            "timing_state": str(self.timing_state),
            "recent_price_path": dict(self.recent_price_path or {}),
            "risk_context": dict(self.risk_context or {}),
            "risk_envelope": dict(self.risk_envelope or {}),
            "gpt_state_note": self.gpt_state_note,
        }


@dataclass
class GptDecision:
    """Decision as returned by the GPT brain layer."""
    action: str  # "long" | "short" | "flat"
    confidence: float
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"action": self.action, "confidence": float(self.confidence), "notes": self.notes}


@dataclass
class RiskDecision:
    """Decision after applying risk engine constraints on GPT's suggestion."""
    approved: bool
    side: str  # "long" | "short" | "flat"
    position_size: float
    leverage: float
    stop_loss_price: Optional[float]
    take_profit_price: Optional[float]
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _safe_get(d: Dict[str, Any], key: str, default: Any) -> Any:
    return d.get(key, default)


def validate_snapshot_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    """Coerce and validate a snapshot dict so it is safe to send to GPT.

    - Ensures all required keys are present.
    - Coerces enum-like fields to allowed string sets.
    - Fills missing optional values with None or "unknown".
    - Never raises for normal missing data.
    """
    if d is None:
        d = {}

    out: Dict[str, Any] = {}

    out["timestamp"] = float(_safe_get(d, "timestamp", 0.0))
    out["symbol"] = str(_safe_get(d, "symbol", "UNKNOWN"))
    out["price"] = float(_safe_get(d, "price", 0.0))

    # Core enums
    trend_enum = coerce_enum(str(_safe_get(d, "trend", "unknown")), Trend, Trend.UNKNOWN)
    out["trend"] = enum_to_str(trend_enum)

    range_enum = coerce_enum(str(_safe_get(d, "range_position", "unknown")), RangePosition, RangePosition.UNKNOWN)
    out["range_position"] = enum_to_str(range_enum)

    vol_enum = coerce_enum(str(_safe_get(d, "volatility_mode", "unknown")), VolatilityMode, VolatilityMode.UNKNOWN)
    out["volatility_mode"] = enum_to_str(vol_enum)

    # Flow fields
    flow = dict(_safe_get(d, "flow", {}) or {})
    skew_enum = coerce_enum(str(flow.get("skew_bias", "unknown")), SkewBias, SkewBias.UNKNOWN)
    flow["skew_bias"] = enum_to_str(skew_enum)
    out["flow"] = flow

    # Microstructure: keep whatever keys we have, but normalise a few
    micro = dict(_safe_get(d, "microstructure", {}) or {})
    micro["shape_bias"] = str(micro.get("shape_bias", "none"))
    try:
        micro["shape_score"] = float(micro.get("shape_score", 0.0) or 0.0)
    except (TypeError, ValueError):
        micro["shape_score"] = 0.0
    out["microstructure"] = micro

    # Liquidity context
    liq = dict(_safe_get(d, "liquidity_context", {}) or {})
    out["liquidity_context"] = {
        "liquidity_above": liq.get("liquidity_above"),
        "liquidity_below": liq.get("liquidity_below"),
    }

    # Fib context
    fib = dict(_safe_get(d, "fib_context", {}) or {})
    out["fib_context"] = {
        "macro_zone": fib.get("macro_zone", "unknown"),
        "micro_zone": fib.get("micro_zone", "unknown"),
    }

    out["danger_mode"] = bool(_safe_get(d, "danger_mode", False))

    timing_enum = coerce_enum(str(_safe_get(d, "timing_state", "unknown")), TimingState, TimingState.UNKNOWN)
    out["timing_state"] = enum_to_str(timing_enum)

    # New: recent_price_path and risk_context and gpt_state_note
    rpp = dict(_safe_get(d, "recent_price_path", {}) or {})
    # Normalise some known keys but keep flexible
    for key in ("ret_1", "ret_5", "ret_15"):
        if key in rpp:
            try:
                rpp[key] = float(rpp[key])
            except (TypeError, ValueError):
                rpp[key] = 0.0
    rpp["impulse_state"] = str(rpp.get("impulse_state", "unknown"))
    out["recent_price_path"] = rpp

    risk_ctx = dict(_safe_get(d, "risk_context", {}) or {})
    try:
        risk_ctx["equity"] = float(risk_ctx.get("equity", 0.0) or 0.0)
    except (TypeError, ValueError):
        risk_ctx["equity"] = 0.0
    try:
        risk_ctx["max_drawdown"] = float(risk_ctx.get("max_drawdown", 0.0) or 0.0)
    except (TypeError, ValueError):
        risk_ctx["max_drawdown"] = 0.0
    risk_ctx.setdefault("open_positions_summary", risk_ctx.get("open_positions_summary", []))
    risk_ctx["last_action"] = str(risk_ctx.get("last_action", "flat"))
    try:
        risk_ctx["last_confidence"] = float(risk_ctx.get("last_confidence", 0.0) or 0.0)
    except (TypeError, ValueError):
        risk_ctx["last_confidence"] = 0.0
    out["risk_context"] = risk_ctx

    # Risk envelope with safe defaults and light coercion
    risk_env_raw = _safe_get(d, "risk_envelope", None)
    risk_env_provided = "risk_envelope" in d
    risk_env_dict = risk_env_raw if isinstance(risk_env_raw, dict) else {}

    def _coerce_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    risk_env: Dict[str, Any] = {}
    for key in (
        "max_notional",
        "max_leverage",
        "max_risk_per_trade_pct",
        "min_stop_distance_pct",
        "max_stop_distance_pct",
        "max_daily_loss_pct",
    ):
        risk_env[key] = _coerce_float(risk_env_dict.get(key, 0.0), 0.0)

    default_note = "risk_envelope not provided"
    if not risk_env_provided:
        note_value = default_note
    else:
        note_value = risk_env_dict.get("note", default_note)
    risk_env["note"] = str(note_value if note_value is not None else default_note)
    out["risk_envelope"] = risk_env

    note = _safe_get(d, "gpt_state_note", None)
    out["gpt_state_note"] = None if note is None else str(note)

    return out
