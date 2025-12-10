# tw5/schemas.py

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Run modes for the TW-5 engine
# ---------------------------------------------------------------------------


class RunMode(str, Enum):
    """Execution mode for the TW-5 engine.

    - ADVISOR: snapshot -> GPT -> plan -> clamp, but do NOT send orders.
    - AGENT_CONFIRM: same, but require a human/UX confirmation before execution.
    - AUTO: fully automated - snapshot -> GPT -> clamp -> executor.
    """

    ADVISOR = "advisor"
    AGENT_CONFIRM = "agent_confirm"
    AUTO = "auto"


# ---------------------------------------------------------------------------
# Tiny TW-5 market snapshot schema
# ---------------------------------------------------------------------------


@dataclass
class Tw5Snapshot:
    """Minimal, human-like market panel for TW-5.

    All enum-ish fields are plain strings so the JSON is simple and
    GPT-friendly. Builders are responsible for picking stable vocabularies.

    Fields:
        symbol:      Trading symbol, e.g. "BTCUSDT".
        timeframe:   Candlestick timeframe, e.g. "1m", "5m".
        timestamp:   Unix timestamp (seconds since epoch) for this snapshot.
        price:       Last traded price at snapshot time.

        trend_1h:    "up" | "down" | "sideways" | "unknown".
        trend_4h:    "up" | "down" | "sideways" | "unknown".

        range_low_7d:        Lowest price observed over the recent window.
        range_high_7d:       Highest price observed over the recent window.
        range_position_7d:   "low" | "mid" | "high" | "extreme_low"
                             | "extreme_high" | "unknown".

        swing_low:   Anchor low price for fibs over the current swing.
        swing_high:  Anchor high price for fibs over the current swing.

        fib_0_382:   0.382 retracement level.
        fib_0_5:     0.5 retracement level.
        fib_0_618:   0.618 retracement level.
        fib_0_786:   0.786 retracement level.

        vol_mode:    "low" | "normal" | "high" | "explosive" | "unknown".
        atr_pct:     ATR-like avg move as a fraction of price (e.g. 0.01 = 1%).

        last_impulse_direction: "up" | "down" | "chop" | "unknown".
        last_impulse_size_pct:  Size of last impulse as pct of price.

        swept_prev_high:       True if we recently swept a prior high and rejected.
        swept_prev_low:        True if we recently swept a prior low and rejected.

        position_side:         "flat" | "long" | "short".
        position_size:         Absolute position size in contracts (or units).
        position_entry_price:  Average entry price for current position, or None if flat.
    """

    # Identity / timing
    symbol: str
    timeframe: str
    timestamp: float
    price: float

    # Trend / range
    trend_1h: str
    trend_4h: str

    range_low_7d: float
    range_high_7d: float
    range_position_7d: str

    # Swing / fib
    swing_low: float
    swing_high: float
    fib_0_382: float
    fib_0_5: float
    fib_0_618: float
    fib_0_786: float

    # Volatility / path
    vol_mode: str
    atr_pct: float

    # Microstructure-lite
    last_impulse_direction: str
    last_impulse_size_pct: float
    swept_prev_high: bool
    swept_prev_low: bool

    # Position
    position_side: str
    position_size: float
    position_entry_price: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable dict representation."""
        return asdict(self)


# ---------------------------------------------------------------------------
# Order plan schema returned by GPT or the TW-5 stub
# ---------------------------------------------------------------------------


@dataclass
class TPLevel:
    """Single take-profit level for a leg.

    Fields:
        price:      Absolute price for this take-profit target.
        size_frac:  Fraction of the leg's size to close here (0–1).
        tag:        Optional label, e.g. "1R", "2R", "partial_exit".
    """

    price: float
    size_frac: float
    tag: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class OrderLeg:
    """One entry/exit leg within an OrderPlan.

    Fields:
        id:          Stable identifier for this leg (for logging / replay).
        entry_type:  "limit" | "market".
        entry_price: Absolute entry price.
        entry_tag:   Optional label, e.g. "fib_0.382", "range_low".
        size_frac:   Fraction of the *overall* position size allocated to this leg (0–1).
        stop_loss:   Absolute stop-loss price.
        take_profits: Ordered list of TP levels for this leg.
    """

    id: str
    entry_type: str
    entry_price: float
    entry_tag: str
    size_frac: float
    stop_loss: float
    take_profits: List[TPLevel] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "entry_type": self.entry_type,
            "entry_price": self.entry_price,
            "entry_tag": self.entry_tag,
            "size_frac": self.size_frac,
            "stop_loss": self.stop_loss,
            "take_profits": [tp.to_dict() for tp in self.take_profits],
        }


@dataclass
class OrderPlan:
    """Structured order plan generated by GPT or the TW-5 stub.

    Fields:
        mode:                "enter" | "manage" | "flat".
        side:                "long" | "short" | "flat".
        legs:                Individual legs making up the overall plan.
        max_total_size_frac: Intended aggressiveness of the plan (0–1).
        confidence:          0–1 confidence score from GPT.
        rationale:           Free-text reasoning for logging / debugging.
    """

    mode: str
    side: str
    legs: List[OrderLeg] = field(default_factory=list)
    max_total_size_frac: float = 1.0
    confidence: float = 0.5
    rationale: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "side": self.side,
            "legs": [leg.to_dict() for leg in self.legs],
            "max_total_size_frac": self.max_total_size_frac,
            "confidence": self.confidence,
            "rationale": self.rationale,
        }

    @classmethod
    def empty_flat(cls, rationale: str = "") -> "OrderPlan":
        """Return a flat, no-op plan (useful for clamps / safety / no-call ticks)."""
        return cls(
            mode="flat",
            side="flat",
            legs=[],
            max_total_size_frac=0.0,
            confidence=0.0,
            rationale=rationale,
        )


# ---------------------------------------------------------------------------
# Result of passing a plan through the risk clamp
# ---------------------------------------------------------------------------


@dataclass
class RiskClampResult:
    """Output of the TW-5 risk clamp.

    Fields:
        approved:      Whether this plan is allowed to proceed to execution.
        reason:        Human-readable explanation (for logs/UI).
        original_plan: The plan as returned by GPT or stub.
        clamped_plan:  The plan after size/stop tightening, if approved.
    """

    approved: bool
    reason: str
    original_plan: OrderPlan
    clamped_plan: Optional[OrderPlan] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "approved": self.approved,
            "reason": self.reason,
            "original_plan": self.original_plan.to_dict(),
            "clamped_plan": self.clamped_plan.to_dict() if self.clamped_plan is not None else None,
        }


# ---------------------------------------------------------------------------
# Pending limit order for replay simulation
# ---------------------------------------------------------------------------


@dataclass
class PendingOrder:
    """Tracks a pending limit order in replay simulation.

    This allows limit orders to persist across bars until filled or cancelled,
    providing realistic backtesting where pullback entries can fill on future bars.

    Fields:
        plan:          The order plan (enter/manage) that generated this order.
        clamp_result:  The risk clamp result for this plan.
        created_ts:    Timestamp when the order was created.
        created_idx:   Bar index when the order was created.
        expires_idx:   Bar index when this order expires (None = no expiry).
        snapshot:      The snapshot at the time the order was created.
    """

    plan: OrderPlan
    clamp_result: RiskClampResult
    created_ts: float
    created_idx: int
    expires_idx: Optional[int] = None
    snapshot: Optional[Tw5Snapshot] = None

    def is_expired(self, current_idx: int) -> bool:
        """Check if this order has expired based on bar index."""
        if self.expires_idx is None:
            return False
        return current_idx >= self.expires_idx
