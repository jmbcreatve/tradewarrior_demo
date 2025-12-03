from enum import Enum


class Trend(Enum):
    UP = "up"
    DOWN = "down"
    SIDEWAYS = "sideways"
    UNKNOWN = "unknown"


class RangePosition(Enum):
    LOW = "low"
    MID = "mid"
    HIGH = "high"
    EXTREME_LOW = "extreme_low"
    EXTREME_HIGH = "extreme_high"
    UNKNOWN = "unknown"


class VolatilityMode(Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    EXPLOSIVE = "explosive"
    UNKNOWN = "unknown"


class SkewBias(Enum):
    CALL_DOMINANT = "call_dominant"
    PUT_DOMINANT = "put_dominant"
    NEUTRAL = "neutral"
    UNKNOWN = "unknown"


class LiquidityLevel(Enum):
    THIN = "thin"
    NORMAL = "normal"
    THICK = "thick"
    UNKNOWN = "unknown"


class TimingState(Enum):
    AVOID = "avoid"
    CAUTIOUS = "cautious"
    NORMAL = "normal"
    AGGRESSIVE = "aggressive"
    UNKNOWN = "unknown"


class Side(Enum):
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


class ExecutionMode(Enum):
    """
    High-level execution wiring mode.

    SIM        -> Pure mock adapters (no real network).
    HL_TESTNET -> Hyperliquid testnet (real venue, play money).
    HL_MAINNET -> Hyperliquid mainnet (real capital) – currently disabled.
    """
    SIM = "sim"
    HL_TESTNET = "hl_testnet"
    HL_MAINNET = "hl_mainnet"


def enum_to_str(value: Enum) -> str:
    """Convert an Enum to its snapshot string representation."""
    return value.value if isinstance(value, Enum) else str(value)


def coerce_enum(value: str, enum_cls, default) -> Enum:
    """Coerce an arbitrary string to the given enum class, or return default."""
    if isinstance(value, enum_cls):
        return value
    if not isinstance(value, str):
        return default
    for member in enum_cls:
        if member.value == value or member.name.lower() == value.lower():
            return member
    return default
