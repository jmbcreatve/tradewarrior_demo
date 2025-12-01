import argparse
import json
from typing import TypeVar

from config import Config, load_config
from enums import TimingState, VolatilityMode
from risk_envelope import compute_risk_envelope

try:
    from state_memory import DEFAULT_STATE
except Exception:  # noqa: BLE001
    DEFAULT_STATE = {"equity": 10_000.0}


EnumType = TypeVar("EnumType", bound=type)


def _coerce_enum(value: str, enum_cls: EnumType, fallback):
    """Convert a string name/value to an enum member, falling back safely."""
    if isinstance(value, enum_cls):
        return value
    if isinstance(value, str):
        member = enum_cls.__members__.get(value.upper())
        if member:
            return member
        for candidate in enum_cls:
            if candidate.value.lower() == value.lower():
                return candidate
    return fallback


def _resolve_equity(explicit_equity: float | None) -> float:
    if explicit_equity is not None:
        return float(explicit_equity)
    default_equity = 10_000.0
    try:
        if isinstance(DEFAULT_STATE, dict):
            default_equity = float(DEFAULT_STATE.get("equity", default_equity))
    except Exception:  # noqa: BLE001
        default_equity = 10_000.0
    return default_equity


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print the computed risk envelope for the provided inputs.",
    )
    parser.add_argument(
        "--vol",
        default="UNKNOWN",
        help="VolatilityMode name (e.g. HIGH, NORMAL, LOW). Defaults to UNKNOWN.",
    )
    parser.add_argument(
        "--timing",
        default="NORMAL",
        help="TimingState name (e.g. NORMAL, CAUTIOUS, AVOID). Defaults to NORMAL.",
    )
    parser.add_argument(
        "--danger",
        action="store_true",
        default=False,
        help="Enable danger mode (forces max risk to zero).",
    )
    parser.add_argument(
        "--equity",
        type=float,
        default=None,
        help="Optional account equity override. Defaults to state memory or 10,000.",
    )
    args = parser.parse_args()

    cfg: Config = load_config()
    vol_enum = _coerce_enum(args.vol, VolatilityMode, VolatilityMode.UNKNOWN)
    timing_enum = _coerce_enum(args.timing, TimingState, TimingState.NORMAL)
    equity_value = _resolve_equity(args.equity)

    env = compute_risk_envelope(
        config=cfg,
        equity=equity_value,
        volatility_mode=vol_enum,
        danger_mode=bool(args.danger),
        timing_state=timing_enum,
    )

    print(json.dumps(env.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
