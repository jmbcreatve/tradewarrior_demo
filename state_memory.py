from __future__ import annotations

import json
import os
from copy import deepcopy
from typing import Any, Dict, List

from config import Config
from logger_utils import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

STATE_VERSION = 1

DEFAULT_STATE: Dict[str, Any] = {
    # Core account metrics
    "equity": 10_000.0,
    "max_drawdown": 0.0,

    # Daily P&L tracking
    "daily_start_equity": None,     # float: equity at start of trading day
    "daily_start_timestamp": None,  # float: UTC timestamp when day started
    "daily_pnl": 0.0,              # float: current day's P&L (equity - daily_start_equity)
    "trading_halted": False,        # bool: circuit breaker flag

    # Positions summary: adapters can store lightweight position views here.
    "open_positions_summary": [],  # type: List[Dict[str, Any]]

    # Last GPT / risk context
    "last_decision": None,         # unified Decision object as dict
    "last_envelope": None,         # last RiskEnvelope as dict
    "last_gpt_decision": None,      # raw dict from GptDecision.to_dict()

    # Snapshot tracking
    "last_snapshot": None,          # last snapshot dict processed
    "last_gpt_snapshot": None,      # snapshot at last GPT call
    "last_gpt_call_ts": None,       # last GPT snapshot timestamp (same unit as snapshot["timestamp"])
    "last_gpt_equity": 0.0,         # equity at last GPT call
    "trades_since_last_gpt": 0,     # integer counter, will be refined later

    # GPT call throttling
    "gpt_call_timestamps": [],      # wall-time seconds
    "last_gpt_call_walltime": 0.0,

    # Symbol / meta
    "symbol": None,
    "gpt_state_note": None,
    "run_id": None,
    "snapshot_id": 0,
    "state_version": STATE_VERSION,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return []


def _coerce_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _reset_daily_tracking(state: Dict[str, Any], current_equity: float, current_timestamp: float) -> None:
    """Reset daily tracking metrics at start of new trading day."""
    state["daily_start_equity"] = current_equity
    state["daily_start_timestamp"] = current_timestamp
    state["daily_pnl"] = 0.0
    state["trading_halted"] = False
    logger.info(
        "Daily tracking reset: start_equity=%.2f, timestamp=%.2f",
        current_equity,
        current_timestamp,
    )


def _load_from_disk(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            logger.warning("State file %s did not contain a dict; ignoring.", path)
            return {}
        return data
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to load state file %s: %s", path, exc)
        return {}


def _normalise_state(raw: Dict[str, Any], config: Config | None = None) -> Dict[str, Any]:
    """Merge raw on-disk state with defaults and coerce types."""
    state: Dict[str, Any] = dict(DEFAULT_STATE)

    # Merge first, then coerce/repair.
    state.update(raw or {})
    state["state_version"] = STATE_VERSION

    # Remove legacy keys no longer used.
    state.pop("last_action", None)
    state.pop("last_confidence", None)

    # Symbol: always ensure we have one.
    if not state.get("symbol") and config is not None:
        state["symbol"] = config.symbol
    elif state.get("symbol"):
        state["symbol"] = str(state["symbol"])

    # Run metadata
    run_id_value = state.get("run_id")
    if run_id_value is None or isinstance(run_id_value, str):
        state["run_id"] = run_id_value
    else:
        state["run_id"] = None

    try:
        snapshot_id_val = int(state.get("snapshot_id", DEFAULT_STATE["snapshot_id"]) or 0)
    except (TypeError, ValueError):
        snapshot_id_val = 0
    if snapshot_id_val < 0:
        snapshot_id_val = 0
    state["snapshot_id"] = snapshot_id_val

    # Equity / drawdown
    state["equity"] = _coerce_float(
        state.get("equity", DEFAULT_STATE["equity"]),
        DEFAULT_STATE["equity"],
    )
    state["max_drawdown"] = _coerce_float(
        state.get("max_drawdown", DEFAULT_STATE["max_drawdown"]),
        DEFAULT_STATE["max_drawdown"],
    )

    # Daily P&L tracking
    daily_start_equity = state.get("daily_start_equity")
    if daily_start_equity is None:
        state["daily_start_equity"] = None
    else:
        state["daily_start_equity"] = _coerce_float(daily_start_equity, state["equity"])

    daily_start_ts = state.get("daily_start_timestamp")
    if daily_start_ts is None:
        state["daily_start_timestamp"] = None
    else:
        try:
            state["daily_start_timestamp"] = float(daily_start_ts)
        except (TypeError, ValueError):
            state["daily_start_timestamp"] = None

    state["daily_pnl"] = _coerce_float(
        state.get("daily_pnl", DEFAULT_STATE["daily_pnl"]),
        DEFAULT_STATE["daily_pnl"],
    )

    # Trading halted flag
    trading_halted = state.get("trading_halted")
    if trading_halted is None:
        state["trading_halted"] = DEFAULT_STATE["trading_halted"]
    else:
        state["trading_halted"] = bool(trading_halted)

    # Open positions
    ops = state.get("open_positions_summary", DEFAULT_STATE["open_positions_summary"])
    state["open_positions_summary"] = ops if isinstance(ops, list) else []

    # Last decision / envelope: keep as dict or None
    ld = state.get("last_decision")
    if ld is not None and not isinstance(ld, dict):
        ld = None
    state["last_decision"] = ld

    le = state.get("last_envelope")
    if le is not None and not isinstance(le, dict):
        le = None
    state["last_envelope"] = le

    # Last GPT decision: keep as dict or None
    lgd = state.get("last_gpt_decision")
    if lgd is not None and not isinstance(lgd, dict):
        lgd = None
    state["last_gpt_decision"] = lgd

    # Snapshots
    if state.get("last_snapshot") is not None and not isinstance(state["last_snapshot"], dict):
        state["last_snapshot"] = None
    if state.get("last_gpt_snapshot") is not None and not isinstance(state["last_gpt_snapshot"], dict):
        state["last_gpt_snapshot"] = None

    # GPT snapshot deltas
    lgts = state.get("last_gpt_call_ts")
    if lgts is None:
        state["last_gpt_call_ts"] = None
    else:
        try:
            state["last_gpt_call_ts"] = float(lgts)
        except (TypeError, ValueError):
            state["last_gpt_call_ts"] = None

    state["last_gpt_equity"] = _coerce_float(
        state.get("last_gpt_equity", DEFAULT_STATE["last_gpt_equity"]),
        DEFAULT_STATE["last_gpt_equity"],
    )

    try:
        trades_since = int(state.get("trades_since_last_gpt", DEFAULT_STATE["trades_since_last_gpt"]) or 0)
    except (TypeError, ValueError):
        trades_since = 0
    if trades_since < 0:
        trades_since = 0
    state["trades_since_last_gpt"] = trades_since

    # GPT call timestamps
    state["gpt_call_timestamps"] = [
        _coerce_float(t, 0.0)
        for t in _ensure_list(state.get("gpt_call_timestamps"))
        if _coerce_float(t, -1.0) > 0.0
    ]
    state["last_gpt_call_walltime"] = _coerce_float(
        state.get("last_gpt_call_walltime", DEFAULT_STATE["last_gpt_call_walltime"]),
        DEFAULT_STATE["last_gpt_call_walltime"],
    )

    # gpt_state_note: str or None
    note = state.get("gpt_state_note", None)
    if note is None:
        state["gpt_state_note"] = None
    else:
        state["gpt_state_note"] = str(note)

    return state


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _state_path(config: Config) -> str:
    return config.state_file


def _fresh_state(config: Config) -> Dict[str, Any]:
    """Construct a brand new in-memory state based on defaults + config."""
    state = dict(DEFAULT_STATE)
    state["symbol"] = config.symbol
    state["state_version"] = STATE_VERSION
    # Use config's initial_equity if available (for testnet $1000 accounts)
    state["equity"] = getattr(config, "initial_equity", DEFAULT_STATE["equity"])
    # Make sure list fields are fresh lists, not shared references
    state["open_positions_summary"] = []
    state["gpt_call_timestamps"] = []
    # Daily tracking starts fresh
    state["daily_start_equity"] = None
    state["daily_start_timestamp"] = None
    state["daily_pnl"] = 0.0
    state["trading_halted"] = False
    return state


def reset_state(config: Config | None = None) -> Dict[str, Any]:
    """
    Return a fresh in-memory state based on DEFAULT_STATE.
    If a Config is provided, use it to fill any config-dependent defaults
    (e.g. symbol, initial_equity) but do not read or write the state file.
    """
    state = deepcopy(DEFAULT_STATE)
    if config is not None:
        state["symbol"] = config.symbol
        # Use config's initial_equity if available (for testnet $1000 accounts)
        state["equity"] = getattr(config, "initial_equity", DEFAULT_STATE["equity"])
    state = _normalise_state(state, config)
    state["run_id"] = None
    state["snapshot_id"] = 0
    state["state_version"] = STATE_VERSION
    return state


def _migrate_state_if_needed(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure state['state_version'] exists and matches STATE_VERSION.
    For now, perform minimal, safe migrations and log what happened.
    """
    version = state.get("state_version")
    if version is None:
        logger.info("State: upgrading legacy state; adding state_version=%s.", STATE_VERSION)
        state["state_version"] = STATE_VERSION
        return state
    if version != STATE_VERSION:
        logger.warning(
            "State: migrating state from version %s to %s.",
            version,
            STATE_VERSION,
        )
        state["state_version"] = STATE_VERSION
    return state


def load_state(config: Config) -> Dict[str, Any]:
    """Load state from disk, or return a fresh state if missing/bad.

    Always returns a dict that passes _normalise_state and is safe to mutate.
    """
    path = _state_path(config)
    raw = _load_from_disk(path)
    if not raw:
        logger.info("State: no existing state on disk; starting fresh.")
        state = _fresh_state(config)
    else:
        state = raw
    state = _migrate_state_if_needed(state)
    state = _normalise_state(state, config)
    if raw:
        logger.info("State: loaded from %s (equity=%.2f, max_dd=%.2f).",
                    path, state["equity"], state["max_drawdown"])
    return state


def save_state(state: Dict[str, Any], config: Config) -> None:
    """Persist state to disk atomically."""
    path = _state_path(config)
    tmp_path = f"{path}.tmp"

    # Make a serialisable copy (basic types only).
    to_save: Dict[str, Any] = dict(state)

    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(to_save, f, ensure_ascii=False, sort_keys=True, indent=2)
        os.replace(tmp_path, path)
        logger.info("State: saved to %s.", path)
    except Exception as exc:  # noqa: BLE001
        logger.error("State: failed to save to %s: %s", path, exc)
        # Best effort: remove temp file if something went wrong.
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
