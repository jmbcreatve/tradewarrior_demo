from __future__ import annotations

import json
import os
from typing import Any, Dict, List

from config import Config
from logger_utils import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_STATE: Dict[str, Any] = {
    # Core account metrics
    "equity": 10_000.0,
    "max_drawdown": 0.0,

    # Positions summary: adapters can store lightweight position views here.
    "open_positions_summary": [],  # type: List[Dict[str, Any]]

    # Last GPT / risk context
    "last_action": "flat",
    "last_confidence": 0.0,
    "last_gpt_decision": None,      # raw dict from GptDecision.to_dict()

    # Snapshot tracking
    "last_snapshot": None,          # last snapshot dict processed
    "last_gpt_snapshot": None,      # snapshot at last GPT call

    # GPT call throttling
    "gpt_call_timestamps": [],      # wall-time seconds
    "last_gpt_call_walltime": 0.0,

    # Symbol / meta
    "symbol": None,
    "gpt_state_note": None,
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


def _normalise_state(raw: Dict[str, Any], config: Config) -> Dict[str, Any]:
    """Merge raw on-disk state with defaults and coerce types."""
    state: Dict[str, Any] = dict(DEFAULT_STATE)

    # Merge first, then coerce/repair.
    state.update(raw or {})

    # Symbol: always ensure we have one.
    if not state.get("symbol"):
        state["symbol"] = config.symbol
    else:
        state["symbol"] = str(state["symbol"])

    # Equity / drawdown
    state["equity"] = _coerce_float(
        state.get("equity", DEFAULT_STATE["equity"]),
        DEFAULT_STATE["equity"],
    )
    state["max_drawdown"] = _coerce_float(
        state.get("max_drawdown", DEFAULT_STATE["max_drawdown"]),
        DEFAULT_STATE["max_drawdown"],
    )

    # Open positions
    ops = state.get("open_positions_summary", DEFAULT_STATE["open_positions_summary"])
    state["open_positions_summary"] = ops if isinstance(ops, list) else []

    # Last action / confidence
    last_action = state.get("last_action", DEFAULT_STATE["last_action"])
    if not isinstance(last_action, str) or not last_action:
        last_action = DEFAULT_STATE["last_action"]
    state["last_action"] = last_action

    state["last_confidence"] = _coerce_float(
        state.get("last_confidence", DEFAULT_STATE["last_confidence"]),
        DEFAULT_STATE["last_confidence"],
    )

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
    # Make sure list fields are fresh lists, not shared references
    state["open_positions_summary"] = []
    state["gpt_call_timestamps"] = []
    return state


def load_state(config: Config) -> Dict[str, Any]:
    """Load state from disk, or return a fresh state if missing/bad.

    Always returns a dict that passes _normalise_state and is safe to mutate.
    """
    path = _state_path(config)
    raw = _load_from_disk(path)
    if not raw:
        logger.info("State: no existing state on disk; starting fresh.")
        return _fresh_state(config)
    state = _normalise_state(raw, config)
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
