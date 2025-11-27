import json
import os
from typing import Dict, Any

from config import Config
from logger_utils import get_logger

logger = get_logger(__name__)


# Default state shape. This is merged with whatever is on disk.
DEFAULT_STATE: Dict[str, Any] = {
    # Basic account
    "equity": 10_000.0,
    "max_drawdown": 0.0,
    # Simple open-position summary; adapters can populate this if desired.
    "open_positions_summary": [],
    # Last full GPT decision (action/confidence/notes as a dict).
    "last_gpt_decision": None,
    # Last snapshot we processed (raw snapshot dict).
    "prev_snapshot": None,
    # Symbol we are trading (usually comes from Config.symbol).
    "symbol": None,
    # Short persistent note from the GPT brain about the current regime / context.
    "gpt_state_note": None,
    # Gatekeeper / cost-control bookkeeping
    "gpt_call_timestamps": [],
    "last_gpt_call_walltime": 0.0,
    "last_gpt_snapshot": None,
}


def load_state(config: Config) -> Dict[str, Any]:
    """Load state from disk and merge it with DEFAULT_STATE.

    If the file is missing or invalid, we fall back to DEFAULT_STATE.
    """
    path = config.state_file
    if not os.path.exists(path):
        return dict(DEFAULT_STATE)

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to load state from %s: %s", path, exc)
        return dict(DEFAULT_STATE)

    if not isinstance(data, dict):
        logger.warning("State file %s did not contain a dict; resetting.", path)
        return dict(DEFAULT_STATE)

    state = dict(DEFAULT_STATE)
    state.update(data or {})
    return state


def save_state(state: Dict[str, Any], config: Config) -> None:
    """Atomically persist state to disk as JSON."""
    path = config.state_file
    tmp_path = path + ".tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, sort_keys=True)
        os.replace(tmp_path, path)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to save state to %s: %s", path, exc)
