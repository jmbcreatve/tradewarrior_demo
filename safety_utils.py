from __future__ import annotations

import os
from typing import Any, Dict

from logger_utils import get_logger

logger = get_logger(__name__)

# Default kill switch file name (in project root)
KILL_SWITCH_FILE = ".tradewarrior_kill_switch"


def check_kill_switch(kill_switch_file: str | None = None) -> bool:
    """
    Check if kill switch file exists.

    Args:
        kill_switch_file: Path to kill switch file. If None, uses default KILL_SWITCH_FILE.

    Returns:
        True if kill switch is active (file exists), False otherwise.
    """
    if kill_switch_file is None:
        # Get project root (directory containing this file)
        project_root = os.path.dirname(os.path.abspath(__file__))
        kill_switch_file = os.path.join(project_root, KILL_SWITCH_FILE)
    else:
        kill_switch_file = os.path.abspath(kill_switch_file)

    exists = os.path.exists(kill_switch_file)
    if exists:
        logger.critical(
            "🚨 KILL SWITCH ACTIVE: %s exists. Trading halted immediately.",
            kill_switch_file,
        )
    return exists


def check_trading_halted(state: Dict[str, Any], kill_switch_file: str | None = None) -> tuple[bool, str]:
    """
    Check if trading should be halted (state flag or kill switch).

    Args:
        state: Current state dict
        kill_switch_file: Optional path to kill switch file

    Returns:
        Tuple of (is_halted: bool, reason: str)
    """
    # Kill switch takes precedence
    if check_kill_switch(kill_switch_file):
        return (True, "kill_switch_active")

    # Check state flag
    trading_halted = state.get("trading_halted", False)
    if trading_halted:
        return (True, "circuit_breaker_active")

    return (False, "")

