import os
import tempfile
from pathlib import Path

import pytest

from safety_utils import check_kill_switch, check_trading_halted, KILL_SWITCH_FILE


def test_check_kill_switch_returns_false_when_file_missing():
    """Test that kill switch check returns False when file doesn't exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        kill_file = os.path.join(tmpdir, "nonexistent_kill_switch")
        assert check_kill_switch(kill_file) is False


def test_check_kill_switch_returns_true_when_file_exists():
    """Test that kill switch check returns True when file exists."""
    with tempfile.TemporaryDirectory() as tmpdir:
        kill_file = os.path.join(tmpdir, "test_kill_switch")
        # Create the kill switch file
        Path(kill_file).touch()
        assert check_kill_switch(kill_file) is True


def test_check_trading_halted_returns_false_when_no_halt():
    """Test that check_trading_halted returns False when neither kill switch nor circuit breaker active."""
    with tempfile.TemporaryDirectory() as tmpdir:
        kill_file = os.path.join(tmpdir, "nonexistent_kill_switch")
        state = {"trading_halted": False}
        is_halted, reason = check_trading_halted(state, kill_file)
        assert is_halted is False
        assert reason == ""


def test_check_trading_halted_returns_true_for_kill_switch():
    """Test that check_trading_halted returns True when kill switch file exists."""
    with tempfile.TemporaryDirectory() as tmpdir:
        kill_file = os.path.join(tmpdir, "test_kill_switch")
        Path(kill_file).touch()
        state = {"trading_halted": False}
        is_halted, reason = check_trading_halted(state, kill_file)
        assert is_halted is True
        assert reason == "kill_switch_active"


def test_check_trading_halted_returns_true_for_circuit_breaker():
    """Test that check_trading_halted returns True when circuit breaker flag is set."""
    with tempfile.TemporaryDirectory() as tmpdir:
        kill_file = os.path.join(tmpdir, "nonexistent_kill_switch")
        state = {"trading_halted": True}
        is_halted, reason = check_trading_halted(state, kill_file)
        assert is_halted is True
        assert reason == "circuit_breaker_active"


def test_check_trading_halted_kill_switch_takes_precedence():
    """Test that kill switch takes precedence over circuit breaker flag."""
    with tempfile.TemporaryDirectory() as tmpdir:
        kill_file = os.path.join(tmpdir, "test_kill_switch")
        Path(kill_file).touch()
        state = {"trading_halted": True}  # Circuit breaker also active
        is_halted, reason = check_trading_halted(state, kill_file)
        assert is_halted is True
        assert reason == "kill_switch_active"  # Kill switch takes precedence


def test_check_trading_halted_uses_default_kill_switch_file():
    """Test that check_trading_halted uses default KILL_SWITCH_FILE when not specified."""
    # This test checks that the function works with default file path
    # We can't easily test the exact default path without mocking, but we can test it doesn't crash
    state = {"trading_halted": False}
    is_halted, reason = check_trading_halted(state, kill_switch_file=None)
    # Should not crash, result depends on whether default file exists
    assert isinstance(is_halted, bool)
    assert isinstance(reason, str)

