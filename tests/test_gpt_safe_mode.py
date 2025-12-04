"""
Tests for GPT Safe Mode functionality.

Safe mode policy:
- 3 consecutive GPT errors within 5 minutes triggers safe mode
- When safe mode is active, GPT calls are skipped and decisions are forced FLAT
- Safe mode requires manual reset (does not auto-recover)
- Successful GPT calls reset the error counter (but don't clear safe mode)
"""

import json
import os
import sys
import time
from unittest.mock import patch, MagicMock

import pytest

from config import Config
from enums import Side
from gpt_client import call_gpt, create_safe_mode_decision
from risk_engine import evaluate_risk
from schemas import GptDecision
from state_memory import (
    DEFAULT_STATE,
    reset_state,
    load_state,
    save_state,
    _normalise_state,
    record_gpt_error,
    record_gpt_success,
    clear_gpt_safe_mode,
    is_gpt_safe_mode,
    GPT_SAFE_MODE_ERROR_THRESHOLD,
    GPT_SAFE_MODE_ERROR_WINDOW_SEC,
)


# ---------------------------------------------------------------------------
# State Memory Safe Mode Tests
# ---------------------------------------------------------------------------


def test_default_state_has_safe_mode_fields():
    """Verify DEFAULT_STATE includes all GPT safe mode fields."""
    assert "gpt_safe_mode" in DEFAULT_STATE
    assert "gpt_error_count" in DEFAULT_STATE
    assert "last_gpt_error_timestamp" in DEFAULT_STATE
    
    assert DEFAULT_STATE["gpt_safe_mode"] is False
    assert DEFAULT_STATE["gpt_error_count"] == 0
    assert DEFAULT_STATE["last_gpt_error_timestamp"] is None


def test_reset_state_initializes_safe_mode_fields():
    """Verify reset_state initializes GPT safe mode fields correctly."""
    config = Config()
    state = reset_state(config)
    
    assert state["gpt_safe_mode"] is False
    assert state["gpt_error_count"] == 0
    assert state["last_gpt_error_timestamp"] is None


def test_normalise_state_coerces_safe_mode_fields():
    """Verify _normalise_state handles invalid safe mode field values."""
    config = Config()
    
    # Test with invalid types
    raw = {
        "gpt_safe_mode": "true",  # string instead of bool
        "gpt_error_count": "5",    # string instead of int
        "last_gpt_error_timestamp": "invalid",
    }
    state = _normalise_state(raw, config)
    
    assert state["gpt_safe_mode"] is True  # "true" coerces to True
    assert state["gpt_error_count"] == 5
    assert state["last_gpt_error_timestamp"] is None  # Invalid value becomes None
    
    # Test with negative error count
    raw2 = {"gpt_error_count": -3}
    state2 = _normalise_state(raw2, config)
    assert state2["gpt_error_count"] == 0  # Clamped to 0


def test_is_gpt_safe_mode():
    """Test is_gpt_safe_mode helper function."""
    state = {"gpt_safe_mode": False}
    assert is_gpt_safe_mode(state) is False
    
    state["gpt_safe_mode"] = True
    assert is_gpt_safe_mode(state) is True
    
    # Handle missing key
    state = {}
    assert is_gpt_safe_mode(state) is False


def test_record_gpt_error_increments_count():
    """Test that record_gpt_error increments error count."""
    state = reset_state(Config())
    
    now = time.time()
    record_gpt_error(state, now)
    
    assert state["gpt_error_count"] == 1
    assert state["last_gpt_error_timestamp"] == now
    assert state["gpt_safe_mode"] is False  # Not triggered yet


def test_record_gpt_error_resets_count_after_window():
    """Test that error count resets if last error was outside the window."""
    state = reset_state(Config())
    state["gpt_error_count"] = 2
    
    # Set last error to be outside the window
    old_time = time.time() - GPT_SAFE_MODE_ERROR_WINDOW_SEC - 100
    state["last_gpt_error_timestamp"] = old_time
    
    now = time.time()
    record_gpt_error(state, now)
    
    # Count should reset to 1 (not increment to 3)
    assert state["gpt_error_count"] == 1
    assert state["gpt_safe_mode"] is False


def test_record_gpt_error_triggers_safe_mode():
    """Test that N consecutive errors within window triggers safe mode."""
    state = reset_state(Config())
    
    now = time.time()
    
    # Record errors up to threshold
    for i in range(GPT_SAFE_MODE_ERROR_THRESHOLD):
        triggered = record_gpt_error(state, now + i)
        if i < GPT_SAFE_MODE_ERROR_THRESHOLD - 1:
            assert triggered is False
        else:
            assert triggered is True  # Last error triggers safe mode
    
    assert state["gpt_safe_mode"] is True
    assert state["gpt_error_count"] == GPT_SAFE_MODE_ERROR_THRESHOLD


def test_record_gpt_success_resets_error_count():
    """Test that successful GPT call resets error count."""
    state = reset_state(Config())
    state["gpt_error_count"] = 2
    state["last_gpt_error_timestamp"] = time.time()
    
    record_gpt_success(state)
    
    assert state["gpt_error_count"] == 0
    # Note: last_gpt_error_timestamp is NOT cleared (for audit purposes)


def test_record_gpt_success_does_not_clear_safe_mode():
    """Test that successful GPT call does NOT automatically clear safe mode."""
    state = reset_state(Config())
    state["gpt_safe_mode"] = True
    state["gpt_error_count"] = 5
    
    record_gpt_success(state)
    
    # Error count resets, but safe mode stays active
    assert state["gpt_error_count"] == 0
    assert state["gpt_safe_mode"] is True  # Still active


def test_clear_gpt_safe_mode():
    """Test that clear_gpt_safe_mode properly resets safe mode state."""
    state = reset_state(Config())
    state["gpt_safe_mode"] = True
    state["gpt_error_count"] = 5
    state["last_gpt_error_timestamp"] = time.time()
    
    clear_gpt_safe_mode(state)
    
    assert state["gpt_safe_mode"] is False
    assert state["gpt_error_count"] == 0
    # last_gpt_error_timestamp is kept for audit purposes


# ---------------------------------------------------------------------------
# GPT Client Safe Mode Tests
# ---------------------------------------------------------------------------


def test_create_safe_mode_decision():
    """Test that create_safe_mode_decision returns correct flat decision."""
    decision = create_safe_mode_decision()
    
    assert decision.action == "flat"
    assert decision.confidence == 0.0
    assert "safe_mode" in decision.notes.lower()
    
    # Test with custom reason
    decision2 = create_safe_mode_decision("custom reason")
    assert decision2.notes == "custom reason"


def test_call_gpt_tracks_errors_in_state():
    """Test that call_gpt records errors in state when state is provided."""
    config = Config()
    snapshot = {"symbol": "BTCUSDT", "timestamp": time.time(), "price": 50000.0}
    state = reset_state(config)
    
    # Simulate an error by having an invalid API key
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
        mock_openai = MagicMock()
        mock_openai.ChatCompletion.create.side_effect = Exception("API Error")
        
        with patch.dict(sys.modules, {"openai": mock_openai}):
            decision = call_gpt(config, snapshot, state)
            
            # Should return fallback decision
            assert decision.action == "flat"
            assert decision.confidence == 0.0
            
            # Should have recorded the error
            assert state["gpt_error_count"] == 1
            assert state["last_gpt_error_timestamp"] is not None


def test_call_gpt_tracks_success_in_state():
    """Test that call_gpt records success in state when state is provided."""
    config = Config()
    snapshot = {"symbol": "BTCUSDT", "timestamp": time.time(), "price": 50000.0}
    state = reset_state(config)
    state["gpt_error_count"] = 2  # Start with some errors
    
    mock_response = {
        "choices": [
            {
                "message": {
                    "content": json.dumps({
                        "action": "long",
                        "size": 0.5,
                        "confidence": 0.8,
                        "rationale": "Test successful call"
                    })
                }
            }
        ]
    }
    
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
        mock_openai = MagicMock()
        mock_openai.ChatCompletion.create.return_value = mock_response
        
        with patch.dict(sys.modules, {"openai": mock_openai}):
            decision = call_gpt(config, snapshot, state)
            
            # Should return successful decision
            assert decision.action == "long"
            assert decision.confidence == 0.8
            
            # Should have reset error count
            assert state["gpt_error_count"] == 0


def test_call_gpt_without_state_does_not_track():
    """Test that call_gpt without state param doesn't crash."""
    config = Config()
    snapshot = {"symbol": "BTCUSDT", "timestamp": time.time(), "price": 50000.0}
    
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
        mock_openai = MagicMock()
        mock_openai.ChatCompletion.create.side_effect = Exception("API Error")
        
        with patch.dict(sys.modules, {"openai": mock_openai}):
            # Should not crash, just return fallback
            decision = call_gpt(config, snapshot)  # No state param
            
            assert decision.action == "flat"
            assert decision.confidence == 0.0


# ---------------------------------------------------------------------------
# Risk Engine Safe Mode Tests
# ---------------------------------------------------------------------------


def test_risk_engine_rejects_when_safe_mode_active():
    """Test that risk engine always returns flat when safe mode is active."""
    config = Config()
    snapshot = {
        "symbol": "BTCUSDT",
        "timestamp": time.time(),
        "price": 50000.0,
        "volatility_mode": "normal",
        "range_position": "mid",
        "timing_state": "normal",
        "danger_mode": False,
    }
    state = reset_state(config)
    state["gpt_safe_mode"] = True  # Safe mode active
    
    # Even with a strong long signal from GPT...
    gpt_decision = GptDecision(action="long", confidence=1.0, notes="strong signal")
    
    decision = evaluate_risk(snapshot, gpt_decision, state, config)
    
    # ...should be forced flat
    assert decision.approved is False
    assert decision.side == Side.FLAT.value
    assert decision.position_size == 0.0
    assert "safe_mode" in decision.reason.lower()


def test_risk_engine_allows_trade_when_safe_mode_inactive():
    """Test that risk engine works normally when safe mode is not active."""
    config = Config()
    snapshot = {
        "symbol": "BTCUSDT",
        "timestamp": time.time(),
        "price": 50000.0,
        "volatility_mode": "normal",
        "range_position": "mid",
        "timing_state": "normal",
        "danger_mode": False,
    }
    state = reset_state(config)
    state["gpt_safe_mode"] = False  # Safe mode not active
    
    gpt_decision = GptDecision(action="long", confidence=0.9, notes="good setup")
    
    decision = evaluate_risk(snapshot, gpt_decision, state, config)
    
    # Should allow the trade
    assert decision.approved is True
    assert decision.side == Side.LONG.value
    assert decision.position_size > 0


def test_risk_engine_safe_mode_overrides_all_other_conditions():
    """Test that safe mode overrides even valid trading conditions."""
    config = Config(risk_per_trade=0.02, max_leverage=5.0)
    snapshot = {
        "symbol": "BTCUSDT",
        "timestamp": time.time(),
        "price": 50000.0,
        "volatility_mode": "low",  # Good volatility
        "range_position": "mid",   # Good range
        "timing_state": "aggressive",  # Best timing
        "danger_mode": False,
        "risk_envelope": {
            "max_risk_per_trade_pct": 0.02,
            "max_leverage": 5.0,
            "max_notional": 100000.0,
            "min_stop_distance_pct": 0.005,
            "max_stop_distance_pct": 0.03,
            "max_daily_loss_pct": 0.05,
            "note": "optimal conditions",
        },
    }
    state = reset_state(config)
    state["equity"] = 50000.0  # Plenty of equity
    state["gpt_safe_mode"] = True  # But safe mode is active
    
    # Perfect setup from GPT
    gpt_decision = GptDecision(action="long", confidence=1.0, notes="perfect setup")
    
    decision = evaluate_risk(snapshot, gpt_decision, state, config)
    
    # Safe mode still forces flat
    assert decision.approved is False
    assert decision.side == Side.FLAT.value
    assert "safe_mode" in decision.reason.lower()


# ---------------------------------------------------------------------------
# Integration Tests
# ---------------------------------------------------------------------------


def test_safe_mode_activation_flow():
    """Test the full flow: errors accumulate and trigger safe mode."""
    config = Config()
    state = reset_state(config)
    
    now = time.time()
    
    # Simulate consecutive errors
    assert is_gpt_safe_mode(state) is False
    
    record_gpt_error(state, now)
    assert is_gpt_safe_mode(state) is False
    assert state["gpt_error_count"] == 1
    
    record_gpt_error(state, now + 1)
    assert is_gpt_safe_mode(state) is False
    assert state["gpt_error_count"] == 2
    
    record_gpt_error(state, now + 2)  # Third error triggers safe mode
    assert is_gpt_safe_mode(state) is True
    assert state["gpt_error_count"] == 3
    
    # Verify risk engine now forces flat
    snapshot = {
        "symbol": "BTCUSDT",
        "timestamp": time.time(),
        "price": 50000.0,
        "volatility_mode": "normal",
        "range_position": "mid",
        "timing_state": "normal",
        "danger_mode": False,
    }
    gpt_decision = GptDecision(action="long", confidence=0.9)
    
    decision = evaluate_risk(snapshot, gpt_decision, state, config)
    assert decision.approved is False
    assert "safe_mode" in decision.reason.lower()


def test_safe_mode_manual_reset_flow():
    """Test that safe mode requires manual reset to clear."""
    config = Config()
    state = reset_state(config)
    
    # Trigger safe mode
    now = time.time()
    for i in range(GPT_SAFE_MODE_ERROR_THRESHOLD):
        record_gpt_error(state, now + i)
    
    assert is_gpt_safe_mode(state) is True
    
    # Successful GPT call doesn't clear safe mode
    record_gpt_success(state)
    assert is_gpt_safe_mode(state) is True  # Still active
    
    # Manual reset clears safe mode
    clear_gpt_safe_mode(state)
    assert is_gpt_safe_mode(state) is False
    assert state["gpt_error_count"] == 0


def test_error_window_expiration():
    """Test that errors outside the window don't accumulate."""
    config = Config()
    state = reset_state(config)
    
    # First error
    old_time = time.time() - GPT_SAFE_MODE_ERROR_WINDOW_SEC - 100
    record_gpt_error(state, old_time)
    assert state["gpt_error_count"] == 1
    
    # Second error much later (outside window)
    now = time.time()
    record_gpt_error(state, now)
    
    # Should reset to 1, not accumulate to 2
    assert state["gpt_error_count"] == 1
    assert state["gpt_safe_mode"] is False


def test_safe_mode_persists_across_state_save_load(tmp_path):
    """Test that safe mode state persists across save/load cycles."""
    config = Config()
    config.state_file = str(tmp_path / "test_state.json")
    
    # Create state with safe mode active
    state = reset_state(config)
    state["gpt_safe_mode"] = True
    state["gpt_error_count"] = 5
    state["last_gpt_error_timestamp"] = time.time()
    
    # Save state
    save_state(state, config)
    
    # Load state
    loaded_state = load_state(config)
    
    # Safe mode should persist
    assert loaded_state["gpt_safe_mode"] is True
    assert loaded_state["gpt_error_count"] == 5
    assert loaded_state["last_gpt_error_timestamp"] is not None

