import json
import os
import sys
from unittest.mock import patch, MagicMock
import pytest

from gpt_client import _build_user_message, call_gpt, validate_openai_api_key
from config import Config
from schemas import GptDecision


def test_build_user_message_includes_risk_summary_and_snapshot():
    snapshot = {
        "risk_envelope": {
            "max_notional": 150000,
            "max_leverage": 3,
            "max_risk_per_trade_pct": 1.25,
            "min_stop_distance_pct": 0.5,
            "max_stop_distance_pct": 4.0,
            "max_daily_loss_pct": 2.5,
            "note": "stay conservative",
        },
        "foo": "bar",
    }

    message = _build_user_message(snapshot)

    assert "RISK ENVELOPE" in message
    assert "SNAPSHOT JSON" in message
    assert "150000.0" in message
    assert "stay conservative" in message
    # Ensure the full snapshot JSON is present
    assert json.dumps(snapshot, sort_keys=True) in message


def test_build_user_message_defaults_when_missing_risk_envelope():
    snapshot = {"foo": "bar"}

    message = _build_user_message(snapshot)

    assert "- max_notional: 0.0" in message
    assert "SNAPSHOT JSON" in message
    assert json.dumps(snapshot, sort_keys=True) in message


def test_call_gpt_parses_rationale_from_response():
    """Test that call_gpt correctly parses 'rationale' field from GPT JSON response."""
    config = Config()
    snapshot = {"symbol": "BTCUSDT", "timestamp": 1234567890.0, "price": 50000.0}
    
    # Mock OpenAI response with "rationale" field (as per brain.txt)
    mock_response = {
        "choices": [
            {
                "message": {
                    "content": json.dumps({
                        "action": "long",
                        "size": 0.5,
                        "confidence": 0.75,
                        "rationale": "Uptrend with bullish microstructure and favorable risk context."
                    })
                }
            }
        ]
    }
    
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
        mock_openai = MagicMock()
        mock_openai.ChatCompletion.create.return_value = mock_response
        with patch.dict(sys.modules, {"openai": mock_openai}):
            decision = call_gpt(config, snapshot)
            
            assert isinstance(decision, GptDecision)
            assert decision.action == "long"
            assert decision.confidence == 0.75
            assert decision.notes == "Uptrend with bullish microstructure and favorable risk context."
            assert "rationale" in decision.notes or decision.notes  # Should contain the rationale text


def test_call_gpt_fallback_to_notes_field():
    """Test that call_gpt falls back to 'notes' field if 'rationale' is missing (backward compatibility)."""
    config = Config()
    snapshot = {"symbol": "BTCUSDT", "timestamp": 1234567890.0, "price": 50000.0}
    
    # Mock OpenAI response with "notes" field (old format)
    mock_response = {
        "choices": [
            {
                "message": {
                    "content": json.dumps({
                        "action": "flat",
                        "size": 0.0,
                        "confidence": 0.3,
                        "notes": "Old format with notes field"
                    })
                }
            }
        ]
    }
    
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
        mock_openai = MagicMock()
        mock_openai.ChatCompletion.create.return_value = mock_response
        with patch.dict(sys.modules, {"openai": mock_openai}):
            decision = call_gpt(config, snapshot)
            
            assert isinstance(decision, GptDecision)
            assert decision.action == "flat"
            assert decision.confidence == 0.3
            assert decision.notes == "Old format with notes field"


def test_rationale_flows_to_gpt_state_note():
    """Test that GPT rationale flows through: GPT response -> GptDecision.notes -> state -> next snapshot."""
    config = Config()
    snapshot = {"symbol": "BTCUSDT", "timestamp": 1234567890.0, "price": 50000.0}
    
    expected_rationale = "Test rationale: uptrend with strong structure"
    
    # Mock OpenAI response
    mock_response = {
        "choices": [
            {
                "message": {
                    "content": json.dumps({
                        "action": "long",
                        "size": 0.6,
                        "confidence": 0.7,
                        "rationale": expected_rationale
                    })
                }
            }
        ]
    }
    
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
        mock_openai = MagicMock()
        mock_openai.ChatCompletion.create.return_value = mock_response
        with patch.dict(sys.modules, {"openai": mock_openai}):
            # Step 1: Call GPT and get decision
            decision = call_gpt(config, snapshot)
            assert decision.notes == expected_rationale
            
            # Step 2: Simulate storing in state (as engine.py does)
            state = {}
            state["last_gpt_decision"] = decision.to_dict()
            state["gpt_state_note"] = decision.notes
            
            # Step 3: Verify state contains the rationale
            assert state["gpt_state_note"] == expected_rationale
            assert state["last_gpt_decision"]["notes"] == expected_rationale
            
            # Step 4: Verify it would appear in next snapshot (as build_features.py does)
            next_snapshot_gpt_state_note = state.get("gpt_state_note")
            assert next_snapshot_gpt_state_note == expected_rationale


def test_validate_openai_api_key_success():
    """Test that validate_openai_api_key returns the key when it's set correctly."""
    test_key = "sk-test1234567890abcdef"
    
    with patch.dict(os.environ, {"OPENAI_API_KEY": test_key}, clear=False):
        result = validate_openai_api_key()
        assert result == test_key


def test_validate_openai_api_key_missing():
    """Test that validate_openai_api_key raises RuntimeError when key is missing."""
    with patch.dict(os.environ, {}, clear=True):
        # Remove OPENAI_API_KEY if it exists
        if "OPENAI_API_KEY" in os.environ:
            del os.environ["OPENAI_API_KEY"]
        
        with pytest.raises(RuntimeError, match="OPENAI_API_KEY.*missing or empty"):
            validate_openai_api_key()


def test_validate_openai_api_key_empty():
    """Test that validate_openai_api_key raises RuntimeError when key is empty."""
    with patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=False):
        with pytest.raises(RuntimeError, match="OPENAI_API_KEY.*missing or empty"):
            validate_openai_api_key()
    
    with patch.dict(os.environ, {"OPENAI_API_KEY": "   "}, clear=False):
        with pytest.raises(RuntimeError, match="OPENAI_API_KEY.*missing or empty"):
            validate_openai_api_key()


def test_call_gpt_uses_env_var_for_api_key():
    """Test that call_gpt reads OPENAI_API_KEY from os.environ correctly."""
    config = Config()
    snapshot = {"symbol": "BTCUSDT", "timestamp": 1234567890.0, "price": 50000.0}
    
    test_key = "sk-test-key-from-env-var"
    mock_response = {
        "choices": [
            {
                "message": {
                    "content": json.dumps({
                        "action": "long",
                        "size": 0.5,
                        "confidence": 0.8,
                        "rationale": "Test rationale"
                    })
                }
            }
        ]
    }
    
    with patch.dict(os.environ, {"OPENAI_API_KEY": test_key}):
        # Mock the openai module that gets imported inside call_gpt
        import sys
        mock_openai = MagicMock()
        mock_openai.ChatCompletion.create.return_value = mock_response
        # Inject mock into sys.modules before call_gpt imports it
        with patch.dict(sys.modules, {"openai": mock_openai}):
            decision = call_gpt(config, snapshot)
            
            # Verify ChatCompletion.create was called
            assert mock_openai.ChatCompletion.create.called
            # Verify the API key was set on the mock module
            assert mock_openai.api_key == test_key
            # Verify the decision was made correctly
            assert decision.action == "long"
            assert decision.confidence == 0.8
