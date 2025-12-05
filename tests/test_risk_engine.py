import json
import re

import pytest

from config import Config
from enums import Side
from risk_engine import evaluate_risk, _check_daily_loss_limit
from schemas import GptDecision


def test_risk_rejects_when_price_invalid():
    config = Config()
    snapshot = {"price": 0.0, "volatility_mode": "normal", "range_position": "mid", "timing_state": "normal"}
    state = {"equity": 10_000}

    decision = evaluate_risk(snapshot, GptDecision(action="long", confidence=1.0), state, config)

    assert decision.approved is False
    assert decision.reason == "invalid_price"


def test_risk_rejects_when_equity_missing():
    config = Config()
    snapshot = {"price": 100.0, "volatility_mode": "normal", "range_position": "mid", "timing_state": "normal"}
    state = {"equity": -50}

    decision = evaluate_risk(snapshot, GptDecision(action="long", confidence=1.0), state, config)

    assert decision.approved is False
    assert decision.reason == "no_equity"


def test_risk_rejects_flat_action():
    config = Config()
    snapshot = {"price": 100.0, "volatility_mode": "normal", "range_position": "mid", "timing_state": "normal"}
    state = {"equity": 10_000}

    decision = evaluate_risk(snapshot, GptDecision(action="flat", confidence=1.0), state, config)

    assert decision.approved is False
    assert decision.side == Side.FLAT.value


def test_risk_approves_basic_long_setup():
    config = Config()
    snapshot = {
        "price": 100.0,
        "volatility_mode": "normal",
        "range_position": "mid",
        "timing_state": "normal",
        "danger_mode": False,
    }
    state = {"equity": 10_000}

    decision = evaluate_risk(snapshot, GptDecision(action="long", confidence=0.9), state, config)

    assert decision.approved is True
    assert decision.side == Side.LONG.value
    assert decision.position_size > 0


def test_risk_envelope_shrinks_large_proposal():
    config = Config(risk_per_trade=0.01, max_leverage=5.0)
    snapshot = {
        "price": 100.0,
        "volatility_mode": "normal",
        "range_position": "mid",
        "timing_state": "normal",
        "danger_mode": False,
        "risk_envelope": {
            "max_risk_per_trade_pct": 0.002,
            "max_leverage": 1.5,
            "max_notional": 1000.0,
        },
    }
    state = {"equity": 20_000}

    decision = evaluate_risk(snapshot, GptDecision(action="long", confidence=1.0), state, config)

    assert decision.approved is True
    assert decision.side == Side.LONG.value
    notional = decision.position_size * snapshot["price"]
    assert notional <= snapshot["risk_envelope"]["max_notional"] + 1e-8
    assert decision.leverage <= snapshot["risk_envelope"]["max_leverage"] + 1e-8
    stop_pct = 0.005  # normal vol stop
    risk_pct = (notional * stop_pct) / (decision.leverage * state["equity"])
    assert risk_pct <= snapshot["risk_envelope"]["max_risk_per_trade_pct"] + 1e-6


def test_stop_distance_respects_envelope_min_max():
    config = Config(risk_per_trade=0.01, max_leverage=3.0)
    snapshot = {
        "price": 100.0,
        "volatility_mode": "normal",
        "range_position": "mid",
        "timing_state": "normal",
        "danger_mode": False,
        "risk_envelope": {
            "max_risk_per_trade_pct": 0.02,
            "max_leverage": 3.0,
            "max_notional": 100_000.0,
            "min_stop_distance_pct": 0.02,  # force wider than vol-based stop
            "max_stop_distance_pct": 0.03,
            "max_daily_loss_pct": 0.03,
        },
    }
    state = {"equity": 10_000}

    decision = evaluate_risk(snapshot, GptDecision(action="long", confidence=1.0), state, config)

    assert decision.approved is True
    expected_stop_distance = snapshot["price"] * 0.02  # clamped up to envelope min
    assert decision.stop_loss_price == pytest.approx(snapshot["price"] - expected_stop_distance)
    assert decision.take_profit_price == pytest.approx(snapshot["price"] + 2 * expected_stop_distance)


def test_stop_distance_clamped_to_max_when_vol_wide():
    config = Config(risk_per_trade=0.01, max_leverage=3.0)
    snapshot = {
        "price": 200.0,
        "volatility_mode": "explosive",
        "range_position": "mid",
        "timing_state": "normal",
        "danger_mode": False,
        "risk_envelope": {
            "max_risk_per_trade_pct": 0.02,
            "max_leverage": 5.0,
            "max_notional": 200_000.0,
            "min_stop_distance_pct": 0.001,
            "max_stop_distance_pct": 0.003,  # cap the wide vol stop
            "max_daily_loss_pct": 0.03,
        },
    }
    state = {"equity": 20_000}

    decision = evaluate_risk(snapshot, GptDecision(action="long", confidence=0.9), state, config)

    assert decision.approved is True
    expected_stop_pct = 0.003  # clamped down to envelope max
    expected_stop_distance = snapshot["price"] * expected_stop_pct
    assert decision.stop_loss_price == pytest.approx(snapshot["price"] - expected_stop_distance)
    assert decision.take_profit_price == pytest.approx(snapshot["price"] + 2 * expected_stop_distance)


def test_risk_envelope_blocks_when_zeroed():
    config = Config()
    snapshot = {
        "price": 150.0,
        "volatility_mode": "normal",
        "range_position": "mid",
        "timing_state": "normal",
        "danger_mode": False,
        "risk_envelope": {
            "max_risk_per_trade_pct": 0.0,
            "max_leverage": 0.0,
            "max_notional": 0.0,
        },
    }
    state = {"equity": 10_000}

    decision = evaluate_risk(snapshot, GptDecision(action="short", confidence=0.8), state, config)

    assert decision.approved is False
    assert decision.side == Side.FLAT.value
    assert "risk envelope" in decision.reason.lower()


def test_risk_logs_envelope_and_decision_for_approved_trade(caplog):
    """Verify that approved trades log both risk_envelope and risk_decision info."""
    config = Config(risk_per_trade=0.01, max_leverage=3.0)
    snapshot = {
        "symbol": "BTCUSDT",
        "timestamp": 1000.0,
        "price": 100.0,
        "volatility_mode": "normal",
        "range_position": "mid",
        "timing_state": "normal",
        "danger_mode": False,
        "risk_envelope": {
            "max_risk_per_trade_pct": 0.01,
            "max_leverage": 3.0,
            "max_notional": 5000.0,
            "min_stop_distance_pct": 0.005,
            "max_stop_distance_pct": 0.03,
            "max_daily_loss_pct": 0.03,
            "note": "baseline_vol;timing_normal",
        },
    }
    state = {"equity": 10_000}

    with caplog.at_level("INFO"):
        decision = evaluate_risk(snapshot, GptDecision(action="long", confidence=0.8), state, config)

    assert decision.approved is True
    assert decision.side == Side.LONG.value
    
    # Find the RiskDecision event log
    risk_log_found = False
    for record in caplog.records:
        if "RiskDecision event:" in record.message:
            risk_log_found = True
            msg = record.message
            
            # Verify risk_envelope is present in the log
            assert "risk_envelope" in msg or "'risk_envelope'" in msg, f"risk_envelope not found in log: {msg}"
            
            # Verify envelope note is present
            assert "note" in msg or "'note'" in msg, f"note not found in log: {msg}"
            
            # Verify RiskDecision fields are present
            assert "approved" in msg or "'approved'" in msg, f"approved not found in log: {msg}"
            assert "side" in msg or "'side'" in msg, f"side not found in log: {msg}"
            assert "position_size" in msg or "'position_size'" in msg, f"position_size not found in log: {msg}"
            assert "leverage" in msg or "'leverage'" in msg, f"leverage not found in log: {msg}"
            assert "stop_loss_price" in msg or "'stop_loss_price'" in msg, f"stop_loss_price not found in log: {msg}"
            assert "stop_distance_pct" in msg or "'stop_distance_pct'" in msg, f"stop_distance_pct not found in log: {msg}"
            
            # Verify envelope fields are present
            assert "max_notional" in msg or "'max_notional'" in msg, f"max_notional not found in log: {msg}"
            assert "max_leverage" in msg or "'max_leverage'" in msg, f"max_leverage not found in log: {msg}"
            assert "max_risk_per_trade_pct" in msg or "'max_risk_per_trade_pct'" in msg, f"max_risk_per_trade_pct not found in log: {msg}"
            
            break
    
    assert risk_log_found, "RiskDecision event log not found"
    
    # Verify envelope was stored in state
    assert "last_risk_envelope" in state
    assert state["last_risk_envelope"] is not None
    assert "note" in state["last_risk_envelope"]


def test_risk_logs_envelope_note_for_volatility_tightening(caplog):
    """Verify that envelope note shows volatility-based tightening."""
    config = Config(risk_per_trade=0.01, max_leverage=3.0)
    snapshot = {
        "symbol": "BTCUSDT",
        "timestamp": 1000.0,
        "price": 100.0,
        "volatility_mode": "high",
        "range_position": "mid",
        "timing_state": "normal",
        "danger_mode": False,
    }
    state = {"equity": 10_000}

    with caplog.at_level("INFO"):
        decision = evaluate_risk(snapshot, GptDecision(action="long", confidence=0.8), state, config)

    assert decision.approved is True
    
    # Find the RiskDecision event log
    for record in caplog.records:
        if "RiskDecision event:" in record.message:
            # Check that the log contains envelope info
            assert "risk_envelope" in record.message or "'risk_envelope'" in record.message
            # Check for note field (should contain "trim_for_vol" for high volatility)
            assert "note" in record.message or "'note'" in record.message
            # The note should indicate volatility trimming
            if "trim_for_vol" in record.message:
                break
    
    # Verify envelope note in state contains volatility info
    assert "last_risk_envelope" in state
    assert state["last_risk_envelope"] is not None
    envelope_note = state["last_risk_envelope"].get("note", "")
    assert "trim_for_vol" in envelope_note or "vol" in envelope_note.lower()


def test_check_daily_loss_limit_allows_trading_when_under_limit():
    """Test that daily loss limit check allows trading when under limit."""
    daily_pnl = -100.0  # -$100 loss
    daily_start_equity = 10_000.0  # Started with $10k
    max_daily_loss_pct = 0.03  # 3% limit = $300 max loss

    exceeded = _check_daily_loss_limit(daily_pnl, daily_start_equity, max_daily_loss_pct)
    assert exceeded is False  # -$100 is less than -$300 limit


def test_check_daily_loss_limit_blocks_when_limit_exceeded():
    """Test that daily loss limit check blocks trading when limit exceeded."""
    daily_pnl = -400.0  # -$400 loss
    daily_start_equity = 10_000.0  # Started with $10k
    max_daily_loss_pct = 0.03  # 3% limit = $300 max loss

    exceeded = _check_daily_loss_limit(daily_pnl, daily_start_equity, max_daily_loss_pct)
    assert exceeded is True  # -$400 exceeds -$300 limit


def test_check_daily_loss_limit_allows_when_no_tracking():
    """Test that daily loss limit check allows trading when daily tracking not initialized."""
    daily_pnl = -1000.0  # Large loss
    daily_start_equity = None  # Not initialized
    max_daily_loss_pct = 0.03

    exceeded = _check_daily_loss_limit(daily_pnl, daily_start_equity, max_daily_loss_pct)
    assert exceeded is False  # No tracking means allow trading


def test_risk_rejects_trade_when_daily_loss_limit_exceeded():
    """Test that risk engine rejects trades when daily loss limit is exceeded."""
    config = Config()
    snapshot = {
        "price": 100.0,
        "volatility_mode": "normal",
        "range_position": "mid",
        "timing_state": "normal",
        "danger_mode": False,
        "risk_envelope": {
            "max_risk_per_trade_pct": 0.01,
            "max_leverage": 3.0,
            "max_notional": 5000.0,
            "min_stop_distance_pct": 0.005,
            "max_stop_distance_pct": 0.03,
            "max_daily_loss_pct": 0.03,  # 3% daily loss limit
            "note": "baseline",
        },
    }
    # State with daily loss limit exceeded: -$400 on $10k start = -4% (exceeds 3% limit)
    state = {
        "equity": 9_600,  # Current equity
        "daily_start_equity": 10_000.0,
        "daily_pnl": -400.0,  # -4% loss, exceeds 3% limit
    }

    decision = evaluate_risk(snapshot, GptDecision(action="long", confidence=0.9), state, config)

    assert decision.approved is False
    assert decision.reason == "daily_loss_limit_exceeded"
    assert decision.side == Side.FLAT.value


def test_risk_allows_trade_when_daily_loss_under_limit():
    """Test that risk engine allows trades when daily loss is under limit."""
    config = Config()
    snapshot = {
        "price": 100.0,
        "volatility_mode": "normal",
        "range_position": "mid",
        "timing_state": "normal",
        "danger_mode": False,
        "risk_envelope": {
            "max_risk_per_trade_pct": 0.01,
            "max_leverage": 3.0,
            "max_notional": 5000.0,
            "min_stop_distance_pct": 0.005,
            "max_stop_distance_pct": 0.03,
            "max_daily_loss_pct": 0.03,  # 3% daily loss limit
            "note": "baseline",
        },
    }
    # State with daily loss under limit: -$200 on $10k start = -2% (under 3% limit)
    state = {
        "equity": 9_800,  # Current equity
        "daily_start_equity": 10_000.0,
        "daily_pnl": -200.0,  # -2% loss, under 3% limit
    }

    decision = evaluate_risk(snapshot, GptDecision(action="long", confidence=0.9), state, config)

    assert decision.approved is True
    assert decision.side == Side.LONG.value


def test_risk_logs_envelope_for_rejected_trades(caplog):
    """Verify that rejected trades also log risk_envelope info for auditability."""
    config = Config()
    # Use a snapshot with zeroed envelope to force rejection
    snapshot = {
        "symbol": "BTCUSDT",
        "timestamp": 1000.0,
        "price": 100.0,
        "volatility_mode": "normal",
        "range_position": "mid",
        "timing_state": "normal",
        "danger_mode": False,
        "risk_envelope": {
            "max_risk_per_trade_pct": 0.0,  # Zero to force rejection
            "max_leverage": 0.0,
            "max_notional": 0.0,
            "min_stop_distance_pct": 0.005,
            "max_stop_distance_pct": 0.03,
            "max_daily_loss_pct": 0.03,
            "note": "test_zeroed_envelope",
        },
    }
    state = {"equity": 10_000}

    with caplog.at_level("INFO"):
        decision = evaluate_risk(snapshot, GptDecision(action="long", confidence=0.9), state, config)

    assert decision.approved is False
    
    # Find the RiskDecision event log for rejected trade
    risk_log_found = False
    for record in caplog.records:
        if "RiskDecision event:" in record.message:
            risk_log_found = True
            msg = record.message
            
            # Verify risk_envelope is present even for rejected trades
            assert "risk_envelope" in msg or "'risk_envelope'" in msg, \
                f"risk_envelope not found in rejected trade log: {msg}"
            
            # Verify the rejection is logged
            assert "approved" in msg and "False" in msg, \
                f"approved=False not found in log: {msg}"
            
            break
    
    assert risk_log_found, "RiskDecision event log not found for rejected trade"


def test_risk_logs_envelope_for_gpt_flat_action(caplog):
    """Verify that GPT flat actions also get logged with envelope info."""
    config = Config(risk_per_trade=0.01, max_leverage=3.0)
    snapshot = {
        "symbol": "BTCUSDT",
        "timestamp": 1000.0,
        "price": 100.0,
        "volatility_mode": "normal",
        "range_position": "mid",
        "timing_state": "normal",
        "danger_mode": False,
        "risk_envelope": {
            "max_risk_per_trade_pct": 0.01,
            "max_leverage": 3.0,
            "max_notional": 5000.0,
            "note": "baseline",
        },
    }
    state = {"equity": 10_000}

    with caplog.at_level("INFO"):
        decision = evaluate_risk(snapshot, GptDecision(action="flat", confidence=0.8), state, config)

    assert decision.approved is False
    assert decision.side == Side.FLAT.value
    
    # Verify logging occurred
    found_gpt_flat_log = False
    for record in caplog.records:
        if "GPT requested FLAT" in record.message:
            found_gpt_flat_log = True
            break
    
    assert found_gpt_flat_log, "GPT flat log not found"
