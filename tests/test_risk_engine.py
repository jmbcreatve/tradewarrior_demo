from config import Config
from enums import Side
from risk_engine import evaluate_risk
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
