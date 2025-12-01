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
