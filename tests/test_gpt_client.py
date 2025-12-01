import json

from gpt_client import _build_user_message


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
