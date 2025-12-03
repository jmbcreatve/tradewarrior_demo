from enums import RangePosition, TimingState, VolatilityMode, enum_to_str
from gatekeeper import (
    MAX_GPT_CALLS_PER_HOUR,
    MIN_ABS_PRICE_PCT,
    MIN_SECONDS_FOR_SLOW_TREND_CALL,
    MIN_SHAPE_SCORE_FOR_SLOW_TREND,
    should_call_gpt,
)


def _base_snapshot(price: float, shape_score: float, shape_bias: str, range_pos: str, impulse: str) -> dict:
    return {
        "timestamp": 1_000.0,
        "price": price,
        "range_position": range_pos,
        "volatility_mode": enum_to_str(VolatilityMode.NORMAL),
        "timing_state": enum_to_str(TimingState.NORMAL),
        "danger_mode": False,
        "microstructure": {"shape_score": shape_score, "shape_bias": shape_bias},
        "recent_price_path": {"impulse_state": impulse},
    }


def test_gatekeeper_skips_when_no_move_no_shape():
    snapshot = _base_snapshot(
        price=100.0,
        shape_score=0.0,
        shape_bias="none",
        range_pos=enum_to_str(RangePosition.MID),
        impulse="chop",
    )
    state = {"last_gpt_snapshot": {"price": 100.0}, "gpt_call_timestamps": [], "last_gpt_call_walltime": 0.0}

    decision = should_call_gpt(snapshot, prev_snapshot=snapshot, state=state)

    assert decision["should_call_gpt"] is False
    assert state["gpt_call_timestamps"] == []


def test_gatekeeper_triggers_on_strong_setup():
    snapshot = _base_snapshot(
        price=101.0,
        shape_score=0.8,
        shape_bias="bull",
        range_pos=enum_to_str(RangePosition.EXTREME_LOW),
        impulse="ripping_up",
    )
    state = {"last_gpt_snapshot": {"price": 100.0}, "gpt_call_timestamps": [], "last_gpt_call_walltime": 0.0}

    decision = should_call_gpt(snapshot, prev_snapshot=None, state=state)

    assert decision["should_call_gpt"] is True
    assert state["gpt_call_timestamps"]  # timestamp appended on approval


def test_gatekeeper_skips_when_since_last_gpt_negligible(caplog):
    snapshot = _base_snapshot(
        price=101.0,
        shape_score=0.8,
        shape_bias="bull",
        range_pos=enum_to_str(RangePosition.MID),
        impulse="ripping_up",
    )
    snapshot["since_last_gpt"] = {
        "time_since_last_gpt_sec": 30.0,
        "price_change_pct_since_last_gpt": 0.0001,
        "equity_change_since_last_gpt": 0.0,
        "trades_since_last_gpt": 0,
    }
    state = {"last_gpt_snapshot": {"price": 100.0}, "gpt_call_timestamps": [], "last_gpt_call_walltime": 0.0}

    with caplog.at_level("INFO"):
        decision = should_call_gpt(snapshot, prev_snapshot=None, state=state)

    assert decision["should_call_gpt"] is False
    assert state["gpt_call_timestamps"] == []
    assert any("since_last_gpt" in rec.message or "negligible change" in rec.message for rec in caplog.records)


def test_gatekeeper_triggers_on_slow_trend_timeout():
    """Test that slow drift + long time since last GPT triggers a call via slow trend timeout."""
    # Create a snapshot with slow drift (price change below normal threshold)
    # but above MIN_ABS_PRICE_PCT, and long time since last GPT
    snapshot = _base_snapshot(
        price=100.05,  # Small move: 0.05% (above MIN_ABS_PRICE_PCT 0.05% but below MIN_PRICE_MOVE_PCT 0.1%)
        shape_score=0.25,  # Some shape but below normal threshold (0.3)
        shape_bias="bull",
        range_pos=enum_to_str(RangePosition.MID),  # Not at extreme
        impulse="chop",  # No strong impulse
    )
    # Set since_last_gpt to show long time and small drift
    snapshot["since_last_gpt"] = {
        "time_since_last_gpt_sec": MIN_SECONDS_FOR_SLOW_TREND_CALL + 100.0,  # > 15 minutes
        "price_change_pct_since_last_gpt": 0.0006,  # Above MIN_ABS_PRICE_PCT (0.0005)
        "equity_change_since_last_gpt": 0.0,
        "trades_since_last_gpt": 10,  # Some trades happened
    }
    # Set up state with a previous GPT snapshot
    state = {
        "last_gpt_snapshot": {"price": 100.0},
        "gpt_call_timestamps": [],
        "last_gpt_call_walltime": snapshot["timestamp"] - MIN_SECONDS_FOR_SLOW_TREND_CALL - 100.0,
    }

    decision = should_call_gpt(snapshot, prev_snapshot=snapshot, state=state)

    assert decision["should_call_gpt"] is True
    assert decision["reason"] == "slow_trend_timeout"
    assert len(state["gpt_call_timestamps"]) == 1
    assert state["last_gpt_call_walltime"] == snapshot["timestamp"]


def test_gatekeeper_slow_trend_respects_rate_limits():
    """Test that rate limits still prevent over-calling even with slow trend conditions."""
    # Create a snapshot with slow drift and long time since last GPT
    snapshot = _base_snapshot(
        price=100.05,
        shape_score=MIN_SHAPE_SCORE_FOR_SLOW_TREND,  # Meets slow trend threshold
        shape_bias="bull",
        range_pos=enum_to_str(RangePosition.MID),
        impulse="chop",
    )
    snapshot["since_last_gpt"] = {
        "time_since_last_gpt_sec": MIN_SECONDS_FOR_SLOW_TREND_CALL + 100.0,
        "price_change_pct_since_last_gpt": MIN_ABS_PRICE_PCT + 0.0001,  # Above threshold
        "equity_change_since_last_gpt": 0.0,
        "trades_since_last_gpt": 10,
    }

    # Set up state with max calls already in the last hour
    now_ts = snapshot["timestamp"]
    # Create 12 timestamps in the last hour (max calls)
    call_timestamps = [now_ts - 100.0 * i for i in range(MAX_GPT_CALLS_PER_HOUR)]
    state = {
        "last_gpt_snapshot": {"price": 100.0},
        "gpt_call_timestamps": call_timestamps,
        "last_gpt_call_walltime": call_timestamps[0],
    }

    decision = should_call_gpt(snapshot, prev_snapshot=snapshot, state=state)

    assert decision["should_call_gpt"] is False
    assert decision["reason"] == "max_calls"
    # Timestamps should not have been modified
    assert len(state["gpt_call_timestamps"]) == MAX_GPT_CALLS_PER_HOUR
