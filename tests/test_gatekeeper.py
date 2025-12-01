from enums import RangePosition, TimingState, VolatilityMode, enum_to_str
from gatekeeper import should_call_gpt


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

    assert decision is False
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

    assert decision is True
    assert state["gpt_call_timestamps"]  # timestamp appended on approval
