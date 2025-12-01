from config import load_config
from enums import TimingState, VolatilityMode
from risk_envelope import compute_risk_envelope


def test_envelope_debug_outputs_expected_keys():
    cfg = load_config()
    env = compute_risk_envelope(
        config=cfg,
        equity=12_345.0,
        volatility_mode=VolatilityMode.UNKNOWN,
        danger_mode=False,
        timing_state=TimingState.NORMAL,
    )

    env_dict = env.to_dict()
    expected_keys = {
        "max_notional",
        "max_leverage",
        "max_risk_per_trade_pct",
        "min_stop_distance_pct",
        "max_stop_distance_pct",
        "max_daily_loss_pct",
        "note",
    }

    assert set(env_dict.keys()) == expected_keys
