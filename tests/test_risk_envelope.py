import pytest

from config import Config
from enums import TimingState, VolatilityMode
from risk_envelope import compute_risk_envelope


def test_compute_risk_envelope_normal_regime():
    config = Config(risk_per_trade=0.01, max_leverage=4.0)
    equity = 10_000.0

    env = compute_risk_envelope(
        config=config,
        equity=equity,
        volatility_mode=VolatilityMode.NORMAL,
        danger_mode=False,
        timing_state=TimingState.NORMAL,
    )

    assert 0.0 <= env.max_risk_per_trade_pct <= config.risk_per_trade
    assert 0.0 <= env.max_leverage <= config.max_leverage
    assert env.max_notional == pytest.approx(equity * env.max_leverage)
    assert env.min_stop_distance_pct < env.max_stop_distance_pct

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


def test_compute_risk_envelope_danger_mode_zeroes_risk():
    config = Config(risk_per_trade=0.02, max_leverage=5.0)
    equity = 5_000.0

    env = compute_risk_envelope(
        config=config,
        equity=equity,
        volatility_mode=VolatilityMode.HIGH,
        danger_mode=True,
        timing_state=TimingState.AVOID,
    )

    assert env.max_risk_per_trade_pct == 0.0
    assert env.max_leverage == 0.0
    assert env.max_notional == pytest.approx(0.0)
    assert env.min_stop_distance_pct < env.max_stop_distance_pct
    assert env.max_risk_per_trade_pct <= config.risk_per_trade
    assert env.max_leverage <= config.max_leverage
