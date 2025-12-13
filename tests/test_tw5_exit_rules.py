import types

import pytest

from tw5.exit_rules import (
    compute_tp_absolute_fracs_from_remaining,
    compute_tp_levels,
    compute_trailing_stop,
)


def _cfg(**overrides):
    defaults = {
        "tw5_trail_early_trigger_r": 0.7,
        "tw5_trail_early_stop_r": -0.25,
        "tw5_trail_after_tp1_stop_r": 0.10,
        "tw5_trail_after_tp2_stop_r": 1.0,
        "tw5_trail_runner_giveback_r": 1.0,
    }
    defaults.update(overrides)
    return types.SimpleNamespace(**defaults)


def test_compute_tp_absolute_fracs_from_remaining():
    remaining = [0.30, 0.30, 1.00]
    expected_abs = [0.30, 0.21, 0.49]

    abs_fracs = compute_tp_absolute_fracs_from_remaining(remaining)

    assert abs_fracs == pytest.approx(expected_abs)
    assert sum(abs_fracs) == pytest.approx(1.0)


def test_compute_tp_levels_price_math_long_short():
    r_mults = [1.2, 2.0, 3.0]
    remaining = [0.30, 0.30, 1.00]
    expected_abs = [0.30, 0.21, 0.49]

    long_levels = compute_tp_levels(100.0, 90.0, "long", r_mults, remaining)
    short_levels = compute_tp_levels(100.0, 110.0, "short", r_mults, remaining)

    assert [lvl[0] for lvl in long_levels] == pytest.approx([112.0, 120.0, 130.0])
    assert [lvl[1] for lvl in long_levels] == pytest.approx(expected_abs)
    assert long_levels[0][2] == "1.2R"

    assert [lvl[0] for lvl in short_levels] == pytest.approx([88.0, 80.0, 70.0])
    assert [lvl[1] for lvl in short_levels] == pytest.approx(expected_abs)
    assert short_levels[1][2] == "2R"


def test_compute_trailing_stop_monotonic():
    cfg = _cfg()
    entry = 100.0
    R = 10.0
    # Start with a tight stop already above entry; rules must not widen it.
    initial_stop = 102.0

    stop_after_tp1 = compute_trailing_stop(
        entry=entry,
        initial_stop=initial_stop,
        side="long",
        R=R,
        high_water_price=120.0,  # 2R move
        tp1_hit=True,
        tp2_hit=False,
        cfg=cfg,
    )
    assert stop_after_tp1 == pytest.approx(initial_stop)

    stop_after_tp2 = compute_trailing_stop(
        entry=entry,
        initial_stop=stop_after_tp1,
        side="long",
        R=R,
        high_water_price=130.0,  # 3R move
        tp1_hit=True,
        tp2_hit=True,
        cfg=cfg,
    )
    assert stop_after_tp2 >= stop_after_tp1
    assert stop_after_tp2 == pytest.approx(120.0)
