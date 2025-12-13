from types import SimpleNamespace

from config import Config
from tw5.exit_rules import compute_tp_levels
from tw5.replay import _build_open_pos, _manage_and_process_bar


def test_manage_triggers_by_time_not_bar_count():
    cfg = Config()
    cfg.tw5_manage_interval_sec = 120.0

    # Entry at t=0 with stop=90, tp ladder standard
    tp_levels = compute_tp_levels(
        entry=100.0,
        stop=90.0,
        side="long",
        r_mults=[1.2, 2.0, 3.0],
        remaining_fracs=[0.3, 0.3, 1.0],
    )
    pos = _build_open_pos(
        side="long",
        entry_price=100.0,
        stop_loss=90.0,
        size=1.0,
        tp_levels=tp_levels,
        ts=0.0,
        equity=10_000.0,
        risk_value=100.0,
        snapshot=SimpleNamespace(),
        gpt_called=False,
        gpt_reason="",
        clamp_reason="",
        fee_rate=0.0,
    )

    # First bar at t=60s: should NOT manage yet
    candle1 = {"open": 100.0, "high": 105.0, "low": 99.0, "close": 104.0, "timestamp": 60}
    res1 = _manage_and_process_bar(
        open_pos=pos,
        candle=candle1,
        ts=60.0,
        bar_idx=0,
        manage_every=cfg.tw5_manage_interval_sec,
        fee_rate=0.0,
        slippage_bps=0.0,
        config=cfg,
        fills=[],
    )
    assert res1 is None
    # last_manage_ts should still be 0
    assert pos.get("last_manage_ts", 0.0) == 0.0

    # Second bar at t=180s: >= 120s elapsed, manage should fire and update last_manage_ts
    candle2 = {"open": 104.0, "high": 110.0, "low": 103.0, "close": 109.0, "timestamp": 180}
    res2 = _manage_and_process_bar(
        open_pos=pos,
        candle=candle2,
        ts=180.0,
        bar_idx=1,
        manage_every=cfg.tw5_manage_interval_sec,
        fee_rate=0.0,
        slippage_bps=0.0,
        config=cfg,
        fills=[],
    )
    assert res2 is None
    assert pos.get("last_manage_ts", 0.0) == 180.0
