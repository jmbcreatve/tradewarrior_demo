from types import SimpleNamespace

import pytest

from config import Config
from tw5.exit_rules import compute_tp_levels
from tw5.replay import (
    _build_open_pos,
    _force_exit,
    _manage_and_process_bar,
)


def _make_open_pos(side: str, stop_current: float) -> dict:
    entry = 100.0
    stop = 90.0 if side == "long" else 110.0
    tp_levels = compute_tp_levels(
        entry=entry,
        stop=stop,
        side=side,
        r_mults=[1.2, 2.0, 3.0],
        remaining_fracs=[0.30, 0.30, 1.0],
    )
    pos = _build_open_pos(
        side=side,
        entry_price=entry,
        stop_loss=stop,
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
    pos["stop_current"] = stop_current
    return pos


def test_tp1_then_stop_net_positive_long():
    pos = _make_open_pos("long", stop_current=95.0)
    candle = {"open": 100.0, "high": 112.0, "low": 95.0, "close": 105.0}
    fills = []
    trade = _manage_and_process_bar(
        open_pos=pos,
        candle=candle,
        ts=60.0,
        bar_idx=1,
        manage_every=100,
        fee_rate=0.0,
        slippage_bps=0.0,
        config=Config(),
        fills=fills,
    )
    assert trade is not None
    assert trade["exit_reason"] == "stop_loss"
    assert trade["tp1_hit"] is True
    assert trade["tp2_hit"] is False
    assert trade["tp3_hit"] is False
    assert trade["pnl"] > 0.0


def test_tp1_tp2_then_stop_long():
    pos = _make_open_pos("long", stop_current=95.0)
    candle = {"open": 100.0, "high": 125.0, "low": 95.0, "close": 120.0}
    fills = []
    trade = _manage_and_process_bar(
        open_pos=pos,
        candle=candle,
        ts=60.0,
        bar_idx=1,
        manage_every=100,
        fee_rate=0.0,
        slippage_bps=0.0,
        config=Config(),
        fills=fills,
    )
    assert trade is not None
    assert trade["tp1_hit"] and trade["tp2_hit"]
    assert trade["pnl"] > 0.0


def test_full_tp_ladder_long():
    pos = _make_open_pos("long", stop_current=95.0)
    candle = {"open": 100.0, "high": 135.0, "low": 95.0, "close": 130.0}
    fills = []
    trade = _manage_and_process_bar(
        open_pos=pos,
        candle=candle,
        ts=60.0,
        bar_idx=1,
        manage_every=100,
        fee_rate=0.0,
        slippage_bps=0.0,
        config=Config(),
        fills=fills,
    )
    # Remaining size zero should yield take-profit exit
    assert trade is not None
    assert trade["exit_reason"] == "take_profit"
    assert trade["tp3_hit"] is True
    assert pytest.approx(trade["pnl"]) == trade["pnl"]  # smoke: valid number


def test_tp1_then_stop_net_positive_short():
    pos = _make_open_pos("short", stop_current=101.0)
    candle = {"open": 100.0, "high": 103.0, "low": 88.0, "close": 95.0}
    fills = []
    trade = _manage_and_process_bar(
        open_pos=pos,
        candle=candle,
        ts=60.0,
        bar_idx=1,
        manage_every=100,
        fee_rate=0.0,
        slippage_bps=0.0,
        config=Config(),
        fills=fills,
    )
    assert trade is not None
    assert trade["tp1_hit"] is True
    assert trade["pnl"] > 0.0


def test_force_exit_remaining():
    pos = _make_open_pos("long", stop_current=95.0)
    fills = []
    trade = _force_exit(
        pos,
        price=98.0,
        ts=999.0,
        fee_rate=0.0,
        slippage_bps=0.0,
        fills=fills,
        reason="end_of_data",
    )
    assert trade["exit_reason"] == "end_of_data"
    assert trade["pnl"] != 0.0
