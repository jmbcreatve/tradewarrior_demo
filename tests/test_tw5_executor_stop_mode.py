from types import SimpleNamespace

import pytest

from config import Config
from tw5.executor import Tw5Executor
from tw5.schemas import OrderPlan, OrderLeg, TPLevel


class MockAdapter:
    def __init__(self):
        self.trigger_calls = []
        self.limit_calls = []
        self.close_calls = []
        self.positions = []

    def place_order(self, **kwargs):
        return {"status": "filled", "fill_price": kwargs.get("entry_price", 100.0), "order_id": "entry-1"}

    def place_trigger_order(self, **kwargs):
        self.trigger_calls.append(kwargs)
        return {"response": {"data": {"statuses": [{"resting": {"oid": len(self.trigger_calls)}}]}}, "status": "ok"}

    def place_limit_order(self, **kwargs):
        self.limit_calls.append(kwargs)
        return {"response": {"data": {"statuses": [{"resting": {"oid": len(self.limit_calls)}}]}}, "status": "ok"}

    def get_open_positions(self, symbol):
        return list(self.positions)

    def close_position_or_raise(self, symbol, sz=None):
        self.close_calls.append({"symbol": symbol, "sz": sz})
        self.positions = []
        return {"status": "ok"}


def _plan(side: str) -> OrderPlan:
    leg = OrderLeg(
        id="leg1",
        entry_type="market",
        entry_price=100.0,
        entry_tag="price",
        size_frac=1.0,
        stop_loss=90.0 if side == "long" else 110.0,
        take_profits=[TPLevel(price=120.0, size_frac=1.0, tag="tp")],
    )
    return OrderPlan(mode="enter", side=side, legs=[leg], max_total_size_frac=1.0, confidence=0.5, rationale="test")


def test_stop_market_default():
    adapter = MockAdapter()
    executor = Tw5Executor(adapter)
    cfg = Config()
    snap = SimpleNamespace(symbol="BTCUSDT", price=100.0, timestamp=0.0)
    state = {}

    plan = _plan("long")
    clamp = SimpleNamespace(approved=True, clamped_plan=plan, original_plan=plan, reason="ok")

    executor.maybe_enter_from_plan(snap, clamp, cfg, state)

    assert adapter.trigger_calls, "stop should be placed"
    stop_call = adapter.trigger_calls[0]
    assert stop_call["reduce_only"] is True
    assert stop_call["is_buy"] is False  # closing long -> sell
    assert stop_call["is_market"] is True


def test_stop_market_default_flag_with_mock():
    class AdapterMock:
        def __init__(self):
            self.trigger_calls = []

        def place_order(self, **kwargs):
            return {"status": "filled", "fill_price": 100.0, "order_id": "entry-1"}

        def place_trigger_order(self, **kwargs):
            self.trigger_calls.append(kwargs)
            return {"response": {"data": {"statuses": [{"resting": {"oid": 1}}]}}, "status": "ok"}

        def place_limit_order(self, **kwargs):
            return {"response": {"data": {"statuses": [{"resting": {"oid": 2}}]}}, "status": "ok"}

    adapter = AdapterMock()
    executor = Tw5Executor(adapter)
    cfg = Config()
    snap = SimpleNamespace(symbol="BTCUSDT", price=100.0, timestamp=0.0)
    state = {}

    plan = _plan("long")
    clamp = SimpleNamespace(approved=True, clamped_plan=plan, original_plan=plan, reason="ok")

    executor.maybe_enter_from_plan(snap, clamp, cfg, state)

    assert adapter.trigger_calls, "stop placement should have been attempted"
    assert adapter.trigger_calls[0]["is_market"] is True


def test_flatten_no_flip_verification():
    adapter = MockAdapter()
    adapter.positions = [{"symbol": "BTCUSDT", "side": "long", "size": 1.0}]
    executor = Tw5Executor(adapter)

    executor._flatten_market("BTCUSDT", "long", 1.0)

    assert adapter.close_calls == [{"symbol": "BTCUSDT", "sz": 1.0}]
    # Positions cleared, no exception raised (no flip)
