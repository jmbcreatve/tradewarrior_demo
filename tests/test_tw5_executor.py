from types import SimpleNamespace

import pytest

from config import Config
from tw5.executor import ExecutionState, Tw5Executor
from tw5.schemas import OrderPlan, OrderLeg


class DummyAdapter:
    def __init__(self):
        self.open_positions = []
        self.open_orders = []
        self.cancelled = []
        self.trigger_orders = []
        self.limit_orders = []
        self.market_orders = []

    def get_open_positions(self, symbol):
        return list(self.open_positions)

    def get_open_orders(self, symbol):
        return list(self.open_orders)

    def cancel_orders(self, symbol, oids):
        self.cancelled.append((symbol, list(oids)))

    def place_trigger_order(self, **kwargs):
        self.trigger_orders.append(kwargs)
        return {"oid": len(self.trigger_orders)}

    def place_limit_order(self, **kwargs):
        self.limit_orders.append(kwargs)
        return {"oid": len(self.limit_orders) + 100}

    def place_order(self, **kwargs):
        self.market_orders.append(kwargs)
        return {
            "status": "filled",
            "fill_price": kwargs.get("size", 100.0),
            "order_id": len(self.market_orders) + 1000,
        }


def test_manage_open_position_never_widens_stop():
    adapter = DummyAdapter()
    adapter.open_positions = [{"side": "long", "size": 1.0, "entry_price": 100.0}]
    adapter.open_orders = [{"oid": 1, "limitPx": 105.0}]

    exec_state = ExecutionState(
        symbol="BTCUSDT",
        side="long",
        stop_oid=1,
        tp_oids={"1R": 2},
        entry_price=100.0,
        initial_stop=90.0,
        R=10.0,
        high_water_price=120.0,
        last_manage_ts=0.0,
        position_size_last_seen=1.0,
    )

    state = {"tw5_exec": exec_state.to_dict()}
    snapshot = SimpleNamespace(symbol="BTCUSDT", price=110.0, timestamp=200.0)
    cfg = Config()
    executor = Tw5Executor(adapter)

    executor.manage_open_position(snapshot, cfg, state)

    # Desired trailing stop (97.5) is below current stop (105), so no cancel/replace
    assert adapter.cancelled == []
    assert adapter.trigger_orders == []


def test_sync_reconciliation_clears_stale_state_and_cancels():
    adapter = DummyAdapter()
    adapter.open_positions = []
    adapter.open_orders = []

    exec_state = ExecutionState(
        symbol="ETHUSDT",
        side="long",
        stop_oid=10,
        tp_oids={"tp1": 11},
        entry_price=2000.0,
        initial_stop=1900.0,
        R=100.0,
    )
    state = {"tw5_exec": exec_state.to_dict()}
    snapshot = SimpleNamespace(symbol="ETHUSDT", price=2050.0, timestamp=100.0)
    cfg = Config()
    executor = Tw5Executor(adapter)

    executor.sync_and_reconcile(snapshot, cfg, state)

    # Cancels tracked oids and clears executor state to flat
    assert adapter.cancelled == [("ETHUSDT", [10, 11])]
    cleared = state.get("tw5_exec", {})
    assert cleared.get("side") == "flat"
    assert cleared.get("stop_oid") is None
    assert cleared.get("tp_oids") == {}


def test_maybe_enter_from_plan_halted_no_error():
    """Regression test: ensure exec_state is loaded before halted check."""
    adapter = DummyAdapter()
    executor = Tw5Executor(adapter)

    # Create a state that triggers trading_halted
    state = {"trading_halted": True}

    # Create a valid clamp_result that would normally attempt entry
    plan = OrderPlan(
        mode="enter",
        side="long",
        legs=[
            OrderLeg(
                id="leg1",
                entry_type="market",
                entry_price=100.0,
                entry_tag="test",
                size_frac=0.5,
                stop_loss=90.0,
            )
        ],
    )
    clamp_result = SimpleNamespace(approved=True, clamped_plan=plan)

    snapshot = SimpleNamespace(symbol="BTCUSDT", price=100.0, timestamp=1000.0)
    cfg = Config()

    # This should not raise NameError
    result = executor.maybe_enter_from_plan(snapshot, clamp_result, cfg, state)

    # Should return None and not place any orders
    assert result is None
    assert adapter.market_orders == []
    assert adapter.trigger_orders == []
    assert adapter.limit_orders == []

    # Should record the halt reason in exec_state
    exec_state = state.get("tw5_exec", {})
    assert exec_state.get("last_error") == "halted:circuit_breaker_active"
