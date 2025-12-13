from types import SimpleNamespace

import pytest

from config import Config
from tw5.executor import ExecutionState, Tw5Executor


class DummyAdapter:
    def __init__(self):
        self.open_positions = []
        self.open_orders = []
        self.cancelled = []
        self.trigger_orders = []

    def get_open_positions(self, symbol):
        return list(self.open_positions)

    def get_open_orders(self, symbol):
        return list(self.open_orders)

    def cancel_orders(self, symbol, oids):
        self.cancelled.append((symbol, list(oids)))

    def place_trigger_order(self, **kwargs):
        self.trigger_orders.append(kwargs)
        return {"oid": len(self.trigger_orders)}


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
