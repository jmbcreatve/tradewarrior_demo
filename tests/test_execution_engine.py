import json
import re

from config import Config
from execution_engine import execute_decision
from schemas import RiskDecision
from adapters.base_execution_adapter import BaseExecutionAdapter


class _DummyExecutionAdapter(BaseExecutionAdapter):
    def __init__(self, result):
        self.result = result
        self.called = False
        self.last_order = None

    def get_open_positions(self, symbol: str):
        return []

    def place_order(
        self,
        symbol: str,
        side: str,
        size: float,
        order_type: str = "market",
        stop_loss=None,
        take_profit=None,
        leverage=None,
    ):
        self.called = True
        self.last_order = {
            "symbol": symbol,
            "side": side,
            "size": size,
            "order_type": order_type,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "leverage": leverage,
        }
        return dict(self.result)

    def cancel_all_orders(self, symbol: str) -> None:
        return None


def test_execute_decision_calls_adapter_when_approved():
    config = Config()
    state = {
        "symbol": "BTCUSDT",
        "run_id": "run-123",
        "snapshot_id": "snap-001",
        "last_gpt_decision": {"side": "long", "confidence": 0.9},
    }
    risk_decision = RiskDecision(
        approved=True,
        side="long",
        position_size=1.0,
        leverage=3.0,
        stop_loss_price=99.5,
        take_profit_price=105.5,
        reason="test",
    )
    adapter_result = {"status": "filled", "fill_price": 101.0, "fee_paid": 0.1, "realized_pnl": 1.2}
    adapter = _DummyExecutionAdapter(adapter_result)

    result = execute_decision(risk_decision, config, state, adapter)

    assert adapter.called is True
    assert result["status"] == adapter_result["status"]


def test_execution_logs_envelope_and_decision_for_approved_trade(caplog):
    """Verify that execution logs contain both risk_envelope and risk_decision info."""
    config = Config()
    risk_envelope = {
        "max_notional": 5000.0,
        "max_leverage": 3.0,
        "max_risk_per_trade_pct": 0.01,
        "min_stop_distance_pct": 0.005,
        "max_stop_distance_pct": 0.03,
        "max_daily_loss_pct": 0.03,
        "note": "baseline_vol;timing_normal",
    }
    state = {
        "symbol": "BTCUSDT",
        "run_id": "run-123",
        "snapshot_id": "snap-001",
        "last_gpt_decision": {"side": "long", "confidence": 0.9},
        "last_risk_envelope": risk_envelope,
    }
    risk_decision = RiskDecision(
        approved=True,
        side="long",
        position_size=1.0,
        leverage=3.0,
        stop_loss_price=99.5,
        take_profit_price=105.5,
        reason="test",
    )
    adapter_result = {
        "status": "filled",
        "fill_price": 101.0,
        "avg_fill_price": 101.0,
        "fee_paid": 0.1,
        "realized_pnl": 1.2,
    }
    adapter = _DummyExecutionAdapter(adapter_result)

    with caplog.at_level("INFO"):
        result = execute_decision(risk_decision, config, state, adapter)

    assert result["status"] == "filled"
    
    # Find the Execution event log
    execution_log_found = False
    for record in caplog.records:
        if "Execution event:" in record.message:
            execution_log_found = True
            # Check that the log contains risk_envelope and risk_decision fields
            assert "risk_envelope" in record.message or "'risk_envelope'" in record.message
            assert "position_size" in record.message or "'position_size'" in record.message
            assert "leverage" in record.message or "'leverage'" in record.message
            assert "stop_loss_price" in record.message or "'stop_loss_price'" in record.message
            assert "stop_distance_pct" in record.message or "'stop_distance_pct'" in record.message
            # Check for envelope note
            assert "note" in record.message or "'note'" in record.message
            break
    
    assert execution_log_found, "Execution event log not found"
    
    # Also check execution trace log
    trace_log_found = False
    for record in caplog.records:
        if "Execution trace:" in record.message:
            trace_log_found = True
            assert "risk_envelope" in record.message or "'risk_envelope'" in record.message
            assert "position_size" in record.message or "'position_size'" in record.message
            assert "leverage" in record.message or "'leverage'" in record.message
            assert "stop_distance_pct" in record.message or "'stop_distance_pct'" in record.message
            break
    
    assert trace_log_found, "Execution trace log not found"


def test_execution_logs_envelope_for_rejected_trades(caplog):
    """Verify that rejected trades (approved=False) also log risk_envelope info."""
    config = Config()
    risk_envelope = {
        "max_notional": 5000.0,
        "max_leverage": 3.0,
        "max_risk_per_trade_pct": 0.01,
        "min_stop_distance_pct": 0.005,
        "max_stop_distance_pct": 0.03,
        "max_daily_loss_pct": 0.03,
        "note": "test_rejection_envelope",
    }
    state = {
        "symbol": "BTCUSDT",
        "run_id": "run-456",
        "snapshot_id": "snap-002",
        "last_gpt_decision": {"side": "long", "confidence": 0.8},
        "last_risk_envelope": risk_envelope,
    }
    # Rejected trade
    risk_decision = RiskDecision(
        approved=False,
        side="flat",
        position_size=0.0,
        leverage=0.0,
        stop_loss_price=None,
        take_profit_price=None,
        reason="risk_envelope_zeroed",
    )
    adapter = _DummyExecutionAdapter({"status": "no_trade"})

    with caplog.at_level("INFO"):
        result = execute_decision(risk_decision, config, state, adapter)

    # Adapter should NOT be called for rejected trades
    assert adapter.called is False
    assert result["status"] == "no_trade"
    
    # Find the Execution event log - should still log for rejected trades
    execution_log_found = False
    for record in caplog.records:
        if "Execution event:" in record.message:
            execution_log_found = True
            # Check that risk_envelope is logged even for rejected trades
            assert "risk_envelope" in record.message or "'risk_envelope'" in record.message, \
                f"risk_envelope not found in rejected trade log: {record.message}"
            break
    
    assert execution_log_found, "Execution event log not found for rejected trade"


def test_execution_logs_all_required_fields_for_approved_trade(caplog):
    """Comprehensive test that all required fields are present in execution logs."""
    config = Config()
    risk_envelope = {
        "max_notional": 5000.0,
        "max_leverage": 3.0,
        "max_risk_per_trade_pct": 0.01,
        "min_stop_distance_pct": 0.005,
        "max_stop_distance_pct": 0.03,
        "max_daily_loss_pct": 0.03,
        "note": "comprehensive_test",
    }
    state = {
        "symbol": "ETHUSDT",
        "run_id": "run-789",
        "snapshot_id": "snap-003",
        "last_gpt_decision": {"action": "short", "confidence": 0.75},
        "last_risk_envelope": risk_envelope,
    }
    risk_decision = RiskDecision(
        approved=True,
        side="short",
        position_size=2.5,
        leverage=2.0,
        stop_loss_price=105.0,
        take_profit_price=95.0,
        reason="risk_rules_v2",
    )
    adapter_result = {
        "status": "filled",
        "fill_price": 100.0,
        "avg_fill_price": 100.0,
        "fee_paid": 0.25,
        "realized_pnl": 0.0,
    }
    adapter = _DummyExecutionAdapter(adapter_result)

    with caplog.at_level("INFO"):
        result = execute_decision(risk_decision, config, state, adapter)

    assert result["status"] == "filled"
    
    # Check execution event log has all required fields
    for record in caplog.records:
        if "Execution event:" in record.message:
            msg = record.message
            
            # Required fields for execution event
            required_fields = [
                "type", "symbol", "approved", "side", "position_size",
                "leverage", "stop_loss_price", "take_profit_price",
                "reason", "risk_envelope", "status"
            ]
            for field in required_fields:
                assert field in msg or f"'{field}'" in msg, \
                    f"Required field '{field}' not found in execution log: {msg}"
            break
