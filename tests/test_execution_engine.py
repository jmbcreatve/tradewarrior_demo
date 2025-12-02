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
