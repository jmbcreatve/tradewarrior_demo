from typing import List, Dict, Optional
import time


from .base_execution_adapter import BaseExecutionAdapter


class MockExecutionAdapter(BaseExecutionAdapter):
    """In-memory, instant-fill execution adapter for DEMO/paper trading."""

    def __init__(self):
        self._positions: List[Dict] = []

    def get_open_positions(self, symbol: str) -> List[Dict]:
        return [p for p in self._positions if p.get("symbol") == symbol]

    def place_order(
        self,
        symbol: str,
        side: str,
        size: float,
        order_type: str = "market",
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        leverage: Optional[float] = None,
    ) -> Dict:
        ts = time.time()
        position = {
            "symbol": symbol,
            "side": side,
            "size": size,
            "order_type": order_type,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "leverage": leverage,
            "timestamp": ts,
        }
        self._positions.append(position)
        return {"status": "filled", "position": position}

    def cancel_all_orders(self, symbol: str) -> None:
        # For simplicity, this mock tracks only positions, not pending orders.
        return None
