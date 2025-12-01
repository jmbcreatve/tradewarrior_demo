from typing import Any, Dict, List, Optional
import time

from logger_utils import get_logger

from .base_execution_adapter import BaseExecutionAdapter

logger = get_logger(__name__)


class MockExecutionAdapter(BaseExecutionAdapter):
    """In-memory, instant-fill execution adapter for DEMO/paper trading."""

    def __init__(self) -> None:
        self._positions: List[Dict[str, Any]] = []
        # Basic cost model knobs (bps = 1/10,000).
        self._fee_bps = 0.03  # 0.03% taker-ish fee per side
        self._spread_bps = 1.0  # total bid/ask spread in bps
        self._min_slippage_bps = 0.05
        self._slippage_per_unit_bps = 0.01
        self._max_slippage_bps = 1.0

    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _infer_mid_price(self, stop_loss: Any, take_profit: Any) -> float:
        """
        Risk engine sets SL/TP around the snapshot mid using a 1:2 RR,
        so we can recover an approximate mid = (2*SL + TP)/3 (or symmetric).
        """
        sl = self._safe_float(stop_loss, -1.0)
        tp = self._safe_float(take_profit, -1.0)
        if sl <= 0.0 or tp <= 0.0:
            return 0.0
        mid = (2.0 * sl + tp) / 3.0
        return mid if mid > 0.0 else 0.0

    def _compute_slippage_bps(self, size: float) -> float:
        size_f = self._safe_float(size, 0.0)
        slip = self._min_slippage_bps + size_f * self._slippage_per_unit_bps
        if slip < 0.0:
            slip = 0.0
        if slip > self._max_slippage_bps:
            slip = self._max_slippage_bps
        return slip

    def _apply_price_adjustments(self, mid_price: float, side: str, size: float) -> float:
        spread_frac = max(self._spread_bps, 0.0) / 10_000.0
        half_spread = mid_price * (spread_frac / 2.0)

        slip_frac = self._compute_slippage_bps(size) / 10_000.0

        if side == "long":
            price = (mid_price + half_spread) * (1.0 + slip_frac)
        else:
            price = (mid_price - half_spread) * (1.0 - slip_frac)
        return price if price > 0.0 else 0.0

    def _find_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        for pos in self._positions:
            if pos.get("symbol") == symbol:
                return pos
        return None

    def _set_position(self, symbol: str, position: Optional[Dict[str, Any]]) -> None:
        self._positions = [p for p in self._positions if p.get("symbol") != symbol]
        if position is not None:
            self._positions.append(position)

    def _calc_realized_pnl(self, entry_price: float, exit_price: float, size: float, side: str) -> float:
        if entry_price <= 0.0 or exit_price <= 0.0 or size <= 0.0:
            return 0.0
        if side == "long":
            return (exit_price - entry_price) * size
        return (entry_price - exit_price) * size

    def _update_positions(
        self,
        symbol: str,
        side: str,
        size: float,
        fill_price: float,
        meta: Dict[str, Any],
    ) -> Dict[str, Any]:
        ts = meta.get("timestamp", time.time())
        realized_pnl = 0.0

        existing = self._find_position(symbol)
        if existing is None or self._safe_float(existing.get("size"), 0.0) <= 0.0:
            position = {
                "symbol": symbol,
                "side": side,
                "size": size,
                "entry_price": fill_price,
                "order_type": meta.get("order_type"),
                "stop_loss": meta.get("stop_loss"),
                "take_profit": meta.get("take_profit"),
                "leverage": meta.get("leverage"),
                "timestamp": ts,
            }
            self._set_position(symbol, position)
            return {"realized_pnl": realized_pnl, "position": position}

        # Net with existing position
        existing_side = str(existing.get("side", "")).lower()
        existing_size = self._safe_float(existing.get("size"), 0.0)
        entry_price = self._safe_float(existing.get("entry_price") or existing.get("fill_price"), 0.0)

        if existing_side == side:
            total = existing_size + size
            if total <= 0.0:
                self._set_position(symbol, None)
                return {"realized_pnl": realized_pnl, "position": None}
            new_entry = ((entry_price * existing_size) + (fill_price * size)) / total if total else fill_price
            existing.update(
                {
                    "size": total,
                    "entry_price": new_entry,
                    "timestamp": ts,
                    "order_type": meta.get("order_type", existing.get("order_type")),
                    "stop_loss": meta.get("stop_loss", existing.get("stop_loss")),
                    "take_profit": meta.get("take_profit", existing.get("take_profit")),
                    "leverage": meta.get("leverage", existing.get("leverage")),
                }
            )
            self._set_position(symbol, existing)
            return {"realized_pnl": realized_pnl, "position": existing}

        closing_size = min(size, existing_size)
        realized_pnl = self._calc_realized_pnl(entry_price, fill_price, closing_size, existing_side)
        remaining = existing_size - closing_size

        if remaining > 0.0:
            existing.update({"size": remaining, "timestamp": ts})
            self._set_position(symbol, existing)
            return {"realized_pnl": realized_pnl, "position": existing}

        # Fully closed existing position
        self._set_position(symbol, None)
        flipped = size - closing_size
        new_position = None
        if flipped > 0.0:
            new_position = {
                "symbol": symbol,
                "side": side,
                "size": flipped,
                "entry_price": fill_price,
                "order_type": meta.get("order_type"),
                "stop_loss": meta.get("stop_loss"),
                "take_profit": meta.get("take_profit"),
                "leverage": meta.get("leverage"),
                "timestamp": ts,
            }
            self._set_position(symbol, new_position)

        return {"realized_pnl": realized_pnl, "position": new_position}

    def get_open_positions(self, symbol: str) -> List[Dict[str, Any]]:
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
    ) -> Dict[str, Any]:
        ts = time.time()
        result: Dict[str, Any] = {
            "status": "no_trade",
            "symbol": symbol,
            "side": side,
            "size": size,
            "fill_price": 0.0,
            "fee_paid": 0.0,
            "realized_pnl": 0.0,
        }

        try:
            side_norm = str(side).lower()
            if side_norm not in {"long", "short"}:
                logger.warning("MockExecutionAdapter: invalid side %s; refusing trade.", side)
                result["status"] = "error"
                result["reason"] = "invalid_side"
                return result

            trade_size = self._safe_float(size, 0.0)
            if trade_size <= 0.0:
                logger.warning("MockExecutionAdapter: non-positive size %s; refusing trade.", size)
                result["status"] = "error"
                result["reason"] = "invalid_size"
                return result

            mid_price = self._infer_mid_price(stop_loss, take_profit)
            if mid_price <= 0.0:
                logger.warning("MockExecutionAdapter: missing price hints; cannot compute fill price.")
                result["status"] = "error"
                result["reason"] = "invalid_price"
                return result

            fill_price = self._apply_price_adjustments(mid_price, side_norm, trade_size)
            if fill_price <= 0.0:
                logger.warning("MockExecutionAdapter: computed non-positive fill price; aborting trade.")
                result["status"] = "error"
                result["reason"] = "bad_fill_price"
                return result

            fee_rate = max(self._fee_bps, 0.0) / 10_000.0
            fee_paid = abs(trade_size * fill_price * fee_rate)

            meta = {
                "order_type": order_type,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "leverage": leverage,
                "timestamp": ts,
            }
            pos_update = self._update_positions(symbol, side_norm, trade_size, fill_price, meta)

            result.update(
                {
                    "status": "filled",
                    "fill_price": fill_price,
                    "fee_paid": fee_paid,
                    "realized_pnl": pos_update.get("realized_pnl", 0.0),
                    "position": pos_update.get("position"),
                }
            )
            return result

        except Exception as exc:  # noqa: BLE001
            logger.exception("MockExecutionAdapter.place_order failed: %s", exc)
            result["status"] = "error"
            result["reason"] = "exception"
            return result

    def cancel_all_orders(self, symbol: str) -> None:
        # For simplicity, this mock tracks only positions, not pending orders.
        return None
