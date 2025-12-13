"""Execution adapter wrapper for TW-5.

Responsible for:
- Translating a clamped OrderPlan into concrete orders
- Calling the existing execution adapters
- Maintaining lightweight execution state in the shared state dict
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from config import Config
from logger_utils import get_logger
from safety_utils import check_trading_halted
from .exit_rules import compute_tp_levels, compute_trailing_stop
from .schemas import OrderPlan, TPLevel

logger = get_logger(__name__)

EXEC_STATE_KEY = "tw5_exec"
STOP_EPS = 1e-9


@dataclass
class ExecutionState:
    """
    Minimal TW-5 execution bookkeeping persisted in `state[EXEC_STATE_KEY]`.
    """

    symbol: str = ""
    side: str = "flat"
    entry_order_ids: List[Any] = field(default_factory=list)
    stop_oid: Optional[Any] = None
    tp_oids: Dict[str, Any] = field(default_factory=dict)
    entry_price: Optional[float] = None
    initial_stop: Optional[float] = None
    stop_current: Optional[float] = None
    R: Optional[float] = None
    tp1_hit: bool = False
    tp2_hit: bool = False
    high_water_price: Optional[float] = None
    last_manage_ts: Optional[float] = None
    position_size_last_seen: Optional[float] = None
    last_action: str = ""
    last_error: str = ""
    stop_replace_fail_count: int = 0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExecutionState":
        if not isinstance(data, dict):
            return cls()
        return cls(
            symbol=data.get("symbol", ""),
            side=data.get("side", "flat"),
            entry_order_ids=list(data.get("entry_order_ids", []) or []),
            stop_oid=data.get("stop_oid"),
            tp_oids=dict(data.get("tp_oids", {}) or {}),
            entry_price=_safe_float(data.get("entry_price")),
            initial_stop=_safe_float(data.get("initial_stop")),
            stop_current=_safe_float(data.get("stop_current")),
            R=_safe_float(data.get("R")),
            tp1_hit=bool(data.get("tp1_hit", False)),
            tp2_hit=bool(data.get("tp2_hit", False)),
            high_water_price=_safe_float(data.get("high_water_price")),
            last_manage_ts=_safe_float(data.get("last_manage_ts")),
            position_size_last_seen=_safe_float(data.get("position_size_last_seen")),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def reset(self) -> None:
        self.__dict__.update(
            ExecutionState().__dict__
        )

    @property
    def in_position(self) -> bool:
        return self.side in ("long", "short") and (self.position_size_last_seen or 0.0) > 0.0


def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _extract_oid(resp: Dict[str, Any]) -> Any:
    """
    Extract oid from common HL responses.
    """
    if not isinstance(resp, dict):
        return None
    if "oid" in resp:
        return resp.get("oid")
    try:
        statuses = resp.get("response", {}).get("data", {}).get("statuses", [])
        if statuses:
            status = statuses[0]
            if "filled" in status:
                return status["filled"].get("oid")
            if "resting" in status:
                return status["resting"].get("oid")
            return status.get("oid")
    except Exception:
        return None
    return None


def _order_price(order: Dict[str, Any]) -> Optional[float]:
    """
    Best-effort extraction of limit/stop price from an open order dict.
    """
    for key in ("limit_px", "limitPx", "px", "price"):
        if key in order:
            return _safe_float(order.get(key))
    # HL trigger orders embed triggerPx under orderType.trigger.triggerPx
    try:
        return _safe_float(order.get("orderType", {}).get("trigger", {}).get("triggerPx"))
    except Exception:
        return None


class Tw5Executor:
    """
    Thin wrapper that translates TW-5 plans into execution adapter calls and
    maintains TW-5 execution state in `state[EXEC_STATE_KEY]`.
    """

    def __init__(self, execution_adapter: Any) -> None:
        self._adapter = execution_adapter

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #

    def sync_and_reconcile(self, snapshot: Any, config: Config, state: Dict[str, Any]) -> ExecutionState:
        """
        Reconcile local execution state with exchange reality.
        - If flat: cancel orphan reduce-only orders we tracked; clear state.
        - If in position: ensure we still have a stop + TPs; if missing, re-place using known params.
        """
        exec_state = self._load_state(state)
        hl_orders = self._get_open_orders(snapshot.symbol)
        open_positions = self._get_open_positions(snapshot.symbol)

        in_pos = _select_position(open_positions)
        if not in_pos:
            # Flat: cancel any tracked oids and clear state
            orphan_oids = _collect_tracked_oids(exec_state)
            if orphan_oids:
                logger.info("TW-5 executor: cancelling orphan orders: %s", orphan_oids)
                self._cancel_orders(snapshot.symbol, orphan_oids)
            exec_state.reset()
            state[EXEC_STATE_KEY] = exec_state.to_dict()
            return exec_state

        # Update core position fields
        exec_state.symbol = snapshot.symbol
        exec_state.side = in_pos["side"]
        exec_state.position_size_last_seen = _safe_float(in_pos.get("size"))
        if exec_state.entry_price is None:
            exec_state.entry_price = _safe_float(in_pos.get("entry_price"), snapshot.price)
        if exec_state.high_water_price is None:
            exec_state.high_water_price = exec_state.entry_price

        # Maintain high-water
        exec_state.high_water_price = _update_high_water(exec_state, snapshot.price)

        # Map oid -> order for lookup
        open_order_map = {o.get("oid"): o for o in hl_orders if "oid" in o}
        if exec_state.initial_stop is None and exec_state.stop_oid in open_order_map:
            exec_state.initial_stop = _order_price(open_order_map.get(exec_state.stop_oid, {}))
        if exec_state.R is None and exec_state.entry_price is not None and exec_state.initial_stop is not None:
            exec_state.R = abs(exec_state.entry_price - exec_state.initial_stop)

        # Restore stop if missing but we know where it should be
        has_stop = exec_state.stop_oid is not None and exec_state.stop_oid in open_order_map
        if not has_stop and exec_state.initial_stop is not None and exec_state.R is not None:
            logger.warning("TW-5 executor: stop missing; re-placing protective stop.")
            stop_resp = self._adapter.place_trigger_order(
                symbol=snapshot.symbol,
                is_buy=exec_state.side == "short",
                sz=exec_state.position_size_last_seen or 0.0,
                trigger_px=exec_state.initial_stop,
                reduce_only=True,
                tpsl="sl",
                is_market=True,
                limit_px=exec_state.initial_stop,
            )
            exec_state.stop_oid = _extract_oid(stop_resp)

        # Restore any missing TP orders from derived ladder
        new_tp_oids: Dict[str, Any] = {}
        if exec_state.entry_price is not None and exec_state.initial_stop is not None and exec_state.R is not None:
            tp_levels = compute_tp_levels(
                entry=exec_state.entry_price,
                stop=exec_state.initial_stop,
                side=exec_state.side,
                r_mults=getattr(config, "tw5_tp_r_multipliers", [1.2, 2.0, 3.0]),
                remaining_fracs=getattr(config, "tw5_tp_remaining_fracs", [0.30, 0.30, 1.0]),
            )
            for price, abs_frac, tag in tp_levels:
                existing_oid = exec_state.tp_oids.get(tag)
                if existing_oid in open_order_map:
                    new_tp_oids[tag] = existing_oid
                    continue
                logger.warning("TW-5 executor: TP %s missing; re-placing.", tag)
                tp_resp = self._adapter.place_limit_order(
                    symbol=snapshot.symbol,
                    is_buy=(exec_state.side == "short"),
                    sz=abs_frac * (exec_state.position_size_last_seen or 0.0),
                    limit_px=price,
                    reduce_only=True,
                    tif="Gtc",
                )
                new_tp_oids[tag] = _extract_oid(tp_resp)
            exec_state.tp_oids = new_tp_oids if new_tp_oids else exec_state.tp_oids

        state[EXEC_STATE_KEY] = exec_state.to_dict()
        return exec_state

    def maybe_enter_from_plan(
        self,
        snapshot: Any,
        clamp_result: Any,
        config: Config,
        state: Dict[str, Any],
    ) -> Optional[ExecutionState]:
        """
        Enter a position if flat and plan approved. MVP: market entry + stop/TP ladder.
        """
        exec_state = self._load_state(state)

        halted, reason = check_trading_halted(state)
        if halted:
            logger.warning("TW-5 executor: trading halted (%s); refusing new entries.", reason)
            exec_state.last_error = f"halted:{reason}"
            state[EXEC_STATE_KEY] = exec_state.to_dict()
            return None
        exec_state.last_error = ""
        if exec_state.in_position:
            return exec_state

        if clamp_result is None or not getattr(clamp_result, "approved", False):
            return None

        plan: OrderPlan = getattr(clamp_result, "clamped_plan", None) or getattr(clamp_result, "original_plan", None)
        if plan is None or plan.side not in ("long", "short"):
            return None

        # Aggregate plan sizing/stop
        legs = list(plan.legs or [])
        total_size = sum(max(0.0, getattr(leg, "size_frac", 0.0)) for leg in legs)
        if total_size <= 0.0 or not legs:
            return None

        entry_price = _weighted(legs, "entry_price", snapshot.price)
        stop_loss = _worst_stop(legs, plan.side)
        R = abs(entry_price - stop_loss)
        if R <= 0.0:
            logger.warning("TW-5 executor: invalid R (entry=%.4f stop=%.4f); refusing entry.", entry_price, stop_loss)
            return None

        # Market entry only for MVP
        try:
            resp = self._adapter.place_order(
                symbol=snapshot.symbol,
                side=plan.side,
                size=total_size,
                order_type="market",
            )
            logger.info("TW-5 executor: placed market entry side=%s size=%.4f resp=%s", plan.side, total_size, resp)
        except Exception as e:
            logger.error("TW-5 executor: place_order exception: %s", e, exc_info=True)
            exec_state.last_error = f"entry_exception:{e}"
            state[EXEC_STATE_KEY] = exec_state.to_dict()
            return None

        status = resp.get("status")
        if status not in ("filled", "resting"):
            logger.warning("TW-5 executor: entry not filled (status=%s); skipping exits.", status)
            return None

        fill_price = _safe_float(resp.get("fill_price"), entry_price)
        entry_order_id = resp.get("order_id") or _extract_oid(resp)

        # Compute TP levels and place exits
        tp_levels = compute_tp_levels(
            entry=fill_price,
            stop=stop_loss,
            side=plan.side,
            r_mults=getattr(config, "tw5_tp_r_multipliers", [1.2, 2.0, 3.0]),
            remaining_fracs=getattr(config, "tw5_tp_remaining_fracs", [0.30, 0.30, 1.0]),
        )

        stop_mode = str(getattr(config, "tw5_stop_order_mode", "stop_market"))
        stop_is_market = stop_mode != "stop_limit"
        stop_resp = self._adapter.place_trigger_order(
            symbol=snapshot.symbol,
            is_buy=(plan.side == "short"),
            sz=total_size,
            trigger_px=stop_loss,
            reduce_only=True,
            tpsl="sl",
            is_market=stop_is_market,
            limit_px=stop_loss,
        )
        stop_oid = _extract_oid(stop_resp)
        self._log_stop_placement(
            symbol=snapshot.symbol,
            stop_mode=stop_mode,
            trigger_px=stop_loss,
            limit_px=stop_loss,
            is_market=stop_is_market,
            reduce_only=True,
            oid=stop_oid,
        )
        if stop_oid is None:
            logger.critical("TW-5 executor: stop placement failed, attempting emergency flatten.")
            exec_state.last_error = "stop_place_failed"
            self._flatten_market(snapshot.symbol, plan.side, total_size)
            state[EXEC_STATE_KEY] = exec_state.to_dict()
            return None

        tp_oids: Dict[str, Any] = {}
        for price, abs_frac, tag in tp_levels:
            tp_resp = self._adapter.place_limit_order(
                symbol=snapshot.symbol,
                is_buy=(plan.side == "short"),
                sz=abs_frac * total_size,
                limit_px=price,
                reduce_only=True,
                tif="Gtc",
            )
            tp_oids[tag] = _extract_oid(tp_resp)
            logger.info("TW-5 executor: placed TP %s size=%.4f px=%.4f oid=%s", tag, abs_frac * total_size, price, tp_oids[tag])

        exec_state.symbol = snapshot.symbol
        exec_state.side = plan.side
        exec_state.entry_order_ids = [entry_order_id] if entry_order_id is not None else []
        exec_state.stop_oid = stop_oid
        exec_state.tp_oids = tp_oids
        exec_state.entry_price = fill_price
        exec_state.initial_stop = stop_loss
        exec_state.stop_current = stop_loss
        exec_state.R = R
        exec_state.tp1_hit = False
        exec_state.tp2_hit = False
        exec_state.high_water_price = fill_price
        exec_state.last_manage_ts = snapshot.timestamp
        exec_state.position_size_last_seen = total_size
        exec_state.last_action = "entered"
        exec_state.last_error = ""

        state[EXEC_STATE_KEY] = exec_state.to_dict()
        return exec_state

    def manage_open_position(self, snapshot: Any, config: Config, state: Dict[str, Any]) -> Optional[ExecutionState]:
        """
        Periodic stop management (monotonic tightening). No widening allowed.
        """
        exec_state = self._load_state(state)
        if not exec_state.in_position:
            return None
        if exec_state.entry_price is None or exec_state.initial_stop is None or exec_state.R is None:
            logger.warning("TW-5 executor: missing entry/stop/R; cannot manage trailing stop.")
            return exec_state

        now = snapshot.timestamp or time.time()
        halted, reason = check_trading_halted(state)
        interval = getattr(config, "tw5_manage_interval_sec", 180.0)
        if not halted and exec_state.last_manage_ts is not None and (now - exec_state.last_manage_ts) < interval:
            return exec_state
        if halted and exec_state.last_manage_ts is not None and (now - exec_state.last_manage_ts) < interval:
            # When halted, we still allow tighten-only but keep cadence to avoid spam
            return exec_state

        # Refresh open orders to detect TP hits and current stop price
        hl_orders = self._get_open_orders(exec_state.symbol)
        open_order_map = {o.get("oid"): o for o in hl_orders if "oid" in o}

        # Mark TP hits based on missing OIDs
        tp_oids_list = list(exec_state.tp_oids.values())
        if tp_oids_list:
            if len(tp_oids_list) > 0 and tp_oids_list[0] not in open_order_map:
                exec_state.tp1_hit = True
            if len(tp_oids_list) > 1 and tp_oids_list[1] not in open_order_map:
                exec_state.tp2_hit = True

        # Current stop price (if order still resting)
        current_stop_price = None
        if exec_state.stop_oid is not None and exec_state.stop_oid in open_order_map:
            current_stop_price = _order_price(open_order_map[exec_state.stop_oid])

        # Update high-water from price
        exec_state.high_water_price = _update_high_water(exec_state, snapshot.price)

        desired_stop = compute_trailing_stop(
            entry=exec_state.entry_price,
            initial_stop=exec_state.initial_stop,
            side=exec_state.side,
            R=exec_state.R,
            high_water_price=exec_state.high_water_price,
            tp1_hit=exec_state.tp1_hit,
            tp2_hit=exec_state.tp2_hit,
            cfg=config,
        )

        if halted:
            logger.warning("TW-5 executor: trading halted; tightening-only mode.")
        if not _is_tighter(exec_state.side, current_stop_price, desired_stop):
            exec_state.last_manage_ts = now
            state[EXEC_STATE_KEY] = exec_state.to_dict()
            return exec_state

        # Replace stop with tighter value (with backoff)
        if exec_state.stop_replace_fail_count > 0:
            # Simple backoff: skip if last failure was recent and we haven't waited interval * fail_count
            if exec_state.last_manage_ts is not None and (now - exec_state.last_manage_ts) < interval * exec_state.stop_replace_fail_count:
                state[EXEC_STATE_KEY] = exec_state.to_dict()
                return exec_state

        if exec_state.stop_oid is not None:
            logger.info("TW-5 executor: cancelling old stop oid=%s", exec_state.stop_oid)
            self._cancel_orders(exec_state.symbol, [exec_state.stop_oid])

        stop_mode = str(getattr(config, "tw5_stop_order_mode", "stop_market"))
        stop_is_market = stop_mode != "stop_limit"
        stop_resp = self._adapter.place_trigger_order(
            symbol=exec_state.symbol,
            is_buy=(exec_state.side == "short"),
            sz=exec_state.position_size_last_seen or 0.0,
            trigger_px=desired_stop,
            reduce_only=True,
            tpsl="sl",
            is_market=stop_is_market,
            limit_px=desired_stop,
        )
        exec_state.stop_oid = _extract_oid(stop_resp)
        self._log_stop_placement(
            symbol=exec_state.symbol,
            stop_mode=stop_mode,
            trigger_px=desired_stop,
            limit_px=desired_stop,
            is_market=stop_is_market,
            reduce_only=True,
            oid=exec_state.stop_oid,
        )
        exec_state.last_manage_ts = now
        exec_state.stop_current = desired_stop
        exec_state.last_action = "stop_tightened"
        if exec_state.stop_oid is not None:
            exec_state.last_error = ""
            exec_state.stop_replace_fail_count = 0
        else:
            exec_state.stop_replace_fail_count += 1
            exec_state.last_error = f"stop_replace_failed:{exec_state.stop_replace_fail_count}"
            logger.critical(
                "TW-5 executor: failed to replace stop (fail_count=%d); attempting emergency flatten if above threshold.",
                exec_state.stop_replace_fail_count,
            )
            if exec_state.stop_replace_fail_count >= 3:
                self._flatten_market(exec_state.symbol, exec_state.side, exec_state.position_size_last_seen or 0.0)
        state[EXEC_STATE_KEY] = exec_state.to_dict()
        return exec_state

    # ------------------------------------------------------------------ #
    # Internals                                                         #
    # ------------------------------------------------------------------ #

    def _load_state(self, state: Dict[str, Any]) -> ExecutionState:
        raw = state.get(EXEC_STATE_KEY, {}) or {}
        if not isinstance(raw, dict):
            raw = {}
        exec_state = ExecutionState.from_dict(raw)
        state[EXEC_STATE_KEY] = exec_state.to_dict()
        return exec_state

    def _get_open_positions(self, symbol: str) -> List[Dict[str, Any]]:
        if hasattr(self._adapter, "get_open_positions"):
            try:
                return self._adapter.get_open_positions(symbol) or []
            except Exception as e:
                logger.error("TW-5 executor: get_open_positions failed: %s", e, exc_info=True)
                return []
        return []

    def _get_open_orders(self, symbol: str) -> List[Dict[str, Any]]:
        if hasattr(self._adapter, "get_open_orders"):
            try:
                return self._adapter.get_open_orders(symbol) or []
            except Exception as e:
                logger.error("TW-5 executor: get_open_orders failed: %s", e, exc_info=True)
                return []
        return []

    def _cancel_orders(self, symbol: str, oids: List[Any]) -> None:
        if not oids:
            return
        if hasattr(self._adapter, "cancel_orders"):
            try:
                resp = self._adapter.cancel_orders(symbol, oids)
                logger.info("TW-5 executor: cancel_orders %s resp=%s", oids, resp)
            except Exception as e:
                logger.error("TW-5 executor: cancel_orders failed: %s", e, exc_info=True)

    def _flatten_market(self, symbol: str, side: str, size: float) -> None:
        if size <= 0.0:
            return
        try:
            if hasattr(self._adapter, "close_position_or_raise"):
                resp = self._adapter.close_position_or_raise(symbol, sz=size)
                logger.critical("TW-5 executor: EMERGENCY FLATTEN via close_position_or_raise resp=%s", resp)
            else:
                opposite = "short" if side == "long" else "long"
                resp = self._adapter.place_order(
                    symbol=symbol,
                    side=opposite,
                    size=size,
                    order_type="market",
                )
                logger.critical("TW-5 executor: EMERGENCY FLATTEN via market order resp=%s", resp)
            positions = self._get_open_positions(symbol)
            for pos in positions:
                if pos.get("symbol") == symbol:
                    sz_left = abs(float(pos.get("size", 0.0)))
                    if sz_left > 1e-6:
                        raise RuntimeError(f"flatten_failed: residual size {sz_left}")
        except Exception as e:
            logger.critical("TW-5 executor: EMERGENCY FLATTEN failed: %s", e, exc_info=True)
            raise

    def _log_stop_placement(
        self,
        symbol: str,
        stop_mode: str,
        trigger_px: float,
        limit_px: float,
        is_market: bool,
        reduce_only: bool,
        oid: Any,
    ) -> None:
        logger.info(
            "TW-5 executor: placed stop (mode=%s trigger=%.6f limit=%.6f is_market=%s reduce_only=%s oid=%s)",
            stop_mode,
            trigger_px,
            limit_px,
            is_market,
            reduce_only,
            oid,
        )
        try:
            open_orders = self._get_open_orders(symbol)
            if not open_orders:
                return
            match = None
            for order in open_orders:
                if order.get("oid") == oid:
                    match = order
                    break
            if match is not None:
                logger.info("TW-5 executor: stop verification snapshot oid=%s order=%s", oid, match)
        except Exception:
            logger.debug("TW-5 executor: unable to verify stop order on book.", exc_info=True)


def _collect_tracked_oids(exec_state: ExecutionState) -> List[Any]:
    ids = []
    if exec_state.stop_oid is not None:
        ids.append(exec_state.stop_oid)
    ids.extend([oid for oid in exec_state.tp_oids.values() if oid is not None])
    ids.extend(exec_state.entry_order_ids or [])
    return [oid for oid in ids if oid is not None]


def _select_position(positions: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not positions:
        return None
    # Prefer first valid entry
    for pos in positions:
        if pos.get("side") in ("long", "short") and _safe_float(pos.get("size"), 0.0) > 0.0:
            return pos
    return None


def _weighted(legs: List[TPLevel], attr: str, fallback: float) -> float:
    num = 0.0
    den = 0.0
    for leg in legs:
        w = max(0.0, getattr(leg, "size_frac", 0.0))
        num += w * getattr(leg, attr)
        den += w
    return num / den if den > 0.0 else fallback


def _worst_stop(legs: List[TPLevel], side: str) -> float:
    stops = [getattr(leg, "stop_loss", None) for leg in legs if getattr(leg, "stop_loss", None) is not None]
    if not stops:
        return 0.0
    if side == "long":
        return min(stops)
    return max(stops)


def _update_high_water(exec_state: ExecutionState, price: float) -> float:
    if exec_state.high_water_price is None:
        return price
    if exec_state.side == "long":
        return max(exec_state.high_water_price, price)
    if exec_state.side == "short":
        return min(exec_state.high_water_price, price)
    return exec_state.high_water_price


def _is_tighter(side: str, current_stop: Optional[float], desired_stop: Optional[float]) -> bool:
    if desired_stop is None:
        return False
    if current_stop is None:
        # No existing stop -> placing one is always tightening
        return True
    if side == "long":
        return desired_stop > current_stop + STOP_EPS
    if side == "short":
        return desired_stop < current_stop - STOP_EPS
    return False
