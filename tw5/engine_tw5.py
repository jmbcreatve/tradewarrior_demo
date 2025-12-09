# tw5/engine_tw5.py

"""
TW-5 engine wiring: data -> snapshot -> GPT/stub -> risk_clamp -> (executor) -> state.

This module provides a single-tick entrypoint that the live loop and replay
can both call, keeping parity between modes.

For now, execution is a no-op; we only build snapshot, plan, and clamp.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
import time

from config import Config
from logger_utils import get_logger
from data_router import get_market_data
from state_memory import save_state

from .schemas import Tw5Snapshot, OrderPlan, RiskClampResult, RunMode
from .snapshot_builder import build_tw5_snapshot
from .stub import generate_tw5_stub_plan
from .gpt_client import generate_order_plan_with_gpt
from .risk_clamp import clamp_order_plan
from .gatekeeper import should_call_gpt_tw5

logger = get_logger(__name__)


@dataclass
class TickResult:
    """Summary of a single TW-5 tick for logging / replay."""

    snapshot: Tw5Snapshot
    plan: Optional[OrderPlan]
    clamp_result: Optional[RiskClampResult]
    execution_result: Any
    gpt_called: bool
    gpt_reason: str


def run_tw5_tick(
    config: Config,
    state: Dict[str, Any],
    data_adapters: Dict[str, Any],
    execution_adapter: Any = None,
    run_mode: RunMode = RunMode.AUTO,
    use_stub: bool = True,
    persist_state: bool = False,
) -> TickResult:
    """
    Run a single TW-5 tick.

    Steps:
      1) Fetch market data via get_market_data().
      2) Build Tw5Snapshot.
      3) Gatekeeper: decide whether to call GPT/stub.
      4) If no call -> return flat TickResult.
      5) If call:
           - generate plan (stub or GPT)
           - clamp via risk_clamp
           - (execution: currently NO-OP; executor will be plugged later)
      6) Update TW-5-specific fields in state (last snapshot/plan/clamp, GPT call history).
      7) Optionally persist state via save_state().

    For now, this does not send any real orders even in AUTO mode; execution_result
    is always None. This keeps early testing safe.
    """
    now_ts = time.time()

    # 1) Market data: pass the full adapters dict, get_market_data chooses primary.
    market_data = get_market_data(config, data_adapters, limit=500)

    # 2) Snapshot
    snapshot = build_tw5_snapshot(config, market_data, state)

    # 3) Gatekeeper inputs from state
    last_snapshot_dict = state.get("tw5_last_snapshot")
    last_snapshot: Tw5Snapshot | None = None
    if isinstance(last_snapshot_dict, dict):
        try:
            last_snapshot = Tw5Snapshot(**last_snapshot_dict)
        except Exception:
            last_snapshot = None

    last_gpt_ts = _safe_float(state.get("tw5_last_gpt_ts"), None)
    call_history = _get_gpt_call_history(state)
    gpt_calls_last_hour = _count_calls_last_hour(call_history, now_ts)

    # Gatekeeper thresholds (TW-5 specific config with defaults)
    min_seconds_between_calls = getattr(config, "tw5_min_seconds_between_gpt", 60.0)
    max_calls_per_hour = int(getattr(config, "tw5_max_gpt_calls_per_hour", 12))
    min_atr_move_mult = getattr(config, "tw5_min_atr_move_mult", 0.5)

    should_call, reason = should_call_gpt_tw5(
        snapshot=snapshot,
        last_snapshot=last_snapshot,
        now_ts=now_ts,
        last_gpt_ts=last_gpt_ts,
        gpt_calls_last_hour=gpt_calls_last_hour,
        min_seconds_between_calls=min_seconds_between_calls,
        max_calls_per_hour=max_calls_per_hour,
        min_atr_move_mult=min_atr_move_mult,
    )

    if not should_call:
        # No GPT call; flat TickResult
        _update_state_after_tick(
            state=state,
            snapshot=snapshot,
            plan=None,
            clamp_result=None,
            gpt_called=False,
            gpt_ts=None,
        )
        if persist_state:
            save_state(config, state)

        return TickResult(
            snapshot=snapshot,
            plan=None,
            clamp_result=None,
            execution_result=None,
            gpt_called=False,
            gpt_reason=reason,
        )

    # 4) Generate plan (stub or GPT)
    if use_stub:
        plan = generate_tw5_stub_plan(snapshot, seed=None)
    else:
        plan = generate_order_plan_with_gpt(config, snapshot, state)

    # 5) Clamp plan
    clamp_result = clamp_order_plan(snapshot, plan, state, config)

    # Execution is a NO-OP for now; we only return the clamped plan.
    execution_result = None

    # Update call history
    call_history.append(now_ts)
    state["tw5_gpt_call_history"] = call_history
    state["tw5_last_gpt_ts"] = now_ts

    # 6) Update state snapshot/plan/clamp
    _update_state_after_tick(
        state=state,
        snapshot=snapshot,
        plan=plan,
        clamp_result=clamp_result,
        gpt_called=True,
        gpt_ts=now_ts,
    )

    if persist_state:
        save_state(config, state)

    return TickResult(
        snapshot=snapshot,
        plan=plan,
        clamp_result=clamp_result,
        execution_result=execution_result,
        gpt_called=True,
        gpt_reason=reason,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_float(value: Any, default: float | None) -> float | None:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _get_gpt_call_history(state: Dict[str, Any]) -> list[float]:
    raw = state.get("tw5_gpt_call_history")
    if not isinstance(raw, list):
        return []
    out: list[float] = []
    for x in raw:
        try:
            out.append(float(x))
        except (TypeError, ValueError):
            continue
    return out


def _count_calls_last_hour(history: list[float], now_ts: float) -> int:
    cutoff = now_ts - 3600.0
    return sum(1 for ts in history if ts >= cutoff)


def _update_state_after_tick(
    state: Dict[str, Any],
    snapshot: Tw5Snapshot,
    plan: Optional[OrderPlan],
    clamp_result: Optional[RiskClampResult],
    gpt_called: bool,
    gpt_ts: Optional[float],
) -> None:
    """Store TW-5-specific fields into state for continuity and debugging."""
    state["tw5_last_snapshot"] = snapshot.to_dict()
    if plan is not None:
        state["tw5_last_plan"] = plan.to_dict()
    if clamp_result is not None:
        state["tw5_last_clamp_result"] = clamp_result.to_dict()
    state["tw5_last_gpt_called"] = bool(gpt_called)
    if gpt_ts is not None:
        state["tw5_last_gpt_ts"] = gpt_ts
