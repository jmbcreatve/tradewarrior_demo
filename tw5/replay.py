# tw5/replay.py

"""
Minimal replay harness for TW-5.

This runs the TW-5 spine over an offline candle stream:

    candles (mainnet/testnet) -> snapshot -> gatekeeper
                                -> (stub/GPT) -> risk_clamp -> TickResult[]

Execution is still a NO-OP; we do not send any orders.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Iterable, List, Sequence
import time

from config import Config
from logger_utils import get_logger

from .schemas import RunMode, Tw5Snapshot, OrderPlan, RiskClampResult
from .snapshot_builder import build_tw5_snapshot
from .stub import generate_tw5_stub_plan
from .gpt_client import generate_order_plan_with_gpt
from .risk_clamp import clamp_order_plan
from .gatekeeper import should_call_gpt_tw5
from .engine_tw5 import TickResult  # reuse the same TickResult dataclass

logger = get_logger(__name__)


def run_tw5_replay_from_candles(
    candles: Sequence[Dict[str, Any]],
    config: Config,
    initial_state: Dict[str, Any],
    use_stub: bool = True,
    warmup_bars: int = 100,
) -> List[TickResult]:
    """
    Run TW-5 over an offline candle sequence.

    Args:
        candles:       Sequence of candle dicts with at least "close" and "timestamp".
        config:        Config object (we use symbol, timeframe, risk settings).
        initial_state: Starting state dict (e.g. from load_state()).
        use_stub:      If True, use stub policy instead of GPT.
        warmup_bars:   Number of initial bars to skip before first tick, so
                       snapshot/recent-window logic has some context.

    Returns:
        List[TickResult] for each replay tick.
    """
    if not candles:
        logger.warning("TW-5 replay: no candles provided.")
        return []

    state: Dict[str, Any] = deepcopy(initial_state)
    results: List[TickResult] = []

    # TW-5-specific history fields (same as engine_tw5)
    call_history = _get_gpt_call_history(state)
    last_gpt_ts = _safe_float(state.get("tw5_last_gpt_ts"), None)
    last_snapshot: Tw5Snapshot | None = None

    # Gatekeeper thresholds
    min_seconds_between_calls = getattr(config, "tw5_min_seconds_between_gpt", 60.0)
    max_calls_per_hour = int(getattr(config, "tw5_max_gpt_calls_per_hour", 12))
    min_atr_move_mult = getattr(config, "tw5_min_atr_move_mult", 0.5)

    for idx in range(len(candles)):
        # Build a rolling window up to current bar
        window_start = max(0, idx - warmup_bars)
        window = list(candles[window_start : idx + 1])

        market_data = {"candles": window}
        now_ts = float(window[-1].get("timestamp", time.time()))

        # 1) Snapshot
        snapshot = build_tw5_snapshot(config, market_data, state)

        # 2) Gatekeeper
        gpt_calls_last_hour = _count_calls_last_hour(call_history, now_ts)
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
            # No GPT call; flat tick
            _update_state_after_tick(
                state=state,
                snapshot=snapshot,
                plan=None,
                clamp_result=None,
                gpt_called=False,
                gpt_ts=None,
            )
            results.append(
                TickResult(
                    snapshot=snapshot,
                    plan=None,
                    clamp_result=None,
                    execution_result=None,
                    gpt_called=False,
                    gpt_reason=reason,
                )
            )
            last_snapshot = snapshot
            continue

        # 3) Generate plan
        if use_stub:
            plan = generate_tw5_stub_plan(snapshot, seed=None)
        else:
            plan = generate_order_plan_with_gpt(config, snapshot, state)

        # 4) Clamp
        clamp_result = clamp_order_plan(snapshot, plan, state, config)

        # 5) Update GPT call history
        call_history.append(now_ts)
        last_gpt_ts = now_ts

        # 6) State update (no execution)
        _update_state_after_tick(
            state=state,
            snapshot=snapshot,
            plan=plan,
            clamp_result=clamp_result,
            gpt_called=True,
            gpt_ts=now_ts,
        )
        state["tw5_gpt_call_history"] = call_history
        state["tw5_last_gpt_ts"] = last_gpt_ts

        results.append(
            TickResult(
                snapshot=snapshot,
                plan=plan,
                clamp_result=clamp_result,
                execution_result=None,
                gpt_called=True,
                gpt_reason=reason,
            )
        )
        last_snapshot = snapshot

    logger.info("TW-5 replay: completed %d ticks over %d candles.", len(results), len(candles))
    return results


# ---------------------------------------------------------------------------
# Helpers (mirrored from engine_tw5)
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
    plan: OrderPlan | None,
    clamp_result: RiskClampResult | None,
    gpt_called: bool,
    gpt_ts: float | None,
) -> None:
    state["tw5_last_snapshot"] = snapshot.to_dict()
    if plan is not None:
        state["tw5_last_plan"] = plan.to_dict()
    if clamp_result is not None:
        state["tw5_last_clamp_result"] = clamp_result.to_dict()
    state["tw5_last_gpt_called"] = bool(gpt_called)
    if gpt_ts is not None:
        state["tw5_last_gpt_ts"] = gpt_ts
