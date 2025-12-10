# tw5/replay.py

"""
Replay harness for TW-5.

Two layers:

1) Structural replay (no PnL):
   candles -> Tw5Snapshot -> gatekeeper -> (stub/GPT) -> risk_clamp -> TickResult[]

2) PnL replay on top of TickResult:
   - Simple one-position-at-a-time simulator
   - Risk-per-trade sizing from config
   - Stop/TP per bar, gap-aware exits
   - Basic slippage + fees
   - Equity curve, PnL, max drawdown, win rate, avg R

Execution is still a NO-OP; we never send real orders.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence, Tuple
from copy import deepcopy
import argparse
import csv
import time

from config import Config, load_testnet_config
from logger_utils import get_logger

from .schemas import Tw5Snapshot, OrderPlan, RiskClampResult, PendingOrder
from .snapshot_builder import build_tw5_snapshot
from .stub import generate_tw5_stub_plan
from .gpt_client import generate_order_plan_with_gpt
from .risk_clamp import clamp_order_plan
from .gatekeeper import should_call_gpt_tw5
from .engine_tw5 import TickResult

logger = get_logger(__name__)

# Conservative execution assumptions
DEFAULT_FEE_RATE = 0.0004        # 4 bps per side
DEFAULT_SLIPPAGE_BPS = 0.0003    # 3 bps adverse slippage


# ---------------------------------------------------------------------------
# Replay stats dataclass
# ---------------------------------------------------------------------------


@dataclass
class Tw5ReplayStats:
    """Summary statistics for a TW-5 replay with PnL."""

    equity_curve: List[float]
    trades: List[Dict[str, Any]]
    total_pnl: float
    total_return_pct: float        # fraction, e.g. 0.12 = +12%
    max_drawdown: float            # fraction, e.g. 0.25 = -25% peak-to-trough
    trade_count: int
    win_rate: float                # fraction, e.g. 0.55 = 55%
    avg_r: float                   # average R-multiple


# ---------------------------------------------------------------------------
# Shared helpers (mirroring engine_tw5 + replay_engine where needed)
# ---------------------------------------------------------------------------


def _safe_float(value: Any, default: float | None = 0.0) -> float | None:
    """Convert to float, falling back to default on None/invalid."""
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _get_gpt_call_history(state: Dict[str, Any]) -> List[float]:
    """Extract and sanitise GPT call timestamps from state."""
    raw = state.get("tw5_gpt_call_history")
    if not isinstance(raw, Iterable) or isinstance(raw, (str, bytes)):
        return []
    history: List[float] = []
    for v in raw:
        try:
            if v is None:
                continue
            ts = float(v)
            if ts > 0:
                history.append(ts)
        except (TypeError, ValueError):
            continue
    return history


def _count_calls_last_hour(history: List[float], now_ts: float) -> int:
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
    """Mirror engine_tw5's TW-5-specific state updates."""
    state["tw5_last_snapshot"] = snapshot.to_dict()
    if plan is not None:
        state["tw5_last_plan"] = plan.to_dict()
    if clamp_result is not None:
        state["tw5_last_clamp_result"] = clamp_result.to_dict()
    state["tw5_last_gpt_called"] = bool(gpt_called)
    if gpt_ts is not None:
        state["tw5_last_gpt_ts"] = float(gpt_ts)


def _safe_price(candle: Dict[str, Any], key: str = "close", default: float = 0.0) -> float:
    """Safe float getter for OHLC candles."""
    v = candle.get(key, default)
    try:
        return float(v)
    except (TypeError, ValueError):
        return float(default)


def _compute_pnl(position: Dict[str, Any], exit_price: float) -> float:
    """Linear PnL for one position (no funding)."""
    entry = float(position.get("entry_price", 0.0))
    size = float(position.get("size", 0.0))
    side = str(position.get("side", "flat"))
    if side == "short":
        return (entry - exit_price) * size
    return (exit_price - entry) * size


def _max_drawdown(equity_curve: List[float]) -> float:
    """Peak-to-trough drawdown as a fraction of peak equity."""
    if not equity_curve:
        return 0.0
    peak = equity_curve[0]
    max_dd = 0.0
    for eq in equity_curve:
        if eq > peak:
            peak = eq
        if peak > 0:
            dd = (peak - eq) / peak
            if dd > max_dd:
                max_dd = dd
    return max_dd


def _resolve_exit(position: Dict[str, Any], candle: Dict[str, Any]) -> Tuple[float, str]:
    """
    Decide exit price for a single bar, honoring stop/take-profit if hit.

    Gap-aware, mildly pessimistic:
      - For longs, if price gaps below stop, we exit at the low (worse than stop).
      - For shorts, if price gaps above stop, we exit at the high.

    Returns (exit_price, exit_reason), where exit_reason is:
      - "stop_loss"
      - "take_profit"
      - "bar_close" (no stop/TP hit, end-of-bar mark)
    """
    close_price = _safe_price(candle, "close", position.get("entry_price", 0.0))
    high = _safe_price(candle, "high", close_price)
    low = _safe_price(candle, "low", close_price)

    stop = position.get("stop_loss")
    tp = position.get("take_profit")
    side = str(position.get("side", "flat"))

    if side == "long":
        if stop is not None and low <= stop:
            # If we gap through, assume worst fill at the low
            exit_price = low if low < stop else stop
            return float(exit_price), "stop_loss"
        if tp is not None and high >= tp:
            # Take-profit: we keep this at the TP level for mild conservatism
            return float(tp), "take_profit"
    elif side == "short":
        if stop is not None and high >= stop:
            exit_price = high if high > stop else stop
            return float(exit_price), "stop_loss"
        if tp is not None and low <= tp:
            return float(tp), "take_profit"

    return close_price, "bar_close"


def _effective_risk_per_trade_pct(config: Config) -> float:
    """
    Derive per-trade risk from config.

    For now we only use config.risk_per_trade (0–1 fraction of equity).
    """
    base = _safe_float(getattr(config, "risk_per_trade", 0.0), 0.0) or 0.0
    return float(base)


# ---------------------------------------------------------------------------
# 1) Structural TW-5 replay (no PnL)
# ---------------------------------------------------------------------------


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
        config:        Config object (symbol, timeframe, risk settings).
        initial_state: Starting state dict (e.g. from load_state()).
        use_stub:      If True, use stub policy instead of GPT.
        warmup_bars:   Number of initial bars to use only as context.

    Returns:
        List[TickResult] for each replay tick (1 per candle).
    """
    if not candles:
        logger.warning("TW-5 replay: called with empty candle list.")
        return []

    # Copy state so replay is in-memory only.
    state: Dict[str, Any] = deepcopy(initial_state) if initial_state is not None else {}

    results: List[TickResult] = []
    call_history = _get_gpt_call_history(state)
    last_gpt_ts = _safe_float(state.get("tw5_last_gpt_ts"), None)
    last_snapshot: Tw5Snapshot | None = None

    # TW-5 gating thresholds (with reasonable defaults)
    min_seconds_between_calls = float(getattr(config, "tw5_min_seconds_between_gpt", 60.0))
    max_calls_per_hour = int(getattr(config, "tw5_max_gpt_calls_per_hour", 12))
    min_atr_move_mult = float(getattr(config, "tw5_min_atr_move_mult", 0.5))

    n = len(candles)
    for idx in range(n):
        window_start = max(0, idx - warmup_bars)
        window = list(candles[window_start : idx + 1])
        candle = window[-1]
        now_ts = float(candle.get("timestamp", time.time()))

        market_data = {"candles": window}
        snapshot = build_tw5_snapshot(config, market_data, state)

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

        # Generate plan (stub or GPT)
        if use_stub:
            plan = generate_tw5_stub_plan(snapshot, seed=None)
        else:
            plan = generate_order_plan_with_gpt(config, snapshot, state)

        # Clamp plan
        clamp_result = clamp_order_plan(snapshot, plan, state, config)

        # Update GPT call history
        call_ts = float(snapshot.timestamp or now_ts)
        call_history.append(call_ts)
        last_gpt_ts = call_ts
        state["tw5_gpt_call_history"] = call_history
        state["tw5_last_gpt_ts"] = last_gpt_ts

        _update_state_after_tick(
            state=state,
            snapshot=snapshot,
            plan=plan,
            clamp_result=clamp_result,
            gpt_called=True,
            gpt_ts=call_ts,
        )

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
# 2) PnL replay on top of TickResult
# ---------------------------------------------------------------------------


def run_tw5_replay_with_pnl_from_candles(
    candles: Sequence[Dict[str, Any]],
    config: Config,
    initial_state: Dict[str, Any],
    use_stub: bool = True,
    warmup_bars: int = 100,
    max_hold_bars: int = 1440,
    fee_rate: float = DEFAULT_FEE_RATE,
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS,
    order_lifetime_bars: int = 100,
) -> Tuple[List[TickResult], Tw5ReplayStats]:
    """
    Run TW-5 replay and simulate PnL using a simple one-position model with persistent limit orders.

    Rules:
      - One position at a time.
      - Limit orders persist across bars until filled or expired.
      - Entries only when flat, on approved non-flat plans (enter/manage).
      - Entry must be touchable within the bar's high/low to fill.
      - Position sized by risk_per_trade_pct.
      - Stop/TP checked bar-by-bar via OHLC (gap-aware).
      - Optional max_hold_bars -> "time_expiry" exit.
      - Basic fees + slippage applied.
      - Orders expire after order_lifetime_bars (default: 100).

    Returns:
      (ticks, stats) where `ticks` is the same as run_tw5_replay_from_candles
      and `stats` is a Tw5ReplayStats object.
    """
    ticks = run_tw5_replay_from_candles(
        candles=candles,
        config=config,
        initial_state=initial_state,
        use_stub=use_stub,
        warmup_bars=warmup_bars,
    )

    if not ticks:
        stats = Tw5ReplayStats(
            equity_curve=[],
            trades=[],
            total_pnl=0.0,
            total_return_pct=0.0,
            max_drawdown=0.0,
            trade_count=0,
            win_rate=0.0,
            avg_r=0.0,
        )
        return ticks, stats

    equity_start = float(getattr(config, "initial_equity", 0.0) or 0.0)
    equity = equity_start
    equity_curve: List[float] = [equity]
    trades: List[Dict[str, Any]] = []

    risk_pct = _effective_risk_per_trade_pct(config)
    if risk_pct <= 0.0:
        logger.warning(
            "TW-5 PnL replay: risk_per_trade is not configured (>0). "
            "No trades will be opened."
        )

    open_pos: Dict[str, Any] | None = None
    bars_in_trade = 0
    pending_orders: List[PendingOrder] = []

    for idx, (candle, tick) in enumerate(zip(candles, ticks)):
        ts = float(candle.get("timestamp", 0.0))
        high = _safe_price(candle, "high")
        low = _safe_price(candle, "low")
        close = _safe_price(candle, "close")

        # 0) Check pending limit orders for fills (only if flat and risk configured)
        if open_pos is None and risk_pct > 0.0:
            # Remove expired orders first
            pending_orders = [
                order for order in pending_orders
                if not order.is_expired(idx)
            ]

            # Try to fill pending orders on this bar
            filled_order = None
            for order in pending_orders:
                cp = order.clamp_result.clamped_plan or order.plan
                if not cp or cp.side not in ("long", "short"):
                    continue

                legs = list(cp.legs or [])
                if not legs:
                    continue

                # Calculate weighted entry price from legs
                def _weighted(attr: str, fallback: float) -> float:
                    num = 0.0
                    den = 0.0
                    for leg in legs:
                        w = max(0.0, leg.size_frac)
                        num += w * getattr(leg, attr)
                        den += w
                    return num / den if den > 0.0 else fallback

                entry_price_planned = _weighted("entry_price", close)

                # Check if entry price is within this bar's range (can be filled)
                can_fill = False
                if cp.side == "long":
                    can_fill = low <= entry_price_planned <= high
                elif cp.side == "short":
                    can_fill = low <= entry_price_planned <= high

                if can_fill:
                    filled_order = order
                    break

            # If we found a fillable order, enter the position
            if filled_order is not None:
                cp = filled_order.clamp_result.clamped_plan or filled_order.plan
                legs = list(cp.legs or [])

                # Calculate entry details
                def _weighted(attr: str, fallback: float) -> float:
                    num = 0.0
                    den = 0.0
                    for leg in legs:
                        w = max(0.0, leg.size_frac)
                        num += w * getattr(leg, attr)
                        den += w
                    return num / den if den > 0.0 else fallback

                entry_price_planned = _weighted("entry_price", close)

                # Apply slippage
                if cp.side == "long":
                    entry_price = entry_price_planned * (1.0 + slippage_bps)
                else:
                    entry_price = entry_price_planned * (1.0 - slippage_bps)

                # Calculate stop and TP
                if cp.side == "long":
                    stop_loss = min(leg.stop_loss for leg in legs)
                else:
                    stop_loss = max(leg.stop_loss for leg in legs)

                tp_num = 0.0
                tp_den = 0.0
                for leg in legs:
                    if leg.take_profits:
                        w = max(0.0, leg.size_frac)
                        tp_num += w * leg.take_profits[0].price
                        tp_den += w
                take_profit = tp_num / tp_den if tp_den > 0.0 else None

                # Validate and size the position
                if entry_price > 0.0 and stop_loss > 0.0 and entry_price != stop_loss:
                    risk_per_unit = abs(entry_price - stop_loss)
                    risk_value = equity * risk_pct
                    if risk_value > 0.0 and risk_per_unit > 0.0:
                        size = risk_value / risk_per_unit
                        if size > 0.0:
                            # Open the position
                            open_pos = {
                                "side": cp.side,
                                "entry_price": entry_price,
                                "size": size,
                                "stop_loss": stop_loss,
                                "take_profit": take_profit,
                                "entry_ts": ts,
                                "entry_equity": equity,
                                "risk_value": risk_value,
                            }
                            bars_in_trade = 0

                            # Remove the filled order and cancel all other pending orders
                            pending_orders = []

        # 1) Handle existing position exit first
        if open_pos is not None:
            bars_in_trade += 1
            exit_raw, exit_reason = _resolve_exit(open_pos, candle)

            time_expired = max_hold_bars > 0 and bars_in_trade > max_hold_bars
            if time_expired and exit_reason == "bar_close":
                exit_reason = "time_expiry"
                exit_raw = close

            if exit_reason != "bar_close" or time_expired:
                # Apply slippage and fees
                side = open_pos.get("side")
                size = float(open_pos.get("size", 0.0))
                entry_price = float(open_pos.get("entry_price", 0.0))

                if side == "long":
                    exit_price = exit_raw * (1.0 - slippage_bps)
                else:
                    exit_price = exit_raw * (1.0 + slippage_bps)

                pnl_gross = _compute_pnl(
                    {"entry_price": entry_price, "size": size, "side": side},
                    exit_price,
                )

                notional_entry = abs(entry_price * size)
                notional_exit = abs(exit_price * size)
                fees = fee_rate * (notional_entry + notional_exit)
                pnl = pnl_gross - fees

                entry_equity = float(open_pos.get("entry_equity", equity))
                risk_value = float(open_pos.get("risk_value", 0.0))

                ret_pct = pnl / entry_equity if entry_equity > 0 else 0.0
                r_mult = pnl / risk_value if risk_value > 0 else 0.0

                equity += pnl
                equity_curve.append(equity)

                trades.append(
                    {
                        "side": side,
                        "entry_ts": open_pos.get("entry_ts"),
                        "exit_ts": ts,
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "size": size,
                        "pnl": pnl,
                        "return_pct": ret_pct,
                        "R": r_mult,
                        "exit_reason": exit_reason,
                    }
                )

                open_pos = None
                bars_in_trade = 0

        # 2) Consider new entry only if flat and risk is configured
        if open_pos is not None or risk_pct <= 0.0:
            continue

        clamp = tick.clamp_result
        plan = tick.plan

        if (
            clamp is None
            or not clamp.approved
            or plan is None
            or plan.side not in ("long", "short")
            or plan.mode not in ("enter", "manage")
        ):
            continue

        # Use clamped plan if available, else original
        cp = clamp.clamped_plan or plan
        legs = list(cp.legs or [])
        if not legs:
            continue

        total_size_frac = sum(max(0.0, leg.size_frac) for leg in legs)
        if total_size_frac <= 0.0:
            continue

        def _weighted(attr: str, fallback: float) -> float:
            num = 0.0
            den = 0.0
            for leg in legs:
                w = max(0.0, leg.size_frac)
                num += w * getattr(leg, attr)
                den += w
            return num / den if den > 0.0 else fallback

        snapshot = tick.snapshot
        ref_price = float(snapshot.price)
        entry_price_planned = _weighted("entry_price", ref_price)

        # Check if entry is within this bar's range (can be filled immediately)
        can_fill_immediately = False
        if cp.side == "long":
            can_fill_immediately = low <= entry_price_planned <= high
        elif cp.side == "short":
            can_fill_immediately = low <= entry_price_planned <= high

        if can_fill_immediately:
            # Fill immediately on this bar
            if cp.side == "long":
                entry_price = entry_price_planned * (1.0 + slippage_bps)
            else:
                entry_price = entry_price_planned * (1.0 - slippage_bps)

            # Stop and TP from clamped legs
            if cp.side == "long":
                stop_loss = min(leg.stop_loss for leg in legs)
            else:
                stop_loss = max(leg.stop_loss for leg in legs)

            # Weighted TP from first TP of each leg (optional)
            tp_num = 0.0
            tp_den = 0.0
            for leg in legs:
                if leg.take_profits:
                    w = max(0.0, leg.size_frac)
                    tp_num += w * leg.take_profits[0].price
                    tp_den += w
            take_profit = tp_num / tp_den if tp_den > 0.0 else None

            if entry_price > 0.0 and stop_loss > 0.0 and entry_price != stop_loss:
                risk_per_unit = abs(entry_price - stop_loss)
                risk_value = equity * risk_pct
                if risk_value > 0.0 and risk_per_unit > 0.0:
                    size = risk_value / risk_per_unit
                    if size > 0.0:
                        open_pos = {
                            "side": cp.side,
                            "entry_price": entry_price,
                            "size": size,
                            "stop_loss": stop_loss,
                            "take_profit": take_profit,
                            "entry_ts": ts,
                            "entry_equity": equity,
                            "risk_value": risk_value,
                        }
                        bars_in_trade = 0

                        # Cancel any pending orders since we just entered
                        pending_orders = []
        else:
            # Entry not fillable on this bar - add to pending orders
            # Create a pending order that will persist until filled or expired
            pending_order = PendingOrder(
                plan=plan,
                clamp_result=clamp,
                created_ts=ts,
                created_idx=idx,
                expires_idx=idx + order_lifetime_bars,
                snapshot=snapshot,
            )
            pending_orders.append(pending_order)

    # If still in a trade at the end, close at last candle
    if open_pos is not None and candles:
        last_candle = candles[-1]
        last_ts = float(last_candle.get("timestamp", 0.0))
        last_close = _safe_price(last_candle, "close", open_pos.get("entry_price", 0.0))
        exit_raw, exit_reason = _resolve_exit(open_pos, last_candle)
        if exit_reason == "bar_close":
            exit_reason = "end_of_data"
            exit_raw = last_close

        side = open_pos.get("side")
        size = float(open_pos.get("size", 0.0))
        entry_price = float(open_pos.get("entry_price", 0.0))

        if side == "long":
            exit_price = exit_raw * (1.0 - slippage_bps)
        else:
            exit_price = exit_raw * (1.0 + slippage_bps)

        pnl_gross = _compute_pnl(
            {"entry_price": entry_price, "size": size, "side": side},
            exit_price,
        )
        notional_entry = abs(entry_price * size)
        notional_exit = abs(exit_price * size)
        fees = fee_rate * (notional_entry + notional_exit)
        pnl = pnl_gross - fees

        entry_equity = float(open_pos.get("entry_equity", equity))
        risk_value = float(open_pos.get("risk_value", 0.0))
        ret_pct = pnl / entry_equity if entry_equity > 0 else 0.0
        r_mult = pnl / risk_value if risk_value > 0 else 0.0

        equity += pnl
        equity_curve.append(equity)

        trades.append(
            {
                "side": side,
                "entry_ts": open_pos.get("entry_ts"),
                "exit_ts": last_ts,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "size": size,
                "pnl": pnl,
                "return_pct": ret_pct,
                "R": r_mult,
                "exit_reason": exit_reason,
            }
        )

    trade_count = len(trades)
    total_pnl = sum(t["pnl"] for t in trades)
    if equity_start > 0.0:
        total_return_pct = (equity / equity_start) - 1.0
    else:
        total_return_pct = 0.0

    max_dd = _max_drawdown(equity_curve) if equity_curve else 0.0
    wins = [t for t in trades if t["pnl"] > 0.0]
    win_rate = len(wins) / trade_count if trade_count > 0 else 0.0
    r_values = [t.get("R", 0.0) for t in trades]
    avg_r = sum(r_values) / len(r_values) if r_values else 0.0

    stats = Tw5ReplayStats(
        equity_curve=equity_curve,
        trades=trades,
        total_pnl=total_pnl,
        total_return_pct=total_return_pct,
        max_drawdown=max_dd,
        trade_count=trade_count,
        win_rate=win_rate,
        avg_r=avg_r,
    )
    return ticks, stats


# ---------------------------------------------------------------------------
# CLI entrypoint for quick experimentation
# ---------------------------------------------------------------------------


def _load_csv_candles(path: str) -> List[Dict[str, Any]]:
    candles: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            candles.append(
                {
                    "timestamp": float(row.get("timestamp", 0.0)),
                    "open": float(row.get("open", row.get("close", 0.0))),
                    "high": float(row.get("high", row.get("close", 0.0))),
                    "low": float(row.get("low", row.get("close", 0.0))),
                    "close": float(row.get("close", 0.0)),
                    "volume": float(row.get("volume", 0.0)),
                }
            )
    return candles


def main() -> None:
    parser = argparse.ArgumentParser(description="TW-5 replay with optional PnL simulation.")
    parser.add_argument("--csv", required=True, help="Path to candles CSV with timestamp,open,high,low,close,volume")
    parser.add_argument(
        "--use-gpt",
        action="store_true",
        help="Use GPT policy instead of stub (stub is default and costs $0).",
    )
    parser.add_argument(
        "--equity",
        type=float,
        default=1000.0,
        help="Starting equity for PnL simulation (default: 1000.0).",
    )
    parser.add_argument(
        "--warmup-bars",
        type=int,
        default=200,
        help="Warmup bars for snapshot context (default: 200).",
    )
    parser.add_argument(
        "--max-hold-bars",
        type=int,
        default=1440,
        help="Max bars to hold a position before time-expiry exit (default: 1440 = 1d on 1m).",
    )
    parser.add_argument(
        "--stub",
        action="store_true",
        help="Force stub mode (default). If omitted and --use-gpt is not set, stub is used.",
    )
