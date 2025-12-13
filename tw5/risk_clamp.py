# tw5/risk_clamp.py

"""
Down-only risk clamp for TW-5.

Takes an OrderPlan and applies:
  - per-trade risk caps
  - leverage/notional caps (via config.max_leverage)
  - optional daily loss clamp
  - optional equity floor
  - GPT safe mode and kill switch guards

It MUST NOT:
  - invent new trades
  - widen stops
  - increase leverage beyond config caps
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, List, Optional

from config import Config
from logger_utils import get_logger
from safety_utils import check_trading_halted
from state_memory import is_gpt_safe_mode
from .schemas import OrderPlan, OrderLeg, RiskClampResult, Tw5Snapshot

logger = get_logger(__name__)


def clamp_order_plan(
    snapshot: Tw5Snapshot,
    plan: OrderPlan,
    state: Dict[str, Any],
    config: Config,
) -> RiskClampResult:
    """
    Apply TW-5 risk rules to an OrderPlan.

    Behaviour:
    - If trading is halted (kill switch / circuit breaker) -> veto.
    - If GPT safe mode is active -> veto non-flat plans.
    - If plan is already flat -> approve as-is (flat).
    - If snapshot/price is invalid -> veto.
    - If equity <= 0 -> veto.
    - If equity floor breached -> veto.
    - If daily loss limit exceeded -> veto.
    - Otherwise:
        * shrink stops if they exceed max_stop_pct (never widen)
        * normalise leg size_fracs to sum to 1 (take_profits may be empty; exits are handled downstream)
        * compute allowed max_total_size_frac from risk_per_trade and max_leverage
        * if that shrinks to ~0 -> veto
        * else return clamped plan
    """

    # 0) Trading halted / kill switch
    halted, halt_reason = check_trading_halted(state)
    if halted:
        logger.warning("TW-5 risk_clamp: trading halted (%s). Vetoing plan.", halt_reason)
        return RiskClampResult(
            approved=False,
            reason=f"trading_not_allowed:{halt_reason}",
            original_plan=plan,
            clamped_plan=OrderPlan.empty_flat(f"trading_not_allowed:{halt_reason}"),
        )

    # 1) GPT Safe Mode: no new exposure
    if is_gpt_safe_mode(state):
        if plan.side != "flat":
            logger.warning("TW-5 risk_clamp: GPT safe mode active. Forcing flat.")
        return RiskClampResult(
            approved=False,
            reason="gpt_safe_mode_active",
            original_plan=plan,
            clamped_plan=OrderPlan.empty_flat("gpt_safe_mode_active"),
        )

    # 2) Flat plans are always safe
    if plan.mode == "flat" or plan.side == "flat" or not plan.legs:
        return RiskClampResult(
            approved=True,
            reason="flat_plan",
            original_plan=plan,
            clamped_plan=OrderPlan.empty_flat("flat_plan"),
        )

    if plan.mode not in ("enter", "manage"):
        return RiskClampResult(
            approved=False,
            reason=f"unsupported_mode:{plan.mode}",
            original_plan=plan,
            clamped_plan=OrderPlan.empty_flat(f"unsupported_mode:{plan.mode}"),
        )

    if plan.side not in ("long", "short"):
        return RiskClampResult(
            approved=False,
            reason=f"unsupported_side:{plan.side}",
            original_plan=plan,
            clamped_plan=OrderPlan.empty_flat(f"unsupported_side:{plan.side}"),
        )

    # 3) Basic snapshot sanity
    if snapshot.price <= 0.0:
        return RiskClampResult(
            approved=False,
            reason="invalid_snapshot_price",
            original_plan=plan,
            clamped_plan=OrderPlan.empty_flat("invalid_snapshot_price"),
        )

    # 4) Equity and daily stats
    equity = _safe_float(state.get("equity", getattr(config, "initial_equity", 0.0)), 0.0)
    if equity <= 0.0:
        return RiskClampResult(
            approved=False,
            reason="non_positive_equity",
            original_plan=plan,
            clamped_plan=OrderPlan.empty_flat("non_positive_equity"),
        )

    daily_pnl = _safe_float(state.get("daily_pnl", 0.0), 0.0)
    daily_start_equity = _safe_float(state.get("daily_start_equity"), None)
    start_equity = _safe_float(getattr(config, "initial_equity", None), None)
    if start_equity is None or start_equity <= 0.0:
        start_equity = daily_start_equity or equity

    # Equity floor: TW-5 specific, default 70% of starting equity
    equity_floor_pct = _safe_float(getattr(config, "tw5_equity_floor_pct", 0.7), 0.7)
    if start_equity and equity_floor_pct > 0.0:
        floor = start_equity * equity_floor_pct
        if equity <= floor:
            logger.warning(
                "TW-5 risk_clamp: equity_floor breached (equity=%.2f, floor=%.2f).",
                equity,
                floor,
            )
            return RiskClampResult(
                approved=False,
                reason="equity_floor_breached",
                original_plan=plan,
                clamped_plan=OrderPlan.empty_flat("equity_floor_breached"),
            )

    # Daily loss cap: TW-5 override or 5% default
    max_daily_loss_pct = _safe_float(getattr(config, "tw5_max_daily_loss_pct", 0.05), 0.05)
    effective_daily_start = daily_start_equity or start_equity
    if _check_daily_loss_limit(daily_pnl, effective_daily_start, max_daily_loss_pct):
        logger.warning(
            "TW-5 risk_clamp: daily loss limit exceeded (daily_pnl=%.2f, start_equity=%.2f, max_daily_loss_pct=%.3f).",
            daily_pnl,
            effective_daily_start or 0.0,
            max_daily_loss_pct,
        )
        return RiskClampResult(
            approved=False,
            reason="daily_loss_limit_exceeded",
            original_plan=plan,
            clamped_plan=OrderPlan.empty_flat("daily_loss_limit_exceeded"),
        )

    # 5) Risk parameters from config (with optional TW-5 overrides)
    base_risk_pct = max(_safe_float(getattr(config, "risk_per_trade", 0.0), 0.0), 0.0)
    tw5_risk_override = _safe_float(getattr(config, "tw5_risk_per_trade_pct", 0.0), 0.0)
    risk_per_trade_pct = tw5_risk_override if tw5_risk_override > 0.0 else base_risk_pct
    if risk_per_trade_pct <= 0.0:
        return RiskClampResult(
            approved=False,
            reason="risk_per_trade_not_configured",
            original_plan=plan,
            clamped_plan=OrderPlan.empty_flat("risk_per_trade_not_configured"),
        )

    base_max_leverage = max(_safe_float(getattr(config, "max_leverage", 1.0), 1.0), 0.0)
    tw5_max_lev_override = _safe_float(getattr(config, "tw5_max_leverage", 0.0), 0.0)
    effective_max_leverage = base_max_leverage
    if 0.0 < tw5_max_lev_override < base_max_leverage:
        effective_max_leverage = tw5_max_lev_override

    if effective_max_leverage <= 0.0:
        return RiskClampResult(
            approved=False,
            reason="non_positive_max_leverage",
            original_plan=plan,
            clamped_plan=OrderPlan.empty_flat("non_positive_max_leverage"),
        )

    max_stop_pct = _safe_float(getattr(config, "tw5_max_stop_pct", 0.05), 0.05)
    max_stop_pct = max(0.001, min(max_stop_pct, 0.10))  # between 0.1% and 10%

    # 6) Prepare legs: shrink stops if needed, normalise size_fracs. Take-profits may be empty; executor applies exits.
    valid_legs: List[OrderLeg] = []
    for leg in plan.legs:
        if leg.size_frac <= 0.0 or leg.entry_price <= 0.0:
            continue
        clamped_stop = _clamp_stop_distance(plan.side, leg, max_stop_pct)
        valid_legs.append(replace(leg, stop_loss=clamped_stop))

    if not valid_legs:
        return RiskClampResult(
            approved=False,
            reason="no_valid_legs",
            original_plan=plan,
            clamped_plan=OrderPlan.empty_flat("no_valid_legs"),
        )

    total_size_frac = sum(max(0.0, leg.size_frac) for leg in valid_legs)
    if total_size_frac <= 0.0:
        return RiskClampResult(
            approved=False,
            reason="total_size_frac_zero",
            original_plan=plan,
            clamped_plan=OrderPlan.empty_flat("total_size_frac_zero"),
        )

    # Normalise leg size_fracs to sum to 1.0
    norm_legs: List[OrderLeg] = []
    for leg in valid_legs:
        norm_size = max(0.0, leg.size_frac) / total_size_frac
        norm_legs.append(replace(leg, size_frac=norm_size))

    # 7) Compute risk factor: sum(weight_i * risk_frac_i)
    risk_factor = _compute_plan_risk_factor(norm_legs)
    if risk_factor <= 0.0:
        return RiskClampResult(
            approved=False,
            reason="non_positive_risk_factor",
            original_plan=plan,
            clamped_plan=OrderPlan.empty_flat("non_positive_risk_factor"),
        )

    # Allowed max_total_size_frac from risk constraint:
    # risk_total = equity * effective_max_leverage * max_total_size_frac * risk_factor
    # We require: risk_total <= equity * risk_per_trade_pct
    # => max_total_size_frac <= risk_per_trade_pct / (effective_max_leverage * risk_factor)
    max_size_from_risk = risk_per_trade_pct / (effective_max_leverage * risk_factor)
    max_size_from_risk = max(0.0, min(1.0, max_size_from_risk))

    # Respect GPT's own aggressiveness suggestion
    gpt_max_size = max(0.0, min(1.0, plan.max_total_size_frac))
    clamped_max_total_size_frac = min(gpt_max_size, max_size_from_risk)

    if clamped_max_total_size_frac <= 0.0:
        logger.info(
            "TW-5 risk_clamp: size scaled to zero (risk_per_trade_pct=%.4f, lev=%.2f, risk_factor=%.5f).",
            risk_per_trade_pct,
            effective_max_leverage,
            risk_factor,
        )
        return RiskClampResult(
            approved=False,
            reason="size_scaled_to_zero",
            original_plan=plan,
            clamped_plan=OrderPlan.empty_flat("size_scaled_to_zero"),
        )

    clamped_plan = OrderPlan(
        mode=plan.mode,
        side=plan.side,
        legs=norm_legs,
        max_total_size_frac=clamped_max_total_size_frac,
        confidence=plan.confidence,
        rationale=plan.rationale,
    )

    return RiskClampResult(
        approved=True,
        reason="ok",
        original_plan=plan,
        clamped_plan=clamped_plan,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_float(value: Any, default: Optional[float] = 0.0) -> float:
    try:
        if value is None:
            raise TypeError
        return float(value)
    except (TypeError, ValueError):
        return float(default if default is not None else 0.0)


def _check_daily_loss_limit(
    daily_pnl: float,
    daily_start_equity: Optional[float],
    max_daily_loss_pct: float,
) -> bool:
    """
    Minimal daily loss check.

    Returns True if daily loss limit is exceeded.
    """
    if daily_start_equity is None or daily_start_equity <= 0.0:
        return False
    if max_daily_loss_pct <= 0.0:
        return False

    daily_pnl_fraction = daily_pnl / daily_start_equity
    return daily_pnl_fraction <= -max_daily_loss_pct


def _clamp_stop_distance(
    side: str,
    leg: OrderLeg,
    max_stop_pct: float,
) -> float:
    """
    Ensure stop distance does not exceed max_stop_pct.

    We only tighten stops (move them closer to entry), never widen them.
    """
    entry = leg.entry_price
    stop = leg.stop_loss

    if entry <= 0.0:
        return stop

    if side == "long":
        # Risk as fraction of entry: (entry - stop) / entry
        dist = (entry - stop) / entry
        if dist <= 0.0:
            # Degenerate or protective stop; leave as-is
            return stop
        if dist > max_stop_pct:
            clamped = entry * (1.0 - max_stop_pct)
            return max(clamped, 0.0001)
        return stop

    if side == "short":
        # Risk as fraction: (stop - entry) / entry
        dist = (stop - entry) / entry
        if dist <= 0.0:
            return stop
        if dist > max_stop_pct:
            clamped = entry * (1.0 + max_stop_pct)
            return max(clamped, 0.0001)
        return stop

    return stop


def _compute_plan_risk_factor(legs: List[OrderLeg]) -> float:
    """
    Aggregate risk factor over legs:

        risk_factor = sum(weight_i * risk_frac_i)

    where risk_frac_i = abs(entry - stop) / entry.
    """
    if not legs:
        return 0.0

    total = 0.0
    for leg in legs:
        entry = leg.entry_price
        stop = leg.stop_loss
        if entry <= 0.0:
            continue
        dist = abs(entry - stop) / entry
        if dist <= 0.0:
            continue
        weight = max(0.0, leg.size_frac)
        total += weight * dist

    return total
