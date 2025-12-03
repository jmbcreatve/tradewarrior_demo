from __future__ import annotations

from typing import Any, Dict

from config import Config
from enums import (
    Side,
    VolatilityMode,
    RangePosition,
    TimingState,
    coerce_enum,
)
from logger_utils import get_logger
from risk_envelope import compute_risk_envelope, RiskEnvelope
from schemas import GptDecision, RiskDecision

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _get_float(d: Dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(d.get(key, default) or default)
    except (TypeError, ValueError):
        return default


def _norm_gpt_decision(gpt_decision: Any) -> Dict[str, Any]:
    """
    Accept either a GptDecision instance or a raw dict and normalise it into:

        {"action": "long|short|flat", "confidence": float[0,1], "notes": str}
    """
    if isinstance(gpt_decision, GptDecision):
        data = gpt_decision.to_dict()
    elif isinstance(gpt_decision, dict):
        data = {
            "action": str(gpt_decision.get("action", "flat")).lower(),
            "confidence": _get_float(gpt_decision, "confidence", 0.0),
            "notes": str(gpt_decision.get("notes", "") or ""),
        }
    else:
        data = {"action": "flat", "confidence": 0.0, "notes": ""}

    action = str(data.get("action", "flat")).lower()
    if action not in {"long", "short", "flat"}:
        action = "flat"

    conf = _get_float(data, "confidence", 0.0)
    conf = _clamp(conf, 0.0, 1.0)

    notes = str(data.get("notes", "") or "")

    return {"action": action, "confidence": conf, "notes": notes}


def _vol_stop_pct(vol_enum: VolatilityMode) -> float:
    """
    Map vol regime to a reasonable stop distance as a fraction of price.

    Higher vol → wider stops. Very conservative on purpose.
    """
    if vol_enum == VolatilityMode.LOW:
        return 0.003  # 0.3%
    if vol_enum == VolatilityMode.NORMAL:
        return 0.005  # 0.5%
    if vol_enum == VolatilityMode.HIGH:
        return 0.008  # 0.8%
    if vol_enum == VolatilityMode.EXPLOSIVE:
        return 0.012  # 1.2%
    return 0.006  # fallback


def _check_daily_loss_limit(
    daily_pnl: float,
    daily_start_equity: float,
    max_daily_loss_pct: float,
) -> bool:
    """
    Check if daily loss limit has been exceeded.

    Args:
        daily_pnl: Current day's P&L (can be negative for losses)
        daily_start_equity: Equity at start of trading day
        max_daily_loss_pct: Maximum daily loss as fraction (e.g., 0.03 for 3%)

    Returns:
        True if daily loss limit is exceeded (should halt trading).
    """
    if daily_start_equity is None or daily_start_equity <= 0.0:
        return False  # No daily tracking yet, allow trading

    if max_daily_loss_pct <= 0.0:
        return False  # No limit set, allow trading

    # Calculate daily P&L as fraction of start equity
    daily_pnl_fraction = daily_pnl / daily_start_equity
    # Compare: if daily_pnl_fraction <= -max_daily_loss_pct, limit exceeded
    # Example: -0.04 <= -0.03 is True (4% loss exceeds 3% limit)
    return daily_pnl_fraction <= -max_daily_loss_pct


# ---------------------------------------------------------------------------
# Core risk engine
# ---------------------------------------------------------------------------

def evaluate_risk(
    snapshot: Dict[str, Any],
    gpt_decision: GptDecision,
    state: Dict[str, Any],
    config: Config,
) -> RiskDecision:
    """
    Risk engine: treat GPT as a proposal, enforce hard risk rules.

    Invariants:
    - If GPT says FLAT, we are FLAT (no trade).
    - If price/equity are invalid, no trade.
    - danger_mode or TimingState.AVOID → no trade.
    - Regime/timing can only REDUCE risk, never increase it.
    """

    # --- Normalise GPT decision ----------------------------------------------
    gpt = _norm_gpt_decision(gpt_decision)
    action = gpt["action"]
    confidence = gpt["confidence"]
    symbol = snapshot.get("symbol")
    timestamp = snapshot.get("timestamp")
    price = _get_float(snapshot, "price", 0.0)
    vol_enum = coerce_enum(
        str(snapshot.get("volatility_mode", "unknown")),
        VolatilityMode,
        VolatilityMode.UNKNOWN,
    )
    range_enum = coerce_enum(
        str(snapshot.get("range_position", "mid")),
        RangePosition,
        RangePosition.MID,
    )
    timing_enum = coerce_enum(
        str(snapshot.get("timing_state", "normal")),
        TimingState,
        TimingState.NORMAL,
    )
    danger_mode = bool(snapshot.get("danger_mode", False))
    effective_env: RiskEnvelope | None = None

    def _log_risk_event(decision: RiskDecision, env: RiskEnvelope | None) -> None:
        try:
            # Calculate stop distance if we have stop_loss and price
            stop_distance_pct = None
            if decision.stop_loss_price and price > 0:
                if decision.side == "long":
                    stop_distance_pct = abs((price - decision.stop_loss_price) / price)
                elif decision.side == "short":
                    stop_distance_pct = abs((decision.stop_loss_price - price) / price)
            
            event = {
                "type": "risk_decision",
                "symbol": symbol,
                "timestamp": timestamp,
                "price": price,
                "gpt_action": action,
                "gpt_confidence": confidence,
                "approved": decision.approved,
                "side": decision.side,
                "position_size": decision.position_size,
                "leverage": decision.leverage,
                "stop_loss_price": decision.stop_loss_price,
                "take_profit_price": decision.take_profit_price,
                "stop_distance_pct": stop_distance_pct,
                "reason": decision.reason,
                "risk_envelope": {
                    "max_notional": env.max_notional if env else None,
                    "max_leverage": env.max_leverage if env else None,
                    "max_risk_per_trade_pct": env.max_risk_per_trade_pct if env else None,
                    "min_stop_distance_pct": env.min_stop_distance_pct if env else None,
                    "max_stop_distance_pct": env.max_stop_distance_pct if env else None,
                    "max_daily_loss_pct": env.max_daily_loss_pct if env else None,
                    "note": env.note if env else None,
                },
                "danger_mode": danger_mode,
                "timing_state": timing_enum.value if isinstance(timing_enum, TimingState) else timing_enum,
            }
            logger.info("RiskDecision event: %s", event)
        except Exception:
            logger.info("RiskDecision event logging failed", exc_info=True)

    if action == "flat":
        logger.info("Risk: GPT requested FLAT; no trade.")
        decision = RiskDecision(
            approved=False,
            side=Side.FLAT.value,
            position_size=0.0,
            leverage=0.0,
            stop_loss_price=None,
            take_profit_price=None,
            reason="gpt_flat",
        )
        _log_risk_event(decision, effective_env)
        return decision

    # --- Basic market sanity checks -------------------------------------------
    if price <= 0.0:
        logger.info("Risk: invalid or zero price; no trade.")
        decision = RiskDecision(
            approved=False,
            side=Side.FLAT.value,
            position_size=0.0,
            leverage=0.0,
            stop_loss_price=None,
            take_profit_price=None,
            reason="invalid_price",
        )
        _log_risk_event(decision, effective_env)
        return decision

    # --- Equity / risk budget -------------------------------------------------
    # Use equity from state, falling back to config's initial_equity
    default_equity = getattr(config, "initial_equity", 10_000.0)
    equity = _get_float(state, "equity", default_equity)
    if equity <= 0.0:
        logger.info("Risk: non-positive equity; no trade.")
        decision = RiskDecision(
            approved=False,
            side=Side.FLAT.value,
            position_size=0.0,
            leverage=0.0,
            stop_loss_price=None,
            take_profit_price=None,
            reason="no_equity",
        )
        _log_risk_event(decision, effective_env)
        return decision

    # --- Risk envelope (downward-only caps) ----------------------------------
    risk_env_dict = snapshot.get("risk_envelope") or None
    if isinstance(risk_env_dict, dict) and risk_env_dict:
        def _env_float(key: str) -> float:
            try:
                return float(risk_env_dict.get(key, 0.0) or 0.0)
            except (TypeError, ValueError):
                return 0.0

        env_note = str(risk_env_dict.get("note") or "snapshot risk envelope")
        try:
            effective_env = RiskEnvelope(
                max_notional=_env_float("max_notional"),
                max_leverage=_env_float("max_leverage"),
                max_risk_per_trade_pct=_env_float("max_risk_per_trade_pct"),
                min_stop_distance_pct=_env_float("min_stop_distance_pct"),
                max_stop_distance_pct=_env_float("max_stop_distance_pct"),
                max_daily_loss_pct=_env_float("max_daily_loss_pct"),
                note=env_note,
            )
        except Exception:
            effective_env = compute_risk_envelope(config, equity, vol_enum, danger_mode, timing_enum)
    else:
        effective_env = compute_risk_envelope(config, equity, vol_enum, danger_mode, timing_enum)

    # Store effective envelope in state for execution logging
    if effective_env:
        state["last_risk_envelope"] = effective_env.to_dict()

    # --- Daily loss limit check (circuit breaker) -----------------------------
    daily_pnl = _get_float(state, "daily_pnl", 0.0)
    daily_start_equity = state.get("daily_start_equity")
    if daily_start_equity is not None:
        daily_start_equity = _get_float(state, "daily_start_equity", 0.0)
        max_daily_loss_pct = effective_env.max_daily_loss_pct if effective_env else 0.03

        if _check_daily_loss_limit(daily_pnl, daily_start_equity, max_daily_loss_pct):
            daily_pnl_pct = (daily_pnl / daily_start_equity) * 100.0 if daily_start_equity > 0 else 0.0
            logger.warning(
                "Risk: DAILY LOSS LIMIT EXCEEDED (daily_pnl=%.2f, daily_pnl_pct=%.2f%%, "
                "max_daily_loss_pct=%.2f%%). Circuit breaker activated - no trade allowed.",
                daily_pnl,
                daily_pnl_pct,
                max_daily_loss_pct * 100.0,
            )
            decision = RiskDecision(
                approved=False,
                side=Side.FLAT.value,
                position_size=0.0,
                leverage=0.0,
                stop_loss_price=None,
                take_profit_price=None,
                reason="daily_loss_limit_exceeded",
            )
            _log_risk_event(decision, effective_env)
            return decision

    cfg_risk_pct = max(config.risk_per_trade, 0.0)
    env_risk_pct = max(0.0, effective_env.max_risk_per_trade_pct)
    allowed_risk_pct = min(cfg_risk_pct, env_risk_pct)

    cfg_max_lev = max(float(getattr(config, "max_leverage", 0.0)), 0.0)
    env_max_lev = max(0.0, effective_env.max_leverage)
    allowed_max_lev = min(cfg_max_lev, env_max_lev)

    env_max_notional = float(getattr(effective_env, "max_notional", 0.0) or 0.0)

    if allowed_risk_pct <= 0.0 or allowed_max_lev <= 0.0 or env_max_notional <= 0.0:
        logger.info(
            "Risk: risk envelope forbids new exposure (risk_pct=%.4f, lev=%.2f, max_notional=%.2f).",
            allowed_risk_pct,
            allowed_max_lev,
            env_max_notional,
        )
        decision = RiskDecision(
            approved=False,
            side=Side.FLAT.value,
            position_size=0.0,
            leverage=0.0,
            stop_loss_price=None,
            take_profit_price=None,
            reason="risk envelope forbids new exposure",
        )
        _log_risk_event(decision, effective_env)
        return decision

    base_risk_pct = cfg_risk_pct
    # Start with GPT confidence scaling risk within [0, allowed_risk_pct]
    risk_pct = _clamp(base_risk_pct * confidence, 0.0, allowed_risk_pct)

    regime_mult = 1.0

    # Volatility: trim risk as vol increases
    if vol_enum == VolatilityMode.LOW:
        regime_mult *= 0.75
    elif vol_enum == VolatilityMode.NORMAL:
        regime_mult *= 1.0
    elif vol_enum == VolatilityMode.HIGH:
        regime_mult *= 0.7
    elif vol_enum == VolatilityMode.EXPLOSIVE:
        regime_mult *= 0.5

    # Range extremes: cut risk at extremes
    if range_enum in {RangePosition.EXTREME_HIGH, RangePosition.EXTREME_LOW}:
        regime_mult *= 0.5

    # Timing: AVOID kills risk, CAUTIOUS trims risk
    if timing_enum == TimingState.AVOID:
        regime_mult = 0.0
    elif timing_enum == TimingState.CAUTIOUS:
        regime_mult *= 0.7

    # Danger mode is a hard veto
    if danger_mode:
        regime_mult = 0.0

    effective_risk_pct = _clamp(risk_pct * regime_mult, 0.0, allowed_risk_pct)

    if effective_risk_pct <= 0.0:
        logger.info(
            "Risk: effective risk zero (risk_pct=%.4f, regime_mult=%.3f, vol=%s, "
            "range=%s, timing=%s, danger=%s); no trade.",
            risk_pct,
            regime_mult,
            vol_enum,
            range_enum,
            timing_enum,
            danger_mode,
        )
        decision = RiskDecision(
            approved=False,
            side=Side.FLAT.value,
            position_size=0.0,
            leverage=0.0,
            stop_loss_price=None,
            take_profit_price=None,
            reason="risk_zero",
        )
        _log_risk_event(decision, effective_env)
        return decision

    # --- Position sizing ------------------------------------------------------
    stop_pct = _vol_stop_pct(vol_enum)
    stop_pct = max(stop_pct, 0.001)  # floor at 0.1%

    dollar_risk = equity * effective_risk_pct

    # Leverage: 1x → max_leverage based on confidence,
    # with a hard cap from max_leverage_10x_mode, and trimmed by the envelope.
    max_lev_cfg = max(float(getattr(config, "max_leverage", 1.0)), 0.0)
    max_lev_hard = max(
        float(getattr(config, "max_leverage_10x_mode", max_lev_cfg)),
        max_lev_cfg,
    )

    leverage = 1.0 + (max_lev_cfg - 1.0) * confidence
    leverage = _clamp(leverage, 0.0, min(max_lev_hard, allowed_max_lev))

    # Units: units = $risk * leverage / (price * stop_pct)
    qty = (dollar_risk * leverage) / (price * stop_pct)
    if qty <= 0.0:
        logger.info("Risk: computed position size <= 0; no trade.")
        decision = RiskDecision(
            approved=False,
            side=Side.FLAT.value,
            position_size=0.0,
            leverage=0.0,
            stop_loss_price=None,
            take_profit_price=None,
            reason="qty_zero",
        )
        _log_risk_event(decision, effective_env)
        return decision

    # Global notional cap: never risk more than X% equity notionally.
    max_notional_pct = 0.05
    notional = price * qty
    cfg_notional_cap = equity * max_notional_pct * leverage
    overall_notional_cap = min(cfg_notional_cap, env_max_notional)
    if notional > overall_notional_cap:
        scale = overall_notional_cap / notional
        qty *= scale
        logger.info(
            "Risk: scaled down qty due to notional cap (scale=%.3f, notional=%.2f, cfg_cap=%.2f, env_cap=%.2f).",
            scale,
            notional,
            cfg_notional_cap,
            env_max_notional,
        )

    # --- Stops / targets ------------------------------------------------------
    stop_distance = price * stop_pct

    if action == "long":
        stop_loss = price - stop_distance
        take_profit = price + 2.0 * stop_distance
        side_str = Side.LONG.value
    else:
        stop_loss = price + stop_distance
        take_profit = price - 2.0 * stop_distance
        side_str = Side.SHORT.value

    logger.info(
        "Risk: approving trade side=%s, qty=%.6f, lev=%.2f, risk_pct=%.4f "
        "(conf=%.3f, regime_mult=%.3f, vol=%s, range=%s, danger=%s, timing=%s).",
        side_str,
        qty,
        leverage,
        effective_risk_pct,
        confidence,
        regime_mult,
        vol_enum,
        range_enum,
        danger_mode,
        timing_enum,
    )

    decision = RiskDecision(
        approved=True,
        side=side_str,
        position_size=qty,
        leverage=leverage,
        stop_loss_price=stop_loss,
        take_profit_price=take_profit,
        reason="risk_rules_v2",
    )
    _log_risk_event(decision, effective_env)
    return decision
