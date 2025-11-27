from typing import Dict, Any

from config import Config
from schemas import GptDecision, RiskDecision
from enums import Side
from logger_utils import get_logger

logger = get_logger(__name__)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _get_float(d: Dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(d.get(key, default) or default)
    except (TypeError, ValueError):
        return default


def evaluate_risk(
    snapshot: Dict[str, Any],
    gpt_decision: GptDecision,
    state: Dict[str, Any],
    config: Config,
) -> RiskDecision:
    """Risk engine: treat GPT as a proposal, enforce hard risk rules.

    Key ideas:
    - FLAT is always allowed and costs nothing.
    - GPT's action is a suggestion; we can veto or down-size it.
    - Risk is sized off equity * risk_per_trade, modulated by confidence & regime.
    - Volatility and danger/timing can only REDUCE risk, never increase it.
    """

    # --- Normalize GPT decision ------------------------------------------------
    raw_action = (gpt_decision.action or "flat").strip().lower()
    if raw_action not in ("long", "short", "flat"):
        raw_action = "flat"

    confidence = float(gpt_decision.confidence or 0.0)
    confidence = _clamp(confidence, 0.0, 1.0)

    # If GPT is flat, we are flat. No discussion.
    if raw_action == "flat":
        logger.info("Risk engine: GPT chose FLAT; no trade.")
        return RiskDecision(
            approved=False,
            side=Side.FLAT.value,
            position_size=0.0,
            leverage=0.0,
            stop_loss_price=None,
            take_profit_price=None,
            reason="gpt_flat",
        )

    # --- Basic market sanity checks -------------------------------------------
    price = _get_float(snapshot, "price", 0.0)
    if price <= 0.0:
        logger.info("Risk engine: invalid or zero price; no trade.")
        return RiskDecision(
            approved=False,
            side=Side.FLAT.value,
            position_size=0.0,
            leverage=0.0,
            stop_loss_price=None,
            take_profit_price=None,
            reason="invalid_price",
        )

    trend = str(snapshot.get("trend", "unknown") or "unknown").lower()
    vol_mode = str(snapshot.get("volatility_mode", "unknown") or "unknown").lower()
    danger_mode = bool(snapshot.get("danger_mode", False))
    timing_state = str(snapshot.get("timing_state", "unknown") or "unknown").lower()
    range_position = str(snapshot.get("range_position", "unknown") or "unknown").lower()

    # --- Pull equity and base risk --------------------------------------------
    equity = float(state.get("equity", 10_000.0) or 10_000.0)
    if equity <= 0:
        equity = 10_000.0

    base_risk_pct = float(config.risk_per_trade)
    base_risk_pct = _clamp(base_risk_pct, 0.0001, 0.05)  # between 0.01% and 5% per trade

    # --- Map confidence → risk multiplier -------------------------------------
    if confidence < 0.4:
        # We simply refuse low-conviction directional trades.
        logger.info("Risk engine: confidence %.3f below 0.4; forcing FLAT.", confidence)
        return RiskDecision(
            approved=False,
            side=Side.FLAT.value,
            position_size=0.0,
            leverage=0.0,
            stop_loss_price=None,
            take_profit_price=None,
            reason="low_confidence",
        )
    elif confidence < 0.6:
        conf_mult = 0.5
    elif confidence < 0.8:
        conf_mult = 1.0
    else:
        conf_mult = 1.5  # high conviction, but still capped elsewhere

    # --- Regime-based adjustments (can only reduce risk) ----------------------
    regime_mult = 1.0

    # Timing state
    if timing_state == "avoid":
        logger.info("Risk engine: timing_state=avoid; no new trades.")
        return RiskDecision(
            approved=False,
            side=Side.FLAT.value,
            position_size=0.0,
            leverage=0.0,
            stop_loss_price=None,
            take_profit_price=None,
            reason="timing_avoid",
        )
    elif timing_state == "cautious":
        regime_mult *= 0.5
    elif timing_state == "aggressive":
        regime_mult *= 1.2  # small boost in good windows

    # Danger mode
    if danger_mode:
        # In dangerous environments, we only allow smaller probing size, if at all.
        regime_mult *= 0.35

    # Volatility
    if vol_mode == "low":
        regime_mult *= 0.75
    elif vol_mode == "normal":
        regime_mult *= 1.0
    elif vol_mode == "high":
        regime_mult *= 0.85
    elif vol_mode == "explosive":
        regime_mult *= 0.5

    # Final risk percent for this trade
    effective_risk_pct = base_risk_pct * conf_mult * regime_mult
    effective_risk_pct = _clamp(effective_risk_pct, 0.0, base_risk_pct * 2.0)

    if effective_risk_pct <= 0.0:
        logger.info("Risk engine: effective_risk_pct <= 0; no trade.")
        return RiskDecision(
            approved=False,
            side=Side.FLAT.value,
            position_size=0.0,
            leverage=0.0,
            stop_loss_price=None,
            take_profit_price=None,
            reason="zero_effective_risk",
        )

    # --- Choose a stop distance based on volatility ---------------------------
    if vol_mode == "low":
        stop_pct = 0.0075  # 0.75%
    elif vol_mode == "normal":
        stop_pct = 0.01    # 1%
    elif vol_mode == "high":
        stop_pct = 0.0125  # 1.25%
    elif vol_mode == "explosive":
        stop_pct = 0.015   # 1.5%
    else:
        stop_pct = 0.01

    stop_distance = price * stop_pct
    if stop_distance <= 0:
        stop_distance = price * 0.01

    # --- Compute size from equity + risk -------------------------------------
    risk_notional = equity * effective_risk_pct
    qty = risk_notional / stop_distance if stop_distance > 0 else 0.0
    notional = qty * price

    # --- Decide leverage cap --------------------------------------------------
    # Default leverage cap
    leverage_cap = float(config.max_leverage)

    # Allow rare "10x mode" only in aligned, high-confidence conditions.
    aligned_trend = (
        (raw_action == "long" and trend in ("up", "sideways"))
        or (raw_action == "short" and trend in ("down", "sideways"))
    )

    at_extreme_range = range_position in ("extreme_low", "extreme_high")

    if (
        confidence >= 0.85
        and not danger_mode
        and vol_mode in ("normal", "high")
        and timing_state in ("normal", "aggressive")
        and aligned_trend
        and at_extreme_range
    ):
        leverage_cap = float(getattr(config, "max_leverage_10x_mode", config.max_leverage))

    # Compute implied leverage and clamp.
    raw_leverage = notional / max(equity, 1e-8)
    leverage = _clamp(raw_leverage, 0.0, leverage_cap)

    if qty <= 0 or leverage <= 0:
        logger.info("Risk engine: qty/leverage <= 0; no trade.")
        return RiskDecision(
            approved=False,
            side=Side.FLAT.value,
            position_size=0.0,
            leverage=0.0,
            stop_loss_price=None,
            take_profit_price=None,
            reason="zero_size_or_leverage",
        )

    # --- Build stop-loss / take-profit levels --------------------------------
    if raw_action == "long":
        stop_loss = price - stop_distance
        take_profit = price + 2.0 * stop_distance
        side_str = Side.LONG.value
    else:
        stop_loss = price + stop_distance
        take_profit = price - 2.0 * stop_distance
        side_str = Side.SHORT.value

    logger.info(
        "Risk engine: approving trade side=%s, qty=%.6f, lev=%.2f, risk_pct=%.4f "
        "(conf=%.3f, regime_mult=%.3f, vol=%s, danger=%s, timing=%s).",
        side_str,
        qty,
        leverage,
        effective_risk_pct,
        confidence,
        regime_mult,
        vol_mode,
        danger_mode,
        timing_state,
    )

    return RiskDecision(
        approved=True,
        side=side_str,
        position_size=qty,
        leverage=leverage,
        stop_loss_price=stop_loss,
        take_profit_price=take_profit,
        reason="risk_rules_v1",
    )
