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

    if action == "flat":
        logger.info("Risk: GPT requested FLAT; no trade.")
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
        logger.info("Risk: invalid or zero price; no trade.")
        return RiskDecision(
            approved=False,
            side=Side.FLAT.value,
            position_size=0.0,
            leverage=0.0,
            stop_loss_price=None,
            take_profit_price=None,
            reason="invalid_price",
        )

    # --- Equity / risk budget -------------------------------------------------
    equity = _get_float(state, "equity", 10_000.0)
    if equity <= 0.0:
        logger.info("Risk: non-positive equity; no trade.")
        return RiskDecision(
            approved=False,
            side=Side.FLAT.value,
            position_size=0.0,
            leverage=0.0,
            stop_loss_price=None,
            take_profit_price=None,
            reason="no_equity",
        )

    base_risk_pct = max(config.risk_per_trade, 0.0)
    # Start with GPT confidence scaling risk within [0, base_risk_pct]
    risk_pct = _clamp(base_risk_pct * confidence, 0.0, base_risk_pct)

    # --- Regime modifiers from snapshot --------------------------------------
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

    effective_risk_pct = _clamp(risk_pct * regime_mult, 0.0, base_risk_pct)

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
        return RiskDecision(
            approved=False,
            side=Side.FLAT.value,
            position_size=0.0,
            leverage=0.0,
            stop_loss_price=None,
            take_profit_price=None,
            reason="risk_zero",
        )

    # --- Position sizing ------------------------------------------------------
    stop_pct = _vol_stop_pct(vol_enum)
    stop_pct = max(stop_pct, 0.001)  # floor at 0.1%

    dollar_risk = equity * effective_risk_pct

    # Leverage: 1x → max_leverage based on confidence,
    # with a hard cap from max_leverage_10x_mode.
    max_lev_cfg = max(float(getattr(config, "max_leverage", 1.0)), 1.0)
    max_lev_hard = max(
        float(getattr(config, "max_leverage_10x_mode", max_lev_cfg)),
        max_lev_cfg,
    )

    leverage = 1.0 + (max_lev_cfg - 1.0) * confidence
    leverage = _clamp(leverage, 1.0, max_lev_hard)

    # Units: units = $risk * leverage / (price * stop_pct)
    qty = (dollar_risk * leverage) / (price * stop_pct)
    if qty <= 0.0:
        logger.info("Risk: computed position size <= 0; no trade.")
        return RiskDecision(
            approved=False,
            side=Side.FLAT.value,
            position_size=0.0,
            leverage=0.0,
            stop_loss_price=None,
            take_profit_price=None,
            reason="qty_zero",
        )

    # Global notional cap: never risk more than X% equity notionally.
    max_notional_pct = 0.05
    notional = price * qty
    max_notional = equity * max_notional_pct * leverage
    if notional > max_notional:
        scale = max_notional / notional
        qty *= scale
        logger.info(
            "Risk: scaled down qty due to notional cap (scale=%.3f, notional=%.2f, cap=%.2f).",
            scale,
            notional,
            max_notional,
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

    return RiskDecision(
        approved=True,
        side=side_str,
        position_size=qty,
        leverage=leverage,
        stop_loss_price=stop_loss,
        take_profit_price=take_profit,
        reason="risk_rules_v2",
    )
