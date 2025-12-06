from __future__ import annotations

from typing import Any, Dict

from schemas import GptDecision


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def generate_stub_decision(snapshot: Dict[str, Any]) -> GptDecision:
    """
    Deterministic GPT replacement used only during replay/backtests.

    Rules:
    - bull shape_bias + up trend -> long
    - bear shape_bias + down trend -> short
    - otherwise flat
    - confidence = 0.3 + shape_score * 0.7 (clamped to [0, 1])
    """
    micro = snapshot.get("microstructure") or {}
    shape_bias = str(micro.get("shape_bias", "none")).lower()
    shape_score = _safe_float(micro.get("shape_score", 0.0), 0.0)
    trend = str(snapshot.get("trend", "unknown")).lower()

    action = "flat"
    if shape_bias == "bull" and trend == "up":
        action = "long"
    elif shape_bias == "bear" and trend == "down":
        action = "short"

    confidence = max(0.0, min(1.0, 0.3 + (shape_score * 0.7)))
    notes = f"replay stub: bias={shape_bias}, trend={trend}, shape_score={shape_score:.3f}"

    return GptDecision(action=action, confidence=confidence, notes=notes)


def generate_stub_decision_v2(snapshot: Dict[str, Any]) -> GptDecision:
    """
    Replay-only stub v2 with basic trend/vol/range awareness.

    Regimes:
    - trend_regime: bull if ret_360 > +2%, bear if ret_360 < -2%, else none.
    - vol_regime: low if volatility_mode == LOW, else normal/high.
    - If trend is none and vol is low -> stay flat.
    - Bull trend + price in lower/mid half of range -> long bias (discourage shorts).
    - Bear trend + price in upper/mid half of range -> short bias (discourage longs).
    Confidence still tied to shape_score.
    """
    rpp = snapshot.get("recent_price_path") or {}
    ret_360 = _safe_float(rpp.get("ret_360"), 0.0)
    vol_mode = str(snapshot.get("volatility_mode", "unknown")).lower()
    range_pos = str(snapshot.get("range_position", "mid")).lower()
    micro = snapshot.get("microstructure") or {}
    shape_score = _safe_float(micro.get("shape_score", 0.0), 0.0)

    if ret_360 > 0.02:
        trend_regime = "bull"
    elif ret_360 < -0.02:
        trend_regime = "bear"
    else:
        trend_regime = "none"

    vol_regime = "low" if vol_mode == "low" else "normal"

    lower_half = {"extreme_low", "low", "mid"}
    upper_half = {"extreme_high", "high", "mid"}

    action = "flat"
    if trend_regime == "none" and vol_regime == "low":
        action = "flat"
    elif trend_regime == "bull" and range_pos in lower_half:
        action = "long"
    elif trend_regime == "bear" and range_pos in upper_half:
        action = "short"

    confidence = max(0.0, min(1.0, 0.3 + (shape_score * 0.7)))
    notes = (
        f"replay stub v2: trend={trend_regime}, vol={vol_regime}, "
        f"range={range_pos}, ret360={ret_360:.4f}, shape_score={shape_score:.3f}"
    )
    return GptDecision(action=action, confidence=confidence, notes=notes)
