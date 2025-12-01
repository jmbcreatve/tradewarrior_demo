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
