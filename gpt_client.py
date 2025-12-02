import json
import os
from typing import Dict, Any

from config import Config
from schemas import GptDecision
from logger_utils import get_logger

logger = get_logger(__name__)

_BRAIN_PROMPT_CACHE: str | None = None


def _load_brain_prompt() -> str:
    """Load the brain.txt prompt from disk, with a safe fallback."""
    global _BRAIN_PROMPT_CACHE
    if _BRAIN_PROMPT_CACHE is not None:
        return _BRAIN_PROMPT_CACHE

    try:
        with open("brain.txt", "r", encoding="utf-8") as f:
            text = f.read().strip()
        if not text:
            raise ValueError("brain.txt was empty")
        _BRAIN_PROMPT_CACHE = text
        return _BRAIN_PROMPT_CACHE
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to load brain.txt, using minimal inline instructions: %s", exc)
        _BRAIN_PROMPT_CACHE = (
            "You are the reasoning layer of an automated trading system. "
            "You will receive a JSON snapshot of market state and must respond ONLY with JSON "
            "containing keys: action, confidence, notes."
        )
        return _BRAIN_PROMPT_CACHE


def _safe_float(value: Any, default: float) -> float:
    """Coerce a value to float with a safe default."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _build_user_message(snapshot: Dict[str, Any]) -> str:
    """Build a human-readable risk envelope summary plus the full snapshot JSON."""
    risk_env = snapshot.get("risk_envelope") or {}

    max_notional = _safe_float(risk_env.get("max_notional"), 0.0)
    max_leverage = _safe_float(risk_env.get("max_leverage"), 0.0)
    max_risk_per_trade_pct = _safe_float(risk_env.get("max_risk_per_trade_pct"), 0.0)
    min_stop = _safe_float(risk_env.get("min_stop_distance_pct"), 0.0)
    max_stop = _safe_float(risk_env.get("max_stop_distance_pct"), 0.0)
    max_daily_loss_pct = _safe_float(risk_env.get("max_daily_loss_pct"), 0.0)
    note = str(risk_env.get("note", "n/a"))

    summary = (
        "RISK ENVELOPE (limits you MUST respect):\n"
        f"- max_notional: {max_notional}\n"
        f"- max_leverage: {max_leverage}\n"
        f"- max_risk_per_trade_pct: {max_risk_per_trade_pct}\n"
        f"- min_stop_distance_pct: {min_stop}\n"
        f"- max_stop_distance_pct: {max_stop}\n"
        f"- max_daily_loss_pct: {max_daily_loss_pct}\n"
        f"- note: {note}\n\n"
        "SNAPSHOT JSON:\n"
    )

    return summary + json.dumps(snapshot, sort_keys=True)


def call_gpt(config: Config, snapshot: Dict[str, Any]) -> GptDecision:
    """Call the GPT model defined in the config.

    DEMO behavior:
    - If OPENAI_API_KEY is missing, return a flat, zero-confidence decision.
    - On any error, return a flat 'error fallback' decision.
    """
    def _log_decision_event(decision: GptDecision) -> None:
        try:
            try:
                symbol = snapshot.get("symbol") if isinstance(snapshot, dict) else None
                timestamp = snapshot.get("timestamp") if isinstance(snapshot, dict) else None
                snapshot_id = snapshot.get("snapshot_id") if isinstance(snapshot, dict) else None
            except Exception:
                symbol = None
                timestamp = None
                snapshot_id = None

            event = {
                "type": "gpt_decision",
                "symbol": symbol,
                "timestamp": timestamp,
                "snapshot_id": snapshot_id,
                "side": getattr(decision, "action", None),
                "confidence": getattr(decision, "confidence", None),
                "reason": getattr(decision, "notes", None),
            }
            logger.info("GPT decision: %s", event)
        except Exception:
            logger.exception("Failed to log GPT decision event", exc_info=True)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.warning("OPENAI_API_KEY missing; returning demo flat decision.")
        decision = GptDecision(action="flat", confidence=0.0, notes="demo: no GPT key")
        _log_decision_event(decision)
        return decision

    # Allow overriding the model via env var without changing code.
    model_name = os.getenv("TRADEWARRIOR_GPT_MODEL", config.gpt_model)

    try:
        import openai  # type: ignore[attr-defined]

        openai.api_key = api_key

        system_prompt = _load_brain_prompt()
        user_message = _build_user_message(snapshot)

        response = openai.ChatCompletion.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            max_tokens=config.gpt_max_tokens,
        )

        content = response["choices"][0]["message"]["content"]
        data = json.loads(content)

        action = str(data.get("action", "flat")).lower()
        confidence = float(data.get("confidence", 0.0))
        notes = str(data.get("notes", ""))

        if action not in ("long", "short", "flat"):
            action = "flat"

        confidence = max(0.0, min(1.0, confidence))

        logger.info("GPT model=%s action=%s confidence=%.3f", model_name, action, confidence)
        decision = GptDecision(action=action, confidence=confidence, notes=notes)
        _log_decision_event(decision)
        return decision
    except Exception as exc:  # noqa: BLE001
        logger.warning("GPT call failed, using fallback: %s", exc)
        decision = GptDecision(action="flat", confidence=0.0, notes="error fallback")
        _log_decision_event(decision)
        return decision
