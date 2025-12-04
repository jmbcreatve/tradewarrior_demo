import json
import os
import time
from typing import Dict, Any, Optional

from config import Config
from schemas import GptDecision
from logger_utils import get_logger

logger = get_logger(__name__)

_BRAIN_PROMPT_CACHE: str | None = None
_API_KEY_VALIDATED: bool = False


# ---------------------------------------------------------------------------
# Safe Mode Decision Factory
# ---------------------------------------------------------------------------

def create_safe_mode_decision(reason: str = "gpt_safe_mode_active") -> GptDecision:
    """
    Create a flat GptDecision when GPT safe mode is active.
    
    This is the canonical "do nothing" decision used when:
    - GPT safe mode is active
    - GPT calls are skipped due to repeated failures
    
    Args:
        reason: The reason string to include in notes.
        
    Returns:
        A GptDecision with action="flat", confidence=0.0.
    """
    return GptDecision(
        action="flat",
        confidence=0.0,
        notes=reason,
    )


def validate_openai_api_key() -> str:
    """
    Validate that OPENAI_API_KEY is set and non-empty.
    
    Returns:
        The API key string (never log this value).
        
    Raises:
        RuntimeError: If OPENAI_API_KEY is missing or empty.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or not api_key.strip():
        raise RuntimeError(
            "OPENAI_API_KEY environment variable is missing or empty. "
            "Please set it in your environment or local_env.txt file. "
            "GPT calls require a valid OpenAI API key."
        )
    
    # Log that we found a key (without exposing it)
    key_preview = f"{api_key[:7]}...{api_key[-4:]}" if len(api_key) > 11 else "***"
    logger.info("GPT client: OPENAI_API_KEY found (preview: %s)", key_preview)
    
    return api_key


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
            "containing keys: action, size, confidence, rationale."
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


# GPT output schema expected from brain.txt:
# - action: "long", "short", or "flat" (lowercase)
# - size: float 0.0-1.0 representing relative trade size
# - confidence: float 0.0-1.0 indicating conviction
# - rationale: short human-readable reasoning for the choice
def call_gpt(
    config: Config,
    snapshot: Dict[str, Any],
    state: Optional[Dict[str, Any]] = None,
) -> GptDecision:
    """Call the GPT model defined in the config.
    
    Requires OPENAI_API_KEY to be set (validated at startup).
    On API errors, returns a flat 'error fallback' decision.
    
    If state dict is provided, GPT errors/successes will be tracked for safe mode:
    - Errors increment gpt_error_count and may trigger gpt_safe_mode
    - Successes reset gpt_error_count (but don't clear safe mode)
    
    Args:
        config: Runtime configuration.
        snapshot: Market snapshot to send to GPT.
        state: Optional state dict for error tracking. If None, errors are not tracked.
        
    Returns:
        GptDecision with action, confidence, and notes.
    """
    # Import here to avoid circular imports
    from state_memory import record_gpt_error, record_gpt_success
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

    # Get API key (should have been validated at startup, but check again for safety)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or not api_key.strip():
        logger.error("OPENAI_API_KEY missing during GPT call. This should have been caught at startup.")
        decision = GptDecision(action="flat", confidence=0.0, notes="error: API key missing")
        _log_decision_event(decision)
        # Track error if state provided
        if state is not None:
            record_gpt_error(state, time.time())
        return decision

    # Allow overriding the model via env var without changing code.
    model_name = os.getenv("TRADEWARRIOR_GPT_MODEL", config.gpt_model)

    try:
        import openai  # type: ignore[attr-defined]

        # Set API key (using legacy v0.x API pattern for openai==0.28.1)
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
        # Parse "rationale" from GPT response (as per brain.txt), with fallback to "notes" for backward compatibility
        notes = str(data.get("rationale", data.get("notes", "")))

        if action not in ("long", "short", "flat"):
            action = "flat"

        confidence = max(0.0, min(1.0, confidence))

        logger.info("GPT model=%s action=%s confidence=%.3f", model_name, action, confidence)
        decision = GptDecision(action=action, confidence=confidence, notes=notes)
        _log_decision_event(decision)
        # Track success if state provided
        if state is not None:
            record_gpt_success(state)
        return decision
    except Exception as exc:  # noqa: BLE001
        logger.warning("GPT call failed, using fallback: %s", exc)
        decision = GptDecision(action="flat", confidence=0.0, notes=f"error fallback: {exc}")
        _log_decision_event(decision)
        # Track error if state provided
        if state is not None:
            record_gpt_error(state, time.time())
        return decision
