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


def _build_user_message(snapshot: Dict[str, Any]) -> str:
    """User message is just the raw snapshot JSON."""
    return json.dumps(snapshot, separators=(",", ":"), sort_keys=True)


def call_gpt(config: Config, snapshot: Dict[str, Any]) -> GptDecision:
    """Call the GPT model defined in the config.

    DEMO behavior:
    - If OPENAI_API_KEY is missing, return a flat, zero-confidence decision.
    - On any error, return a flat 'error fallback' decision.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.warning("OPENAI_API_KEY missing; returning demo flat decision.")
        return GptDecision(action="flat", confidence=0.0, notes="demo: no GPT key")

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
        return GptDecision(action=action, confidence=confidence, notes=notes)
    except Exception as exc:  # noqa: BLE001
        logger.warning("GPT call failed, using fallback: %s", exc)
        return GptDecision(action="flat", confidence=0.0, notes="error fallback")
