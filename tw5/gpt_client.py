# tw5/gpt_client.py

"""
TW-5 GPT client: turns a Tw5Snapshot into an OrderPlan via GPT or a stub.

Public API:

- generate_order_plan_with_gpt(config, snapshot, state=None) -> OrderPlan
- generate_order_plan_stub(snapshot, seed=None) -> OrderPlan

This module is TW-5 specific and does NOT depend on the classic GptDecision
schema or brain.txt; it uses its own prompt and OrderPlan schema.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

from config import Config
from logger_utils import get_logger
from state_memory import record_gpt_error, record_gpt_success
from .schemas import Tw5Snapshot, OrderPlan, OrderLeg, TPLevel
from .stub import generate_tw5_stub_plan

logger = get_logger(__name__)

_TW5_SYSTEM_PROMPT_CACHE: Dict[str, str] = {}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_order_plan_stub(snapshot: Tw5Snapshot, seed: Optional[int] = None) -> OrderPlan:
    """
    Thin wrapper around the TW-5 stub policy.

    Used by engine_tw5 and replay when we don't want to burn GPT tokens.
    """
    return generate_tw5_stub_plan(snapshot, seed=seed)


def generate_order_plan_with_gpt(
    config: Config,
    snapshot: Tw5Snapshot,
    state: Optional[Dict[str, Any]] = None,
) -> OrderPlan:
    """
    Call GPT with the TW-5 prompt and return an OrderPlan.

    Behaviour:
    - Requires OPENAI_API_KEY.
    - Uses TRADEWARRIOR_GPT_MODEL (or config.gpt_model) as the model name.
    - On any error (API key missing, API failure, parse error):
        * logs a warning
        * records a GPT error in state (if provided)
        * returns a flat OrderPlan with an error rationale
    - On success:
        * records GPT success in state (if provided)
        * returns a parsed OrderPlan (may still be flat if GPT chooses so)
    """
    # Basic snapshot sanity: if price is invalid, don't even try GPT.
    if snapshot.price <= 0.0:
        logger.warning("TW-5 GPT client: snapshot price <= 0, returning flat plan.")
        return OrderPlan.empty_flat("invalid_snapshot_price_for_gpt")

    # Get API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or not api_key.strip():
        logger.error("TW-5 GPT client: OPENAI_API_KEY missing or empty. Returning flat plan.")
        if state is not None:
            record_gpt_error(state, time.time())
        return OrderPlan.empty_flat("error: OPENAI_API_KEY missing")

    # Choose model: allow override via env, fallback to config.gpt_model
    model_name = os.getenv("TRADEWARRIOR_GPT_MODEL", config.gpt_model)

    # Determine active profile from config, with env var override
    profile = getattr(config, "tw5_prompt_profile", None)
    if not profile:
        profile = os.getenv("TW5_PROMPT_PROFILE", "conservative")
    # Normalize and coerce unknown values to "conservative"
    profile = profile.lower().strip()
    if profile not in ("conservative", "aggressive"):
        profile = "conservative"

    try:
        import openai  # type: ignore[attr-defined]

        # Legacy v0.x style (openai==0.28.x)
        openai.api_key = api_key

        system_prompt = _load_tw5_system_prompt(profile)
        user_message = _build_user_message(snapshot)

        response = openai.ChatCompletion.create(  # type: ignore[attr-defined]
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            max_tokens=config.gpt_max_tokens,
        )

        content = response["choices"][0]["message"]["content"]
        raw = content.strip()
        logger.debug("TW-5 GPT raw content: %s", raw)

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.warning("TW-5 GPT client: JSON decode error: %s", exc)
            if state is not None:
                record_gpt_error(state, time.time())
            return OrderPlan.empty_flat(f"error: JSON decode failed ({exc})")

        plan = _parse_order_plan_from_dict(data)
        if state is not None:
            record_gpt_success(state)
        return plan

    except Exception as exc:  # noqa: BLE001
        logger.warning("TW-5 GPT client: API call failed, returning flat plan: %s", exc)
        if state is not None:
            record_gpt_error(state, time.time())
        return OrderPlan.empty_flat(f"error: GPT call failed ({exc})")


# ---------------------------------------------------------------------------
# System prompt & user message
# ---------------------------------------------------------------------------


def _load_tw5_system_prompt(profile: str) -> str:
    """
    Load (or lazily cache) the TW-5 system prompt from external brain files.

    Args:
        profile: Prompt profile name ("conservative" or "aggressive").
                 Unknown values are coerced to "conservative".

    Returns:
        The system prompt text for the specified profile.

    The prompt is loaded from tw5/tw5_brain_<profile>.txt relative to this module.
    """
    global _TW5_SYSTEM_PROMPT_CACHE

    # Normalize profile
    profile = profile.lower().strip()
    if profile not in ("conservative", "aggressive"):
        profile = "conservative"

    # Check cache
    if profile in _TW5_SYSTEM_PROMPT_CACHE:
        return _TW5_SYSTEM_PROMPT_CACHE[profile]

    # Build path to brain file
    prompt_filename = f"tw5_brain_{profile}.txt"
    prompt_path = Path(__file__).resolve().parent / prompt_filename

    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            prompt = f.read()
        _TW5_SYSTEM_PROMPT_CACHE[profile] = prompt
        logger.debug("Loaded TW-5 system prompt for profile '%s' from %s", profile, prompt_path)
        return prompt
    except FileNotFoundError:
        logger.error(
            "TW-5 brain file not found: %s. Falling back to conservative.",
            prompt_path,
        )
        # Fallback: try conservative if aggressive was requested but not found
        if profile != "conservative":
            return _load_tw5_system_prompt("conservative")
        # If even conservative is missing, raise
        raise


def _build_user_message(snapshot: Tw5Snapshot) -> str:
    """
    Build the user message for GPT.

    We provide:
    - A short natural-language summary of the snapshot context.
    - The full snapshot JSON (Tw5Snapshot.to_dict()).
    """
    s = snapshot

    # Short human summary
    summary_lines = [
        f"Symbol: {s.symbol}",
        f"Timeframe: {s.timeframe}",
        f"Price: {s.price}",
        f"Trend 1h / 4h: {s.trend_1h} / {s.trend_4h}",
        f"7d range: low={s.range_low_7d}, high={s.range_high_7d}, position={s.range_position_7d}",
        f"Swing: low={s.swing_low}, high={s.swing_high}",
        "Fibs: "
        f"0.382={s.fib_0_382}, 0.5={s.fib_0_5}, 0.618={s.fib_0_618}, 0.786={s.fib_0_786}",
        f"Vol: mode={s.vol_mode}, atr_pct={s.atr_pct}",
        f"Last impulse: dir={s.last_impulse_direction}, size_pct={s.last_impulse_size_pct}",
        f"Sweeps: swept_prev_high={s.swept_prev_high}, swept_prev_low={s.swept_prev_low}",
        f"Position: side={s.position_side}, size={s.position_size}, entry={s.position_entry_price}",
        "",
        "Full snapshot JSON:",
    ]

    summary = "\n".join(summary_lines)
    snapshot_json = json.dumps(s.to_dict(), separators=(",", ":"), sort_keys=True)

    return f"{summary}\n{snapshot_json}"
    

# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _parse_order_plan_from_dict(data: Dict[str, Any]) -> OrderPlan:
    """
    Parse an OrderPlan from GPT JSON.

    This is deliberately conservative:
    - Unknown mode/side values are coerced to "flat".
    - Legs missing required fields are dropped.
    - If no valid legs remain for a non-flat plan, returns a flat plan with error rationale.
    """
    if not isinstance(data, dict):
        logger.warning("TW-5 GPT client: root JSON is not an object; returning flat.")
        return OrderPlan.empty_flat("error: root_not_object")

    mode = str(data.get("mode", "flat")).lower()
    side = str(data.get("side", "flat")).lower()

    if mode not in ("enter", "manage", "flat"):
        mode = "flat"
    if side not in ("long", "short", "flat"):
        side = "flat"

    max_total_size_frac = _safe_float(data.get("max_total_size_frac"), 0.0)
    max_total_size_frac = max(0.0, min(1.0, max_total_size_frac))

    confidence = _safe_float(data.get("confidence"), 0.0)
    confidence = max(0.0, min(1.0, confidence))

    rationale = str(data.get("rationale", "") or "")

    legs_raw = data.get("legs") or []
    legs = []
    if isinstance(legs_raw, list):
        for raw in legs_raw:
            leg = _parse_leg(raw)
            if leg is not None:
                legs.append(leg)

    if (mode != "flat" and side != "flat") and not legs:
        logger.warning("TW-5 GPT client: non-flat plan with no valid legs; forcing flat.")
        return OrderPlan.empty_flat("error: no_valid_legs")

    if mode == "flat" or side == "flat":
        # Normalise to a clean flat plan
        return OrderPlan.empty_flat(rationale or "flat")

    return OrderPlan(
        mode=mode,
        side=side,
        legs=legs,
        max_total_size_frac=max_total_size_frac,
        confidence=confidence,
        rationale=rationale,
    )


def _parse_leg(raw: Any) -> Optional[OrderLeg]:
    if not isinstance(raw, dict):
        return None

    try:
        leg_id = str(raw.get("id", "") or "").strip() or "leg1"
        entry_type = str(raw.get("entry_type", "limit")).lower()
        if entry_type not in ("limit", "market"):
            entry_type = "limit"

        entry_price = _safe_float(raw.get("entry_price"), 0.0)
        entry_tag = str(raw.get("entry_tag", "") or "")

        size_frac = _safe_float(raw.get("size_frac"), 0.0)
        size_frac = max(0.0, min(1.0, size_frac))

        stop_loss = _safe_float(raw.get("stop_loss"), 0.0)

        tp_raw = raw.get("take_profits") or []
        tps = []
        if isinstance(tp_raw, list):
            for t in tp_raw:
                tp = _parse_tp(t)
                if tp is not None:
                    tps.append(tp)

        if entry_price <= 0.0 or stop_loss <= 0.0 or size_frac <= 0.0:
            return None

        return OrderLeg(
            id=leg_id,
            entry_type=entry_type,
            entry_price=entry_price,
            entry_tag=entry_tag,
            size_frac=size_frac,
            stop_loss=stop_loss,
            take_profits=tps,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("TW-5 GPT client: failed to parse leg: %s", exc)
        return None


def _parse_tp(raw: Any) -> Optional[TPLevel]:
    if not isinstance(raw, dict):
        return None
    try:
        price = _safe_float(raw.get("price"), 0.0)
        size_frac = _safe_float(raw.get("size_frac"), 0.0)
        size_frac = max(0.0, min(1.0, size_frac))
        tag = str(raw.get("tag", "") or "")
        if price <= 0.0 or size_frac <= 0.0:
            return None
        return TPLevel(price=price, size_frac=size_frac, tag=tag)
    except Exception as exc:  # noqa: BLE001
        logger.warning("TW-5 GPT client: failed to parse TP: %s", exc)
        return None


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
