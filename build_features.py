from __future__ import annotations

from typing import Dict, Any, List, Optional

from config import Config
from enums import (
    Trend,
    RangePosition,
    VolatilityMode,
    SkewBias,
    TimingState,
    coerce_enum,
    enum_to_str,
)
from schemas import MarketSnapshot, validate_snapshot_dict
from risk_envelope import compute_risk_envelope, RiskEnvelope
from shapes_module import detect_shapes
from logger_utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Small numeric helpers
# ---------------------------------------------------------------------------

def _safe_last_candle(candles: List[Dict[str, float]]) -> Optional[Dict[str, float]]:
    if not candles:
        return None
    return candles[-1]


def _get_closes(candles: List[Dict[str, float]]) -> List[float]:
    closes: List[float] = []
    for c in candles:
        try:
            closes.append(float(c["close"]))
        except (KeyError, TypeError, ValueError):
            # If we hit a bad candle, just reuse the last close or 0.0
            closes.append(closes[-1] if closes else 0.0)
    return closes


def _safe_div(num: float, denom: float) -> float:
    if denom == 0:
        return 0.0
    return num / denom


# ---------------------------------------------------------------------------
# Regime / structure computation
# ---------------------------------------------------------------------------

def _compute_trend(candles: List[Dict[str, float]]) -> str:
    """Very simple trend classifier based on start vs end close."""
    if len(candles) < 2:
        return enum_to_str(Trend.UNKNOWN)

    closes = _get_closes(candles)
    first = closes[0]
    last = closes[-1]
    change = _safe_div(last - first, abs(first)) if first else 0.0

    if change > 0.005:
        return enum_to_str(Trend.UP)
    if change < -0.005:
        return enum_to_str(Trend.DOWN)
    return enum_to_str(Trend.SIDEWAYS)


def _compute_range_position(candles: List[Dict[str, float]], price: float) -> str:
    """Where current price sits inside recent high/low range."""
    if not candles:
        return enum_to_str(RangePosition.MID)

    closes = _get_closes(candles)
    low = min(closes)
    high = max(closes)

    if high == low:
        return enum_to_str(RangePosition.MID)

    pct = _safe_div(price - low, high - low)

    if pct <= 0.1:
        return enum_to_str(RangePosition.EXTREME_LOW)
    if pct <= 0.3:
        return enum_to_str(RangePosition.LOW)
    if pct >= 0.9:
        return enum_to_str(RangePosition.EXTREME_HIGH)
    if pct >= 0.7:
        return enum_to_str(RangePosition.HIGH)
    return enum_to_str(RangePosition.MID)


def _compute_volatility_mode(candles: List[Dict[str, float]]) -> str:
    """Simple realized-vol regime based on absolute bar-to-bar returns."""
    if len(candles) < 2:
        return enum_to_str(VolatilityMode.UNKNOWN)

    closes = _get_closes(candles)
    returns: List[float] = []
    for i in range(1, len(closes)):
        prev = closes[i - 1]
        cur = closes[i]
        if prev:
            returns.append(abs(cur - prev) / abs(prev))

    if not returns:
        return enum_to_str(VolatilityMode.UNKNOWN)

    avg_ret = sum(returns) / len(returns)

    if avg_ret < 0.001:
        return enum_to_str(VolatilityMode.LOW)
    if avg_ret < 0.003:
        return enum_to_str(VolatilityMode.NORMAL)
    if avg_ret < 0.007:
        return enum_to_str(VolatilityMode.HIGH)
    return enum_to_str(VolatilityMode.EXPLOSIVE)


def _compute_recent_price_path(candles: List[Dict[str, float]]) -> Dict[str, Any]:
    """Short-horizon returns + crude impulse labeling."""
    result: Dict[str, Any] = {
        "lookback_bars": len(candles),
        "ret_1": 0.0,
        "ret_5": 0.0,
        "ret_15": 0.0,
        "impulse_state": "unknown",
    }
    if not candles:
        return result

    closes = _get_closes(candles)
    last = closes[-1]

    def _ret(n: int) -> float:
        if len(closes) <= n:
            return 0.0
        base = closes[-1 - n]
        if not base:
            return 0.0
        return _safe_div(last - base, abs(base))

    result["ret_1"] = _ret(1)
    result["ret_5"] = _ret(5)
    result["ret_15"] = _ret(15)

    # Toy impulse classifier based on 5-bar return
    r5 = result["ret_5"]
    if r5 > 0.01:
        result["impulse_state"] = "ripping_up"
    elif r5 < -0.01:
        result["impulse_state"] = "ripping_down"
    elif r5 > 0.003:
        result["impulse_state"] = "grinding_up"
    elif r5 < -0.003:
        result["impulse_state"] = "grinding_down"
    else:
        result["impulse_state"] = "chop"

    return result


# ---------------------------------------------------------------------------
# Context layers: flow, liquidity, fib, timing, risk
# ---------------------------------------------------------------------------

def _build_flow_layer(market_data: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize basic flow metrics into a single dict."""
    flow: Dict[str, Any] = {
        "funding": market_data.get("funding"),
        "open_interest": market_data.get("open_interest"),
        "skew": market_data.get("skew"),
    }

    # Optional: very rough skew bias classification here, but the canonical skew_bias
    # is still derived in validate_snapshot_dict via SkewBias.
    skew_val = flow["skew"]
    try:
        skew_f = float(skew_val) if skew_val is not None else 0.0
    except (TypeError, ValueError):
        skew_f = 0.0

    if skew_f > 0.5:
        flow["skew_bias_hint"] = enum_to_str(SkewBias.CALL_DOMINANT)
    elif skew_f < -0.5:
        flow["skew_bias_hint"] = enum_to_str(SkewBias.PUT_DOMINANT)
    else:
        flow["skew_bias_hint"] = enum_to_str(SkewBias.NEUTRAL)

    return flow


def _find_fractal_swing_points(
    candles: List[Dict[str, float]],
    lookback: int = 2,
) -> tuple[List[float], List[float]]:
    """Identify fractal swing highs and lows from OHLC candles.

    A fractal high: a candle whose high is greater than `lookback` candles on each side.
    A fractal low: a candle whose low is less than `lookback` candles on each side.

    Returns (swing_highs, swing_lows) as lists of price levels.
    """
    swing_highs: List[float] = []
    swing_lows: List[float] = []

    n = len(candles)
    if n < 2 * lookback + 1:
        return swing_highs, swing_lows

    for i in range(lookback, n - lookback):
        try:
            h_mid = float(candles[i].get("high", 0.0))
            l_mid = float(candles[i].get("low", 0.0))
        except (TypeError, ValueError):
            continue

        is_swing_high = True
        is_swing_low = True

        for j in range(1, lookback + 1):
            try:
                h_left = float(candles[i - j].get("high", 0.0))
                h_right = float(candles[i + j].get("high", 0.0))
                l_left = float(candles[i - j].get("low", 0.0))
                l_right = float(candles[i + j].get("low", 0.0))
            except (TypeError, ValueError):
                is_swing_high = False
                is_swing_low = False
                break

            if h_mid <= h_left or h_mid <= h_right:
                is_swing_high = False
            if l_mid >= l_left or l_mid >= l_right:
                is_swing_low = False

        if is_swing_high:
            swing_highs.append(h_mid)
        if is_swing_low:
            swing_lows.append(l_mid)

    return swing_highs, swing_lows


def _build_liquidity_context(
    candles: List[Dict[str, float]],
    shapes: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build liquidity context using swing highs/lows from recent candles.

    Computes:
    - liquidity_above: nearest swing high above current price (potential sell-side liquidity)
    - liquidity_below: nearest swing low below current price (potential buy-side liquidity)

    Optionally refines using shapes_module outputs (equal-high/low clusters).
    """
    result: Dict[str, Any] = {
        "liquidity_above": None,
        "liquidity_below": None,
    }

    if not candles:
        return result

    # Get current price from last candle
    try:
        current_price = float(candles[-1].get("close", 0.0))
    except (TypeError, ValueError):
        current_price = 0.0

    if current_price <= 0:
        return result

    # Find fractal swing points
    swing_highs, swing_lows = _find_fractal_swing_points(candles, lookback=2)

    # Also collect recent highs/lows as candidate levels even without fractals
    # This ensures we have levels even in short windows
    recent_highs: List[float] = []
    recent_lows: List[float] = []
    for c in candles[-20:]:  # last 20 candles for recency
        try:
            h = float(c.get("high", 0.0))
            l = float(c.get("low", 0.0))
            if h > 0:
                recent_highs.append(h)
            if l > 0:
                recent_lows.append(l)
        except (TypeError, ValueError):
            continue

    # Combine swing levels with recent extreme levels
    all_high_levels = list(set(swing_highs))
    all_low_levels = list(set(swing_lows))

    # If shapes indicate equal-high/low clusters, add the most recent cluster price
    if shapes:
        # eq_high_cluster means there's a cluster of equal highs recently
        # We add the recent max high as a cluster level
        if shapes.get("eq_high_cluster") and recent_highs:
            cluster_high = max(recent_highs[-5:]) if len(recent_highs) >= 5 else max(recent_highs)
            if cluster_high not in all_high_levels:
                all_high_levels.append(cluster_high)

        if shapes.get("eq_low_cluster") and recent_lows:
            cluster_low = min(recent_lows[-5:]) if len(recent_lows) >= 5 else min(recent_lows)
            if cluster_low not in all_low_levels:
                all_low_levels.append(cluster_low)

        # If there was a sweep, the sweep level is significant liquidity
        # sweep_up means we took out a high but closed back below - that high is key
        if shapes.get("sweep_up") and recent_highs:
            sweep_level = max(recent_highs)
            if sweep_level not in all_high_levels:
                all_high_levels.append(sweep_level)

        if shapes.get("sweep_down") and recent_lows:
            sweep_level = min(recent_lows)
            if sweep_level not in all_low_levels:
                all_low_levels.append(sweep_level)

    # Find nearest level above current price
    levels_above = [h for h in all_high_levels if h > current_price]
    if levels_above:
        result["liquidity_above"] = min(levels_above)  # nearest above

    # Find nearest level below current price
    levels_below = [l for l in all_low_levels if l < current_price]
    if levels_below:
        result["liquidity_below"] = max(levels_below)  # nearest below

    return result


def _build_fib_context(candles: List[Dict[str, float]]) -> Dict[str, Any]:
    """Placeholder macro/micro fib zones.

    Later this will look at higher timeframe structure; for now, we just
    distinguish 'macro_discount' / 'macro_premium' around the mid of the range.
    """
    if not candles:
        return {
            "macro_zone": "unknown",
            "micro_zone": "unknown",
        }

    closes = _get_closes(candles)
    low = min(closes)
    high = max(closes)
    mid = (low + high) / 2 if (high != low) else low
    last = closes[-1]

    if last <= mid:
        macro = "macro_discount"
    else:
        macro = "macro_premium"

    # Micro is intentionally dumb for now.
    micro = "micro_discount" if last <= mid else "micro_premium"

    return {
        "macro_zone": macro,
        "micro_zone": micro,
    }


def _compute_danger_mode(
    vol_mode: str,
    recent_path: Dict[str, Any],
) -> bool:
    """Danger flag used by gatekeeper + risk.

    For now:
    - EXPLOSIVE vol or 'ripping' impulses set danger_mode True.
    """
    if vol_mode == enum_to_str(VolatilityMode.EXPLOSIVE):
        return True

    impulse = str(recent_path.get("impulse_state", "unknown"))
    if impulse in ("ripping_up", "ripping_down"):
        return True

    return False


def _classify_market_session(timestamp: float) -> str:
    """Classify market session based on UTC timestamp.
    
    Returns: "ASIA", "EUROPE", "US", or "OFF_HOURS"
    
    Session hours (UTC):
    - ASIA: 00:00-08:00 UTC (Tokyo session)
    - EUROPE: 07:00-16:00 UTC (London session, overlaps with ASIA)
    - US: 13:00-22:00 UTC (New York session, overlaps with EUROPE)
    - OFF_HOURS: everything else
    
    When sessions overlap, we prioritize: US > EUROPE > ASIA
    """
    if timestamp <= 0:
        return "OFF_HOURS"
    
    try:
        from datetime import datetime, timezone
        dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
        hour = dt.hour
    except (ValueError, OSError, OverflowError):
        return "OFF_HOURS"
    
    # US session: 13:00-22:00 UTC (highest priority)
    if 13 <= hour < 22:
        return "US"
    
    # EUROPE session: 07:00-16:00 UTC
    if 7 <= hour < 16:
        return "EUROPE"
    
    # ASIA session: 00:00-08:00 UTC
    if 0 <= hour < 8:
        return "ASIA"
    
    # OFF_HOURS: 22:00-00:00 UTC (after US close, before ASIA open)
    return "OFF_HOURS"


def _compute_timing_state(timestamp: float) -> tuple[str, str]:
    """Compute timing state based on market session.
    
    Returns: (market_session, timing_state_enum_str)
    - market_session: "ASIA", "EUROPE", "US", or "OFF_HOURS"
    - timing_state_enum_str: enum string for TimingState
    
    Conservative mapping:
    - OFF_HOURS -> AVOID (low liquidity)
    - ASIA -> CAUTIOUS (lower liquidity than US/EU)
    - EUROPE -> NORMAL (good liquidity)
    - US -> NORMAL (best liquidity)
    """
    session = _classify_market_session(timestamp)
    
    # Map session to TimingState enum with conservative rules
    if session == "OFF_HOURS":
        timing_enum = TimingState.AVOID
    elif session == "ASIA":
        timing_enum = TimingState.CAUTIOUS
    elif session == "EUROPE":
        timing_enum = TimingState.NORMAL
    elif session == "US":
        timing_enum = TimingState.NORMAL
    else:
        timing_enum = TimingState.UNKNOWN
    
    return session, enum_to_str(timing_enum)


def _build_risk_context(state: Dict[str, Any], config: Config | None = None) -> Dict[str, Any]:
    """
    Compress account/position state into a small dict.
    
    Uses equity from state if available, otherwise falls back to config.initial_equity
    (if config provided) or a safe default. This ensures risk_context reflects the
    current equity correctly and does not silently assume a hardcoded starting equity.
    """
    risk: Dict[str, Any] = {}

    # Determine fallback equity: prefer config.initial_equity, then safe default
    default_equity = 10_000.0
    if config is not None:
        default_equity = getattr(config, "initial_equity", default_equity)

    try:
        risk["equity"] = float(state.get("equity", default_equity) or default_equity)
    except (TypeError, ValueError):
        risk["equity"] = default_equity

    try:
        risk["max_drawdown"] = float(state.get("max_drawdown", 0.0) or 0.0)
    except (TypeError, ValueError):
        risk["max_drawdown"] = 0.0

    ops = state.get("open_positions_summary", [])
    risk["open_positions_summary"] = ops if isinstance(ops, list) else []

    # Derive last action/confidence from the last execution/risk outcome (not raw GPT)
    try:
        last_conf = float(state.get("last_confidence", 0.0) or 0.0)
    except (TypeError, ValueError):
        last_conf = 0.0

    last_action_state = state.get("last_action")
    last_action = str(last_action_state) if isinstance(last_action_state, str) else ""

    # Fallback to last execution result if state.last_action is missing
    if not last_action:
        last_exec = state.get("last_execution_result") or {}
        if isinstance(last_exec, dict):
            status = str(last_exec.get("status", "")).lower()
            side = last_exec.get("side", "")
            if side and status not in {"no_trade", "skipped", "gatekeeper_skipped", "safe_mode_flat", "dry_run"}:
                last_action = str(side)

    # Fallback to last risk decision if still empty
    if not last_action:
        last_risk_decision = state.get("last_risk_decision") or {}
        if isinstance(last_risk_decision, dict):
            side = last_risk_decision.get("side", "")
            if side and last_risk_decision.get("approved", False):
                last_action = str(side)

    if not last_action:
        last_action = "flat"

    risk["last_action"] = last_action
    risk["last_confidence"] = last_conf

    return risk


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_snapshot(config: Config, market_data: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a snapshot dict from raw market data + persistent state.

    Contract:
    - Returns a JSON-safe dict that matches the shape described in brain.txt
      and schemas.validate_snapshot_dict.
    - The rest of the pipeline (gatekeeper, GPT, risk, execution) expects a
      plain dict and uses snapshot.get(...).
    """
    candles: List[Dict[str, float]] = list(market_data.get("candles") or [])
    last_candle = _safe_last_candle(candles)

    if last_candle is None:
        price = 0.0
        ts = 0.0
    else:
        try:
            price = float(last_candle["close"])
        except (KeyError, TypeError, ValueError):
            price = 0.0
        try:
            ts = float(last_candle.get("timestamp", 0.0) or 0.0)
        except (TypeError, ValueError):
            ts = 0.0

    trend = _compute_trend(candles)
    range_pos = _compute_range_position(candles, price)
    vol_mode = _compute_volatility_mode(candles)
    recent_path = _compute_recent_price_path(candles)

    flow = _build_flow_layer(market_data)
    shapes = detect_shapes(candles)
    liquidity_context = _build_liquidity_context(candles, shapes=shapes)
    fib_context = _build_fib_context(candles)

    danger_mode = _compute_danger_mode(vol_mode, recent_path)
    market_session, timing_state = _compute_timing_state(ts)

    risk_context = _build_risk_context(state, config)
    try:
        equity = float(risk_context.get("equity", 0.0) or 0.0)
    except (TypeError, ValueError):
        equity = 0.0

    last_gpt_ts = state.get("last_gpt_call_ts")
    last_gpt_equity = state.get("last_gpt_equity")
    last_gpt_snapshot = state.get("last_gpt_snapshot") or {}
    prev_price = last_gpt_snapshot.get("price")

    try:
        now_ts = float(ts)
    except (TypeError, ValueError):
        now_ts = 0.0

    if last_gpt_ts is None:
        time_since = 0.0
    else:
        try:
            time_since = max(0.0, now_ts - float(last_gpt_ts))
        except (TypeError, ValueError):
            time_since = 0.0

    if prev_price not in (None, 0):
        try:
            price_change_pct = (price - float(prev_price)) / float(prev_price)
        except (TypeError, ValueError, ZeroDivisionError):
            price_change_pct = 0.0
    else:
        price_change_pct = 0.0

    try:
        equity_change = float(equity) - float(last_gpt_equity or 0.0)
    except (TypeError, ValueError):
        equity_change = 0.0

    trades_since = int(state.get("trades_since_last_gpt", 0) or 0)

    since_last_gpt = {
        "time_since_last_gpt_sec": time_since,
        "price_change_pct_since_last_gpt": price_change_pct,
        "equity_change_since_last_gpt": equity_change,
        "trades_since_last_gpt": trades_since,
    }

    vol_enum = coerce_enum(vol_mode, VolatilityMode, VolatilityMode.UNKNOWN)
    timing_enum = coerce_enum(timing_state, TimingState, TimingState.UNKNOWN)

    risk_envelope: RiskEnvelope = compute_risk_envelope(
        config=config,
        equity=equity,
        volatility_mode=vol_enum,
        danger_mode=danger_mode,
        timing_state=timing_enum,
    )

    gpt_state_note = state.get("gpt_state_note")
    run_id = state.get("run_id")
    snapshot_id = int(state.get("snapshot_id", 0) or 0)
    if snapshot_id <= 0:
        snapshot_id = 0

    snap = MarketSnapshot(
        symbol=str(state.get("symbol", config.symbol)),
        timestamp=ts,
        price=price,
        trend=trend,
        range_position=range_pos,
        volatility_mode=vol_mode,
        flow=flow,
        microstructure=shapes,
        liquidity_context=liquidity_context,
        fib_context=fib_context,
        danger_mode=danger_mode,
        timing_state=timing_state,
        market_session=market_session,
        recent_price_path=recent_path,
        risk_context=risk_context,
        risk_envelope=risk_envelope.to_dict(),
        gpt_state_note=gpt_state_note,
        since_last_gpt=since_last_gpt,
    )

    # Important: callers expect a dict, not the dataclass, so we normalise via
    # to_dict() + validate_snapshot_dict().
    snap_dict = snap.to_dict()
    snap_dict["market_session"] = market_session  # Add raw session label
    snap_dict["risk_envelope"] = risk_envelope.to_dict()
    snap_dict["since_last_gpt"] = since_last_gpt
    snap_dict = validate_snapshot_dict(snap_dict)
    snap_dict["run_id"] = run_id
    snap_dict["snapshot_id"] = snapshot_id
    return snap_dict
