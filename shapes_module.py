from __future__ import annotations

from typing import Any, Dict, List


def _coerce_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _get_ohlc(candle: Dict[str, Any]) -> Dict[str, float]:
    """Return a safe OHLC view for a single candle."""
    return {
        "open": _coerce_float(candle.get("open")),
        "high": _coerce_float(candle.get("high")),
        "low": _coerce_float(candle.get("low")),
        "close": _coerce_float(candle.get("close")),
    }


def detect_shapes(candles: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Heuristic ICT-ish shape detection for DEMO.

    This is intentionally simple and robust:
    - Works with any list of dict-like OHLC candles.
    - Never raises; always returns the same schema.
    - Outputs a small set of microstructure booleans + a summary bias/score.

    Returned keys (all always present):
      fvg_up / fvg_down
      fract_high / fract_low
      sweep_up / sweep_down
      bos_direction / choch_direction
      eq_high_cluster / eq_low_cluster
      displacement_up / displacement_down
      order_block_bias
      compression_active
      wick_exhaustion_up / wick_exhaustion_down
      shape_score (0.0–1.0)
      shape_bias ("bull" | "bear" | "none")
    """
    n = len(candles)
    if n == 0:
        return {
            "fvg_up": False,
            "fvg_down": False,
            "fract_high": False,
            "fract_low": False,
            "sweep_up": False,
            "sweep_down": False,
            "bos_direction": "none",
            "choch_direction": "none",
            "eq_high_cluster": False,
            "eq_low_cluster": False,
            "displacement_up": False,
            "displacement_down": False,
            "order_block_bias": "none",
            "compression_active": False,
            "wick_exhaustion_up": False,
            "wick_exhaustion_down": False,
            "shape_score": 0.0,
            "shape_bias": "none",
        }

    # Build a cleaned OHLC list for the last ~50 bars.
    lookback = min(n, 50)
    recent_raw = candles[-lookback:]
    ohlc = [_get_ohlc(c) for c in recent_raw]

    last = ohlc[-1]
    prev = ohlc[-2] if len(ohlc) > 1 else last
    prev2 = ohlc[-3] if len(ohlc) > 2 else prev

    # ------------------------------------------------------------------
    # Displacement: is the last bar a strong directional bar?
    # ------------------------------------------------------------------
    body = abs(last["close"] - last["open"])
    full = max(last["high"] - last["low"], 1e-8)
    body_frac = body / full

    rel_move = 0.0
    if prev["close"]:
        rel_move = abs(last["close"] - prev["close"]) / max(abs(prev["close"]), 1e-8)

    displacement_threshold = 0.005  # 0.5%

    up_move = last["close"] > prev["close"]
    down_move = last["close"] < prev["close"]

    displacement_up = up_move and rel_move > displacement_threshold and body_frac > 0.5
    displacement_down = down_move and rel_move > displacement_threshold and body_frac > 0.5

    # ------------------------------------------------------------------
    # Simple fractal high/low using only the last few bars.
    # ------------------------------------------------------------------
    fract_high = False
    fract_low = False
    if len(ohlc) >= 3:
        h1 = ohlc[-3]["high"]
        h2 = ohlc[-2]["high"]
        h3 = ohlc[-1]["high"]
        l1 = ohlc[-3]["low"]
        l2 = ohlc[-2]["low"]
        l3 = ohlc[-1]["low"]

        fract_high = h2 > h1 and h2 > h3
        fract_low = l2 < l1 and l2 < l3

    # ------------------------------------------------------------------
    # Sweeps: did we take out prior highs/lows and close back inside?
    # ------------------------------------------------------------------
    highs = [c["high"] for c in ohlc[:-1]] or [last["high"]]
    lows = [c["low"] for c in ohlc[:-1]] or [last["low"]]

    prior_max = max(highs)
    prior_min = min(lows)

    sweep_up = last["high"] > prior_max and last["close"] < prior_max
    sweep_down = last["low"] < prior_min and last["close"] > prior_min

    # ------------------------------------------------------------------
    # Equal highs/lows cluster: any tight cluster in the last few bars?
    # ------------------------------------------------------------------
    def _has_eq_cluster(values: List[float], tol: float = 0.0005) -> bool:
        if len(values) < 2:
            return False
        base = values[-1]
        matches = sum(
            1 for v in values[-5:]
            if abs(v - base) / max(abs(base), 1e-8) <= tol
        )
        return matches >= 2

    eq_high_cluster = _has_eq_cluster([c["high"] for c in ohlc])
    eq_low_cluster = _has_eq_cluster([c["low"] for c in ohlc])

    # ------------------------------------------------------------------
    # Very simple FVG: 3-candle pattern with "gap" (body to wick).
    # ------------------------------------------------------------------
    fvg_up = False
    fvg_down = False
    if len(ohlc) >= 3:
        a, b, c = ohlc[-3], ohlc[-2], ohlc[-1]
        # Up FVG: low of bar C > high of bar A (approx)
        if c["low"] > a["high"]:
            fvg_up = True
        # Down FVG: high of bar C < low of bar A (approx)
        if c["high"] < a["low"]:
            fvg_down = True

    # ------------------------------------------------------------------
    # Wick exhaustion: long wicks against move.
    # ------------------------------------------------------------------
    up_wick = last["high"] - max(last["open"], last["close"])
    down_wick = min(last["open"], last["close"]) - last["low"]
    wick_exhaustion_up = up_wick > 2.0 * body and down_move
    wick_exhaustion_down = down_wick > 2.0 * body and up_move

    # ------------------------------------------------------------------
    # BOS/CHOCH: very crude using last swings.
    # ------------------------------------------------------------------
    bos_direction = "none"
    choch_direction = "none"

    if rel_move > displacement_threshold:
        swing_high = max(highs)
        swing_low = min(lows)
        if last["close"] > swing_high:
            bos_direction = "up"
        elif last["close"] < swing_low:
            bos_direction = "down"
    # CHOCH reserved for later refinement.

    # ------------------------------------------------------------------
    # Order block bias & overall shape bias/score.
    # ------------------------------------------------------------------
    bull_signals = 0
    bear_signals = 0

    # Displacement in direction
    if displacement_up:
        bull_signals += 1
    if displacement_down:
        bear_signals += 1

    # Sweeps can be contrarian
    if sweep_down:
        bull_signals += 1
    if sweep_up:
        bear_signals += 1

    # Fractals at extremes
    if fract_low:
        bull_signals += 1
    if fract_high:
        bear_signals += 1

    # FVG in direction
    if fvg_up:
        bull_signals += 1
    if fvg_down:
        bear_signals += 1

    total_signals = max(bull_signals + bear_signals, 1)
    if bull_signals > bear_signals:
        shape_bias = "bull"
    elif bear_signals > bull_signals:
        shape_bias = "bear"
    else:
        shape_bias = "none"

    # Score ~ how many aligned signals we have, capped.
    shape_score = (max(bull_signals, bear_signals) / total_signals) * min(
        total_signals / 4.0, 1.0
    )

    order_block_bias = shape_bias
    compression_active = False  # reserved for future logic

    return {
        "fvg_up": fvg_up,
        "fvg_down": fvg_down,
        "fract_high": fract_high,
        "fract_low": fract_low,
        "sweep_up": sweep_up,
        "sweep_down": sweep_down,
        "bos_direction": bos_direction,
        "choch_direction": choch_direction,
        "eq_high_cluster": eq_high_cluster,
        "eq_low_cluster": eq_low_cluster,
        "displacement_up": displacement_up,
        "displacement_down": displacement_down,
        "order_block_bias": order_block_bias,
        "compression_active": compression_active,
        "wick_exhaustion_up": wick_exhaustion_up,
        "wick_exhaustion_down": wick_exhaustion_down,
        "shape_score": shape_score,
        "shape_bias": shape_bias,
    }
