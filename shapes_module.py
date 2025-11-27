from typing import List, Dict, Any


def detect_shapes(candles: List[Dict[str, float]]) -> Dict[str, Any]:
    """Very simple, heuristic shape detection for DEMO.

    This is NOT production-quality ICT logic; it just returns a consistent
    microstructure dict given a list of OHLCV candles.
    """
    if not candles:
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

    last = candles[-1]
    prev = candles[-2] if len(candles) > 1 else last

    up_move = last["close"] > prev["close"]
    down_move = last["close"] < prev["close"]

    shape_bias = "bull" if up_move else "bear" if down_move else "none"
    shape_score = 0.3
    if abs(last["close"] - prev["close"]) / max(prev["close"], 1e-8) > 0.005:
        shape_score = 0.7

    return {
        "fvg_up": up_move,
        "fvg_down": down_move,
        "fract_high": last["high"] > prev["high"],
        "fract_low": last["low"] < prev["low"],
        "sweep_up": False,
        "sweep_down": False,
        "bos_direction": "up" if up_move else "down" if down_move else "none",
        "choch_direction": "none",
        "eq_high_cluster": False,
        "eq_low_cluster": False,
        "displacement_up": up_move,
        "displacement_down": down_move,
        "order_block_bias": shape_bias,
        "compression_active": False,
        "wick_exhaustion_up": False,
        "wick_exhaustion_down": False,
        "shape_score": shape_score,
        "shape_bias": shape_bias,
    }
