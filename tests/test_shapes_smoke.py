from shapes_module import detect_shapes


# Fixed, deterministic OHLC candles for smoke testing the shape detector.
STABLE_CANDLES = [
    {"open": 100.0, "high": 101.0, "low": 99.5, "close": 100.4},
    {"open": 100.4, "high": 101.2, "low": 100.0, "close": 101.0},
    {"open": 101.0, "high": 102.0, "low": 100.8, "close": 101.8},
    {"open": 101.8, "high": 102.2, "low": 101.0, "close": 101.2},
    {"open": 101.2, "high": 101.6, "low": 100.6, "close": 100.9},
    {"open": 100.9, "high": 102.4, "low": 100.8, "close": 102.1},
    {"open": 102.1, "high": 103.0, "low": 101.9, "close": 102.8},
    {"open": 102.8, "high": 103.5, "low": 102.0, "close": 102.2},
    {"open": 102.2, "high": 102.6, "low": 101.5, "close": 102.5},
    {"open": 102.5, "high": 103.8, "low": 102.0, "close": 103.6},
    {"open": 103.6, "high": 104.0, "low": 103.0, "close": 103.2},
    {"open": 103.2, "high": 104.5, "low": 103.1, "close": 104.4},
]

EXPECTED_KEYS = {
    "fvg_up",
    "fvg_down",
    "fract_high",
    "fract_low",
    "sweep_up",
    "sweep_down",
    "bos_direction",
    "choch_direction",
    "eq_high_cluster",
    "eq_low_cluster",
    "displacement_up",
    "displacement_down",
    "order_block_bias",
    "compression_active",
    "wick_exhaustion_up",
    "wick_exhaustion_down",
    "shape_score",
    "shape_bias",
}

BOOL_KEYS = {
    "fvg_up",
    "fvg_down",
    "fract_high",
    "fract_low",
    "sweep_up",
    "sweep_down",
    "eq_high_cluster",
    "eq_low_cluster",
    "displacement_up",
    "displacement_down",
    "compression_active",
    "wick_exhaustion_up",
    "wick_exhaustion_down",
}

STRING_KEYS = {"bos_direction", "choch_direction", "order_block_bias", "shape_bias"}


def test_detect_shapes_is_deterministic():
    first = detect_shapes(STABLE_CANDLES)
    second = detect_shapes(STABLE_CANDLES)

    assert first == second


def test_detect_shapes_has_core_keys():
    shapes = detect_shapes(STABLE_CANDLES)

    assert isinstance(shapes, dict)
    assert set(shapes.keys()) == EXPECTED_KEYS
    assert "shape_bias" in shapes
    assert "shape_score" in shapes

    assert shapes["shape_bias"] in {"bull", "bear", "none"}
    assert isinstance(shapes["shape_score"], float)

    for key in BOOL_KEYS:
        assert isinstance(shapes[key], bool)
    for key in STRING_KEYS:
        assert isinstance(shapes[key], str)
