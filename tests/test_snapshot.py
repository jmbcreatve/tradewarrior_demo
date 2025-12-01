from config import Config
from build_features import build_snapshot
from schemas import validate_snapshot_dict


def test_validate_snapshot_fills_defaults():
    raw = {
        "symbol": None,
        "price": "nan",
        "microstructure": {"shape_score": "oops", "shape_bias": 123},
        "flow": {"skew_bias": "bad_value"},
        "risk_context": {"equity": "oops", "max_drawdown": "oops"},
    }

    snap = validate_snapshot_dict(raw)

    required_keys = {
        "timestamp",
        "symbol",
        "price",
        "trend",
        "range_position",
        "volatility_mode",
        "flow",
        "microstructure",
        "liquidity_context",
        "fib_context",
        "danger_mode",
        "timing_state",
        "recent_price_path",
        "risk_context",
        "gpt_state_note",
    }
    assert required_keys.issubset(set(snap.keys()))
    assert isinstance(snap["price"], float)
    assert snap["microstructure"]["shape_score"] == 0.0
    assert snap["microstructure"]["shape_bias"] == "123"
    assert isinstance(snap["risk_context"]["equity"], float)
    assert snap["risk_context"]["open_positions_summary"] == []


def test_build_snapshot_returns_normalized_dict():
    config = Config()
    candles = [
        {"open": 100, "high": 105, "low": 99, "close": 104, "timestamp": 1_000},
        {"open": 104, "high": 107, "low": 103, "close": 106, "timestamp": 1_001},
    ]
    market_data = {"candles": candles, "funding": 0.01, "open_interest": 1000, "skew": 0.2}
    state = {"symbol": "TEST", "equity": 5_000}

    snap = build_snapshot(config, market_data, state)

    assert snap["symbol"] == "TEST"
    assert snap["price"] == 106.0
    assert snap["recent_price_path"]["lookback_bars"] == len(candles)
    assert snap["risk_context"]["equity"] == 5_000
    assert "microstructure" in snap and "shape_score" in snap["microstructure"]
