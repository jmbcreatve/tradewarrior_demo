from datetime import datetime, timezone
from config import Config
from build_features import build_snapshot, _find_fractal_swing_points, _build_liquidity_context, _classify_market_session, _compute_timing_state
from schemas import validate_snapshot_dict
from enums import TimingState, enum_to_str


# ---------------------------------------------------------------------------
# TW_CANON 5.1 Snapshot Schema Compliance Tests
# ---------------------------------------------------------------------------

def test_snapshot_schema_all_required_fields_present():
    """
    TW_CANON 5.1: Verify all required snapshot fields are present.
    
    This test ensures the snapshot schema matches TW_CANON exactly.
    Any missing field should fail loudly.
    """
    config = Config()
    candles = [
        {"open": 100, "high": 105, "low": 99, "close": 104, "timestamp": 1_000},
        {"open": 104, "high": 107, "low": 103, "close": 106, "timestamp": 1_001},
    ]
    market_data = {"candles": candles, "funding": 0.01, "open_interest": 1000, "skew": 0.2}
    state = {"symbol": "TEST", "equity": 5_000}
    
    snap = build_snapshot(config, market_data, state)
    
    # TW_CANON 5.1 required fields
    required_fields = {
        "symbol": str,
        "timestamp": float,
        "price": float,
        "trend": str,
        "range_position": str,
        "volatility_mode": str,
        "flow": dict,
        "microstructure": dict,
        "liquidity_context": dict,
        "fib_context": dict,
        "htf_context": dict,
        "danger_mode": bool,
        "timing_state": str,
        "market_session": str,
        "recent_price_path": dict,
        "risk_context": dict,
        "risk_envelope": dict,
        "since_last_gpt": dict,
        "gpt_state_note": (str, type(None)),  # str or None
    }
    
    for field, expected_type in required_fields.items():
        assert field in snap, f"TW_CANON 5.1 violation: missing required field '{field}'"
        
        if isinstance(expected_type, tuple):
            # Multiple allowed types (e.g., str or None)
            assert isinstance(snap[field], expected_type), \
                f"TW_CANON 5.1 type violation: '{field}' expected {expected_type}, got {type(snap[field])}"
        else:
            assert isinstance(snap[field], expected_type), \
                f"TW_CANON 5.1 type violation: '{field}' expected {expected_type}, got {type(snap[field])}"


def test_snapshot_flow_dict_structure():
    """TW_CANON 5.1: Verify flow dict has required fields."""
    config = Config()
    candles = [{"open": 100, "high": 105, "low": 99, "close": 104, "timestamp": 1_000}]
    market_data = {"candles": candles, "funding": 0.01, "open_interest": 1000, "skew": 0.2}
    state = {"symbol": "TEST", "equity": 5_000}
    
    snap = build_snapshot(config, market_data, state)
    flow = snap["flow"]
    
    # TW_CANON 5.1: flow dict must have funding, open_interest, skew, skew_bias
    required_flow_fields = ["funding", "open_interest", "skew", "skew_bias"]
    for field in required_flow_fields:
        assert field in flow, f"TW_CANON 5.1 violation: flow missing required field '{field}'"


def test_snapshot_microstructure_dict_structure():
    """TW_CANON 5.1: Verify microstructure dict has required fields."""
    config = Config()
    candles = [
        {"open": 100, "high": 105, "low": 99, "close": 104, "timestamp": 1_000},
        {"open": 104, "high": 107, "low": 103, "close": 106, "timestamp": 1_001},
    ]
    market_data = {"candles": candles, "funding": 0.01, "open_interest": 1000, "skew": 0.2}
    state = {"symbol": "TEST", "equity": 5_000}
    
    snap = build_snapshot(config, market_data, state)
    micro = snap["microstructure"]
    
    # TW_CANON 5.1: microstructure must have sweep_up, sweep_down, fvg_up, fvg_down,
    # choch_direction, compression_active, shape_bias, shape_score
    required_micro_fields = [
        "sweep_up", "sweep_down", "fvg_up", "fvg_down",
        "choch_direction", "compression_active", "shape_bias", "shape_score"
    ]
    for field in required_micro_fields:
        assert field in micro, f"TW_CANON 5.1 violation: microstructure missing required field '{field}'"
    
    # shape_score must be float
    assert isinstance(micro["shape_score"], (int, float)), \
        f"shape_score must be numeric, got {type(micro['shape_score'])}"
    
    # shape_bias must be string
    assert isinstance(micro["shape_bias"], str), \
        f"shape_bias must be string, got {type(micro['shape_bias'])}"


def test_snapshot_liquidity_context_structure():
    """TW_CANON 5.1: Verify liquidity_context has required fields."""
    config = Config()
    candles = [{"open": 100, "high": 105, "low": 99, "close": 104, "timestamp": 1_000}]
    market_data = {"candles": candles, "funding": 0.01, "open_interest": 1000, "skew": 0.2}
    state = {"symbol": "TEST", "equity": 5_000}
    
    snap = build_snapshot(config, market_data, state)
    liq = snap["liquidity_context"]
    
    # TW_CANON 5.1: liquidity_above, liquidity_below (floats or None)
    assert "liquidity_above" in liq, "Missing liquidity_above"
    assert "liquidity_below" in liq, "Missing liquidity_below"
    
    # Values must be float or None
    for key in ["liquidity_above", "liquidity_below"]:
        val = liq[key]
        assert val is None or isinstance(val, (int, float)), \
            f"{key} must be float or None, got {type(val)}"


def test_snapshot_fib_context_structure():
    """TW_CANON 5.1: Verify fib_context has required fields."""
    config = Config()
    candles = [{"open": 100, "high": 105, "low": 99, "close": 104, "timestamp": 1_000}]
    market_data = {"candles": candles, "funding": 0.01, "open_interest": 1000, "skew": 0.2}
    state = {"symbol": "TEST", "equity": 5_000}
    
    snap = build_snapshot(config, market_data, state)
    fib = snap["fib_context"]
    
    # TW_CANON 5.1: macro_zone, micro_zone
    assert "macro_zone" in fib, "Missing macro_zone"
    assert "micro_zone" in fib, "Missing micro_zone"
    assert isinstance(fib["macro_zone"], str), "macro_zone must be string"
    assert isinstance(fib["micro_zone"], str), "micro_zone must be string"


def test_snapshot_htf_context_structure():
    """TW_CANON 5.1: Verify htf_context has required fields."""
    config = Config()
    candles = [{"open": 100, "high": 105, "low": 99, "close": 104, "timestamp": 1_000}]
    market_data = {"candles": candles, "funding": 0.01, "open_interest": 1000, "skew": 0.2}
    state = {"symbol": "TEST", "equity": 5_000}
    
    snap = build_snapshot(config, market_data, state)
    htf = snap["htf_context"]
    
    # TW_CANON 5.1: trend_1h, range_pos_1h (placeholder is fine but keys must exist)
    assert "trend_1h" in htf, "Missing trend_1h"
    assert "range_pos_1h" in htf, "Missing range_pos_1h"
    assert isinstance(htf["trend_1h"], str), "trend_1h must be string"
    assert isinstance(htf["range_pos_1h"], str), "range_pos_1h must be string"


def test_snapshot_recent_price_path_structure():
    """TW_CANON 5.1: Verify recent_price_path has required fields."""
    config = Config()
    candles = [
        {"open": 100, "high": 105, "low": 99, "close": 104, "timestamp": 1_000},
        {"open": 104, "high": 107, "low": 103, "close": 106, "timestamp": 1_001},
    ]
    market_data = {"candles": candles, "funding": 0.01, "open_interest": 1000, "skew": 0.2}
    state = {"symbol": "TEST", "equity": 5_000}
    
    snap = build_snapshot(config, market_data, state)
    rpp = snap["recent_price_path"]
    
    # TW_CANON 5.1: ret_1, ret_5, ret_15, impulse_state, lookback_bars
    required_rpp_fields = ["ret_1", "ret_5", "ret_15", "impulse_state", "lookback_bars"]
    for field in required_rpp_fields:
        assert field in rpp, f"Missing recent_price_path.{field}"
    
    # Returns must be float
    for ret_field in ["ret_1", "ret_5", "ret_15"]:
        assert isinstance(rpp[ret_field], (int, float)), f"{ret_field} must be numeric"
    
    # lookback_bars must be int
    assert isinstance(rpp["lookback_bars"], int), "lookback_bars must be int"
    
    # impulse_state must be string
    assert isinstance(rpp["impulse_state"], str), "impulse_state must be string"


def test_snapshot_risk_context_structure():
    """TW_CANON 5.1: Verify risk_context has required fields."""
    config = Config()
    candles = [{"open": 100, "high": 105, "low": 99, "close": 104, "timestamp": 1_000}]
    market_data = {"candles": candles, "funding": 0.01, "open_interest": 1000, "skew": 0.2}
    state = {"symbol": "TEST", "equity": 5_000, "max_drawdown": 0.02}
    
    snap = build_snapshot(config, market_data, state)
    risk_ctx = snap["risk_context"]
    
    # TW_CANON 5.1: equity, max_drawdown, open_positions_summary, last_action, last_confidence
    required_risk_fields = ["equity", "max_drawdown", "open_positions_summary", "last_action", "last_confidence"]
    for field in required_risk_fields:
        assert field in risk_ctx, f"Missing risk_context.{field}"
    
    # equity, max_drawdown, last_confidence must be float
    assert isinstance(risk_ctx["equity"], (int, float)), "equity must be numeric"
    assert isinstance(risk_ctx["max_drawdown"], (int, float)), "max_drawdown must be numeric"
    assert isinstance(risk_ctx["last_confidence"], (int, float)), "last_confidence must be numeric"
    
    # open_positions_summary must be list
    assert isinstance(risk_ctx["open_positions_summary"], list), "open_positions_summary must be list"
    
    # last_action must be string
    assert isinstance(risk_ctx["last_action"], str), "last_action must be string"


def test_snapshot_risk_envelope_structure():
    """TW_CANON 5.3: Verify risk_envelope has all required fields."""
    config = Config()
    candles = [{"open": 100, "high": 105, "low": 99, "close": 104, "timestamp": 1_000}]
    market_data = {"candles": candles, "funding": 0.01, "open_interest": 1000, "skew": 0.2}
    state = {"symbol": "TEST", "equity": 5_000}
    
    snap = build_snapshot(config, market_data, state)
    risk_env = snap["risk_envelope"]
    
    # TW_CANON 5.3 required fields
    required_env_fields = [
        "max_leverage",
        "max_notional",
        "max_risk_per_trade_pct",
        "min_stop_distance_pct",
        "max_stop_distance_pct",
        "max_daily_loss_pct",
        "note",
    ]
    for field in required_env_fields:
        assert field in risk_env, f"TW_CANON 5.3 violation: risk_envelope missing '{field}'"
    
    # All numeric fields must be float
    numeric_fields = [
        "max_leverage", "max_notional", "max_risk_per_trade_pct",
        "min_stop_distance_pct", "max_stop_distance_pct", "max_daily_loss_pct"
    ]
    for field in numeric_fields:
        assert isinstance(risk_env[field], (int, float)), f"{field} must be numeric"
    
    # note must be string
    assert isinstance(risk_env["note"], str), "risk_envelope.note must be string"


def test_snapshot_since_last_gpt_structure():
    """TW_CANON 5.1: Verify since_last_gpt has required fields."""
    config = Config()
    candles = [{"open": 100, "high": 105, "low": 99, "close": 104, "timestamp": 1_000}]
    market_data = {"candles": candles, "funding": 0.01, "open_interest": 1000, "skew": 0.2}
    state = {"symbol": "TEST", "equity": 5_000}
    
    snap = build_snapshot(config, market_data, state)
    slg = snap["since_last_gpt"]
    
    # TW_CANON 5.1: time_since_last_gpt_sec, price_change_pct_since_last_gpt,
    # equity_change_since_last_gpt, trades_since_last_gpt
    required_slg_fields = [
        "time_since_last_gpt_sec",
        "price_change_pct_since_last_gpt",
        "equity_change_since_last_gpt",
        "trades_since_last_gpt",
    ]
    for field in required_slg_fields:
        assert field in slg, f"Missing since_last_gpt.{field}"
    
    # time/price/equity must be float
    float_fields = ["time_since_last_gpt_sec", "price_change_pct_since_last_gpt", "equity_change_since_last_gpt"]
    for field in float_fields:
        assert isinstance(slg[field], (int, float)), f"{field} must be numeric"
    
    # trades_since_last_gpt must be int
    assert isinstance(slg["trades_since_last_gpt"], int), "trades_since_last_gpt must be int"


def test_snapshot_enum_values_valid():
    """TW_CANON 5.1: Verify enum fields have valid values."""
    config = Config()
    candles = [
        {"open": 100, "high": 105, "low": 99, "close": 104, "timestamp": 1_000},
        {"open": 104, "high": 107, "low": 103, "close": 106, "timestamp": 1_001},
    ]
    market_data = {"candles": candles, "funding": 0.01, "open_interest": 1000, "skew": 0.2}
    state = {"symbol": "TEST", "equity": 5_000}
    
    snap = build_snapshot(config, market_data, state)
    
    # trend: "up" | "down" | "sideways" | "unknown"
    valid_trends = {"up", "down", "sideways", "unknown"}
    assert snap["trend"] in valid_trends, f"Invalid trend: {snap['trend']}"
    
    # range_position: "extreme_low" | "low" | "mid" | "high" | "extreme_high" | "unknown"
    valid_range_positions = {"extreme_low", "low", "mid", "high", "extreme_high", "unknown"}
    assert snap["range_position"] in valid_range_positions, f"Invalid range_position: {snap['range_position']}"
    
    # volatility_mode: "low" | "normal" | "high" | "explosive" | "unknown"
    valid_vol_modes = {"low", "normal", "high", "explosive", "unknown"}
    assert snap["volatility_mode"] in valid_vol_modes, f"Invalid volatility_mode: {snap['volatility_mode']}"
    
    # timing_state: "avoid" | "cautious" | "normal" | "aggressive" | "unknown"
    valid_timing_states = {"avoid", "cautious", "normal", "aggressive", "unknown"}
    assert snap["timing_state"] in valid_timing_states, f"Invalid timing_state: {snap['timing_state']}"
    
    # market_session: "ASIA" | "EUROPE" | "US" | "OFF_HOURS"
    valid_sessions = {"ASIA", "EUROPE", "US", "OFF_HOURS"}
    assert snap["market_session"] in valid_sessions, f"Invalid market_session: {snap['market_session']}"
    
    # danger_mode: bool
    assert isinstance(snap["danger_mode"], bool), "danger_mode must be bool"


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
        "htf_context",
        "danger_mode",
        "timing_state",
        "market_session",
        "recent_price_path",
        "risk_context",
        "risk_envelope",
        "since_last_gpt",
        "gpt_state_note",
    }
    assert required_keys.issubset(set(snap.keys()))
    assert isinstance(snap["price"], float)
    assert snap["microstructure"]["shape_score"] == 0.0
    assert snap["microstructure"]["shape_bias"] == "123"
    assert isinstance(snap["risk_context"]["equity"], float)
    assert snap["risk_context"]["open_positions_summary"] == []
    # HTF context should be present with safe defaults
    assert "htf_context" in snap
    assert isinstance(snap["htf_context"], dict)
    assert "trend_1h" in snap["htf_context"]
    assert "range_pos_1h" in snap["htf_context"]
    assert isinstance(snap["htf_context"]["trend_1h"], str)
    assert isinstance(snap["htf_context"]["range_pos_1h"], str)
    risk_env = snap["risk_envelope"]
    numeric_env_keys = {
        "max_notional",
        "max_leverage",
        "max_risk_per_trade_pct",
        "min_stop_distance_pct",
        "max_stop_distance_pct",
        "max_daily_loss_pct",
    }
    assert numeric_env_keys.issubset(set(risk_env.keys()))
    for key in numeric_env_keys:
        assert isinstance(risk_env[key], float)
        assert risk_env[key] == 0.0
    assert risk_env["note"] == "risk_envelope not provided"
    slg = snap["since_last_gpt"]
    slg_keys = {
        "time_since_last_gpt_sec",
        "price_change_pct_since_last_gpt",
        "equity_change_since_last_gpt",
        "trades_since_last_gpt",
    }
    assert slg_keys.issubset(set(slg.keys()))
    assert isinstance(slg["time_since_last_gpt_sec"], float)
    assert isinstance(slg["price_change_pct_since_last_gpt"], float)
    assert isinstance(slg["equity_change_since_last_gpt"], float)
    assert isinstance(slg["trades_since_last_gpt"], int)


def test_build_snapshot_includes_gpt_state_note_from_state():
    """Test that gpt_state_note from state appears in the snapshot."""
    config = Config()
    candles = [
        {"open": 100, "high": 105, "low": 99, "close": 104, "timestamp": 1_000},
        {"open": 104, "high": 107, "low": 103, "close": 106, "timestamp": 1_001},
    ]
    market_data = {"candles": candles, "funding": 0.01, "open_interest": 1000, "skew": 0.2}
    
    # State with gpt_state_note from previous GPT decision
    expected_note = "Previous decision: uptrend with strong bullish structure, staying long"
    state = {
        "symbol": "TEST",
        "equity": 5_000,
        "gpt_state_note": expected_note,
    }
    
    snap = build_snapshot(config, market_data, state)
    
    # Verify gpt_state_note appears in snapshot
    assert "gpt_state_note" in snap
    assert snap["gpt_state_note"] == expected_note


def test_build_snapshot_handles_missing_gpt_state_note():
    """Test that build_snapshot handles missing gpt_state_note gracefully."""
    config = Config()
    candles = [
        {"open": 100, "high": 105, "low": 99, "close": 104, "timestamp": 1_000},
    ]
    market_data = {"candles": candles, "funding": 0.01, "open_interest": 1000, "skew": 0.2}
    state = {"symbol": "TEST", "equity": 5_000}  # No gpt_state_note
    
    snap = build_snapshot(config, market_data, state)
    
    # Verify gpt_state_note is None when not in state
    assert "gpt_state_note" in snap
    assert snap["gpt_state_note"] is None


def test_validate_snapshot_normalizes_risk_envelope():
    raw = {
        "risk_envelope": {
            "max_notional": "100000",
            "max_leverage": "2",
            "max_risk_per_trade_pct": "1.5",
            "min_stop_distance_pct": "0.25",
            "max_stop_distance_pct": "1.5",
            "max_daily_loss_pct": "3.5",
            "note": "provided envelope",
        }
    }

    snap = validate_snapshot_dict(raw)

    risk_env = snap["risk_envelope"]
    assert risk_env["max_notional"] == 100000.0
    assert risk_env["max_leverage"] == 2.0
    assert risk_env["max_risk_per_trade_pct"] == 1.5
    assert risk_env["min_stop_distance_pct"] == 0.25
    assert risk_env["max_stop_distance_pct"] == 1.5
    assert risk_env["max_daily_loss_pct"] == 3.5
    assert risk_env["note"] == "provided envelope"


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
    # HTF context should be present
    assert "htf_context" in snap
    assert isinstance(snap["htf_context"], dict)
    assert "trend_1h" in snap["htf_context"]
    assert "range_pos_1h" in snap["htf_context"]
    assert isinstance(snap["htf_context"]["trend_1h"], str)
    assert isinstance(snap["htf_context"]["range_pos_1h"], str)
    # With only 2 candles, HTF should default to unknown/mid
    assert snap["htf_context"]["trend_1h"] in ("up", "down", "sideways", "unknown")
    assert snap["htf_context"]["range_pos_1h"] in ("low", "mid", "high")
    risk_env = snap["risk_envelope"]
    expected_risk_env_keys = {
        "max_notional",
        "max_leverage",
        "max_risk_per_trade_pct",
        "min_stop_distance_pct",
        "max_stop_distance_pct",
        "max_daily_loss_pct",
        "note",
    }
    assert expected_risk_env_keys.issubset(set(risk_env.keys()))
    for key in expected_risk_env_keys - {"note"}:
        assert isinstance(risk_env[key], float)
    assert isinstance(risk_env["note"], str)
    slg = snap["since_last_gpt"]
    slg_keys = {
        "time_since_last_gpt_sec",
        "price_change_pct_since_last_gpt",
        "equity_change_since_last_gpt",
        "trades_since_last_gpt",
    }
    assert slg_keys.issubset(set(slg.keys()))
    assert isinstance(slg["time_since_last_gpt_sec"], float)
    assert isinstance(slg["price_change_pct_since_last_gpt"], float)
    assert isinstance(slg["equity_change_since_last_gpt"], float)
    assert isinstance(slg["trades_since_last_gpt"], int)


# ---------------------------------------------------------------------------
# Liquidity context tests
# ---------------------------------------------------------------------------

def test_liquidity_context_exists_in_snapshot():
    """Test that liquidity_context is present with required keys."""
    config = Config()
    candles = [
        {"open": 100, "high": 105, "low": 99, "close": 104, "timestamp": 1_000},
        {"open": 104, "high": 107, "low": 103, "close": 106, "timestamp": 1_001},
    ]
    market_data = {"candles": candles, "funding": 0.01, "open_interest": 1000, "skew": 0.2}
    state = {"symbol": "TEST", "equity": 5_000}

    snap = build_snapshot(config, market_data, state)

    assert "liquidity_context" in snap
    liq = snap["liquidity_context"]
    assert "liquidity_above" in liq
    assert "liquidity_below" in liq


def test_liquidity_context_with_clear_swing_levels():
    """Test liquidity_context identifies swing highs/lows from synthetic candles.

    Creates a price series with a clear swing high above current price
    and a clear swing low below current price.
    """
    # Build a candle series: goes up to 110, dips to 102, rallies to 108, then pulls back to 105
    # Fractal swing high should be at 110 (higher than neighbors)
    # Fractal swing low should be at 102 (lower than neighbors)
    candles = [
        {"open": 100, "high": 101, "low": 99, "close": 100, "timestamp": 1},   # baseline
        {"open": 100, "high": 105, "low": 99, "close": 104, "timestamp": 2},   # up
        {"open": 104, "high": 110, "low": 104, "close": 109, "timestamp": 3},  # swing HIGH at 110
        {"open": 109, "high": 109, "low": 103, "close": 104, "timestamp": 4},  # down
        {"open": 104, "high": 105, "low": 102, "close": 103, "timestamp": 5},  # swing LOW at 102
        {"open": 103, "high": 108, "low": 103, "close": 107, "timestamp": 6},  # up
        {"open": 107, "high": 108, "low": 105, "close": 105, "timestamp": 7},  # current: close at 105
    ]

    liq = _build_liquidity_context(candles)

    # Current price is 105
    # Swing high at 110 is above 105 -> liquidity_above should be 110
    # Swing low at 102 is below 105 -> liquidity_below should be 102
    assert liq["liquidity_above"] == 110.0
    assert liq["liquidity_below"] == 102.0


def test_liquidity_context_no_level_above():
    """Test liquidity_context when current price is at or above all swing highs."""
    # All candles have highs below or equal to current close
    candles = [
        {"open": 100, "high": 101, "low": 99, "close": 100, "timestamp": 1},
        {"open": 100, "high": 102, "low": 99, "close": 101, "timestamp": 2},
        {"open": 101, "high": 103, "low": 100, "close": 102, "timestamp": 3},  # swing high at 103
        {"open": 102, "high": 102, "low": 100, "close": 101, "timestamp": 4},
        {"open": 101, "high": 101, "low": 99, "close": 100, "timestamp": 5},   # swing low at 99
        {"open": 100, "high": 104, "low": 100, "close": 103, "timestamp": 6},
        {"open": 103, "high": 110, "low": 103, "close": 110, "timestamp": 7},  # current at 110, highest
    ]

    liq = _build_liquidity_context(candles)

    # Current price is 110, which is >= all swing highs
    assert liq["liquidity_above"] is None
    # Swing low at 99 and 100 are below 110
    assert liq["liquidity_below"] is not None
    assert liq["liquidity_below"] < 110


def test_liquidity_context_no_level_below():
    """Test liquidity_context when current price is at or below all swing lows."""
    # Price series where current price is lowest
    # Need clear fractal swing high: middle bar's high must be > both neighbors' highs
    candles = [
        {"open": 110, "high": 111, "low": 109, "close": 110, "timestamp": 1},
        {"open": 110, "high": 113, "low": 109, "close": 112, "timestamp": 2},
        {"open": 112, "high": 118, "low": 111, "close": 117, "timestamp": 3},  # swing HIGH at 118 (> 113 and > 114)
        {"open": 117, "high": 114, "low": 110, "close": 111, "timestamp": 4},
        {"open": 111, "high": 112, "low": 108, "close": 109, "timestamp": 5},
        {"open": 109, "high": 110, "low": 100, "close": 101, "timestamp": 6},
        {"open": 101, "high": 102, "low": 95, "close": 95, "timestamp": 7},    # current at 95, lowest
    ]

    liq = _build_liquidity_context(candles)

    # Current price is 95, which is <= all swing lows
    assert liq["liquidity_below"] is None
    # Swing high at 118 is above 95
    assert liq["liquidity_above"] is not None
    assert liq["liquidity_above"] > 95


def test_liquidity_context_empty_candles():
    """Test liquidity_context returns None values for empty candles."""
    liq = _build_liquidity_context([])

    assert liq["liquidity_above"] is None
    assert liq["liquidity_below"] is None


def test_liquidity_context_with_eq_cluster_shapes():
    """Test that eq_high_cluster and eq_low_cluster in shapes refine liquidity levels."""
    # Simple candle set
    candles = [
        {"open": 100, "high": 105, "low": 99, "close": 104, "timestamp": 1},
        {"open": 104, "high": 108, "low": 103, "close": 107, "timestamp": 2},
        {"open": 107, "high": 108, "low": 104, "close": 105, "timestamp": 3},
        {"open": 105, "high": 107, "low": 102, "close": 103, "timestamp": 4},
        {"open": 103, "high": 106, "low": 101, "close": 104, "timestamp": 5},
    ]

    # With eq_high_cluster, the cluster high should be added as liquidity
    shapes_with_cluster = {"eq_high_cluster": True, "eq_low_cluster": False}
    liq = _build_liquidity_context(candles, shapes=shapes_with_cluster)

    assert "liquidity_above" in liq
    assert "liquidity_below" in liq
    # Should have some level above (cluster high from recent bars)
    # The max high in last 5 candles is 108, current close is 104
    assert liq["liquidity_above"] is not None
    assert liq["liquidity_above"] > 104


def test_find_fractal_swing_points_basic():
    """Test the fractal swing point detection helper."""
    # Clear swing high at index 2, swing low at index 4
    candles = [
        {"high": 100, "low": 98},
        {"high": 102, "low": 99},
        {"high": 110, "low": 101},  # swing high
        {"high": 105, "low": 100},
        {"high": 103, "low": 95},   # swing low
        {"high": 104, "low": 97},
        {"high": 106, "low": 98},
    ]

    swing_highs, swing_lows = _find_fractal_swing_points(candles, lookback=2)

    assert 110 in swing_highs
    assert 95 in swing_lows


def test_find_fractal_swing_points_insufficient_data():
    """Test fractal detection with too few candles."""
    candles = [
        {"high": 100, "low": 98},
        {"high": 102, "low": 99},
    ]

    swing_highs, swing_lows = _find_fractal_swing_points(candles, lookback=2)

    # Not enough candles for lookback=2 (need at least 5)
    assert swing_highs == []
    assert swing_lows == []


def test_classify_market_session_asia():
    """Test ASIA session classification (00:00-08:00 UTC)."""
    # Test ASIA session: 2024-01-15 02:00 UTC
    dt = datetime(2024, 1, 15, 2, 0, 0, tzinfo=timezone.utc)
    ts = dt.timestamp()
    session = _classify_market_session(ts)
    assert session == "ASIA"


def test_classify_market_session_europe():
    """Test EUROPE session classification (07:00-16:00 UTC)."""
    # Test EUROPE session: 2024-01-15 10:00 UTC (not overlapping with US)
    dt = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
    ts = dt.timestamp()
    session = _classify_market_session(ts)
    assert session == "EUROPE"


def test_classify_market_session_us():
    """Test US session classification (13:00-22:00 UTC)."""
    # Test US session: 2024-01-15 15:00 UTC
    dt = datetime(2024, 1, 15, 15, 0, 0, tzinfo=timezone.utc)
    ts = dt.timestamp()
    session = _classify_market_session(ts)
    assert session == "US"


def test_classify_market_session_off_hours():
    """Test OFF_HOURS classification (22:00-00:00 UTC)."""
    # Test OFF_HOURS: 2024-01-15 23:00 UTC
    dt = datetime(2024, 1, 15, 23, 0, 0, tzinfo=timezone.utc)
    ts = dt.timestamp()
    session = _classify_market_session(ts)
    assert session == "OFF_HOURS"


def test_classify_market_session_overlap_priorities():
    """Test that session overlap prioritizes US > EUROPE > ASIA."""
    # 14:00 UTC overlaps EUROPE and US, should prioritize US
    dt = datetime(2024, 1, 15, 14, 0, 0, tzinfo=timezone.utc)
    ts = dt.timestamp()
    session = _classify_market_session(ts)
    assert session == "US"
    
    # 08:00 UTC overlaps ASIA and EUROPE, should prioritize EUROPE
    dt = datetime(2024, 1, 15, 8, 0, 0, tzinfo=timezone.utc)
    ts = dt.timestamp()
    session = _classify_market_session(ts)
    assert session == "EUROPE"


def test_compute_timing_state_mapping():
    """Test that market sessions map to correct TimingState enum values."""
    # OFF_HOURS -> AVOID
    dt = datetime(2024, 1, 15, 23, 0, 0, tzinfo=timezone.utc)
    ts = dt.timestamp()
    session, timing_state = _compute_timing_state(ts)
    assert session == "OFF_HOURS"
    assert timing_state == enum_to_str(TimingState.AVOID)
    
    # ASIA -> CAUTIOUS
    dt = datetime(2024, 1, 15, 2, 0, 0, tzinfo=timezone.utc)
    ts = dt.timestamp()
    session, timing_state = _compute_timing_state(ts)
    assert session == "ASIA"
    assert timing_state == enum_to_str(TimingState.CAUTIOUS)
    
    # EUROPE -> NORMAL
    dt = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
    ts = dt.timestamp()
    session, timing_state = _compute_timing_state(ts)
    assert session == "EUROPE"
    assert timing_state == enum_to_str(TimingState.NORMAL)
    
    # US -> NORMAL
    dt = datetime(2024, 1, 15, 15, 0, 0, tzinfo=timezone.utc)
    ts = dt.timestamp()
    session, timing_state = _compute_timing_state(ts)
    assert session == "US"
    assert timing_state == enum_to_str(TimingState.NORMAL)


def test_build_snapshot_includes_market_session():
    """Test that build_snapshot includes market_session and timing_state."""
    config = Config()
    
    # Test with ASIA session timestamp
    dt = datetime(2024, 1, 15, 2, 0, 0, tzinfo=timezone.utc)
    ts = dt.timestamp()
    
    candles = [
        {"open": 100, "high": 105, "low": 99, "close": 104, "timestamp": ts},
    ]
    market_data = {"candles": candles, "funding": 0.01, "open_interest": 1000, "skew": 0.2}
    state = {"symbol": "TEST", "equity": 5_000}
    
    snap = build_snapshot(config, market_data, state)
    
    # Verify market_session is present
    assert "market_session" in snap
    assert snap["market_session"] == "ASIA"
    
    # Verify timing_state is set correctly
    assert "timing_state" in snap
    assert snap["timing_state"] == enum_to_str(TimingState.CAUTIOUS)


def test_build_snapshot_timing_state_changes_by_session():
    """Test that timing_state changes appropriately for different sessions."""
    config = Config()
    state = {"symbol": "TEST", "equity": 5_000}
    
    # Test OFF_HOURS -> AVOID
    dt_off = datetime(2024, 1, 15, 23, 0, 0, tzinfo=timezone.utc)
    candles_off = [{"open": 100, "high": 105, "low": 99, "close": 104, "timestamp": dt_off.timestamp()}]
    market_data_off = {"candles": candles_off, "funding": 0.01, "open_interest": 1000, "skew": 0.2}
    snap_off = build_snapshot(config, market_data_off, state)
    assert snap_off["market_session"] == "OFF_HOURS"
    assert snap_off["timing_state"] == enum_to_str(TimingState.AVOID)
    
    # Test US -> NORMAL
    dt_us = datetime(2024, 1, 15, 15, 0, 0, tzinfo=timezone.utc)
    candles_us = [{"open": 100, "high": 105, "low": 99, "close": 104, "timestamp": dt_us.timestamp()}]
    market_data_us = {"candles": candles_us, "funding": 0.01, "open_interest": 1000, "skew": 0.2}
    snap_us = build_snapshot(config, market_data_us, state)
    assert snap_us["market_session"] == "US"
    assert snap_us["timing_state"] == enum_to_str(TimingState.NORMAL)
    
    # Verify they're different
    assert snap_off["timing_state"] != snap_us["timing_state"]


def test_validate_snapshot_includes_market_session():
    """Test that validate_snapshot_dict handles market_session field."""
    raw = {
        "symbol": "TEST",
        "price": 100.0,
        "timestamp": 1000.0,
        "market_session": "US",
    }
    
    snap = validate_snapshot_dict(raw)
    
    assert "market_session" in snap
    assert snap["market_session"] == "US"
    
    # Test with invalid session defaults to OFF_HOURS
    raw_invalid = {
        "symbol": "TEST",
        "price": 100.0,
        "timestamp": 1000.0,
        "market_session": "INVALID",
    }
    
    snap_invalid = validate_snapshot_dict(raw_invalid)
    assert snap_invalid["market_session"] == "OFF_HOURS"
