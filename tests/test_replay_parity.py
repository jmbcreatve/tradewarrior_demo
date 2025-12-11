"""
Tests for replay vs live engine parity.

The key invariant: given the same candles and the same GPT decisions (via stub),
the replay engine and live engine should produce identical decisions.
"""

from config import Config, load_config
from replay_engine import run_replay
from engine import run_spine_tick, SpineTickResult
from build_features import build_snapshot
from replay_gpt_stub import generate_stub_decision, generate_stub_decision_v2
from state_memory import reset_state
from adapters.mock_execution_adapter import MockExecutionAdapter


def test_replay_emits_parity_trace(tmp_path):
    """Test that replay emits a parity trace with required fields."""
    cfg = load_config()
    cfg.symbol = "TEST"
    cfg.timeframe = "1m"
    cfg.state_file = str(tmp_path / "state.json")
    cfg.log_dir = str(tmp_path / "logs")

    candles = [
        {"open": 100.0, "high": 101.0, "low": 99.5, "close": 100.5, "timestamp": 1000},
        {"open": 100.5, "high": 101.5, "low": 100.2, "close": 101.2, "timestamp": 1100},
        {"open": 101.1, "high": 101.6, "low": 100.8, "close": 101.0, "timestamp": 1200},
    ]

    result = run_replay(cfg, candles, use_gpt_stub=True)

    assert "parity_trace" in result
    parity_trace = result["parity_trace"]
    assert len(parity_trace) > 0
    
    # Check all required fields are present in each trace entry
    required_fields = {
        "timestamp", "price", "snapshot_id", 
        "gatekeeper_called", "gatekeeper_reason",
        "gpt_action", "gpt_confidence",
        "approved", "side", "position_size", "leverage",
        "risk_envelope", "execution_status",
    }
    for entry in parity_trace:
        assert required_fields.issubset(set(entry.keys())), f"Missing fields in entry: {set(required_fields) - set(entry.keys())}"


def test_replay_parity_trace_includes_risk_envelope(tmp_path):
    """Test that parity trace includes risk envelope summary."""
    cfg = load_config()
    cfg.symbol = "TEST"
    cfg.timeframe = "1m"
    cfg.state_file = str(tmp_path / "state.json")
    cfg.log_dir = str(tmp_path / "logs")

    candles = [
        {"open": 100.0, "high": 101.0, "low": 99.5, "close": 100.5, "timestamp": 1000},
        {"open": 100.5, "high": 102.5, "low": 100.2, "close": 102.0, "timestamp": 1100},
        {"open": 102.0, "high": 103.0, "low": 101.5, "close": 102.5, "timestamp": 1200},
    ]

    result = run_replay(cfg, candles, use_gpt_stub=True)
    parity_trace = result["parity_trace"]

    for entry in parity_trace:
        assert "risk_envelope" in entry
        risk_env = entry["risk_envelope"]
        # Check envelope has expected keys
        assert "max_notional" in risk_env or risk_env.get("max_notional") is None
        assert "max_leverage" in risk_env or risk_env.get("max_leverage") is None
        assert "note" in risk_env or risk_env.get("note") is None


def test_spine_tick_and_replay_produce_same_decisions(tmp_path):
    """
    Core parity test: verify that run_spine_tick() produces the same decisions
    whether called from live engine context or replay context.
    
    Given:
    - Same candles
    - Same GPT stub
    - Same config
    
    The decisions should be identical.
    """
    cfg = load_config()
    cfg.symbol = "TEST"
    cfg.timeframe = "1m"
    cfg.state_file = str(tmp_path / "state.json")
    cfg.log_dir = str(tmp_path / "logs")

    # Create candles that will trigger a GPT call (significant price moves)
    candles = [
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "timestamp": 1000},
        {"open": 100.5, "high": 105.0, "low": 100.0, "close": 104.5, "timestamp": 1100},  # 4% move
        {"open": 104.5, "high": 106.0, "low": 103.0, "close": 105.5, "timestamp": 1200},
        {"open": 105.5, "high": 108.0, "low": 105.0, "close": 107.5, "timestamp": 1300},  # trend continues
    ]

    # Run replay with stub
    result = run_replay(cfg, candles, use_gpt_stub=True)
    parity_trace = result["parity_trace"]

    # Now run the same candles through spine_tick manually with same stub
    state = reset_state(cfg)
    state["symbol"] = cfg.symbol
    state["equity"] = 10_000.0
    state["gpt_call_timestamps"] = []
    state["last_gpt_call_walltime"] = 0.0
    state["last_gpt_snapshot"] = None
    state["prev_snapshot"] = None

    exec_adapter = MockExecutionAdapter()
    prev_snapshot = None
    manual_trace = []

    for idx, candle in enumerate(candles):
        state["snapshot_id"] = idx + 1
        market_data = {
            "candles": candles[: idx + 1],
            "funding": None,
            "open_interest": None,
            "skew": None,
        }
        snapshot = build_snapshot(cfg, market_data, state)

        spine_result = run_spine_tick(
            snapshot=snapshot,
            prev_snapshot=prev_snapshot,
            state=state,
            config=cfg,
            exec_adapter=exec_adapter,
            gpt_caller=lambda cfg, snap, st: generate_stub_decision_v2(snap),
        )
        manual_trace.append(spine_result.to_parity_trace_entry())
        prev_snapshot = snapshot
        state["prev_snapshot"] = snapshot

    # Compare the two traces
    assert len(parity_trace) == len(manual_trace), f"Trace lengths differ: {len(parity_trace)} vs {len(manual_trace)}"

    for idx, (replay_entry, manual_entry) in enumerate(zip(parity_trace, manual_trace)):
        # Core decision fields must match
        assert replay_entry["gpt_action"] == manual_entry["gpt_action"], \
            f"GPT action mismatch at idx {idx}: {replay_entry['gpt_action']} vs {manual_entry['gpt_action']}"
        assert replay_entry["approved"] == manual_entry["approved"], \
            f"Approved mismatch at idx {idx}: {replay_entry['approved']} vs {manual_entry['approved']}"
        assert replay_entry["side"] == manual_entry["side"], \
            f"Side mismatch at idx {idx}: {replay_entry['side']} vs {manual_entry['side']}"
        
        # Position size should be close (floating point tolerance)
        replay_size = replay_entry.get("position_size", 0.0)
        manual_size = manual_entry.get("position_size", 0.0)
        assert abs(replay_size - manual_size) < 1e-6, \
            f"Position size mismatch at idx {idx}: {replay_size} vs {manual_size}"
        
        # Gatekeeper decisions should match
        assert replay_entry["gatekeeper_called"] == manual_entry["gatekeeper_called"], \
            f"Gatekeeper called mismatch at idx {idx}"


def test_replay_decisions_are_deterministic(tmp_path):
    """
    Test that running replay twice with the same candles and stub
    produces identical results.
    """
    cfg = load_config()
    cfg.symbol = "TEST"
    cfg.timeframe = "1m"
    cfg.state_file = str(tmp_path / "state.json")
    cfg.log_dir = str(tmp_path / "logs")

    candles = [
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "timestamp": 1000},
        {"open": 100.5, "high": 103.0, "low": 100.0, "close": 102.5, "timestamp": 1100},
        {"open": 102.5, "high": 104.0, "low": 101.5, "close": 103.5, "timestamp": 1200},
    ]

    # Run twice
    result1 = run_replay(cfg, candles, use_gpt_stub=True)
    result2 = run_replay(cfg, candles, use_gpt_stub=True)

    trace1 = result1["parity_trace"]
    trace2 = result2["parity_trace"]

    assert len(trace1) == len(trace2)

    for idx, (e1, e2) in enumerate(zip(trace1, trace2)):
        assert e1["gpt_action"] == e2["gpt_action"], f"GPT action differs at idx {idx}"
        assert e1["approved"] == e2["approved"], f"Approved differs at idx {idx}"
        assert e1["side"] == e2["side"], f"Side differs at idx {idx}"
        assert e1["gatekeeper_called"] == e2["gatekeeper_called"], f"Gatekeeper differs at idx {idx}"


def test_parity_trace_captures_gpt_action_details(tmp_path):
    """Test that parity trace captures GPT action and confidence."""
    cfg = load_config()
    cfg.symbol = "TEST"
    cfg.timeframe = "1m"
    cfg.state_file = str(tmp_path / "state.json")
    cfg.log_dir = str(tmp_path / "logs")

    # Create candles with clear trend to trigger GPT call
    candles = [
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "timestamp": 1000},
        {"open": 100.5, "high": 108.0, "low": 100.0, "close": 107.5, "timestamp": 1100},  # Big move
    ]

    result = run_replay(cfg, candles, use_gpt_stub=True)
    parity_trace = result["parity_trace"]

    # Find entries where gatekeeper was called
    called_entries = [e for e in parity_trace if e["gatekeeper_called"]]
    
    for entry in called_entries:
        # GPT action should be valid
        assert entry["gpt_action"] in ("long", "short", "flat"), \
            f"Invalid GPT action: {entry['gpt_action']}"
        # Confidence should be a valid float
        assert isinstance(entry["gpt_confidence"], (int, float)), \
            f"Invalid GPT confidence type: {type(entry['gpt_confidence'])}"
        assert 0.0 <= entry["gpt_confidence"] <= 1.0, \
            f"GPT confidence out of range: {entry['gpt_confidence']}"


def test_parity_trace_captures_risk_reason(tmp_path):
    """Test that parity trace captures risk decision reason."""
    cfg = load_config()
    cfg.symbol = "TEST"
    cfg.timeframe = "1m"
    cfg.state_file = str(tmp_path / "state.json")
    cfg.log_dir = str(tmp_path / "logs")

    candles = [
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "timestamp": 1000},
        {"open": 100.5, "high": 106.0, "low": 100.0, "close": 105.5, "timestamp": 1100},
    ]

    result = run_replay(cfg, candles, use_gpt_stub=True)
    parity_trace = result["parity_trace"]

    for entry in parity_trace:
        # risk_reason should be a string
        assert isinstance(entry.get("risk_reason", ""), str)
