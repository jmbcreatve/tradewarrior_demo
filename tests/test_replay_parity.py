from config import load_config
from replay_engine import run_replay


def test_replay_emits_parity_trace(tmp_path):
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
    for entry in parity_trace:
        assert all(key in entry for key in ("timestamp", "price", "approved", "side"))
