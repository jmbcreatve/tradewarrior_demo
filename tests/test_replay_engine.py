from config import load_config
from replay_engine import _write_replay_exports, run_replay


def test_replay_exports_written(tmp_path):
    cfg = load_config()
    cfg.symbol = "TEST"
    cfg.timeframe = "1m"
    cfg.state_file = str(tmp_path / "state.json")
    cfg.log_dir = str(tmp_path / "logs")

    candles = [
        {"open": 100.0, "high": 100.4, "low": 99.8, "close": 100.0, "timestamp": 1000},
        {"open": 100.0, "high": 100.7, "low": 99.9, "close": 100.5, "timestamp": 1100},
        {"open": 100.6, "high": 101.3, "low": 100.5, "close": 101.2, "timestamp": 1200},
    ]

    result = run_replay(cfg, candles, use_gpt_stub=True)
    run_id = result.get("run_id") or "test_run"
    parity_trace = result.get("parity_trace", [])
    _write_replay_exports(
        result,
        cfg,
        out_dir=str(tmp_path / "analytics"),
        run_id=run_id,
        parity_trace=parity_trace,
        features_entry=result.get("features_at_entry", []),
        features_exit=result.get("features_at_exit", []),
        source_path=None,
        stub_used=bool(result.get("stub_used", False)),
    )

    run_path = tmp_path / "analytics" / "runs" / run_id
    trades_path = run_path / "trades.csv"
    equity_path = run_path / "equity.csv"
    parity_path = run_path / "parity.jsonl"
    config_path = run_path / "run_config.json"
    summary_path = run_path / "summary.json"

    assert run_path.exists()
    assert trades_path.exists()
    assert equity_path.exists()
    assert parity_path.exists()
    assert config_path.exists()
    assert summary_path.exists()

    # Equity curve should include header + at least one datapoint
    assert len(equity_path.read_text().strip().splitlines()) > 1
    # Trades may be empty; ensure file has a header
    assert len(trades_path.read_text().strip().splitlines()) >= 1
    # Parity trace should contain at least one entry
    parity_lines = [line for line in parity_path.read_text().splitlines() if line.strip()]
    assert parity_lines
