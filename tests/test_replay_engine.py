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
    _write_replay_exports(result, cfg, out_dir=str(tmp_path))

    trades_path = tmp_path / f"trades_{cfg.symbol}_{cfg.timeframe}.csv"
    equity_path = tmp_path / f"equity_{cfg.symbol}_{cfg.timeframe}.csv"

    assert trades_path.exists()
    assert equity_path.exists()
    assert len(trades_path.read_text().strip().splitlines()) > 1
    assert len(equity_path.read_text().strip().splitlines()) > 1
