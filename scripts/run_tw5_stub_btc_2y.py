"""
Stub-only TW-5 replay harness for long BTC tapes.

Loads 1m BTC candles (CSV or JSON/NDJSON), runs the TW-5 stub over the
full sequence, and writes exports under analytics/runs/<run_id>/.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

# Ensure project root is on sys.path when running as a script
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import Config
from enums import ExecutionMode
from tw5.replay import (
    export_tw5_replay_to_run_folder,
    run_tw5_replay_from_candles,
    run_tw5_replay_with_pnl_from_candles,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TW-5 stub replay on BTC 1m data.")
    parser.add_argument(
        "--candles-path",
        required=True,
        help="Path to BTC 1m candles (CSV with timestamp,open,high,low,close[,volume] or JSON list/NDJSON).",
    )
    parser.add_argument(
        "--run-id",
        default="tw5_stub_btc_2y",
        help="Run identifier for export folder (default: tw5_stub_btc_2y).",
    )
    parser.add_argument(
        "--pnl",
        action="store_true",
        help="If set, run PnL replay (with stops/TP ladder) instead of structural-only.",
    )
    return parser.parse_args()


def _normalize_candle(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure required fields exist and are numeric."""
    timestamp = raw.get("timestamp")
    close = raw.get("close", raw.get("price"))
    if timestamp is None or close is None:
        raise ValueError(f"Candle missing required fields: {raw}")
    try:
        ts_val = float(timestamp)
        close_val = float(close)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid candle values: {raw}") from exc

    out = dict(raw)
    out["timestamp"] = ts_val
    out["close"] = close_val
    # Best-effort fill OHLC if missing
    out.setdefault("open", close_val)
    out.setdefault("high", close_val)
    out.setdefault("low", close_val)
    return out


def _load_candles(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"No candle file at {path}")

    suffix = path.suffix.lower()
    candles: List[Dict[str, Any]] = []

    if suffix == ".csv":
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                candles.append(_normalize_candle(row))
        return candles

    if suffix == ".json":
        content = path.read_text(encoding="utf-8")
        # Try standard JSON (list or wrapped dict)
        try:
            data = json.loads(content)
            if isinstance(data, dict) and "candles" in data:
                data = data["candles"]
            if isinstance(data, list):
                for row in data:
                    if isinstance(row, dict):
                        candles.append(_normalize_candle(row))
                return candles
        except json.JSONDecodeError:
            pass

        # Fallback: NDJSON
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                candles.append(_normalize_candle(row))
        if candles:
            return candles
        raise ValueError(f"Unsupported JSON structure in {path}")

    raise ValueError(f"Unsupported candle file extension: {suffix}")


def _build_config() -> Config:
    cfg = Config()
    cfg.symbol = "BTCUSDT"
    cfg.timeframe = "1m"
    cfg.replay_mode = True
    cfg.execution_mode = ExecutionMode.SIM
    cfg.initial_equity = 10_000.0
    cfg.tw5_prompt_profile = "conservative"
    return cfg


def main() -> None:
    args = _parse_args()
    candle_path = Path(args.candles_path).expanduser().resolve()
    run_id = args.run_id

    candles = _load_candles(candle_path)
    print(f"Loaded {len(candles)} candles from {candle_path}")

    config = _build_config()
    initial_state: Dict[str, Any] = {}

    if args.pnl:
        ticks, stats = run_tw5_replay_with_pnl_from_candles(
            candles=candles,
            config=config,
            initial_state=initial_state,
            use_stub=True,
            warmup_bars=500,
        )
    else:
        ticks = run_tw5_replay_from_candles(
            candles=candles,
            config=config,
            initial_state=initial_state,
            use_stub=True,
            warmup_bars=500,
        )
        stats = None

    run_path = export_tw5_replay_to_run_folder(
        ticks=ticks,
        config=config,
        stats=stats,
        run_id=run_id,
        base_dir="analytics/runs",
        source_path=str(candle_path),
        parity_trace=None,
        stub_used=True,
    )

    print(f"TW-5 stub replay complete.")
    print(f"run_id: {run_id}")
    print(f"run folder: {run_path}")
    print(f"ticks: {len(ticks)}")
    if stats:
        print(f"final_equity: {config.initial_equity + stats.total_pnl:.2f} | trades: {stats.trade_count}")
    else:
        print(f"final_equity (no PnL sim): {config.initial_equity:.2f}")


if __name__ == "__main__":
    main()
