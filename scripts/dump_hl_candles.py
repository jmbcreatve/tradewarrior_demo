#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List

from adapters.liqdata import HyperliquidDataAdapter

DEFAULT_OUT_PATH = Path("data/BTCUSDT_1m_hl.csv")
FIELDNAMES = ["timestamp", "open", "high", "low", "close", "volume"]


def _coerce_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _write_csv(candles: List[Dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for candle in candles:
            writer.writerow(
                {
                    "timestamp": int(_coerce_float(candle.get("timestamp", 0.0))),
                    "open": _coerce_float(candle.get("open", 0.0)),
                    "high": _coerce_float(candle.get("high", 0.0)),
                    "low": _coerce_float(candle.get("low", 0.0)),
                    "close": _coerce_float(candle.get("close", 0.0)),
                    "volume": _coerce_float(candle.get("volume", 0.0)),
                }
            )


def dump_candles(symbol: str, timeframe: str, limit: int, out_path: Path) -> None:
    adapter = HyperliquidDataAdapter(use_testnet=True)
    if getattr(adapter, "_info", None) is None:
        raise SystemExit(
            "Hyperliquid SDK not available. Install hyperliquid-python-sdk before dumping candles."
        )

    candles = adapter.fetch_recent_candles(symbol, timeframe, limit)
    if not candles:
        raise SystemExit(f"No candles returned for {symbol} {timeframe} (testnet).")

    _write_csv(candles, out_path)
    print(f"Wrote {len(candles)} candles for {symbol} {timeframe} to {out_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dump Hyperliquid testnet candles to CSV.")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Symbol to fetch (default: BTCUSDT)")
    parser.add_argument("--timeframe", type=str, default="1m", help="Timeframe to fetch (default: 1m)")
    parser.add_argument("--limit", type=int, default=1000, help="Number of candles to fetch (default: 1000)")
    parser.add_argument(
        "--out",
        type=str,
        default=str(DEFAULT_OUT_PATH),
        help="Output CSV path (default: data/BTCUSDT_1m_hl.csv)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    dump_candles(
        symbol=args.symbol,
        timeframe=args.timeframe,
        limit=args.limit,
        out_path=Path(args.out),
    )
