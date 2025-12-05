#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Any, Dict, List

from adapters.liqdata import HyperliquidDataAdapter, _normalize_symbol, _normalize_timeframe

DEFAULT_OUT_PATH = Path("dumps/hl_mainnet_btcusdt_1m_30d.csv")
FIELDNAMES = ["timestamp", "open", "high", "low", "close", "volume"]
DEFAULT_DAYS = 30
DEFAULT_CHUNK = 1000  # candles per request window
TF_SECONDS = {
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "1h": 3600,
    "4h": 14400,
    "1d": 86400,
}


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


def _fetch_range(
    adapter: HyperliquidDataAdapter,
    symbol: str,
    timeframe: str,
    days: int,
    chunk_size: int,
) -> List[Dict[str, Any]]:
    if getattr(adapter, "_info", None) is None:
        raise SystemExit(
            "Hyperliquid SDK not available. Install hyperliquid-python-sdk before dumping candles."
        )

    hl_symbol = _normalize_symbol(symbol)
    hl_tf = _normalize_timeframe(timeframe)
    interval_secs = TF_SECONDS.get(hl_tf, 60)
    interval_ms = interval_secs * 1000

    end_ms = int(time.time() * 1000)
    start_ms = end_ms - days * 86400 * 1000
    chunk_ms = max(1, chunk_size) * interval_ms

    candles: Dict[int, Dict[str, Any]] = {}
    cursor = start_ms
    while cursor < end_ms:
        window_end = min(cursor + chunk_ms, end_ms)
        try:
            raw = adapter._info.candles_snapshot(hl_symbol, hl_tf, cursor, window_end)
        except Exception as e:
            raise SystemExit(f"Failed to fetch candles: {e}") from e

        for c in raw:
            ts = int(c.get("t", 0)) // 1000
            candles[ts] = {
                "timestamp": ts,
                "open": _coerce_float(c.get("o")),
                "high": _coerce_float(c.get("h")),
                "low": _coerce_float(c.get("l")),
                "close": _coerce_float(c.get("c")),
                "volume": _coerce_float(c.get("v")),
            }
        cursor = window_end

    ordered = [v for _, v in sorted(candles.items(), key=lambda kv: kv[0])]
    return ordered


def dump_candles(symbol: str, timeframe: str, days: int, chunk_size: int, out_path: Path) -> None:
    adapter = HyperliquidDataAdapter(use_testnet=False)
    candles = _fetch_range(adapter, symbol, timeframe, days, chunk_size)
    if not candles:
        raise SystemExit(f"No candles returned for {symbol} {timeframe} over {days}d (mainnet).")

    _write_csv(candles, out_path)
    print(f"Wrote {len(candles)} candles for {symbol} {timeframe} to {out_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dump Hyperliquid mainnet candles to CSV.")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Symbol to fetch (default: BTCUSDT)")
    parser.add_argument("--timeframe", type=str, default="1m", help="Timeframe to fetch (default: 1m)")
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS, help="Number of days to fetch (default: 30)")
    parser.add_argument(
        "--chunk",
        type=int,
        default=DEFAULT_CHUNK,
        help="Candles per request window (default: 1000, matches HL API chunking).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(DEFAULT_OUT_PATH),
        help="Output CSV path (default: dumps/hl_mainnet_btcusdt_1m_30d.csv)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    dump_candles(
        symbol=args.symbol,
        timeframe=args.timeframe,
        days=args.days,
        chunk_size=args.chunk,
        out_path=Path(args.out),
    )
