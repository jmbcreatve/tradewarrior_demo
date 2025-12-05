from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from adapters.base_execution_adapter import BaseExecutionAdapter
from adapters.mock_data_adapter import MockDataAdapter
from build_features import build_snapshot
from config import Config, load_config
from gpt_client import call_gpt
from logger_utils import get_logger
from replay_gpt_stub import generate_stub_decision
from state_memory import load_state
from schemas import GptDecision

# Import the shared spine tick function for live/replay parity
from engine import run_spine_tick, SpineTickResult

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Lightweight replay execution adapter
# ---------------------------------------------------------------------------


class ReplayExecutionAdapter(BaseExecutionAdapter):
    """In-memory adapter so we can reuse execute_decision without touching brokers."""

    def __init__(self) -> None:
        self._positions: List[Dict[str, Any]] = []
        self._current_price: Optional[float] = None

    def set_current_price(self, price: float) -> None:
        self._current_price = float(price)

    def get_open_positions(self, symbol: str) -> List[Dict[str, Any]]:
        return [p for p in self._positions if p.get("symbol") == symbol]

    def place_order(
        self,
        symbol: str,
        side: str,
        size: float,
        order_type: str = "market",
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        leverage: Optional[float] = None,
    ) -> Dict[str, Any]:
        position = {
            "symbol": symbol,
            "side": side,
            "size": size,
            "order_type": order_type,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "leverage": leverage,
            "entry_price": self._current_price,
        }
        self._positions.append(position)
        return {"status": "filled", "position": position}

    def cancel_all_orders(self, symbol: str) -> None:
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_price(candle: Dict[str, Any], key: str = "close", default: float = 0.0) -> float:
    return _safe_float(candle.get(key, default), default)


def _resolve_exit(position: Dict[str, Any], candle: Dict[str, Any]) -> Tuple[float, str]:
    """Decide exit price for a single-bar hold, honoring stop/take-profit if hit."""
    close_price = _safe_price(candle, "close", position.get("entry_price", 0.0))
    high = _safe_price(candle, "high", close_price)
    low = _safe_price(candle, "low", close_price)

    stop = position.get("stop_loss")
    tp = position.get("take_profit")
    side = str(position.get("side", "flat"))

    if side == "long":
        if stop is not None and low <= stop:
            return float(stop), "stop_loss"
        if tp is not None and high >= tp:
            return float(tp), "take_profit"
    elif side == "short":
        if stop is not None and high >= stop:
            return float(stop), "stop_loss"
        if tp is not None and low <= tp:
            return float(tp), "take_profit"

    return close_price, "bar_close"


def _compute_pnl(position: Dict[str, Any], exit_price: float) -> float:
    entry = _safe_float(position.get("entry_price"), 0.0)
    size = _safe_float(position.get("size"), 0.0)
    side = str(position.get("side", "flat"))
    if side == "short":
        return (entry - exit_price) * size
    return (exit_price - entry) * size


def _max_drawdown(equity_curve: List[float]) -> float:
    peak = equity_curve[0] if equity_curve else 0.0
    max_dd = 0.0
    for eq in equity_curve:
        if eq > peak:
            peak = eq
        if peak > 0:
            dd = (peak - eq) / peak
            if dd > max_dd:
                max_dd = dd
    return max_dd


def _load_json_candles(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "candles" in data:
        data = data["candles"]
    if not isinstance(data, list):
        raise ValueError("JSON file must contain a list of candle objects or a {\"candles\": [...]} wrapper.")
    return [dict(c) for c in data]


def _load_csv_candles(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [dict(row) for row in reader]


def load_candles(path: str | Path) -> List[Dict[str, Any]]:
    """Load candles from a JSON or CSV file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"No candle file at {p}")
    if p.suffix.lower() == ".json":
        return _load_json_candles(p)
    if p.suffix.lower() == ".csv":
        return _load_csv_candles(p)
    raise ValueError(f"Unsupported candle file type: {p.suffix}")


def generate_mock_candles(symbol: str, timeframe: str, limit: int = 300, start_price: float = 30_000.0) -> List[Dict[str, Any]]:
    adapter = MockDataAdapter(start_price=start_price)
    return adapter.fetch_recent_candles(symbol=symbol, timeframe=timeframe, limit=limit)


def _write_replay_exports(
    result: Dict[str, Any],
    config: Config,
    out_dir: str,
) -> None:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    symbol = str(getattr(config, "symbol", "unknown"))
    timeframe = str(getattr(config, "timeframe", "unknown"))

    trades_path = out_path / f"trades_{symbol}_{timeframe}.csv"
    equity_path = out_path / f"equity_{symbol}_{timeframe}.csv"

    trade_fields = [
        "side",
        "entry_price",
        "exit_price",
        "size",
        "entry_ts",
        "exit_ts",
        "pnl",
        "return_pct",
        "exit_reason",
    ]
    trades = result.get("trades") or []
    defaults = {
        "side": "",
        "entry_price": 0,
        "exit_price": 0,
        "size": 0,
        "entry_ts": 0,
        "exit_ts": 0,
        "pnl": 0,
        "return_pct": 0,
        "exit_reason": "",
    }

    with trades_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=trade_fields)
        writer.writeheader()
        for trade in trades:
            row = {field: defaults[field] for field in trade_fields}
            if isinstance(trade, dict):
                for field in trade_fields:
                    value = trade.get(field, defaults[field])
                    row[field] = value if value not in {None, ""} else defaults[field]
            writer.writerow(row)

    equity_curve = result.get("equity_curve") or []
    with equity_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "equity"])
        for idx, equity in enumerate(equity_curve):
            writer.writerow([idx, equity])


# ---------------------------------------------------------------------------
# GPT caller wrapper for replay (stub or real)
# ---------------------------------------------------------------------------


def _make_gpt_caller(use_stub: bool) -> Callable[[Config, Dict[str, Any], Dict[str, Any]], GptDecision]:
    """Create a GPT caller function for replay (either stub or real)."""
    if use_stub:
        def stub_caller(config: Config, snapshot: Dict[str, Any], state: Dict[str, Any]) -> GptDecision:
            return generate_stub_decision(snapshot)
        return stub_caller
    else:
        return call_gpt


# ---------------------------------------------------------------------------
# Core replay harness
# ---------------------------------------------------------------------------


def run_replay(config: Config, candles: List[Dict[str, Any]], use_gpt_stub: bool = False) -> Dict[str, Any]:
    """
    Run a full backtest loop over a list of candle dicts for one symbol/timeframe.

    Uses the shared run_spine_tick() function to ensure live/replay parity.
    The same gatekeeper → GPT → risk → execution path as live engine,
    but with a replay execution adapter and optional GPT stub.
    """
    state = load_state(config)
    # Start from a clean, in-memory state so replay does not mutate live files.
    state["symbol"] = config.symbol
    state["equity"] = _safe_float(state.get("equity", 10_000.0), 10_000.0)
    state["max_drawdown"] = 0.0
    state["open_positions_summary"] = []
    state["gpt_call_timestamps"] = []
    state["last_gpt_call_walltime"] = 0.0
    state["last_gpt_snapshot"] = None
    state["prev_snapshot"] = None
    state["snapshot_id"] = 0

    # Decide whether to use the deterministic GPT stub for this replay run.
    stub_active = bool(use_gpt_stub or not os.getenv("OPENAI_API_KEY"))
    if not use_gpt_stub and stub_active:
        logger.warning("OPENAI_API_KEY missing; using GPT stub for replay.")
    mode_msg = "GPT stub (deterministic)" if stub_active else "real GPT client"
    print(f"Replay GPT mode: {mode_msg}")

    # Create the appropriate GPT caller
    gpt_caller = _make_gpt_caller(stub_active)

    exec_adapter = ReplayExecutionAdapter()

    equity = float(state["equity"])
    equity_curve: List[float] = [equity]
    trades: List[Dict[str, Any]] = []
    open_position: Optional[Dict[str, Any]] = None
    prev_snapshot: Optional[Dict[str, Any]] = None
    parity_trace: List[Dict[str, Any]] = []

    for idx, candle in enumerate(candles):
        ts = _safe_float(candle.get("timestamp", idx), float(idx))

        # If we were in a trade, close it on this bar (1-bar hold with stop/TP checks).
        if open_position:
            exit_price, exit_reason = _resolve_exit(open_position, candle)
            pnl = _compute_pnl(open_position, exit_price)
            trade = {
                "side": open_position["side"],
                "entry_price": _safe_float(open_position.get("entry_price")),
                "exit_price": exit_price,
                "size": _safe_float(open_position.get("size")),
                "entry_ts": open_position.get("timestamp"),
                "exit_ts": ts,
                "pnl": pnl,
                "return_pct": pnl / open_position["equity_at_entry"] if open_position.get("equity_at_entry") else 0.0,
                "exit_reason": exit_reason,
            }
            trades.append(trade)
            equity += pnl
            state["open_positions_summary"] = []
            open_position = None

        # Update equity stats before generating the next snapshot.
        state["equity"] = equity
        state["snapshot_id"] = idx + 1

        market_data: Dict[str, Any] = {
            "candles": candles[: idx + 1],
            "funding": None,
            "open_interest": None,
            "skew": None,
        }

        snapshot = build_snapshot(config, market_data, state)
        exec_adapter.set_current_price(snapshot.get("price", 0.0))

        # --- Use shared spine tick for live/replay parity ---
        spine_result: SpineTickResult = run_spine_tick(
            snapshot=snapshot,
            prev_snapshot=prev_snapshot,
            state=state,
            config=config,
            exec_adapter=exec_adapter,
            gpt_caller=gpt_caller,
        )

        # Record parity trace entry using the standardized format
        parity_trace.append(spine_result.to_parity_trace_entry())

        # Handle position tracking based on spine result
        if spine_result.approved and spine_result.side != "flat":
            position = {
                "symbol": config.symbol,
                "side": spine_result.side,
                "size": spine_result.position_size,
                "entry_price": snapshot.get("price", 0.0),
                "timestamp": ts,
                "stop_loss": spine_result.stop_loss_price,
                "take_profit": spine_result.take_profit_price,
                "leverage": spine_result.leverage,
                "equity_at_entry": equity,
            }
            open_position = position
            state["open_positions_summary"] = [
                {
                    "symbol": config.symbol,
                    "side": position["side"],
                    "size": position["size"],
                    "entry_price": position["entry_price"],
                    "timestamp": ts,
                }
            ]
        elif spine_result.gatekeeper_called:
            # Gatekeeper called but trade not approved - clear positions
            state["open_positions_summary"] = []

        prev_snapshot = snapshot
        state["prev_snapshot"] = snapshot
        equity_curve.append(equity)

    # If we ended with an open trade, force-close it at the last seen price.
    if open_position and candles:
        final_exit_price, exit_reason = _resolve_exit(open_position, candles[-1])
        pnl = _compute_pnl(open_position, final_exit_price)
        trade = {
            "side": open_position["side"],
            "entry_price": _safe_float(open_position.get("entry_price")),
            "exit_price": final_exit_price,
            "size": _safe_float(open_position.get("size")),
            "entry_ts": open_position.get("timestamp"),
            "exit_ts": _safe_float(candles[-1].get("timestamp", len(candles) - 1)),
            "pnl": pnl,
            "return_pct": pnl / open_position["equity_at_entry"] if open_position.get("equity_at_entry") else 0.0,
            "exit_reason": exit_reason,
        }
        trades.append(trade)
        equity += pnl
        state["equity"] = equity
        state["open_positions_summary"] = []
        equity_curve.append(equity)
        open_position = None

    winners = sum(1 for t in trades if t["pnl"] > 0)
    losers = sum(1 for t in trades if t["pnl"] < 0)
    max_drawdown = _max_drawdown(equity_curve)

    stats = {
        "trades": len(trades),
        "winners": winners,
        "losers": losers,
        "win_rate": winners / len(trades) if trades else 0.0,
        "final_equity": equity,
        "max_drawdown": max_drawdown,
    }

    return {"equity_curve": equity_curve, "trades": trades, "stats": stats, "parity_trace": parity_trace}


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------


def _print_summary(stats: Dict[str, Any]) -> None:
    print("Replay complete.")
    print(f"Trades: {stats['trades']} | Win rate: {stats['win_rate'] * 100:.1f}%")
    print(f"Final equity: {stats['final_equity']:.2f}")
    print(f"Max drawdown: {stats['max_drawdown'] * 100:.2f}%")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TradeWarrior replay/backtest harness")
    parser.add_argument("--file", type=str, help="Path to candle JSON/CSV file")
    parser.add_argument(
        "--candles-csv",
        type=str,
        default=None,
        help="Path to a candles CSV (timestamp,open,high,low,close,volume).",
    )
    parser.add_argument("--limit", type=int, default=300, help="Number of candles to generate when using mock data")
    parser.add_argument("--start-price", type=float, default=30_000.0, help="Starting price for mock data generation")
    parser.add_argument("--stub", action="store_true", help="Use deterministic GPT stub instead of live GPT")
    parser.add_argument(
        "--out-dir",
        type=str,
        default="analytics",
        help="Directory to write replay exports (trades/equity).",
    )
    parser.add_argument(
        "--parity-out",
        type=str,
        default=None,
        help="Optional path to write replay parity trace as JSONL.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    cfg = load_config()

    candle_path = args.candles_csv or args.file
    if candle_path:
        candles = load_candles(candle_path)
    else:
        candles = generate_mock_candles(cfg.symbol, cfg.timeframe, limit=args.limit, start_price=args.start_price)

    result = run_replay(cfg, candles, use_gpt_stub=args.stub)
    if args.parity_out:
        parity_path = Path(args.parity_out)
        parity_path.parent.mkdir(parents=True, exist_ok=True)
        with parity_path.open("w", encoding="utf-8") as f:
            for entry in result.get("parity_trace", []):
                f.write(json.dumps(entry) + "\n")
        print(
            f"Wrote parity trace with {len(result.get('parity_trace', []))} entries "
            f"to {args.parity_out}"
        )
    _print_summary(result["stats"])
    _write_replay_exports(result, cfg, args.out_dir)
    print(f"Replay exports written to {args.out_dir}")
