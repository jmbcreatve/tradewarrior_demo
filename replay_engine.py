from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from adapters.base_execution_adapter import BaseExecutionAdapter
from adapters.mock_data_adapter import MockDataAdapter
from build_features import build_snapshot
from config import Config, load_config
from execution_engine import execute_decision
from gatekeeper import should_call_gpt
from gpt_client import call_gpt
from logger_utils import get_logger
from replay_gpt_stub import generate_stub_decision
from risk_engine import evaluate_risk
from state_memory import load_state

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


# ---------------------------------------------------------------------------
# Core replay harness
# ---------------------------------------------------------------------------


def run_replay(config: Config, candles: List[Dict[str, Any]], use_gpt_stub: bool = False) -> Dict[str, Any]:
    """
    Run a full backtest loop over a list of candle dicts for one symbol/timeframe.

    Reuses the same build_snapshot -> gatekeeper -> GPT -> risk -> execution path
    as the live engine, but executes fills locally and tracks equity per step.
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

    # Decide whether to use the deterministic GPT stub for this replay run.
    stub_active = bool(use_gpt_stub or not os.getenv("OPENAI_API_KEY"))
    if not use_gpt_stub and stub_active:
        logger.warning("OPENAI_API_KEY missing; using GPT stub for replay.")
    mode_msg = "GPT stub (deterministic)" if stub_active else "real GPT client"
    print(f"Replay GPT mode: {mode_msg}")

    exec_adapter = ReplayExecutionAdapter()

    equity = float(state["equity"])
    equity_curve: List[float] = [equity]
    trades: List[Dict[str, Any]] = []
    open_position: Optional[Dict[str, Any]] = None
    prev_snapshot: Optional[Dict[str, Any]] = None

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

        market_data: Dict[str, Any] = {
            "candles": candles[: idx + 1],
            "funding": None,
            "open_interest": None,
            "skew": None,
        }

        snapshot = build_snapshot(config, market_data, state)

        if not should_call_gpt(snapshot, prev_snapshot, state):
            prev_snapshot = snapshot
            equity_curve.append(equity)
            continue

        if stub_active:
            gpt_decision = generate_stub_decision(snapshot)
        else:
            gpt_decision = call_gpt(config, snapshot)
        state["last_gpt_decision"] = gpt_decision.to_dict()
        state["gpt_state_note"] = gpt_decision.notes
        state["last_action"] = gpt_decision.action
        state["last_confidence"] = gpt_decision.confidence

        risk_decision = evaluate_risk(snapshot, gpt_decision, state, config)

        exec_adapter.set_current_price(snapshot.get("price", 0.0))
        execution_result = execute_decision(risk_decision, config, state, exec_adapter)
        logger.info("Replay execution result: %s", execution_result)

        if risk_decision.approved and risk_decision.side != "flat":
            position = {
                "symbol": config.symbol,
                "side": risk_decision.side,
                "size": risk_decision.position_size,
                "entry_price": snapshot.get("price", 0.0),
                "timestamp": ts,
                "stop_loss": risk_decision.stop_loss_price,
                "take_profit": risk_decision.take_profit_price,
                "leverage": risk_decision.leverage,
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
        else:
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

    return {"equity_curve": equity_curve, "trades": trades, "stats": stats}


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
    parser.add_argument("--limit", type=int, default=300, help="Number of candles to generate when using mock data")
    parser.add_argument("--start-price", type=float, default=30_000.0, help="Starting price for mock data generation")
    parser.add_argument("--stub", action="store_true", help="Use deterministic GPT stub instead of live GPT")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    cfg = load_config()

    if args.file:
        candles = load_candles(args.file)
    else:
        candles = generate_mock_candles(cfg.symbol, cfg.timeframe, limit=args.limit, start_price=args.start_price)

    result = run_replay(cfg, candles, use_gpt_stub=args.stub)
    _print_summary(result["stats"])
