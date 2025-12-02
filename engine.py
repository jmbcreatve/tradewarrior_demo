import sys
import time
import uuid
from pprint import pprint
from typing import Any, Dict

from config import Config, load_config
from data_router import (
    build_data_adapters,
    build_execution_adapters,
    get_market_data,
)
from build_features import build_snapshot
from gatekeeper import should_call_gpt
from gpt_client import call_gpt
from risk_engine import evaluate_risk
from execution_engine import execute_decision
from state_memory import load_state, save_state, reset_state
from logger_utils import get_logger

RUN_ID = uuid.uuid4().hex
logger = get_logger(__name__)


def _log_config_summary(cfg: Config) -> None:
    summary = {
        "symbol": cfg.symbol,
        "timeframe": cfg.timeframe,
        "execution_mode": str(getattr(cfg, "execution_mode", "")),
        "risk_per_trade": cfg.risk_per_trade,
        "max_leverage": cfg.max_leverage,
        "loop_sleep_seconds": cfg.loop_sleep_seconds,
    }
    logger.info("Config summary: %s", summary)


def _run_tick(cfg: Config, state: Dict[str, Any], dry_run: bool) -> Dict[str, Any]:
    """
    Run one full pipeline tick: state -> data -> snapshot -> gatekeeper -> GPT -> risk -> (maybe) execution -> state.
    Mutates and returns state.
    """
    t0 = time.perf_counter()
    if not state.get("run_id"):
        state["run_id"] = RUN_ID
    state.setdefault("symbol", cfg.symbol)

    data_adapters = build_data_adapters(cfg)
    exec_adapters = build_execution_adapters(cfg)

    market_data = get_market_data(cfg, data_adapters, limit=100)

    prev_snapshot_id = int(state.get("snapshot_id", 0) or 0)
    snapshot_id = prev_snapshot_id + 1
    state["snapshot_id"] = snapshot_id

    try:
        snapshot = build_snapshot(cfg, market_data, state)
        t_snap = time.perf_counter()
    except Exception:
        state["snapshot_id"] = prev_snapshot_id
        logger.critical("Snapshot build or validation failed", exc_info=True)
        return state

    prev_snapshot = state.get("prev_snapshot")
    if not should_call_gpt(snapshot, prev_snapshot, state):
        logger.info("Engine: gatekeeper skipped GPT call.")
        t_decision = t_snap
        dt_snapshot = t_snap - t0
        dt_decision = t_decision - t_snap
        dt_risk = 0.0
        dt_execution = 0.0
        context = {
            "run_id": state.get("run_id"),
            "snapshot_id": state.get("snapshot_id"),
            "symbol": state.get("symbol"),
            "mode": str(getattr(cfg, "execution_mode", "")),
            "dt_snapshot": dt_snapshot,
            "dt_decision": dt_decision,
            "dt_risk": dt_risk,
            "dt_execution": dt_execution,
        }
        state["prev_snapshot"] = snapshot
        save_state(state, cfg)
        logger.info("Tick summary: %s", context)
        return state

    gpt_decision = call_gpt(cfg, snapshot)
    t_decision = time.perf_counter()
    state["last_gpt_decision"] = gpt_decision.to_dict()
    state["gpt_state_note"] = gpt_decision.notes

    risk_decision = evaluate_risk(snapshot, gpt_decision, state, cfg)
    t_risk = time.perf_counter()
    state["last_risk_decision"] = risk_decision.to_dict()

    exec_adapter = exec_adapters.get(cfg.primary_execution_id)
    if exec_adapter is None:
        logger.warning(
            "Primary execution adapter %s not found; using mock.",
            cfg.primary_execution_id,
        )
        exec_adapter = exec_adapters["mock"]

    if dry_run:
        execution_result: Dict[str, Any] = {
            "status": "dry_run",
            "side": risk_decision.side,
            "position_size": risk_decision.position_size,
            "leverage": risk_decision.leverage,
            "stop_loss_price": risk_decision.stop_loss_price,
            "take_profit_price": risk_decision.take_profit_price,
            "reason": risk_decision.reason,
        }
        logger.info(
            "Dry run: would execute symbol=%s side=%s size=%s leverage=%s stop_loss=%s take_profit=%s",
            state.get("symbol", cfg.symbol),
            risk_decision.side,
            risk_decision.position_size,
            risk_decision.leverage,
            risk_decision.stop_loss_price,
            risk_decision.take_profit_price,
        )
    else:
        execution_result = execute_decision(risk_decision, cfg, state, exec_adapter)

    t_exec = time.perf_counter()

    dt_snapshot = t_snap - t0
    dt_decision = t_decision - t_snap
    dt_risk = t_risk - t_decision
    dt_execution = t_exec - t_risk

    state["last_execution_result"] = execution_result
    state["prev_snapshot"] = snapshot
    save_state(state, cfg)
    context = {
        "run_id": state.get("run_id"),
        "snapshot_id": state.get("snapshot_id"),
        "symbol": state.get("symbol"),
        "mode": str(getattr(cfg, "execution_mode", "")),
        "dt_snapshot": dt_snapshot,
        "dt_decision": dt_decision,
        "dt_risk": dt_risk,
        "dt_execution": dt_execution,
    }
    logger.info("Tick summary: %s", context)
    return state


def run_once(
    config: Config | None = None,
    *,
    state: Dict[str, Any] | None = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    Run a single end-to-end iteration of the pipeline.
    Returns a dict containing snapshot, gpt_decision, risk_decision, execution_result.
    """
    cfg = config or load_config()
    if state is None:
        state = load_state(cfg)

    try:
        state = _run_tick(cfg, state, dry_run)
    except Exception:
        logger.exception("Unhandled error in engine tick")
        raise

    return {
        "snapshot": state.get("prev_snapshot"),
        "gpt_decision": state.get("last_gpt_decision"),
        "risk_decision": state.get("last_risk_decision"),
        "execution_result": state.get("last_execution_result"),
    }


def run_forever(
    config: Config | None = None,
    *,
    state: Dict[str, Any] | None = None,
    dry_run: bool = False,
) -> None:
    """Simple infinite loop around _run_tick for DEMO/testing."""
    cfg = config or load_config()
    if state is None:
        state = load_state(cfg)

    while True:
        try:
            state = _run_tick(cfg, state, dry_run)
        except KeyboardInterrupt:
            logger.info("Received KeyboardInterrupt, stopping loop.")
            break
        except Exception:
            logger.exception("Unhandled error in engine tick")
            continue
        time.sleep(cfg.loop_sleep_seconds)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TradeWarrior demo engine")
    parser.add_argument(
        "--mode",
        choices=["once", "loop"],
        default="once",
        help="run mode: 'once' (single cycle) or 'loop' (keep running)",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=None,
        help="override Config.loop_sleep_seconds (seconds between cycles in loop mode)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="optional path to a config file",
    )
    parser.add_argument(
        "--reset-state",
        action="store_true",
        help="start from a fresh in-memory state",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="compute GPT + risk but skip actual execution",
    )

    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.sleep is not None:
        cfg.loop_sleep_seconds = float(args.sleep)

    _log_config_summary(cfg)
    initial_state = reset_state(cfg) if args.reset_state else load_state(cfg)

    if args.mode == "once":
        try:
            result = run_once(cfg, state=initial_state, dry_run=args.dry_run)
            pprint(result)
        except Exception:
            logger.exception("Unhandled error in engine tick")
            sys.exit(1)
    else:
        run_forever(cfg, state=initial_state, dry_run=args.dry_run)
