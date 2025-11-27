import time
from typing import Any, Dict

from config import Config, load_config
from data_router import build_data_adapters, build_execution_adapters, get_market_data
from build_features import build_snapshot
from gatekeeper import should_call_gpt
from gpt_client import call_gpt
from risk_engine import evaluate_risk
from execution_engine import execute_decision
from state_memory import load_state, save_state
from logger_utils import get_logger

logger = get_logger(__name__)


def run_once(config: Config | None = None) -> Dict[str, Any]:
    """Run a single end-to-end iteration of the pipeline.

    Returns a dict containing snapshot, gpt_decision, risk_decision, execution_result.
    """
    if config is None:
        config = load_config()

    data_adapters = build_data_adapters(config)
    exec_adapters = build_execution_adapters(config)

    state = load_state(config)
    # Ensure we always track the symbol we are trading.
    state.setdefault("symbol", config.symbol)

    market_data = get_market_data(config, data_adapters, limit=100)
    snapshot = build_snapshot(config, market_data, state)

    prev_snapshot = state.get("prev_snapshot")
    if not should_call_gpt(snapshot, prev_snapshot, state):
        logger.info("Engine: gatekeeper skipped GPT call.")
        state["prev_snapshot"] = snapshot
        save_state(state, config)
        return {
            "snapshot": snapshot,
            "gpt_decision": None,
            "risk_decision": None,
            "execution_result": None,
        }

    gpt_decision = call_gpt(config, snapshot)
    state["last_gpt_decision"] = gpt_decision.to_dict()
    # Persist a short note so GPT can keep some context between calls.
    state["gpt_state_note"] = gpt_decision.notes

    risk_decision = evaluate_risk(snapshot, gpt_decision, state, config)

    exec_adapter = exec_adapters.get(config.primary_execution_id)
    if exec_adapter is None:
        logger.warning(
            "Primary execution adapter %s not found; using mock.",
            config.primary_execution_id,
        )
        exec_adapter = exec_adapters["mock"]

    execution_result = execute_decision(risk_decision, config, state, exec_adapter)

    state["prev_snapshot"] = snapshot
    save_state(state, config)

    return {
        "snapshot": snapshot,
        "gpt_decision": gpt_decision.to_dict(),
        "risk_decision": risk_decision.to_dict(),
        "execution_result": execution_result,
    }


def run_forever(config: Config | None = None) -> None:
    """Simple infinite loop around run_once for DEMO/testing."""
    if config is None:
        config = load_config()

    while True:
        try:
            result = run_once(config)
            logger.info("Loop result: %s", result)
        except KeyboardInterrupt:
            logger.info("Received KeyboardInterrupt, stopping loop.")
            break
        except Exception as exc:  # noqa: BLE001
            logger.exception("Engine iteration failed: %s", exc)
        time.sleep(config.loop_sleep_seconds)


if __name__ == "__main__":
    # CLI entrypoint:
    # - `python engine.py`        -> loop forever (uses Config.loop_sleep_seconds between cycles)
    # - `python engine.py once`  -> run a single cycle and pretty-print the result
    import argparse
    from pprint import pprint

    parser = argparse.ArgumentParser(description="TradeWarrior demo engine")
    parser.add_argument(
        "mode",
        nargs="?",
        choices=["once", "loop"],
        default="loop",
        help="run mode: 'once' (single cycle) or 'loop' (default: loop forever)",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=None,
        help="override Config.loop_sleep_seconds (seconds between cycles in loop mode)",
    )

    args = parser.parse_args()
    cfg = load_config()
    if args.sleep is not None:
        cfg.loop_sleep_seconds = float(args.sleep)

    if args.mode == "once":
        pprint(run_once(cfg))
    else:
        run_forever(cfg)
