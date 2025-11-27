from typing import Iterable, Dict, Any, List

from config import Config
from build_features import build_snapshot
from gatekeeper import should_call_gpt
from gpt_client import call_gpt
from risk_engine import evaluate_risk
from execution_engine import execute_decision
from state_memory import load_state, save_state
from logger_utils import get_logger

logger = get_logger(__name__)


def replay_from_candles(config: Config, candle_stream: Iterable[List[Dict[str, Any]]]) -> Dict[str, Any]:
    """Very simple replay loop given an iterable of candle batches.

    Each element of candle_stream is a list of candle dicts compatible with the data adapters.
    This reuses the normal gatekeeper + GPT + risk flow, but uses a dummy execution layer.
    """
    state = load_state(config)
    prev_snapshot = state.get("prev_snapshot")
    equity = float(state.get("equity", 10_000.0) or 10_000.0)
    equity_curve: List[float] = [equity]

    for candles in candle_stream:
        market_data: Dict[str, Any] = {
            "candles": candles,
            "funding": None,
            "open_interest": None,
            "skew": None,
        }

        snapshot = build_snapshot(config, market_data, state)

        if not should_call_gpt(snapshot, prev_snapshot, state):
            prev_snapshot = snapshot
            continue

        gpt_decision = call_gpt(config, snapshot)
        state["last_gpt_decision"] = gpt_decision.to_dict()
        state["gpt_state_note"] = gpt_decision.notes

        risk_decision = evaluate_risk(snapshot, gpt_decision, state, config)

        # In replay we don't use a real execution adapter; we just log the risk decision.
        logger.info("Replay risk_decision: %s", risk_decision.to_dict())

        prev_snapshot = snapshot
        equity_curve.append(state.get("equity", equity_curve[-1]))

    save_state(state, config)
    return {"equity_curve": equity_curve}
