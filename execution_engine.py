from typing import Dict, Any

from config import Config
from schemas import RiskDecision
from adapters.base_execution_adapter import BaseExecutionAdapter
from logger_utils import get_logger

logger = get_logger(__name__)


def execute_decision(
    risk_decision: RiskDecision,
    config: Config,
    state: Dict[str, Any],
    execution_adapter: BaseExecutionAdapter,
) -> Dict[str, Any]:
    """Execute a RiskDecision via the given adapter.

    In paper_trading mode, this is expected to use a mock adapter only.
    """
    if not risk_decision.approved or risk_decision.side == "flat":
        logger.info("Execution: no trade (approved=%s, side=%s).", risk_decision.approved, risk_decision.side)
        return {"status": "no_trade", "reason": risk_decision.reason}

    result = execution_adapter.place_order(
        symbol=state.get("symbol", "BTCUSDT"),
        side=risk_decision.side,
        size=risk_decision.position_size,
        stop_loss=risk_decision.stop_loss_price,
        take_profit=risk_decision.take_profit_price,
        leverage=risk_decision.leverage,
    )

    logger.info("Execution result: %s", result)
    return result
