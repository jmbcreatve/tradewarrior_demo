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
    def _log_execution_event(execution_result: Dict[str, Any]) -> None:
        try:
            event = {
                "type": "execution",
                "symbol": state.get("symbol") or execution_result.get("symbol"),
                "side": risk_decision.side,
                "position_size": risk_decision.position_size,
                "leverage": risk_decision.leverage,
                "stop_loss_price": risk_decision.stop_loss_price,
                "take_profit_price": risk_decision.take_profit_price,
                "status": execution_result.get("status"),
                "fill_price": execution_result.get("fill_price") or execution_result.get("avg_fill_price"),
                "avg_fill_price": execution_result.get("avg_fill_price"),
                "fee_paid": execution_result.get("fee_paid"),
                "realized_pnl": execution_result.get("realized_pnl"),
                "position": execution_result.get("position_summary") or execution_result.get("position"),
            }
            logger.info("Execution event: %s", event)
        except Exception:
            logger.info("Execution event logging failed", exc_info=True)

    if not risk_decision.approved or risk_decision.side == "flat":
        logger.info("Execution: no trade (approved=%s, side=%s).", risk_decision.approved, risk_decision.side)
        result = {"status": "no_trade", "reason": risk_decision.reason}
        _log_execution_event(result)
        return result

    result = execution_adapter.place_order(
        symbol=state.get("symbol", "BTCUSDT"),
        side=risk_decision.side,
        size=risk_decision.position_size,
        stop_loss=risk_decision.stop_loss_price,
        take_profit=risk_decision.take_profit_price,
        leverage=risk_decision.leverage,
    )

    logger.info("Execution result: %s", result)
    _log_execution_event(result)
    return result
