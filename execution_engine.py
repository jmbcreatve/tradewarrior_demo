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
    raw_gpt_decision = state.get("last_gpt_decision")
    gpt_decision_dict = raw_gpt_decision if isinstance(raw_gpt_decision, dict) else {}

    def _log_execution_event(execution_result: Dict[str, Any]) -> None:
        try:
            # Get risk envelope from state if available
            risk_envelope = state.get("last_risk_envelope")
            
            # Calculate stop distance if we have stop_loss and can infer price
            stop_distance_pct = None
            if risk_decision.stop_loss_price and risk_decision.position_size > 0:
                # Try to infer price from execution result or use a placeholder
                fill_price = execution_result.get("fill_price") or execution_result.get("avg_fill_price")
                if fill_price and fill_price > 0:
                    if risk_decision.side == "long":
                        stop_distance_pct = abs((fill_price - risk_decision.stop_loss_price) / fill_price)
                    elif risk_decision.side == "short":
                        stop_distance_pct = abs((risk_decision.stop_loss_price - fill_price) / fill_price)
            
            event = {
                "type": "execution",
                "symbol": state.get("symbol") or execution_result.get("symbol"),
                "approved": risk_decision.approved,
                "side": risk_decision.side,
                "position_size": risk_decision.position_size,
                "leverage": risk_decision.leverage,
                "stop_loss_price": risk_decision.stop_loss_price,
                "take_profit_price": risk_decision.take_profit_price,
                "stop_distance_pct": stop_distance_pct,
                "reason": risk_decision.reason,
                "risk_envelope": risk_envelope,
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

    def _log_execution_trace(execution_result: Dict[str, Any], execution_status: str) -> None:
        try:
            gpt_side = gpt_decision_dict.get("side") or gpt_decision_dict.get("action")
            risk_envelope = state.get("last_risk_envelope")
            
            # Calculate stop distance if we have stop_loss and can infer price
            stop_distance_pct = None
            if risk_decision.stop_loss_price:
                fill_price = execution_result.get("fill_price") or execution_result.get("avg_fill_price")
                if fill_price and fill_price > 0:
                    if risk_decision.side == "long":
                        stop_distance_pct = abs((fill_price - risk_decision.stop_loss_price) / fill_price)
                    elif risk_decision.side == "short":
                        stop_distance_pct = abs((risk_decision.stop_loss_price - fill_price) / fill_price)
            
            event = {
                "type": "execution_trace",
                "symbol": state.get("symbol"),
                "run_id": state.get("run_id"),
                "snapshot_id": state.get("snapshot_id"),
                "approved": risk_decision.approved,
                "side": risk_decision.side,
                "position_size": risk_decision.position_size,
                "leverage": risk_decision.leverage,
                "stop_loss_price": risk_decision.stop_loss_price,
                "take_profit_price": risk_decision.take_profit_price,
                "stop_distance_pct": stop_distance_pct,
                "reason": risk_decision.reason,
                "risk_envelope": risk_envelope,
                "gpt_side": gpt_side,
                "gpt_confidence": gpt_decision_dict.get("confidence"),
                "execution_status": execution_status,
            }
            if execution_status != "skipped":
                event["fill_price"] = execution_result.get("fill_price") or execution_result.get("avg_fill_price")
                event["avg_fill_price"] = execution_result.get("avg_fill_price")
                event["fee_paid"] = execution_result.get("fee_paid")
                event["realized_pnl"] = execution_result.get("realized_pnl")

            logger.info("Execution trace: %s", event)
        except Exception:
            logger.info("Execution trace logging failed", exc_info=True)

    if not risk_decision.approved or risk_decision.side == "flat":
        logger.info("Execution: no trade (approved=%s, side=%s).", risk_decision.approved, risk_decision.side)
        result = {"status": "no_trade", "reason": risk_decision.reason}
        _log_execution_event(result)
        _log_execution_trace(result, "skipped")
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
    _log_execution_trace(result, result.get("status"))
    return result
