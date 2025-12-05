import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pprint import pprint
from typing import Any, Dict, Optional, Callable

from config import Config, load_config
from data_router import (
    build_data_adapters,
    build_execution_adapters,
    get_market_data,
)
from build_features import build_snapshot
from gatekeeper import should_call_gpt
from gpt_client import call_gpt, validate_openai_api_key, create_safe_mode_decision
from risk_engine import evaluate_risk, _check_daily_loss_limit
from execution_engine import execute_decision
from state_memory import load_state, save_state, reset_state, _reset_daily_tracking, is_gpt_safe_mode
from safety_utils import check_trading_halted
from logger_utils import get_logger
from schemas import GptDecision, RiskDecision
from adapters.base_execution_adapter import BaseExecutionAdapter

RUN_ID = uuid.uuid4().hex
logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Shared Tick Result (for replay/live parity)
# ---------------------------------------------------------------------------

@dataclass
class SpineTickResult:
    """
    Result of a single spine tick (snapshot → gatekeeper → GPT → risk → execution).
    
    Used by both live engine and replay to ensure parity. All fields that matter
    for decision auditing are captured here.
    """
    # Core snapshot info
    timestamp: float = 0.0
    price: float = 0.0
    snapshot_id: int = 0
    
    # Gatekeeper result
    gatekeeper_called: bool = False
    gatekeeper_reason: str = "not_called"
    
    # GPT decision (None if gatekeeper skipped)
    gpt_decision: Optional[GptDecision] = None
    gpt_action: str = "flat"
    gpt_confidence: float = 0.0
    
    # Risk decision
    risk_decision: Optional[RiskDecision] = None
    approved: bool = False
    side: str = "flat"
    position_size: float = 0.0
    leverage: float = 0.0
    stop_loss_price: Optional[float] = None
    take_profit_price: Optional[float] = None
    risk_reason: str = ""
    
    # Risk envelope summary (compact)
    risk_envelope_summary: Dict[str, Any] = field(default_factory=dict)
    
    # Execution result
    execution_status: str = "skipped"
    fill_price: Optional[float] = None
    realized_pnl: Optional[float] = None
    
    # Flags
    safe_mode_active: bool = False
    
    def to_parity_trace_entry(self) -> Dict[str, Any]:
        """Convert to a dict suitable for parity trace logging."""
        return {
            "timestamp": self.timestamp,
            "price": self.price,
            "snapshot_id": self.snapshot_id,
            "gatekeeper_called": self.gatekeeper_called,
            "gatekeeper_reason": self.gatekeeper_reason,
            "gpt_action": self.gpt_action,
            "gpt_confidence": self.gpt_confidence,
            "approved": self.approved,
            "side": self.side,
            "position_size": self.position_size,
            "leverage": self.leverage,
            "stop_loss_price": self.stop_loss_price,
            "take_profit_price": self.take_profit_price,
            "risk_reason": self.risk_reason,
            "risk_envelope": self.risk_envelope_summary,
            "execution_status": self.execution_status,
            "fill_price": self.fill_price,
            "realized_pnl": self.realized_pnl,
            "safe_mode_active": self.safe_mode_active,
        }


# ---------------------------------------------------------------------------
# Shared Spine Tick Function (live and replay parity)
# ---------------------------------------------------------------------------

def run_spine_tick(
    snapshot: Dict[str, Any],
    prev_snapshot: Optional[Dict[str, Any]],
    state: Dict[str, Any],
    config: Config,
    exec_adapter: BaseExecutionAdapter,
    gpt_caller: Optional[Callable[[Config, Dict[str, Any], Dict[str, Any]], GptDecision]] = None,
) -> SpineTickResult:
    """
    Run the core trading spine: gatekeeper → GPT → risk → execution.
    
    This is the shared single-tick function used by both live engine and replay
    to ensure decision parity. The only difference is the GPT caller and execution
    adapter provided.
    
    Args:
        snapshot: Current market snapshot dict.
        prev_snapshot: Previous snapshot (for gatekeeper comparison).
        state: Mutable state dict (will be updated with GPT/risk decisions).
        config: Runtime configuration.
        exec_adapter: Execution adapter for placing orders.
        gpt_caller: Optional GPT caller function. If None, uses call_gpt.
                    For replay with stub, pass generate_stub_decision.
    
    Returns:
        SpineTickResult with all decision info for parity tracing.
    """
    if gpt_caller is None:
        gpt_caller = call_gpt
    
    result = SpineTickResult(
        timestamp=snapshot.get("timestamp", 0.0),
        price=snapshot.get("price", 0.0),
        snapshot_id=snapshot.get("snapshot_id", 0),
    )
    
    # Extract risk envelope summary for parity trace
    risk_env = snapshot.get("risk_envelope") or {}
    result.risk_envelope_summary = {
        "max_notional": risk_env.get("max_notional"),
        "max_leverage": risk_env.get("max_leverage"),
        "max_risk_pct": risk_env.get("max_risk_per_trade_pct"),
        "max_daily_loss_pct": risk_env.get("max_daily_loss_pct"),
        "note": risk_env.get("note"),
    }
    
    # --- Check GPT Safe Mode ---
    if is_gpt_safe_mode(state):
        result.safe_mode_active = True
        result.gatekeeper_reason = "safe_mode_active"
        
        gpt_decision = create_safe_mode_decision("gpt_safe_mode_active")
        result.gpt_decision = gpt_decision
        result.gpt_action = gpt_decision.action
        result.gpt_confidence = gpt_decision.confidence
        
        state["last_gpt_decision"] = gpt_decision.to_dict()
        state["gpt_state_note"] = gpt_decision.notes
        
        risk_decision = evaluate_risk(snapshot, gpt_decision, state, config)
        result.risk_decision = risk_decision
        result.approved = risk_decision.approved
        result.side = risk_decision.side
        result.position_size = risk_decision.position_size
        result.leverage = risk_decision.leverage
        result.stop_loss_price = risk_decision.stop_loss_price
        result.take_profit_price = risk_decision.take_profit_price
        result.risk_reason = risk_decision.reason
        
        state["last_risk_decision"] = risk_decision.to_dict()
        result.execution_status = "safe_mode_flat"
        state["last_action"] = risk_decision.side if risk_decision.approved else "flat"
        state["last_confidence"] = 0.0
        
        logger.warning(
            "SpineTick: GPT SAFE MODE ACTIVE - skipping GPT, forcing FLAT."
        )
        return result
    
    # --- Gatekeeper check ---
    gatekeeper_result = should_call_gpt(snapshot, prev_snapshot, state)
    result.gatekeeper_called = gatekeeper_result.get("should_call_gpt", False)
    result.gatekeeper_reason = gatekeeper_result.get("reason", "unknown")
    
    if not result.gatekeeper_called:
        result.execution_status = "gatekeeper_skipped"
        logger.info("SpineTick: gatekeeper skipped GPT (reason=%s).", result.gatekeeper_reason)
        return result
    
    # --- GPT call ---
    gpt_decision = gpt_caller(config, snapshot, state)
    result.gpt_decision = gpt_decision
    result.gpt_action = gpt_decision.action
    result.gpt_confidence = gpt_decision.confidence
    
    state["last_gpt_decision"] = gpt_decision.to_dict()
    state["gpt_state_note"] = gpt_decision.notes
    try:
        state["last_gpt_call_ts"] = float(snapshot.get("timestamp") or time.time())
    except (TypeError, ValueError):
        state["last_gpt_call_ts"] = 0.0
    try:
        equity_for_gpt = float(state.get("equity", getattr(config, "initial_equity", 0.0)) or getattr(config, "initial_equity", 0.0))
    except (TypeError, ValueError):
        equity_for_gpt = getattr(config, "initial_equity", 0.0)
    state["last_gpt_equity"] = equity_for_gpt
    state["trades_since_last_gpt"] = 0
    state["last_gpt_snapshot"] = snapshot
    
    # --- Risk evaluation ---
    risk_decision = evaluate_risk(snapshot, gpt_decision, state, config)
    result.risk_decision = risk_decision
    result.approved = risk_decision.approved
    result.side = risk_decision.side
    result.position_size = risk_decision.position_size
    result.leverage = risk_decision.leverage
    result.stop_loss_price = risk_decision.stop_loss_price
    result.take_profit_price = risk_decision.take_profit_price
    result.risk_reason = risk_decision.reason
    
    state["last_risk_decision"] = risk_decision.to_dict()
    
    # --- Execution ---
    execution_result = execute_decision(risk_decision, config, state, exec_adapter)
    result.execution_status = execution_result.get("status", "unknown")
    result.fill_price = execution_result.get("fill_price") or execution_result.get("avg_fill_price")
    result.realized_pnl = execution_result.get("realized_pnl")
    exec_status = str(result.execution_status or "").lower()
    non_trade_statuses = {"no_trade", "skipped", "gatekeeper_skipped", "safe_mode_flat", "dry_run"}
    trade_executed = (
        risk_decision.approved
        and risk_decision.position_size > 0
        and exec_status not in non_trade_statuses
    )
    if trade_executed:
        try:
            trades_since = int(state.get("trades_since_last_gpt", 0) or 0)
        except (TypeError, ValueError):
            trades_since = 0
        state["trades_since_last_gpt"] = trades_since + 1
        state["last_action"] = risk_decision.side
        try:
            state["last_confidence"] = float(getattr(gpt_decision, "confidence", 0.0) or 0.0)
        except (TypeError, ValueError):
            state["last_confidence"] = 0.0
    else:
        state["last_action"] = "flat"
        state["last_confidence"] = 0.0
    
    logger.info(
        "SpineTick: gpt_action=%s, approved=%s, side=%s, size=%.6f, status=%s",
        result.gpt_action,
        result.approved,
        result.side,
        result.position_size,
        result.execution_status,
    )
    
    return result


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
    Run one full pipeline tick: state -> data -> snapshot -> spine (gatekeeper → GPT → risk → execution) -> state.
    
    Uses the shared run_spine_tick() function to ensure live/replay parity.
    Mutates and returns state.
    """
    t0 = time.perf_counter()
    if not state.get("run_id"):
        state["run_id"] = RUN_ID
    state.setdefault("symbol", cfg.symbol)

    # --- Safety checks: kill switch and circuit breaker -----------------------
    is_halted, halt_reason = check_trading_halted(state)
    if is_halted:
        logger.warning(
            "Engine: Trading halted (reason=%s). Skipping tick.",
            halt_reason,
        )
        # Still save state and return
        save_state(state, cfg)
        return state

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

    # --- Select execution adapter ---
    exec_adapter = exec_adapters.get(cfg.primary_execution_id)
    if exec_adapter is None:
        logger.warning(
            "Primary execution adapter %s not found; using mock.",
            cfg.primary_execution_id,
        )
        exec_adapter = exec_adapters["mock"]

    # --- Dry run uses a no-op adapter wrapper ---
    class DryRunAdapter(BaseExecutionAdapter):
        """Wrapper that logs but doesn't execute."""
        def __init__(self, inner: BaseExecutionAdapter):
            self._inner = inner
            
        def get_open_positions(self, symbol: str):
            return self._inner.get_open_positions(symbol)
            
        def place_order(self, symbol, side, size, order_type="market", stop_loss=None, take_profit=None, leverage=None):
            logger.info(
                "Dry run: would execute symbol=%s side=%s size=%s leverage=%s stop_loss=%s take_profit=%s",
                symbol, side, size, leverage, stop_loss, take_profit,
            )
            return {
                "status": "dry_run",
                "side": side,
                "position_size": size,
                "leverage": leverage,
                "stop_loss_price": stop_loss,
                "take_profit_price": take_profit,
            }
            
        def cancel_all_orders(self, symbol: str):
            return self._inner.cancel_all_orders(symbol)

    actual_adapter = DryRunAdapter(exec_adapter) if dry_run else exec_adapter

    # --- Run the shared spine tick ---
    spine_result = run_spine_tick(
        snapshot=snapshot,
        prev_snapshot=prev_snapshot,
        state=state,
        config=cfg,
        exec_adapter=actual_adapter,
        gpt_caller=call_gpt,
    )
    
    t_spine = time.perf_counter()

    # --- Update equity from execution results --------------------------------
    if spine_result.realized_pnl is not None:
        try:
            pnl_value = float(spine_result.realized_pnl)
            old_equity = state.get("equity", 0.0)
            state["equity"] = old_equity + pnl_value

            # Update daily P&L
            daily_start_equity = state.get("daily_start_equity")
            if daily_start_equity is not None:
                state["daily_pnl"] = state["equity"] - daily_start_equity
                logger.info(
                    "Equity updated: %.2f -> %.2f (pnl=%.2f, daily_pnl=%.2f)",
                    old_equity,
                    state["equity"],
                    pnl_value,
                    state["daily_pnl"],
                )
            else:
                logger.info(
                    "Equity updated: %.2f -> %.2f (pnl=%.2f)",
                    old_equity,
                    state["equity"],
                    pnl_value,
                )
        except (TypeError, ValueError):
            logger.warning("Invalid realized_pnl in spine result: %s", spine_result.realized_pnl)

    # --- Check daily loss limit after execution -------------------------------
    daily_pnl = state.get("daily_pnl", 0.0)
    daily_start_equity = state.get("daily_start_equity")
    if daily_start_equity is not None:
        risk_env = state.get("last_risk_envelope")
        if risk_env and isinstance(risk_env, dict):
            max_daily_loss_pct = risk_env.get("max_daily_loss_pct", 0.03)
            if _check_daily_loss_limit(daily_pnl, daily_start_equity, max_daily_loss_pct):
                state["trading_halted"] = True
                daily_pnl_pct = (daily_pnl / daily_start_equity) * 100.0 if daily_start_equity > 0 else 0.0
                logger.critical(
                    "🚨 CIRCUIT BREAKER ACTIVATED: Daily loss limit exceeded "
                    "(daily_pnl=%.2f, daily_pnl_pct=%.2f%%, max=%.2f%%). Trading halted.",
                    daily_pnl,
                    daily_pnl_pct,
                    max_daily_loss_pct * 100.0,
                )

    t_exec = time.perf_counter()

    dt_snapshot = t_snap - t0
    dt_spine = t_spine - t_snap
    dt_post = t_exec - t_spine

    # Build execution result dict from spine result for state storage
    execution_result: Dict[str, Any] = {
        "status": spine_result.execution_status,
        "side": spine_result.side,
        "position_size": spine_result.position_size,
        "fill_price": spine_result.fill_price,
        "realized_pnl": spine_result.realized_pnl,
    }

    state["last_execution_result"] = execution_result
    state["prev_snapshot"] = snapshot
    save_state(state, cfg)
    
    context = {
        "run_id": state.get("run_id"),
        "snapshot_id": state.get("snapshot_id"),
        "symbol": state.get("symbol"),
        "mode": str(getattr(cfg, "execution_mode", "")),
        "gpt_safe_mode": spine_result.safe_mode_active,
        "gatekeeper_reason": spine_result.gatekeeper_reason,
        "gpt_action": spine_result.gpt_action,
        "approved": spine_result.approved,
        "dt_snapshot": dt_snapshot,
        "dt_spine": dt_spine,
        "dt_post": dt_post,
    }
    logger.info("Tick summary: %s", context)
    return state


def _initialize_daily_tracking(state: Dict[str, Any]) -> None:
    """Initialize or reset daily tracking if it's a new day."""
    now = time.time()
    daily_start_ts = state.get("daily_start_timestamp")

    # Check if we need to reset (None or new day)
    reset_needed = False
    if daily_start_ts is None:
        reset_needed = True
    else:
        # Check if it's a new UTC day
        try:
            start_date = datetime.fromtimestamp(daily_start_ts, tz=timezone.utc).date()
            current_date = datetime.fromtimestamp(now, tz=timezone.utc).date()
            if start_date != current_date:
                reset_needed = True
        except (ValueError, OSError):
            # Invalid timestamp, reset
            reset_needed = True

    if reset_needed:
        current_equity = state.get("equity", 0.0)
        _reset_daily_tracking(state, current_equity, now)


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

    # Initialize daily tracking if needed
    _initialize_daily_tracking(state)

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

    # Initialize daily tracking if needed
    _initialize_daily_tracking(state)

    while True:
        try:
            # Re-initialize daily tracking at start of each loop iteration
            # (in case day changed during sleep)
            _initialize_daily_tracking(state)
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
    
    # Validate OpenAI API key (fails fast if missing)
    try:
        validate_openai_api_key()
    except RuntimeError as e:
        logger.error(f"❌ {e}")
        logger.error("Please set OPENAI_API_KEY in environment before running engine.")
        sys.exit(1)
    
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
