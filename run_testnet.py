#!/usr/bin/env python3
"""
TradeWarrior Testnet Runner

This script starts TradeWarrior configured for Hyperliquid testnet trading.
It provides clear logging that the system is running in TESTNET mode with play money.

Usage:
    python run_testnet.py              # Run continuously (default)
    python run_testnet.py --once       # Run single tick (for testing)
    python run_testnet.py --dry-run    # Compute decisions but skip execution
    python run_testnet.py --reset      # Start with fresh state

Required environment:
    - twp3_testnet.env file with:
        HL_TESTNET_PRIVATE_KEY=0x...
    
    - (Optional) local_env.txt or environment with:
        OPENAI_API_KEY=sk-...
"""

from __future__ import annotations

import argparse
import os
import sys
from pprint import pprint

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import load_testnet_config, _load_env_file
from engine import _run_tick, _initialize_daily_tracking
from state_memory import load_state, reset_state, save_state
from risk_envelope import compute_risk_envelope
from enums import VolatilityMode, TimingState
from safety_utils import KILL_SWITCH_FILE, check_trading_halted
from logger_utils import get_logger
from gpt_client import validate_openai_api_key
import time

logger = get_logger(__name__)


# ===========================================================================
# TESTNET BANNER
# ===========================================================================

TESTNET_BANNER = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║   ████████╗███████╗███████╗████████╗███╗   ██╗███████╗████████╗             ║
║   ╚══██╔══╝██╔════╝██╔════╝╚══██╔══╝████╗  ██║██╔════╝╚══██╔══╝             ║
║      ██║   █████╗  ███████╗   ██║   ██╔██╗ ██║█████╗     ██║                ║
║      ██║   ██╔══╝  ╚════██║   ██║   ██║╚██╗██║██╔══╝     ██║                ║
║      ██║   ███████╗███████║   ██║   ██║ ╚████║███████╗   ██║                ║
║      ╚═╝   ╚══════╝╚══════╝   ╚═╝   ╚═╝  ╚═══╝╚══════╝   ╚═╝                ║
║                                                                              ║
║   ⚠️  HYPERLIQUID TESTNET MODE - PLAY MONEY ONLY ⚠️                          ║
║                                                                              ║
║   • Exchange: Hyperliquid Testnet                                            ║
║   • API Endpoint: testnet.hyperliquid.xyz                                    ║
║   • Real orders will be placed on TESTNET (not mainnet)                      ║
║   • This uses PLAY MONEY - no real funds at risk                             ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""


def _log_config_summary(cfg, state) -> None:
    """Log key configuration values for visibility."""
    # Compute risk envelope to get max_daily_loss_pct
    equity = state.get("equity", cfg.initial_equity)
    risk_env = compute_risk_envelope(
        cfg,
        equity,
        VolatilityMode.NORMAL,
        danger_mode=False,
        timing_state=TimingState.NORMAL,
    )

    # Get daily P&L info
    daily_pnl = state.get("daily_pnl", 0.0)
    daily_start_equity = state.get("daily_start_equity")
    daily_pnl_pct = None
    if daily_start_equity is not None and daily_start_equity > 0:
        daily_pnl_pct = (daily_pnl / daily_start_equity) * 100.0

    summary = {
        "mode": "TESTNET",
        "is_testnet": cfg.is_testnet,
        "execution_mode": str(cfg.execution_mode),
        "symbol": cfg.symbol,
        "timeframe": cfg.timeframe,
        "initial_equity": f"${cfg.initial_equity:,.2f}",
        "current_equity": f"${equity:,.2f}",
        "risk_per_trade": f"{cfg.risk_per_trade * 100:.1f}%",
        "max_leverage": f"{cfg.max_leverage}x",
        "max_daily_loss_pct": f"{risk_env.max_daily_loss_pct * 100:.1f}%",
        "daily_pnl": f"${daily_pnl:,.2f}" + (f" ({daily_pnl_pct:+.2f}%)" if daily_pnl_pct is not None else ""),
        "primary_data_source": cfg.primary_data_source_id,
        "primary_execution": cfg.primary_execution_id,
        "state_file": cfg.state_file,
        "kill_switch_file": KILL_SWITCH_FILE,
    }
    
    logger.info("=" * 60)
    logger.info("TESTNET Configuration Summary:")
    for key, value in summary.items():
        logger.info(f"  {key}: {value}")
    logger.info("=" * 60)


def _log_periodic_status(cfg, state) -> None:
    """Log periodic status: daily P&L, safety status, and trading state."""
    equity = state.get("equity", cfg.initial_equity)
    daily_pnl = state.get("daily_pnl", 0.0)
    daily_start_equity = state.get("daily_start_equity")
    
    # Compute risk envelope for max loss
    risk_env = compute_risk_envelope(
        cfg,
        equity,
        VolatilityMode.NORMAL,
        danger_mode=False,
        timing_state=TimingState.NORMAL,
    )
    max_loss_pct = risk_env.max_daily_loss_pct * 100.0
    
    # Check safety status
    is_halted, halt_reason = check_trading_halted(state)
    
    logger.info("")
    logger.info("=" * 60)
    logger.info("PERIODIC STATUS UPDATE:")
    logger.info(f"  Current equity: ${equity:,.2f}")
    
    if daily_start_equity is not None:
        daily_pnl_pct = (daily_pnl / daily_start_equity) * 100.0 if daily_start_equity > 0 else 0.0
        logger.info(f"  Daily P&L: ${daily_pnl:,.2f} ({daily_pnl_pct:+.2f}%)")
        logger.info(f"  Max daily loss: {max_loss_pct:.1f}%")
        
        # Show remaining loss capacity
        remaining_loss_capacity = max(0.0, (max_loss_pct / 100.0) * daily_start_equity + daily_pnl)
        logger.info(f"  Remaining loss capacity: ${remaining_loss_capacity:,.2f}")
    else:
        logger.info("  Daily tracking: Not initialized")
    
    # Safety status
    if is_halted:
        logger.warning(f"  ⚠️  TRADING HALTED: {halt_reason}")
    else:
        logger.info("  ✓ Trading active (no safety triggers)")
    
    logger.info("=" * 60)
    logger.info("")


def run_single_tick(cfg, state, dry_run: bool = False):
    """
    Run a single testnet tick and return the result.
    
    This helper function is accessible for tests and single-tick runs.
    """
    _initialize_daily_tracking(state)
    result_state = _run_tick(cfg, state, dry_run)
    
    return {
        "snapshot": result_state.get("prev_snapshot"),
        "gpt_decision": result_state.get("last_gpt_decision"),
        "risk_decision": result_state.get("last_risk_decision"),
        "execution_result": result_state.get("last_execution_result"),
    }


def run_testnet_forever(cfg, state, dry_run: bool = False) -> None:
    """
    Run testnet trading loop forever with periodic status logging.
    
    This wraps the engine's _run_tick in a loop with:
    - Periodic status logging (daily P&L, safety status)
    - Graceful shutdown on KeyboardInterrupt
    - Respects all safety rails (daily loss limit, kill switch)
    """
    _initialize_daily_tracking(state)
    
    tick_count = 0
    last_status_log_tick = 0
    STATUS_LOG_INTERVAL = 10  # Log status every 10 ticks
    
    logger.info("🚀 Starting TESTNET trading loop (Ctrl+C to stop)...")
    logger.info(f"   Sleep interval: {cfg.loop_sleep_seconds}s between ticks")
    logger.info(f"   Status updates every {STATUS_LOG_INTERVAL} ticks")
    logger.info("")
    
    try:
        while True:
            tick_count += 1
            
            # Re-initialize daily tracking at start of each loop iteration
            # (in case day changed during sleep)
            _initialize_daily_tracking(state)
            
            # Check if we should log periodic status
            if tick_count - last_status_log_tick >= STATUS_LOG_INTERVAL:
                _log_periodic_status(cfg, state)
                last_status_log_tick = tick_count
            
            # Run the tick (includes all safety checks)
            try:
                state = _run_tick(cfg, state, dry_run)
            except KeyboardInterrupt:
                raise  # Re-raise to be caught by outer handler
            except Exception:
                logger.exception("Unhandled error in engine tick, continuing...")
                # Continue loop even on errors
            
            # Sleep between ticks
            time.sleep(cfg.loop_sleep_seconds)
            
    except KeyboardInterrupt:
        logger.info("")
        logger.info("⏹️  TESTNET loop stopped by user (KeyboardInterrupt)")
        # Save state before exiting
        save_state(state, cfg)
        logger.info("✅ State saved before shutdown")


def main():
    parser = argparse.ArgumentParser(
        description="TradeWarrior Hyperliquid Testnet Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_testnet.py              # Continuous trading loop (default)
    python run_testnet.py --once       # Single tick (for testing)
    python run_testnet.py --dry-run    # Preview without execution
    python run_testnet.py --reset      # Fresh start
    python run_testnet.py --equity 500 # Custom equity ($500)
        """,
    )
    
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run single tick instead of continuous loop (for testing)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute GPT + risk decisions but skip actual order execution",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Start from fresh state (ignore existing state_testnet.json)",
    )
    parser.add_argument(
        "--equity",
        type=float,
        default=1000.0,
        help="Initial equity for position sizing (default: $1000)",
    )
    parser.add_argument(
        "--env-file",
        type=str,
        default="twp3_testnet.env",
        help="Path to testnet environment file (default: twp3_testnet.env)",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=None,
        help="Override loop sleep interval (seconds between ticks)",
    )

    args = parser.parse_args()

    # Print testnet banner
    print(TESTNET_BANNER)

    # Load OpenAI key from local_env.txt if present
    _load_env_file("local_env.txt")

    # Validate OpenAI API key (fails fast if missing)
    try:
        validate_openai_api_key()
    except RuntimeError as e:
        logger.error(f"❌ {e}")
        logger.error(
            "Please set OPENAI_API_KEY in local_env.txt or environment before running testnet."
        )
        sys.exit(1)

    # Load testnet configuration
    try:
        cfg = load_testnet_config(
            env_file=args.env_file,
            equity=args.equity,
        )
    except ValueError as e:
        logger.error(f"❌ Configuration error: {e}")
        logger.error(
            f"Please ensure {args.env_file} exists with required credentials:\n"
            "  HL_TESTNET_PRIVATE_KEY=0x..."
        )
        sys.exit(1)

    # Override sleep if specified
    if args.sleep is not None:
        cfg.loop_sleep_seconds = args.sleep

    # Load or reset state
    if args.reset:
        logger.info("🔄 Starting with fresh state (--reset flag)")
        state = reset_state(cfg)
    else:
        state = load_state(cfg)

    logger.info(
        f"💰 Starting equity: ${state.get('equity', cfg.initial_equity):,.2f}"
    )

    # Log configuration (after state is loaded)
    _log_config_summary(cfg, state)

    # Testnet safety confirmations
    logger.info("")
    logger.info("🔐 SAFETY CHECKS:")
    logger.info(f"   ✓ is_testnet = {cfg.is_testnet}")
    logger.info(f"   ✓ execution_mode = {cfg.execution_mode}")
    # Get wallet address from adapter if available (for display only)
    wallet_key = os.getenv('HL_TESTNET_PRIVATE_KEY', 'NOT SET')
    logger.info(f"   ✓ Private key: {'SET' if wallet_key != 'NOT SET' else 'NOT SET'} ({'***' + wallet_key[-4:] if len(wallet_key) > 4 else 'N/A'})")
    logger.info(f"   ✓ Kill switch: {KILL_SWITCH_FILE} (create this file to halt trading)")
    
    # Show daily P&L status
    daily_pnl = state.get("daily_pnl", 0.0)
    daily_start_equity = state.get("daily_start_equity")
    if daily_start_equity is not None:
        daily_pnl_pct = (daily_pnl / daily_start_equity) * 100.0 if daily_start_equity > 0 else 0.0
        risk_env = compute_risk_envelope(
            cfg,
            state.get("equity", cfg.initial_equity),
            VolatilityMode.NORMAL,
            danger_mode=False,
            timing_state=TimingState.NORMAL,
        )
        max_loss_pct = risk_env.max_daily_loss_pct * 100.0
        logger.info(f"   ✓ Daily P&L: ${daily_pnl:,.2f} ({daily_pnl_pct:+.2f}%) / Max loss: {max_loss_pct:.1f}%")
        if state.get("trading_halted", False):
            logger.warning("   ⚠️  TRADING HALTED (circuit breaker active)")
    else:
        logger.info("   ✓ Daily tracking: Not initialized (will start on first tick)")
    logger.info("")

    # Run the engine
    try:
        if args.once:
            logger.info("🚀 Running single TESTNET tick...")
            result = run_single_tick(cfg, state, dry_run=args.dry_run)
            logger.info("")
            logger.info("=" * 60)
            logger.info("TICK RESULT:")
            logger.info("=" * 60)
            pprint(result)
        else:
            # Default: run forever with periodic status logging
            run_testnet_forever(cfg, state, dry_run=args.dry_run)

    except KeyboardInterrupt:
        # This should be caught by run_testnet_forever, but handle here as fallback
        logger.info("")
        logger.info("⏹️  TESTNET loop stopped by user (KeyboardInterrupt)")
    except Exception as e:
        logger.exception(f"❌ Fatal error: {e}")
        sys.exit(1)

    logger.info("")
    logger.info("✅ TradeWarrior TESTNET session ended.")


if __name__ == "__main__":
    main()

