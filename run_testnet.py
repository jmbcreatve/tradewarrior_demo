#!/usr/bin/env python3
"""
TradeWarrior Testnet Runner

This script starts TradeWarrior configured for Hyperliquid testnet trading.
It provides clear logging that the system is running in TESTNET mode with play money.

Usage:
    python run_testnet.py              # Run once (single tick)
    python run_testnet.py --loop       # Run continuously
    python run_testnet.py --dry-run    # Compute decisions but skip execution
    python run_testnet.py --reset      # Start with fresh state

Required environment:
    - twp3_testnet.env file with:
        HL_TESTNET_MAIN_WALLET_ADDRESS=0x...
        HL_TESTNET_API_WALLET_PRIVATE_KEY=0x...
    
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
from engine import run_once, run_forever
from state_memory import load_state, reset_state
from logger_utils import get_logger

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


def _log_config_summary(cfg) -> None:
    """Log key configuration values for visibility."""
    summary = {
        "mode": "TESTNET",
        "is_testnet": cfg.is_testnet,
        "execution_mode": str(cfg.execution_mode),
        "symbol": cfg.symbol,
        "timeframe": cfg.timeframe,
        "initial_equity": f"${cfg.initial_equity:,.2f}",
        "risk_per_trade": f"{cfg.risk_per_trade * 100:.1f}%",
        "max_leverage": f"{cfg.max_leverage}x",
        "primary_data_source": cfg.primary_data_source_id,
        "primary_execution": cfg.primary_execution_id,
        "state_file": cfg.state_file,
    }
    
    logger.info("=" * 60)
    logger.info("TESTNET Configuration Summary:")
    for key, value in summary.items():
        logger.info(f"  {key}: {value}")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="TradeWarrior Hyperliquid Testnet Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_testnet.py              # Single tick
    python run_testnet.py --loop       # Continuous trading loop
    python run_testnet.py --dry-run    # Preview without execution
    python run_testnet.py --reset      # Fresh start
    python run_testnet.py --equity 500 # Custom equity ($500)
        """,
    )
    
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Run continuously (loop mode) instead of single tick",
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

    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        logger.warning(
            "⚠️  OPENAI_API_KEY not set. GPT calls will fail. "
            "Set it in local_env.txt or environment."
        )

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
            "  HL_TESTNET_MAIN_WALLET_ADDRESS=0x...\n"
            "  HL_TESTNET_API_WALLET_PRIVATE_KEY=0x..."
        )
        sys.exit(1)

    # Override sleep if specified
    if args.sleep is not None:
        cfg.loop_sleep_seconds = args.sleep

    # Log configuration
    _log_config_summary(cfg)

    # Load or reset state
    if args.reset:
        logger.info("🔄 Starting with fresh state (--reset flag)")
        state = reset_state(cfg)
    else:
        state = load_state(cfg)

    logger.info(
        f"💰 Starting equity: ${state.get('equity', cfg.initial_equity):,.2f}"
    )

    # Testnet safety confirmations
    logger.info("")
    logger.info("🔐 SAFETY CHECKS:")
    logger.info(f"   ✓ is_testnet = {cfg.is_testnet}")
    logger.info(f"   ✓ execution_mode = {cfg.execution_mode}")
    logger.info(f"   ✓ Wallet: {os.getenv('HL_TESTNET_MAIN_WALLET_ADDRESS', 'NOT SET')[:10]}...")
    logger.info("")

    # Run the engine
    try:
        if args.loop:
            logger.info("🚀 Starting TESTNET trading loop (Ctrl+C to stop)...")
            logger.info(f"   Sleep interval: {cfg.loop_sleep_seconds}s between ticks")
            run_forever(cfg, state=state, dry_run=args.dry_run)
        else:
            logger.info("🚀 Running single TESTNET tick...")
            result = run_once(cfg, state=state, dry_run=args.dry_run)
            logger.info("")
            logger.info("=" * 60)
            logger.info("TICK RESULT:")
            logger.info("=" * 60)
            pprint(result)

    except KeyboardInterrupt:
        logger.info("")
        logger.info("⏹️  Stopped by user (KeyboardInterrupt)")
    except Exception as e:
        logger.exception(f"❌ Fatal error: {e}")
        sys.exit(1)

    logger.info("")
    logger.info("✅ TradeWarrior TESTNET session ended.")


if __name__ == "__main__":
    main()

