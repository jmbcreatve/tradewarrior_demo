#!/usr/bin/env python3
"""
One-shot readiness check for TW-5 Hyperliquid testnet execution.

Verifies required env vars, adapter health, and prints account value,
open positions, and open orders.
"""

from __future__ import annotations

import os
import sys

# Ensure project root on path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from adapters.liqexec import HyperliquidTestnetExecutionAdapter  # noqa: E402


def main() -> None:
    required = ["HL_TESTNET_PRIVATE_KEY"]
    missing = [env for env in required if not os.getenv(env)]
    if missing:
        print(f"Missing required env vars: {missing}")
        sys.exit(1)

    adapter = HyperliquidTestnetExecutionAdapter(use_testnet=True)
    if not adapter.health_check():
        print("Health check FAILED.")
        sys.exit(2)

    acct_val = 0.0
    if hasattr(adapter, "get_account_value_usd"):
        try:
            acct_val = adapter.get_account_value_usd()
        except Exception as exc:  # pragma: no cover
            print(f"Failed to fetch account value: {exc}")

    print(f"Account value (USD): {acct_val:.2f}")

    positions = []
    if hasattr(adapter, "get_open_positions"):
        try:
            positions = adapter.get_open_positions("BTCUSDT") or []
        except Exception as exc:  # pragma: no cover
            print(f"Failed to fetch positions: {exc}")
    print(f"Open positions: {positions}")

    orders = []
    if hasattr(adapter, "get_open_orders"):
        try:
            orders = adapter.get_open_orders("BTCUSDT") or []
        except Exception as exc:  # pragma: no cover
            print(f"Failed to fetch open orders: {exc}")
    print(f"Open orders: {orders}")

    print("TW-5 testnet readiness: OK")


if __name__ == "__main__":
    main()
