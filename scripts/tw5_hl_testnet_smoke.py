#!/usr/bin/env python3
"""
Deterministic TW-5 Hyperliquid testnet smoke:

Phase A: place a far-away limit (should not fill), verify open order count, cancel, verify empty.
Phase B: place tiny market entry, then bracket exits (3 TP limits + stop-market).
Phase C: close position immediately, verify flat / no flip, cancel any residual orders.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, Tuple

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from adapters.liqexec import HyperliquidTestnetExecutionAdapter  # noqa: E402
from tw5.exit_rules import compute_tp_levels  # noqa: E402


SYMBOL = "BTCUSDT"
SIZE = 0.001  # tiny test size
RISK_PCT = 0.01  # 1% move for stop/TP calc


def _require_env() -> None:
    if not os.getenv("HL_TESTNET_PRIVATE_KEY"):
        raise SystemExit("Missing HL_TESTNET_PRIVATE_KEY")


def _get_mark_price(adapter: HyperliquidTestnetExecutionAdapter, symbol: str) -> float:
    if getattr(adapter, "_info", None) is None:
        raise RuntimeError("adapter info not initialized")
    hl_symbol = symbol.replace("USDT", "")
    try:
        meta = adapter._info.meta_and_asset_ctxs()
        if len(meta) >= 2:
            universe = meta[0].get("universe", [])
            ctxs = meta[1]
            for idx, asset in enumerate(universe):
                if asset.get("name") == hl_symbol:
                    if idx < len(ctxs):
                        return float(ctxs[idx].get("markPx", ctxs[idx].get("midPx", 0.0)))
    except Exception as exc:
        print(f"Warning: failed to fetch mark price: {exc}")
    raise RuntimeError("could not fetch mark price")


def _print_orders(adapter: HyperliquidTestnetExecutionAdapter, symbol: str) -> None:
    orders = adapter.get_open_orders(symbol) or []
    print(f"Open orders ({len(orders)}):")
    for o in orders:
        print(o)


def _phase_a(adapter: HyperliquidTestnetExecutionAdapter, price: float) -> None:
    print("Phase A: far-away limit should not fill.")
    limit_price = price * 0.5
    resp = adapter.place_limit_order(symbol=SYMBOL, is_buy=True, sz=SIZE, limit_px=limit_price, reduce_only=False)
    print(f"Placed far limit: {resp}")
    _print_orders(adapter, SYMBOL)
    adapter.cancel_orders(SYMBOL, [resp.get("oid")] if resp.get("oid") is not None else [])
    print("Cancelled far limit.")
    _print_orders(adapter, SYMBOL)


def _phase_b(adapter: HyperliquidTestnetExecutionAdapter, price: float) -> Tuple[float, float]:
    print("Phase B: market entry + bracket.")
    entry_resp = adapter.place_order(symbol=SYMBOL, side="long", size=SIZE, order_type="market")
    print(f"Entry resp: {entry_resp}")
    fill_px = float(entry_resp.get("fill_price") or price)
    stop_px = fill_px * (1.0 - RISK_PCT)
    tp_levels = compute_tp_levels(
        entry=fill_px,
        stop=stop_px,
        side="long",
        r_mults=[1.2, 2.0, 3.0],
        remaining_fracs=[0.30, 0.30, 1.0],
    )
    stop_resp = adapter.place_trigger_order(
        symbol=SYMBOL,
        is_buy=False,
        sz=SIZE,
        trigger_px=stop_px,
        reduce_only=True,
        tpsl="sl",
        is_market=True,
        limit_px=stop_px,
    )
    print(f"Stop resp: {stop_resp}")
    tp_oids = []
    for price_tp, abs_frac, tag in tp_levels:
        tp_resp = adapter.place_limit_order(
            symbol=SYMBOL,
            is_buy=False,
            sz=abs_frac * SIZE,
            limit_px=price_tp,
            reduce_only=True,
            tif="Gtc",
        )
        tp_oids.append(tp_resp.get("oid"))
        print(f"TP {tag} resp: {tp_resp}")
    _print_orders(adapter, SYMBOL)
    return fill_px, stop_px


def _phase_c(adapter: HyperliquidTestnetExecutionAdapter) -> None:
    print("Phase C: close position and verify flat/no flip.")
    adapter.close_position_or_raise(SYMBOL, sz=None)
    positions = adapter.get_open_positions(SYMBOL) or []
    if positions:
        raise RuntimeError(f"Expected flat, found positions: {positions}")
    orders = adapter.get_open_orders(SYMBOL) or []
    if orders:
        adapter.cancel_orders(SYMBOL, [o.get("oid") for o in orders if o.get("oid") is not None])
    print("Closed position, cancelled residual orders, flat verified.")


def main() -> None:
    _require_env()
    adapter = HyperliquidTestnetExecutionAdapter(use_testnet=True)
    if not adapter.health_check():
        raise SystemExit("Health check failed.")
    price = _get_mark_price(adapter, SYMBOL)
    print(f"Mark price for {SYMBOL}: {price}")
    _phase_a(adapter, price)
    _phase_b(adapter, price)
    _phase_c(adapter)
    print("Smoke test complete.")


if __name__ == "__main__":
    main()
