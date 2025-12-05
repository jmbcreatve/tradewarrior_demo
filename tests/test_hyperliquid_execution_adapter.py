"""Unit tests for HyperliquidTestnetExecutionAdapter with mocked Hyperliquid SDK."""

import os
from unittest.mock import MagicMock, patch

import pytest

from adapters.liqexec import HyperliquidTestnetExecutionAdapter


@pytest.fixture
def adapter_with_mocks():
    with patch.dict(os.environ, {"HL_TESTNET_PRIVATE_KEY": "0xabcdef1234567890"}):
        with patch("adapters.liqexec.HL_SDK_AVAILABLE", True):
            with patch("adapters.liqexec.Account") as mock_account_cls, \
                 patch("adapters.liqexec.Exchange") as mock_exchange_cls, \
                 patch("adapters.liqexec.Info") as mock_info_cls:
                mock_account = MagicMock()
                mock_account.address = "0x1234"
                mock_account_cls.from_key.return_value = mock_account

                mock_exchange = MagicMock()
                mock_exchange_cls.return_value = mock_exchange

                mock_info = MagicMock()
                mock_info_cls.return_value = mock_info

                adapter = HyperliquidTestnetExecutionAdapter(use_testnet=True, base_url="https://test.api")
                yield adapter, {"account": mock_account, "exchange": mock_exchange, "info": mock_info}


def test_init_success(adapter_with_mocks):
    adapter, mocks = adapter_with_mocks
    assert adapter._exchange is mocks["exchange"]
    assert adapter._info is mocks["info"]
    assert adapter._account.address == "0x1234"


def test_place_order_places_protective_orders(adapter_with_mocks):
    adapter, mocks = adapter_with_mocks
    mocks["exchange"].update_leverage.return_value = {"status": "ok"}
    mocks["exchange"].market_open.return_value = {
        "status": "ok",
        "response": {"data": {"statuses": [{"filled": {"totalPx": "50000", "totalSz": "100", "oid": "oid-1"}}]}},
    }
    mocks["exchange"].bulk_orders.return_value = {"status": "ok", "response": {"data": {"ids": [1, 2]}}}

    result = adapter.place_order(
        symbol="BTCUSDT",
        side="long",
        size=1.0,
        order_type="market",
        stop_loss=49000.0,
        take_profit=51000.0,
        leverage=3,
    )

    mocks["exchange"].market_open.assert_called_once()
    args, kwargs = mocks["exchange"].market_open.call_args
    assert kwargs["name"] == "BTC"
    assert kwargs["is_buy"] is True
    assert kwargs["sz"] == 1.0

    mocks["exchange"].bulk_orders.assert_called_once()
    orders_arg = mocks["exchange"].bulk_orders.call_args[0][0]
    assert len(orders_arg) == 2
    assert all(order["reduce_only"] for order in orders_arg)
    assert {order["order_type"]["trigger"]["tpsl"] for order in orders_arg} == {"sl", "tp"}

    assert result["status"] == "filled"
    assert "protective_orders" in result


def test_protective_order_failure_flattens_and_returns_no_trade(adapter_with_mocks):
    adapter, mocks = adapter_with_mocks
    mocks["exchange"].market_open.return_value = {
        "status": "ok",
        "response": {"data": {"statuses": [{"filled": {"totalPx": "5000", "totalSz": "10", "oid": "oid-2"}}]}},
    }
    mocks["exchange"].bulk_orders.side_effect = RuntimeError("boom")
    mocks["exchange"].market_close.return_value = {"status": "ok"}

    result = adapter.place_order(
        symbol="ETHUSDT",
        side="short",
        size=2.0,
        order_type="market",
        stop_loss=2500.0,
        take_profit=2300.0,
    )

    mocks["exchange"].market_close.assert_called_once()
    assert result["status"] == "no_trade"
    assert result["reason"] == "protective_order_failure"


def test_cancel_all_orders_filters_by_symbol(adapter_with_mocks):
    adapter, mocks = adapter_with_mocks
    mocks["info"].frontend_open_orders.return_value = [
        {"coin": "BTC", "oid": 1},
        {"coin": "ETH", "oid": 2},
    ]
    mocks["exchange"].bulk_cancel.return_value = {"status": "ok"}

    adapter.cancel_all_orders("BTCUSDT")

    mocks["exchange"].bulk_cancel.assert_called_once_with([{"coin": "BTC", "oid": 1}])


def test_invalid_side_returns_no_trade(adapter_with_mocks):
    adapter, mocks = adapter_with_mocks

    result = adapter.place_order(symbol="BTCUSDT", side="invalid", size=1.0)

    assert result["status"] == "no_trade"
    assert result["reason"] == "invalid_side"
    mocks["exchange"].market_open.assert_not_called()
