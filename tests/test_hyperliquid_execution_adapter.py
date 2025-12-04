"""Unit tests for HyperliquidTestnetExecutionAdapter with mocked Hyperliquid SDK."""

import os
from unittest.mock import MagicMock, patch

import pytest

from adapters.liqexec import HyperliquidTestnetExecutionAdapter


class TestHyperliquidExecutionAdapter:
    """Test HyperliquidTestnetExecutionAdapter with mocked SDK."""

    @patch.dict(os.environ, {
        "HL_TESTNET_PRIVATE_KEY": "0xabcdef1234567890",
    })
    @patch("adapters.liqexec.HL_SDK_AVAILABLE", True)
    @patch("adapters.liqexec.Exchange")
    @patch("adapters.liqexec.Info")
    @patch("adapters.liqexec.Wallet")
    def test_init_success(self, mock_wallet_class, mock_info_class, mock_exchange_class):
        """Test successful initialization with valid credentials."""
        mock_wallet = MagicMock()
        mock_wallet.address = "0x1234567890abcdef"
        mock_wallet_class.return_value = mock_wallet
        mock_exchange = MagicMock()
        mock_exchange_class.return_value = mock_exchange
        mock_info = MagicMock()
        mock_info_class.return_value = mock_info
        
        # Use base_url to avoid needing to patch constants
        adapter = HyperliquidExecutionAdapter(use_testnet=True, base_url="https://api.hyperliquid-testnet.xyz")
        
        assert adapter._exchange is not None
        assert adapter._info is not None
        assert adapter._wallet is not None
        assert adapter._use_testnet is True
        mock_wallet_class.assert_called_once_with("0xabcdef1234567890")
        mock_exchange_class.assert_called_once_with(mock_wallet, base_url="https://api.hyperliquid-testnet.xyz")
        mock_info_class.assert_called_once()

    @patch.dict(os.environ, {}, clear=True)
    @patch("adapters.liqexec.HL_SDK_AVAILABLE", True)
    def test_init_missing_credentials(self):
        """Test initialization fails gracefully when HL_TESTNET_PRIVATE_KEY is missing."""
        adapter = HyperliquidExecutionAdapter(use_testnet=True)
        
        assert adapter._exchange is None
        assert adapter._info is None
        assert adapter._wallet is None

    @patch("adapters.liqexec.HL_SDK_AVAILABLE", False)
    def test_init_sdk_not_available(self):
        """Test initialization when SDK is not installed."""
        adapter = HyperliquidExecutionAdapter(use_testnet=True)
        
        assert adapter._exchange is None
        assert adapter._info is None
        assert adapter._wallet is None

    @patch("adapters.liqexec.HL_SDK_AVAILABLE", True)
    def test_init_rejects_non_testnet(self):
        """Test that adapter rejects non-testnet mode for safety."""
        with pytest.raises(ValueError, match="only supports testnet"):
            HyperliquidExecutionAdapter(use_testnet=False)

    @patch.dict(os.environ, {
        "HL_TESTNET_PRIVATE_KEY": "0xabcdef1234567890",
    })
    @patch("adapters.liqexec.HL_SDK_AVAILABLE", True)
    @patch("adapters.liqexec.Exchange")
    @patch("adapters.liqexec.Info")
    @patch("adapters.liqexec.Wallet")
    def test_health_check_success(self, mock_wallet_class, mock_info_class, mock_exchange_class):
        """Test health check returns True when user state can be fetched."""
        mock_wallet = MagicMock()
        mock_wallet.address = "0x1234567890abcdef"
        mock_wallet_class.return_value = mock_wallet
        mock_exchange = MagicMock()
        mock_exchange_class.return_value = mock_exchange
        mock_info = MagicMock()
        mock_info.user_state.return_value = {"assetPositions": []}
        mock_info_class.return_value = mock_info
        
        adapter = HyperliquidExecutionAdapter(use_testnet=True, base_url="https://api.hyperliquid-testnet.xyz")
        
        assert adapter.health_check() is True
        mock_info.user_state.assert_called_once_with("0x1234567890abcdef")

    @patch.dict(os.environ, {
        "HL_TESTNET_PRIVATE_KEY": "0xabcdef1234567890",
    })
    @patch("adapters.liqexec.HL_SDK_AVAILABLE", True)
    @patch("adapters.liqexec.Exchange")
    @patch("adapters.liqexec.Info")
    @patch("adapters.liqexec.Wallet")
    def test_health_check_failure(self, mock_wallet_class, mock_info_class, mock_exchange_class):
        """Test health check returns False on error."""
        mock_wallet = MagicMock()
        mock_wallet.address = "0x1234567890abcdef"
        mock_wallet_class.return_value = mock_wallet
        mock_exchange = MagicMock()
        mock_exchange_class.return_value = mock_exchange
        mock_info = MagicMock()
        mock_info.user_state.side_effect = Exception("Connection error")
        mock_info_class.return_value = mock_info
        
        adapter = HyperliquidExecutionAdapter(use_testnet=True)
        
        assert adapter.health_check() is False

    @patch.dict(os.environ, {
        "HL_TESTNET_PRIVATE_KEY": "0xabcdef1234567890",
    })
    @patch("adapters.liqexec.HL_SDK_AVAILABLE", True)
    @patch("adapters.liqexec.Exchange")
    @patch("adapters.liqexec.Info")
    @patch("adapters.liqexec.Wallet")
    def test_place_order_market_success(self, mock_wallet_class, mock_info_class, mock_exchange_class):
        """Test successful market order placement."""
        mock_wallet = MagicMock()
        mock_wallet.address = "0x1234567890abcdef"
        mock_wallet_class.return_value = mock_wallet
        mock_exchange = MagicMock()
        mock_exchange.update_leverage.return_value = {"status": "ok"}
        mock_exchange.market_order.return_value = {
            "status": "ok",
            "response": {
                "data": {
                    "statuses": [{
                        "filled": {
                            "totalSz": "100.0",
                            "totalPx": "50000.0",
                            "oid": "order-123",
                        }
                    }]
                }
            }
        }
        mock_exchange_class.return_value = mock_exchange
        mock_info = MagicMock()
        mock_info_class.return_value = mock_info
        
        adapter = HyperliquidExecutionAdapter(use_testnet=True)
        
        result = adapter.place_order(
            symbol="BTCUSDT",
            side="long",
            size=1.0,
            order_type="market",
            leverage=2.0,
        )
        
        assert result["status"] == "filled"
        assert result["fill_price"] == 500.0  # totalPx / totalSz
        assert result["order_id"] == "order-123"
        assert result["fee_paid"] > 0
        mock_exchange.market_order.assert_called_once_with(
            coin="BTC",
            is_buy=True,
            sz=1.0,
        )

    @patch.dict(os.environ, {
        "HL_TESTNET_PRIVATE_KEY": "0xabcdef1234567890",
    })
    @patch("adapters.liqexec.HL_SDK_AVAILABLE", True)
    @patch("adapters.liqexec.Exchange")
    @patch("adapters.liqexec.Info")
    @patch("adapters.liqexec.Wallet")
    def test_place_order_rejects_non_market(self, mock_wallet_class, mock_info_class, mock_exchange_class):
        """Test that non-market orders are rejected."""
        mock_wallet = MagicMock()
        mock_wallet.address = "0x1234567890abcdef"
        mock_wallet_class.return_value = mock_wallet
        mock_exchange = MagicMock()
        mock_exchange_class.return_value = mock_exchange
        mock_info = MagicMock()
        mock_info_class.return_value = mock_info
        
        adapter = HyperliquidExecutionAdapter(use_testnet=True)
        
        result = adapter.place_order(
            symbol="BTCUSDT",
            side="long",
            size=1.0,
            order_type="limit",  # Not market
        )
        
        assert result["status"] == "no_trade"
        assert result["reason"] == "unsupported_order_type"
        mock_exchange.market_order.assert_not_called()

    @patch.dict(os.environ, {
        "HL_TESTNET_PRIVATE_KEY": "0xabcdef1234567890",
    })
    @patch("adapters.liqexec.HL_SDK_AVAILABLE", True)
    @patch("adapters.liqexec.Exchange")
    @patch("adapters.liqexec.Info")
    @patch("adapters.liqexec.Wallet")
    def test_place_order_rejects_invalid_side(self, mock_wallet_class, mock_info_class, mock_exchange_class):
        """Test that invalid side is rejected."""
        mock_wallet = MagicMock()
        mock_wallet.address = "0x1234567890abcdef"
        mock_wallet_class.return_value = mock_wallet
        mock_exchange = MagicMock()
        mock_exchange_class.return_value = mock_exchange
        mock_info = MagicMock()
        mock_info_class.return_value = mock_info
        
        adapter = HyperliquidExecutionAdapter(use_testnet=True)
        
        result = adapter.place_order(
            symbol="BTCUSDT",
            side="invalid",
            size=1.0,
        )
        
        assert result["status"] == "no_trade"
        assert result["reason"] == "invalid_side"

    @patch.dict(os.environ, {
        "HL_TESTNET_PRIVATE_KEY": "0xabcdef1234567890",
    })
    @patch("adapters.liqexec.HL_SDK_AVAILABLE", True)
    @patch("adapters.liqexec.Exchange")
    @patch("adapters.liqexec.Info")
    @patch("adapters.liqexec.Wallet")
    def test_place_order_handles_api_error(self, mock_wallet_class, mock_info_class, mock_exchange_class):
        """Test that API errors are handled gracefully."""
        mock_wallet = MagicMock()
        mock_wallet.address = "0x1234567890abcdef"
        mock_wallet_class.return_value = mock_wallet
        mock_exchange = MagicMock()
        mock_exchange.market_order.return_value = {
            "status": "err",
            "response": {
                "data": {
                    "statuses": [{
                        "resting": {
                            "message": "Insufficient balance"
                        }
                    }]
                }
            }
        }
        mock_exchange_class.return_value = mock_exchange
        mock_info = MagicMock()
        mock_info_class.return_value = mock_info
        
        adapter = HyperliquidExecutionAdapter(use_testnet=True)
        
        result = adapter.place_order(
            symbol="BTCUSDT",
            side="long",
            size=1.0,
        )
        
        assert result["status"] == "no_trade"
        assert "order_rejected" in result["reason"]

    @patch.dict(os.environ, {
        "HL_TESTNET_PRIVATE_KEY": "0xabcdef1234567890",
    })
    @patch("adapters.liqexec.HL_SDK_AVAILABLE", True)
    @patch("adapters.liqexec.Exchange")
    @patch("adapters.liqexec.Info")
    @patch("adapters.liqexec.Wallet")
    def test_place_order_handles_exception(self, mock_wallet_class, mock_info_class, mock_exchange_class):
        """Test that exceptions are caught and return no_trade."""
        mock_wallet = MagicMock()
        mock_wallet.address = "0x1234567890abcdef"
        mock_wallet_class.return_value = mock_wallet
        mock_exchange = MagicMock()
        mock_exchange.market_order.side_effect = Exception("Network error")
        mock_exchange_class.return_value = mock_exchange
        mock_info = MagicMock()
        mock_info_class.return_value = mock_info
        
        adapter = HyperliquidExecutionAdapter(use_testnet=True)
        
        result = adapter.place_order(
            symbol="BTCUSDT",
            side="long",
            size=1.0,
        )
        
        assert result["status"] == "no_trade"
        assert "exception" in result["reason"]

    @patch.dict(os.environ, {
        "HL_TESTNET_PRIVATE_KEY": "0xabcdef1234567890",
    })
    @patch("adapters.liqexec.HL_SDK_AVAILABLE", True)
    @patch("adapters.liqexec.Exchange")
    @patch("adapters.liqexec.Info")
    @patch("adapters.liqexec.Wallet")
    def test_get_open_positions(self, mock_wallet_class, mock_info_class, mock_exchange_class):
        """Test fetching open positions."""
        mock_wallet = MagicMock()
        mock_wallet.address = "0x1234567890abcdef"
        mock_wallet_class.return_value = mock_wallet
        mock_exchange = MagicMock()
        mock_exchange_class.return_value = mock_exchange
        mock_info = MagicMock()
        mock_info.user_state.return_value = {
            "assetPositions": [{
                "position": {
                    "coin": "BTC",
                    "szi": "1.5",
                    "entryPx": "50000.0",
                    "leverage": {"value": "2.0"},
                    "unrealizedPnl": "100.0",
                }
            }]
        }
        mock_info_class.return_value = mock_info
        
        adapter = HyperliquidExecutionAdapter(use_testnet=True)
        
        positions = adapter.get_open_positions("BTCUSDT")
        
        assert len(positions) == 1
        assert positions[0]["symbol"] == "BTCUSDT"
        assert positions[0]["side"] == "long"
        assert positions[0]["size"] == 1.5
        assert positions[0]["entry_price"] == 50000.0
        assert positions[0]["leverage"] == 2.0

    @patch.dict(os.environ, {
        "HL_TESTNET_PRIVATE_KEY": "0xabcdef1234567890",
    })
    @patch("adapters.liqexec.HL_SDK_AVAILABLE", True)
    @patch("adapters.liqexec.Exchange")
    @patch("adapters.liqexec.Info")
    @patch("adapters.liqexec.Wallet")
    def test_cancel_all_orders(self, mock_wallet_class, mock_info_class, mock_exchange_class):
        """Test cancelling all orders."""
        mock_wallet = MagicMock()
        mock_wallet.address = "0x1234567890abcdef"
        mock_wallet_class.return_value = mock_wallet
        mock_exchange = MagicMock()
        mock_exchange.cancel.return_value = {"status": "ok"}
        mock_exchange_class.return_value = mock_exchange
        mock_info = MagicMock()
        mock_info_class.return_value = mock_info
        
        adapter = HyperliquidExecutionAdapter(use_testnet=True)
        
        adapter.cancel_all_orders("BTCUSDT")
        
        mock_exchange.cancel.assert_called_once_with("BTC")

    @patch.dict(os.environ, {
        "HL_TESTNET_PRIVATE_KEY": "0xabcdef1234567890",
    })
    @patch("adapters.liqexec.HL_SDK_AVAILABLE", True)
    @patch("adapters.liqexec.Exchange")
    @patch("adapters.liqexec.Info")
    @patch("adapters.liqexec.Wallet")
    def test_symbol_normalization(self, mock_wallet_class, mock_info_class, mock_exchange_class):
        """Test that various symbol formats are normalized correctly."""
        mock_wallet = MagicMock()
        mock_wallet.address = "0x1234567890abcdef"
        mock_wallet_class.return_value = mock_wallet
        mock_exchange = MagicMock()
        mock_exchange.market_order.return_value = {
            "status": "ok",
            "response": {
                "data": {
                    "statuses": [{
                        "filled": {
                            "totalSz": "1.0",
                            "totalPx": "50000.0",
                            "oid": "order-123",
                        }
                    }]
                }
            }
        }
        mock_exchange_class.return_value = mock_exchange
        mock_info = MagicMock()
        mock_info_class.return_value = mock_info
        
        adapter = HyperliquidExecutionAdapter(use_testnet=True)
        
        # Test various symbol formats
        for symbol in ["BTCUSDT", "BTC-USD", "BTC"]:
            result = adapter.place_order(
                symbol=symbol,
                side="long",
                size=1.0,
            )
            # Should call with normalized "BTC"
            calls = [call for call in mock_exchange.market_order.call_args_list if call[1]["coin"] == "BTC"]
            assert len(calls) > 0

