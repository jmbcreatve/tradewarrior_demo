"""Integration tests for HyperliquidExecutionAdapter with real testnet API.

These tests require:
- HL_TESTNET_MAIN_WALLET_ADDRESS environment variable
- HL_TESTNET_API_WALLET_PRIVATE_KEY environment variable
- Network connectivity to Hyperliquid testnet

To run these tests:
    pytest tests/test_hyperliquid_execution_adapter_integration.py -v --run-integration

Or set environment variable:
    RUN_HL_INTEGRATION_TESTS=1 pytest tests/test_hyperliquid_execution_adapter_integration.py -v
"""

import os
import pytest

from adapters.liqexec import HyperliquidExecutionAdapter


# Skip all tests in this file unless explicitly enabled
pytestmark = pytest.mark.skipif(
    not os.getenv("RUN_HL_INTEGRATION_TESTS") and not os.getenv("--run-integration"),
    reason="Integration tests require RUN_HL_INTEGRATION_TESTS=1 or --run-integration flag"
)


@pytest.fixture
def testnet_adapter():
    """Create a testnet adapter if credentials are available."""
    wallet = os.getenv("HL_TESTNET_MAIN_WALLET_ADDRESS")
    key = os.getenv("HL_TESTNET_API_WALLET_PRIVATE_KEY")
    
    if not wallet or not key:
        pytest.skip("Missing HL_TESTNET_MAIN_WALLET_ADDRESS or HL_TESTNET_API_WALLET_PRIVATE_KEY")
    
    adapter = HyperliquidExecutionAdapter(use_testnet=True)
    
    if not adapter.health_check():
        pytest.skip("Hyperliquid testnet adapter health check failed")
    
    return adapter


class TestHyperliquidExecutionAdapterIntegration:
    """Integration tests that hit real Hyperliquid testnet API."""
    
    def test_health_check(self, testnet_adapter):
        """Test that health check works with real API."""
        assert testnet_adapter.health_check() is True
    
    def test_get_open_positions(self, testnet_adapter):
        """Test fetching open positions from testnet."""
        positions = testnet_adapter.get_open_positions("BTCUSDT")
        # Should return a list (may be empty)
        assert isinstance(positions, list)
        # If positions exist, verify structure
        if positions:
            pos = positions[0]
            assert "symbol" in pos
            assert "side" in pos
            assert "size" in pos
            assert pos["side"] in ["long", "short"]
    
    def test_place_small_market_order(self, testnet_adapter):
        """Test placing a very small market order on testnet.
        
        WARNING: This will place a real testnet order. Use minimal size.
        """
        # Use a very small size to minimize testnet balance impact
        result = testnet_adapter.place_order(
            symbol="BTCUSDT",
            side="long",
            size=0.001,  # Very small size
            order_type="market",
            leverage=1.0,
        )
        
        # Should either fill or return a clear error
        assert result["status"] in ["filled", "no_trade", "resting"]
        assert "symbol" in result
        assert "side" in result
        
        if result["status"] == "filled":
            assert result["fill_price"] > 0
            assert result["fee_paid"] >= 0
    
    def test_cancel_all_orders(self, testnet_adapter):
        """Test cancelling all orders (should not fail even if no orders exist)."""
        # This should not raise an exception
        testnet_adapter.cancel_all_orders("BTCUSDT")
    
    def test_rejects_non_market_order(self, testnet_adapter):
        """Test that non-market orders are rejected."""
        result = testnet_adapter.place_order(
            symbol="BTCUSDT",
            side="long",
            size=0.001,
            order_type="limit",  # Not market
        )
        
        assert result["status"] == "no_trade"
        assert result["reason"] == "unsupported_order_type"
    
    def test_rejects_invalid_side(self, testnet_adapter):
        """Test that invalid side is rejected."""
        result = testnet_adapter.place_order(
            symbol="BTCUSDT",
            side="invalid",
            size=0.001,
        )
        
        assert result["status"] == "no_trade"
        assert result["reason"] == "invalid_side"
    
    def test_rejects_zero_size(self, testnet_adapter):
        """Test that zero size is rejected."""
        result = testnet_adapter.place_order(
            symbol="BTCUSDT",
            side="long",
            size=0.0,
        )
        
        assert result["status"] == "no_trade"
        assert result["reason"] == "invalid_size"

