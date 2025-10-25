"""Unit tests for configurable price aggregation functionality.

This module tests the configurable price aggregation feature to ensure
it works correctly with different precision settings and configurations.
"""

from decimal import Decimal
from unittest.mock import Mock

import pytest

from src.core.liquidity_peaks_analyzer import LiquidityPeaksAnalyzer
from src.core.price_aggregator import PriceAggregator
from src.utils.config import PriceAggregationConfig


class TestPriceAggregator:
    """Test cases for the PriceAggregator class."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.test_bids = [
            Mock(price=Decimal("95000.50"), quantity=Decimal("1.5")),
            Mock(price=Decimal("95001.20"), quantity=Decimal("2.3")),
            Mock(price=Decimal("95002.80"), quantity=Decimal("0.8")),
            Mock(price=Decimal("95003.10"), quantity=Decimal("1.2")),
        ]
        self.test_asks = [
            Mock(price=Decimal("95100.30"), quantity=Decimal("1.8")),
            Mock(price=Decimal("95101.70"), quantity=Decimal("2.1")),
            Mock(price=Decimal("95102.40"), quantity=Decimal("1.5")),
            Mock(price=Decimal("95103.90"), quantity=Decimal("0.9")),
        ]

    def test_init_with_default_parameters(self):
        """Test PriceAggregator initialization with default parameters."""
        aggregator = PriceAggregator()
        assert aggregator.precision == Decimal("1.0")
        assert aggregator.enabled is True
        assert aggregator.max_price_levels == 5000

    def test_init_with_custom_parameters(self):
        """Test PriceAggregator initialization with custom parameters."""
        aggregator = PriceAggregator(
            precision=0.5, enabled=False, max_price_levels=1000
        )
        assert aggregator.precision == Decimal("0.5")
        assert aggregator.enabled is False
        assert aggregator.max_price_levels == 1000

    def test_init_with_invalid_precision(self):
        """Test PriceAggregator initialization with invalid precision."""
        with pytest.raises(ValueError, match="Precision must be positive"):
            PriceAggregator(precision=0)

        with pytest.raises(ValueError, match="Precision must be positive"):
            PriceAggregator(precision=-1.0)

    def test_init_with_invalid_max_levels(self):
        """Test PriceAggregator initialization with invalid max levels."""
        with pytest.raises(ValueError, match="Max price levels must be positive"):
            PriceAggregator(max_price_levels=0)

        with pytest.raises(ValueError, match="Max price levels must be positive"):
            PriceAggregator(max_price_levels=-100)

    def test_aggregate_with_1_dollar_precision(self):
        """Test aggregation with $1 precision."""
        aggregator = PriceAggregator(precision=1.0)
        aggregated_bids, aggregated_asks = aggregator.aggregate_order_book_levels(
            self.test_bids, self.test_asks
        )

        # Check bid aggregation (prices should be rounded down to nearest dollar)
        assert Decimal("95000") in aggregated_bids
        assert Decimal("95001") in aggregated_bids
        assert Decimal("95002") in aggregated_bids
        assert Decimal("95003") in aggregated_bids

        # Check volume aggregation
        assert aggregated_bids[Decimal("95000")] == Decimal("1.5")
        assert aggregated_bids[Decimal("95001")] == Decimal("2.3")
        assert aggregated_bids[Decimal("95002")] == Decimal("0.8")
        assert aggregated_bids[Decimal("95003")] == Decimal("1.2")

        # Check ask aggregation
        assert Decimal("95100") in aggregated_asks
        assert Decimal("95101") in aggregated_asks
        assert Decimal("95102") in aggregated_asks
        assert Decimal("95103") in aggregated_asks

    def test_aggregate_with_0_1_precision(self):
        """Test aggregation with $0.1 precision."""
        aggregator = PriceAggregator(precision=0.1)

        # Create test data with more precise prices
        precise_bids = [
            Mock(price=Decimal("95000.53"), quantity=Decimal("1.5")),
            Mock(price=Decimal("95000.58"), quantity=Decimal("2.3")),
            Mock(price=Decimal("95001.22"), quantity=Decimal("0.8")),
        ]

        aggregated_bids, _ = aggregator.aggregate_order_book_levels(precise_bids, [])

        # Prices should be rounded down to nearest 0.1
        assert Decimal("95000.5") in aggregated_bids
        assert Decimal("95001.2") in aggregated_bids

        # Volumes should be aggregated for same price levels
        assert aggregated_bids[Decimal("95000.5")] == Decimal("3.8")  # 1.5 + 2.3
        assert aggregated_bids[Decimal("95001.2")] == Decimal("0.8")

    def test_aggregate_with_disabled_aggregation(self):
        """Test behavior when aggregation is disabled."""
        aggregator = PriceAggregator(enabled=False)
        aggregated_bids, aggregated_asks = aggregator.aggregate_order_book_levels(
            self.test_bids, self.test_asks
        )

        # Should return original data as dictionaries
        assert len(aggregated_bids) == len(self.test_bids)
        assert len(aggregated_asks) == len(self.test_asks)

        # Check that original prices are preserved
        for _i, bid in enumerate(self.test_bids):
            assert bid.price in aggregated_bids
            assert aggregated_bids[bid.price] == bid.quantity

    def test_max_price_levels_limiting(self):
        """Test that max price levels is respected."""
        # Create many test levels
        many_bids = [
            Mock(price=Decimal(f"9500{i}.50"), quantity=Decimal("1.0"))
            for i in range(100)
        ]
        many_asks = [
            Mock(price=Decimal(f"9510{i}.30"), quantity=Decimal("1.0"))
            for i in range(100)
        ]

        aggregator = PriceAggregator(max_price_levels=10)
        aggregated_bids, aggregated_asks = aggregator.aggregate_order_book_levels(
            many_bids, many_asks
        )

        # Should be limited to 10 levels each
        assert len(aggregated_bids) <= 10
        assert len(aggregated_asks) <= 10

        # Should keep the most relevant levels (highest bids, lowest asks)
        bid_prices = sorted(aggregated_bids.keys(), reverse=True)
        ask_prices = sorted(aggregated_asks.keys())

        # For bids, should keep highest prices
        assert bid_prices == sorted(bid_prices, reverse=True)
        # For asks, should keep lowest prices
        assert ask_prices == sorted(ask_prices)

    def test_get_aggregation_stats(self):
        """Test aggregation statistics calculation."""
        aggregator = PriceAggregator(precision=1.0)
        aggregated_bids, aggregated_asks = aggregator.aggregate_order_book_levels(
            self.test_bids, self.test_asks
        )

        stats = aggregator.get_aggregation_stats(
            self.test_bids, self.test_asks, aggregated_bids, aggregated_asks
        )

        # Check required stats fields
        assert "original_levels" in stats
        assert "aggregated_levels" in stats
        assert "reduction_ratio" in stats
        assert "original_volume" in stats
        assert "aggregated_volume" in stats
        assert "volume_preservation_percent" in stats
        assert "precision" in stats
        assert "enabled" in stats
        assert "max_levels" in stats

        # Check values
        assert stats["original_levels"] == 8  # 4 bids + 4 asks
        assert stats["aggregated_levels"] == 8  # No aggregation with different prices
        assert stats["precision"] == 1.0
        assert stats["enabled"] is True
        assert stats["max_levels"] == 5000
        assert (
            abs(stats["volume_preservation_percent"] - 100.0) < 0.1
        )  # All volume preserved (allowing small rounding errors)

    def test_round_down_to_precision(self):
        """Test price rounding down to precision."""
        aggregator = PriceAggregator(precision=0.5)

        # Test various price levels
        test_cases = [
            (Decimal("95000.50"), Decimal("95000.5")),
            (Decimal("95000.99"), Decimal("95000.5")),
            (Decimal("95001.20"), Decimal("95001.0")),
            (Decimal("95001.49"), Decimal("95001.0")),
        ]

        for input_price, expected_price in test_cases:
            result = aggregator._round_down_to_precision(input_price)
            assert result == expected_price

    def test_invalid_level_filtering(self):
        """Test filtering of invalid depth levels."""
        aggregator = PriceAggregator(precision=1.0)

        # Mix of valid and invalid levels
        mixed_bids = [
            Mock(price=Decimal("95000.50"), quantity=Decimal("1.5")),  # Valid
            Mock(price=Decimal("-100"), quantity=Decimal("1.0")),  # Invalid price
            Mock(
                price=Decimal("95001.50"), quantity=Decimal("-0.5")
            ),  # Invalid quantity
            Mock(price=Decimal("95002.50"), quantity=Decimal("0")),  # Invalid quantity
        ]

        aggregated_bids, _ = aggregator.aggregate_order_book_levels(mixed_bids, [])

        # Should only include valid levels
        assert len(aggregated_bids) == 1
        assert Decimal("95000") in aggregated_bids
        assert aggregated_bids[Decimal("95000")] == Decimal("1.5")


class TestLiquidityPeaksAnalyzerIntegration:
    """Test integration of configurable price aggregation with LiquidityPeaksAnalyzer."""

    def test_with_configured_aggregation(self):
        """Test LiquidityPeaksAnalyzer with configured aggregation."""
        from src.core.models import DepthSnapshot

        # Create test depth snapshot
        snapshot = DepthSnapshot(
            symbol="BTCFDUSD",
            timestamp=None,
            bids=[
                Mock(price=Decimal("95000.30"), quantity=Decimal("1.5")),
                Mock(price=Decimal("95000.80"), quantity=Decimal("2.3")),
                Mock(price=Decimal("95001.20"), quantity=Decimal("0.8")),
            ],
            asks=[
                Mock(price=Decimal("95100.40"), quantity=Decimal("1.8")),
                Mock(price=Decimal("95100.90"), quantity=Decimal("2.1")),
                Mock(price=Decimal("95101.30"), quantity=Decimal("1.5")),
            ],
        )

        # Configure with $1 aggregation
        aggregation_config = {
            "precision": 1.0,
            "enabled": True,
            "max_price_levels": 5000,
        }

        analyzer = LiquidityPeaksAnalyzer(
            min_volume_threshold=0.5, price_aggregation_config=aggregation_config
        )

        result = analyzer.analyze_liquidity_peaks(snapshot)

        # Should have aggregated results
        assert "liquidity_peaks" in result
        assert "bid_aggregation" in result
        assert "ask_aggregation" in result

        # Check that aggregation was applied (prices should be rounded)
        bid_aggregation = result["bid_aggregation"]
        ask_aggregation = result["ask_aggregation"]

        # Prices should be rounded to nearest dollar
        assert Decimal("95000") in bid_aggregation
        assert Decimal("95001") in bid_aggregation
        assert Decimal("95100") in ask_aggregation
        assert Decimal("95101") in ask_aggregation

        # Volumes should be aggregated
        assert bid_aggregation[Decimal("95000")] == Decimal("3.8")  # 1.5 + 2.3

    def test_with_disabled_aggregation(self):
        """Test LiquidityPeaksAnalyzer with disabled aggregation."""
        from src.core.models import DepthSnapshot

        snapshot = DepthSnapshot(
            symbol="BTCFDUSD",
            timestamp=None,
            bids=[Mock(price=Decimal("95000.50"), quantity=Decimal("1.5"))],
            asks=[Mock(price=Decimal("95100.50"), quantity=Decimal("1.8"))],
        )

        # Configure with disabled aggregation
        aggregation_config = {
            "precision": 1.0,
            "enabled": False,
            "max_price_levels": 5000,
        }

        analyzer = LiquidityPeaksAnalyzer(price_aggregation_config=aggregation_config)

        result = analyzer.analyze_liquidity_peaks(snapshot)

        # Should still have results but with original prices
        assert "liquidity_peaks" in result
        bid_aggregation = result["bid_aggregation"]
        ask_aggregation = result["ask_aggregation"]

        # Original prices should be preserved
        assert Decimal("95000.50") in bid_aggregation
        assert Decimal("95100.50") in ask_aggregation


class TestConfigurationIntegration:
    """Test integration with configuration system."""

    def test_price_aggregation_config_model(self):
        """Test PriceAggregationConfig model."""
        config = PriceAggregationConfig()

        assert config.precision == 1.0
        assert config.enabled is True
        assert config.max_price_levels == 5000

        # Test custom values
        custom_config = PriceAggregationConfig(
            precision=0.1, enabled=False, max_price_levels=1000
        )

        assert custom_config.precision == 0.1
        assert custom_config.enabled is False
        assert custom_config.max_price_levels == 1000

    def test_full_configuration_loading(self):
        """Test loading full configuration with price aggregation settings."""
        from src.utils.config import AnalyzerConfig, PriceAggregationConfig

        # Test direct configuration creation
        price_agg_config = PriceAggregationConfig(
            precision=0.5, enabled=True, max_price_levels=2000
        )

        from src.utils.config import DeepSeekConfig

        deepseek_config = DeepSeekConfig(enable=False, api_key="test")
        analyzer_config = AnalyzerConfig(
            deepseek=deepseek_config, price_aggregation=price_agg_config
        )

        assert analyzer_config.price_aggregation.precision == 0.5
        assert analyzer_config.price_aggregation.enabled is True
        assert analyzer_config.price_aggregation.max_price_levels == 2000

    def test_configuration_validation(self):
        """Test configuration validation."""
        from src.utils.config import PriceAggregationConfig

        # Test valid configuration
        valid_config = PriceAggregationConfig(
            precision=1.0, enabled=True, max_price_levels=5000
        )
        assert valid_config.precision == 1.0
        assert valid_config.enabled is True
        assert valid_config.max_price_levels == 5000

        # Test invalid precision (should raise validation error)
        with pytest.raises(
            ValueError, match="Price aggregation precision must be positive"
        ):
            PriceAggregationConfig(precision=-1.0)

        # Test invalid max_price_levels (should raise validation error)
        with pytest.raises(ValueError, match="Max price levels must be positive"):
            PriceAggregationConfig(max_price_levels=0)


if __name__ == "__main__":
    pytest.main([__file__])
