"""Tests for analyzer type fixes."""

import pytest
from decimal import Decimal
from datetime import datetime

from src.core.analyzers import (
    _to_decimal,
    _safe_decimal_division,
    DepthSnapshotAnalyzer,
    OrderFlowAnalyzer,
    MarketAnalyzer
)
from src.core.models import (
    DepthSnapshot,
    DepthLevel,
    MinuteTradeData,
    PriceLevelData
)


class TestAnalyzerFixes:
    """Test analyzer type error fixes."""

    def test_to_decimal_conversion(self):
        """Test _to_decimal function handles various types."""
        # Test Decimal input
        d = Decimal('10.5')
        assert _to_decimal(d) == d

        # Test float input
        assert _to_decimal(10.5) == Decimal('10.5')
        assert _to_decimal(10) == Decimal('10')

        # Test string input
        assert _to_decimal('10.5') == Decimal('10.5')
        assert _to_decimal('10') == Decimal('10')

        # Test invalid input
        with pytest.raises(TypeError):
            _to_decimal([1, 2, 3])

    def test_safe_decimal_division(self):
        """Test safe decimal division function."""
        numerator = Decimal('10')

        # Test division by float
        result = _safe_decimal_division(numerator, 2.0)
        assert result == Decimal('5.0')

        # Test division by int
        result = _safe_decimal_division(numerator, 2)
        assert result == Decimal('5.0')

    def test_depth_analyzer_with_decimal_values(self):
        """Test depth analyzer handles Decimal values correctly."""
        analyzer = DepthSnapshotAnalyzer(min_volume_threshold=0.1, price_zone_size=0.5)

        # Create test data with Decimal values
        bids = [
            DepthLevel(price=Decimal('50000.0'), quantity=Decimal('1.5')),
            DepthLevel(price=Decimal('49999.0'), quantity=Decimal('2.0')),
            DepthLevel(price=Decimal('49998.0'), quantity=Decimal('0.8')),
        ]

        asks = [
            DepthLevel(price=Decimal('50001.0'), quantity=Decimal('1.0')),
            DepthLevel(price=Decimal('50002.0'), quantity=Decimal('1.2')),
            DepthLevel(price=Decimal('50003.0'), quantity=Decimal('0.5')),
        ]

        snapshot = DepthSnapshot(
            symbol='BTCFDUSD',
            timestamp=datetime.now(),
            bids=bids,
            asks=asks
        )

        # Should not raise type errors
        support, resistance = analyzer.analyze_support_resistance(snapshot)

        assert isinstance(support, list)
        assert isinstance(resistance, list)

        # Test liquidity vacuum zones
        vacuum_zones = analyzer.identify_liquidity_vacuum_zones(snapshot)
        assert isinstance(vacuum_zones, list)

    def test_order_flow_analyzer_with_dict_data(self):
        """Test order flow analyzer handles dict data structures."""
        analyzer = OrderFlowAnalyzer()

        # Create test trade data with dict price levels
        trade_data = MinuteTradeData(timestamp=datetime.now())

        # Add price levels as dict
        trade_data.price_levels[Decimal('50000')] = {
            'delta': Decimal('0.5'),
            'total_volume': Decimal('10.0')
        }
        trade_data.price_levels[Decimal('50100')] = {
            'delta': Decimal('-0.3'),
            'total_volume': Decimal('8.0')
        }

        # Should not raise type errors
        poc_levels = analyzer._find_poc_levels([trade_data])
        assert isinstance(poc_levels, list)

    def test_market_analyzer_error_handling(self):
        """Test market analyzer handles errors gracefully."""
        analyzer = MarketAnalyzer()

        # Test with no data
        result = analyzer.analyze_market(None, [], 'BTCFDUSD')
        assert result.symbol == 'BTCFDUSD'
        assert len(result.support_levels) == 0

        # Test with malformed data
        from src.core.models import SupportResistanceLevel

        # Create a support level that might cause issues
        bad_support = SupportResistanceLevel(
            price=Decimal('50000'),
            strength=0.5,
            level_type='support',
            volume_at_level=Decimal('10'),
            confirmation_count=1,
            last_confirmed=datetime.now()
        )

        # Create a snapshot with minimal data
        snapshot = DepthSnapshot(
            symbol='BTCFDUSD',
            timestamp=datetime.now(),
            bids=[DepthLevel(price=Decimal('50000'), quantity=Decimal('1'))],
            asks=[DepthLevel(price=Decimal('50001'), quantity=Decimal('1'))]
        )

        # Should handle errors gracefully
        result = analyzer.analyze_market(snapshot, [], 'BTCFDUSD')
        assert result is not None
        assert result.symbol == 'BTCFDUSD'

    def test_type_consistency_in_calculations(self):
        """Test that all calculations maintain type consistency."""
        analyzer = DepthSnapshotAnalyzer()

        # Test volume calculations
        levels = [
            DepthLevel(price=Decimal('50000'), quantity=Decimal('1.5')),
            DepthLevel(price=Decimal('50001'), quantity=Decimal('2.5')),
        ]

        total_volume = sum(level.quantity for level in levels)
        assert isinstance(total_volume, Decimal)
        assert total_volume == Decimal('4.0')

        # Test zone calculations
        zone_size = Decimal('0.5')
        zones = analyzer._group_by_price_zones(levels, zone_size)
        assert isinstance(zones, dict)

        for zone_price, zone_levels in zones.items():
            assert isinstance(zone_price, Decimal)
            assert isinstance(zone_levels, list)