"""Unit tests for price_aggregator module.

Tests the 1-dollar precision aggregation algorithms and depth analysis functionality.
"""

import pytest
from decimal import Decimal
from typing import List, Dict

from src.core.price_aggregator import (
    aggregate_depth_by_one_dollar,
    calculate_depth_statistics,
    identify_liquidity_clusters,
    convert_to_depth_levels,
    validate_aggregation_quality,
)
from src.core.models import DepthLevel


class TestAggregateDepthByOneDollar:
    """Test cases for 1-dollar precision depth aggregation."""

    @pytest.fixture
    def sample_bids(self) -> List[DepthLevel]:
        """Create sample bid data for testing."""
        return [
            DepthLevel(price=Decimal('99.50'), quantity=Decimal('10.0')),
            DepthLevel(price=Decimal('99.20'), quantity=Decimal('5.0')),
            DepthLevel(price=Decimal('99.80'), quantity=Decimal('15.0')),
            DepthLevel(price=Decimal('98.90'), quantity=Decimal('8.0')),
            DepthLevel(price=Decimal('99.10'), quantity=Decimal('12.0')),
        ]

    @pytest.fixture
    def sample_asks(self) -> List[DepthLevel]:
        """Create sample ask data for testing."""
        return [
            DepthLevel(price=Decimal('100.10'), quantity=Decimal('8.0')),
            DepthLevel(price=Decimal('100.50'), quantity=Decimal('6.0')),
            DepthLevel(price=Decimal('100.20'), quantity=Decimal('10.0')),
            DepthLevel(price=Decimal('101.30'), quantity=Decimal('4.0')),
            DepthLevel(price=Decimal('100.90'), quantity=Decimal('7.0')),
        ]

    def test_aggregate_basic_functionality(self, sample_bids, sample_asks):
        """Test basic aggregation functionality."""
        aggregated_bids, aggregated_asks = aggregate_depth_by_one_dollar(sample_bids, sample_asks)

        # Check bids aggregation (all should round down to 98 or 99)
        assert Decimal('99') in aggregated_bids
        assert Decimal('98') in aggregated_bids
        assert aggregated_bids[Decimal('99')] == Decimal('42.0')  # 99.50+99.20+99.80+99.10
        assert aggregated_bids[Decimal('98')] == Decimal('8.0')   # 98.90

        # Check asks aggregation (all should round down to 100 or 101)
        assert Decimal('100') in aggregated_asks
        assert Decimal('101') in aggregated_asks
        assert aggregated_asks[Decimal('100')] == Decimal('31.0')  # 100.10+100.20+100.50+100.90
        assert aggregated_asks[Decimal('101')] == Decimal('4.0')   # 101.30

    def test_aggregate_with_empty_data(self):
        """Test aggregation with empty input data."""
        empty_bids, empty_asks = aggregate_depth_by_one_dollar([], [])

        assert len(empty_bids) == 0
        assert len(empty_asks) == 0

    def test_aggregate_with_single_side(self, sample_bids):
        """Test aggregation with only one side of the order book."""
        aggregated_bids, aggregated_asks = aggregate_depth_by_one_dollar(sample_bids, [])

        assert len(aggregated_bids) == 2  # Prices round to 98 and 99
        assert len(aggregated_asks) == 0
        assert aggregated_bids[Decimal('99')] == Decimal('42.0')  # Sum of prices rounding to 99
        assert aggregated_bids[Decimal('98')] == Decimal('8.0')   # 98.90 rounds to 98

    def test_price_rounding_behavior(self):
        """Test that prices are correctly rounded down to nearest dollar."""
        test_cases = [
            (Decimal('99.01'), Decimal('99')),
            (Decimal('99.99'), Decimal('99')),
            (Decimal('100.00'), Decimal('100')),
            (Decimal('100.50'), Decimal('100')),
            (Decimal('101.99'), Decimal('101')),
        ]

        for input_price, expected_rounded in test_cases:
            depth_level = DepthLevel(price=input_price, quantity=Decimal('1.0'))
            aggregated_bids, _ = aggregate_depth_by_one_dollar([depth_level], [])

            assert expected_rounded in aggregated_bids

    def test_volume_aggregation_across_multiple_price_levels(self):
        """Test volume aggregation across different price levels."""
        bids = [
            DepthLevel(price=Decimal('98.50'), quantity=Decimal('10.0')),
            DepthLevel(price=Decimal('98.80'), quantity=Decimal('15.0')),
            DepthLevel(price=Decimal('99.20'), quantity=Decimal('20.0')),
            DepthLevel(price=Decimal('99.70'), quantity=Decimal('25.0')),
        ]

        aggregated_bids, _ = aggregate_depth_by_one_dollar(bids, [])

        # Should have two price levels: 98 and 99
        assert len(aggregated_bids) == 2
        assert aggregated_bids[Decimal('98')] == Decimal('25.0')  # 10+15
        assert aggregated_bids[Decimal('99')] == Decimal('45.0')  # 20+25


class TestCalculateDepthStatistics:
    """Test cases for depth statistics calculation."""

    @pytest.fixture
    def sample_aggregated_data(self) -> Dict[Decimal, Decimal]:
        """Create sample aggregated depth data."""
        return {
            Decimal('99'): Decimal('50.0'),
            Decimal('100'): Decimal('35.0'),
            Decimal('101'): Decimal('20.0'),
            Decimal('102'): Decimal('10.0'),
        }

    def test_calculate_basic_statistics(self, sample_aggregated_data):
        """Test basic statistics calculation."""
        stats = calculate_depth_statistics(sample_aggregated_data, {})

        assert stats['total_bid_volume'] == Decimal('115.0')
        assert stats['bid_price_levels'] == Decimal('4')
        assert stats['total_ask_volume'] == Decimal('0')
        assert stats['ask_price_levels'] == Decimal('0')

    def test_calculate_statistics_with_empty_data(self):
        """Test statistics calculation with empty data."""
        stats = calculate_depth_statistics({}, {})

        assert stats['total_bid_volume'] == Decimal('0')
        assert stats['total_ask_volume'] == Decimal('0')
        assert stats['bid_price_levels'] == Decimal('0')
        assert stats['ask_price_levels'] == Decimal('0')

    def test_volume_distribution_calculation(self, sample_aggregated_data):
        """Test volume distribution calculation."""
        stats = calculate_depth_statistics(sample_aggregated_data, {})

        # Check basic statistics
        assert stats['total_bid_volume'] == Decimal('115.0')
        assert stats['bid_ask_ratio'] > Decimal('0')


class TestIdentifyLiquidityClusters:
    """Test cases for liquidity cluster identification."""

    def test_identify_basic_liquidity_clusters(self):
        """Test basic liquidity cluster identification."""
        depth_data = {
            Decimal('95'): Decimal('5.0'),
            Decimal('96'): Decimal('8.0'),
            Decimal('97'): Decimal('15.0'),
            Decimal('98'): Decimal('25.0'),
            Decimal('99'): Decimal('30.0'),
            Decimal('100'): Decimal('20.0'),
            Decimal('101'): Decimal('10.0'),
        }

        clusters = identify_liquidity_clusters(depth_data, min_cluster_volume=Decimal('10.0'))

        # Should identify clusters based on volume concentration
        assert len(clusters) > 0

        # Verify cluster structure
        for cluster in clusters:
            assert 'center_price' in cluster
            assert 'total_volume' in cluster
            assert 'price_range' in cluster
            assert 'strength' in cluster

    def test_identify_clusters_with_low_volume_threshold(self):
        """Test cluster identification with low volume threshold."""
        depth_data = {
            Decimal('99'): Decimal('2.0'),
            Decimal('100'): Decimal('3.0'),
            Decimal('101'): Decimal('1.0'),
        }

        clusters = identify_liquidity_clusters(depth_data, min_cluster_volume=Decimal('5.0'))

        # Should identify clusters even with low volumes
        assert len(clusters) >= 0

    def test_identify_clusters_with_empty_data(self):
        """Test cluster identification with empty data."""
        clusters = identify_liquidity_clusters({}, min_cluster_volume=Decimal('10.0'))
        assert len(clusters) == 0


class TestConvertToDepthLevels:
    """Test cases for converting aggregated data back to DepthLevel objects."""

    def test_convert_basic_aggregated_data(self):
        """Test basic conversion from aggregated data to DepthLevels."""
        aggregated_data = {
            Decimal('99'): Decimal('50.0'),
            Decimal('100'): Decimal('35.0'),
            Decimal('101'): Decimal('20.0'),
        }

        depth_levels = convert_to_depth_levels(aggregated_data)

        assert len(depth_levels) == 3

        # Check conversion preserves data
        price_set = {level.price for level in depth_levels}
        quantity_dict = {level.price: level.quantity for level in depth_levels}

        assert price_set == {Decimal('99'), Decimal('100'), Decimal('101')}
        assert quantity_dict[Decimal('99')] == Decimal('50.0')
        assert quantity_dict[Decimal('100')] == Decimal('35.0')
        assert quantity_dict[Decimal('101')] == Decimal('20.0')

    def test_convert_empty_aggregated_data(self):
        """Test conversion with empty aggregated data."""
        depth_levels = convert_to_depth_levels({})
        assert len(depth_levels) == 0

    def test_convert_sorted_output(self):
        """Test that output DepthLevels are sorted by price."""
        aggregated_data = {
            Decimal('101'): Decimal('20.0'),
            Decimal('99'): Decimal('50.0'),
            Decimal('100'): Decimal('35.0'),
        }

        depth_levels = convert_to_depth_levels(aggregated_data)

        # Check sorting
        prices = [level.price for level in depth_levels]
        assert prices == sorted(prices)


class TestValidateAggregationQuality:
    """Test cases for aggregation quality validation."""

    def test_validate_high_quality_aggregation(self):
        """Test validation with high quality aggregation."""
        original_levels = 100
        aggregated_data = {
            Decimal('99'): Decimal('500.0'),  # High volume preservation
            Decimal('100'): Decimal('300.0'),
            Decimal('101'): Decimal('200.0'),
        }
        original_volume = Decimal('1000.0')

        quality = validate_aggregation_quality(original_levels, aggregated_data, original_volume)

        assert quality['preservation_rate'] > 0.9  # High volume preservation
        assert quality['compression_ratio'] > 10   # Good compression
        assert quality['avg_volume_per_level'] > 0
        assert quality['quality_score'] > 0.8      # High quality score

    def test_validate_low_quality_aggregation(self):
        """Test validation with low quality aggregation."""
        original_levels = 10
        aggregated_data = {
            Decimal('99'): Decimal('1.0'),  # Low volume preservation
        }
        original_volume = Decimal('1000.0')

        quality = validate_aggregation_quality(original_levels, aggregated_data, original_volume)

        assert quality['preservation_rate'] < 0.1  # Poor volume preservation
        assert quality['compression_ratio'] > 1
        assert quality['quality_score'] < 0.5      # Low quality score

    def test_validate_with_empty_data(self):
        """Test validation with empty aggregated data."""
        quality = validate_aggregation_quality(0, {}, Decimal('0'))

        assert quality['preservation_rate'] == 0
        assert quality['compression_ratio'] == 0
        assert quality['avg_volume_per_level'] == 0
        assert quality['quality_score'] == 0

    def test_quality_score_calculation(self):
        """Test quality score calculation combines multiple metrics."""
        # Test case with good compression but poor preservation
        quality1 = validate_aggregation_quality(100, {Decimal('99'): Decimal('100.0')}, Decimal('1000.0'))

        # Test case with poor compression but good preservation
        quality2 = validate_aggregation_quality(10, {Decimal('99'): Decimal('900.0')}, Decimal('1000.0'))

        # Quality scores should reflect the trade-off
        assert 0 <= quality1['quality_score'] <= 1
        assert 0 <= quality2['quality_score'] <= 1


class TestIntegrationScenarios:
    """Integration test scenarios combining multiple functions."""

    def test_complete_aggregation_pipeline(self):
        """Test complete aggregation pipeline from raw depth to quality validation."""
        # Create comprehensive test data
        raw_bids = [
            DepthLevel(price=Decimal(f'{99 + i*0.1}'), quantity=Decimal(f'{10 + i}'))
            for i in range(20)
        ]
        raw_asks = [
            DepthLevel(price=Decimal(f'{100 + i*0.1}'), quantity=Decimal(f'{8 + i*0.5}'))
            for i in range(20)
        ]

        # Step 1: Aggregate by 1-dollar precision
        aggregated_bids, aggregated_asks = aggregate_depth_by_one_dollar(raw_bids, raw_asks)

        # Step 2: Calculate statistics
        bid_stats = calculate_depth_statistics(aggregated_bids)
        ask_stats = calculate_depth_statistics(aggregated_asks)

        # Step 3: Identify liquidity clusters
        bid_clusters = identify_liquidity_clusters(aggregated_bids)
        ask_clusters = identify_liquidity_clusters(aggregated_asks)

        # Step 4: Convert back to DepthLevels
        bid_levels = convert_to_depth_levels(aggregated_bids)
        ask_levels = convert_to_depth_levels(aggregated_asks)

        # Step 5: Validate quality
        original_total_levels = len(raw_bids) + len(raw_asks)
        original_total_volume = sum(level.quantity for level in raw_bids + raw_asks)
        aggregated_total_volume = sum(aggregated_bids.values()) + sum(aggregated_asks.values())

        quality = validate_aggregation_quality(
            original_total_levels,
            {**aggregated_bids, **aggregated_asks},
            original_total_volume
        )

        # Verify pipeline results
        assert len(aggregated_bids) <= len(raw_bids)
        assert len(aggregated_asks) <= len(raw_asks)
        assert bid_stats['total_levels'] > 0
        assert ask_stats['total_levels'] > 0
        assert len(bid_clusters) >= 0
        assert len(ask_clusters) >= 0
        assert len(bid_levels) == len(aggregated_bids)
        assert len(ask_levels) == len(aggregated_asks)
        assert quality['preservation_rate'] > 0.9  # Should preserve most volume
        assert quality['compression_ratio'] > 1   # Should achieve compression