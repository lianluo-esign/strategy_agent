"""Unit tests for normal distribution analyzer."""

import pytest
from decimal import Decimal
from src.core.normal_distribution_analyzer import (
    OrderBookAggregator,
    NormalDistributionAnalyzer,
    NormalDistributionPeakAnalyzer,
    convert_to_decimal_format,
)


class TestOrderBookAggregator:
    """Test cases for OrderBookAggregator."""

    def setup_method(self):
        """Set up test fixtures."""
        self.aggregator = OrderBookAggregator(price_precision=1.0)

    def test_aggregate_to_dollar_precision_basic(self):
        """Test basic aggregation functionality."""
        order_book_data = {
            'bids': [(100.7, 1.5), (100.3, 2.0), (101.2, 0.8)],
            'asks': [(102.1, 1.2), (102.8, 0.9), (103.5, 1.1)]
        }

        aggregated_bids, aggregated_asks = self.aggregator.aggregate_to_dollar_precision(
            order_book_data
        )

        # Bids should be floored to nearest dollar
        assert 100.0 in aggregated_bids
        assert 101.0 in aggregated_bids
        assert aggregated_bids[100.0] == 3.5  # 1.5 + 2.0
        assert aggregated_bids[101.0] == 0.8

        # Asks should be floored to nearest dollar
        assert 102.0 in aggregated_asks
        assert 103.0 in aggregated_asks
        assert aggregated_asks[102.0] == 2.1  # 1.2 + 0.9
        assert aggregated_asks[103.0] == 1.1

    def test_aggregate_empty_order_book(self):
        """Test aggregation with empty order book."""
        order_book_data = {'bids': [], 'asks': []}

        aggregated_bids, aggregated_asks = self.aggregator.aggregate_to_dollar_precision(
            order_book_data
        )

        assert aggregated_bids == {}
        assert aggregated_asks == {}

    def test_aggregate_missing_data(self):
        """Test aggregation with missing bid/ask data."""
        order_book_data = {'bids': [(100.5, 1.0)]}

        aggregated_bids, aggregated_asks = self.aggregator.aggregate_to_dollar_precision(
            order_book_data
        )

        assert aggregated_bids == {100.0: 1.0}
        assert aggregated_asks == {}

    def test_round_to_dollar(self):
        """Test price rounding functionality."""
        assert self.aggregator._round_to_dollar(100.1) == 100.0
        assert self.aggregator._round_to_dollar(100.9) == 100.0
        assert self.aggregator._round_to_dollar(101.0) == 101.0
        assert self.aggregator._round_to_dollar(0.9) == 0.0

    def test_different_price_precision(self):
        """Test aggregator with different price precision."""
        aggregator = OrderBookAggregator(price_precision=0.5)
        order_book_data = {'bids': [(100.7, 1.0), (100.3, 1.0)]}

        aggregated_bids, _ = aggregator.aggregate_to_dollar_precision(order_book_data)

        # With 0.5 precision using floor:
        # 100.7 -> floor(100.7/0.5)*0.5 = floor(201.4)*0.5 = 201*0.5 = 100.5
        # 100.3 -> floor(100.3/0.5)*0.5 = floor(200.6)*0.5 = 200*0.5 = 100.0
        assert 100.5 in aggregated_bids
        assert 100.0 in aggregated_bids
        assert aggregated_bids[100.5] == 1.0
        assert aggregated_bids[100.0] == 1.0


class TestNormalDistributionAnalyzer:
    """Test cases for NormalDistributionAnalyzer."""

    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = NormalDistributionAnalyzer(confidence_level=0.95)

    def test_find_peak_interval_basic(self):
        """Test basic peak interval finding."""
        price_quantities = {
            100.0: 1.0,
            101.0: 2.0,
            102.0: 5.0,  # Peak
            103.0: 3.0,
            104.0: 1.0,
        }

        mean_price, lower_bound, upper_bound = self.analyzer.find_peak_interval(price_quantities)

        assert mean_price is not None
        assert lower_bound is not None
        assert upper_bound is not None
        assert lower_bound <= mean_price <= upper_bound
        # Mean should be close to 102 due to higher weight
        assert abs(mean_price - 102.0) < 1.0

    def test_find_peak_interval_empty(self):
        """Test peak interval finding with empty data."""
        result = self.analyzer.find_peak_interval({})
        assert result == (None, None, None)

    def test_find_peak_interval_single_point(self):
        """Test peak interval finding with single data point."""
        price_quantities = {100.0: 1.0}

        mean_price, lower_bound, upper_bound = self.analyzer.find_peak_interval(price_quantities)

        assert mean_price == 100.0
        assert lower_bound == 100.0
        assert upper_bound == 100.0

    def test_find_peak_interval_zero_variance(self):
        """Test peak interval finding with zero variance."""
        price_quantities = {100.0: 1.0, 100.0: 2.0}  # Same price

        mean_price, lower_bound, upper_bound = self.analyzer.find_peak_interval(price_quantities)

        assert mean_price == 100.0
        assert lower_bound == 100.0
        assert upper_bound == 100.0

    def test_analyze_distribution_peaks(self):
        """Test complete distribution peak analysis."""
        aggregated_bids = {100.0: 2.0, 101.0: 5.0, 102.0: 1.0}
        aggregated_asks = {105.0: 1.0, 106.0: 4.0, 107.0: 2.0}

        results = self.analyzer.analyze_distribution_peaks(aggregated_bids, aggregated_asks)

        assert 'bids' in results
        assert 'asks' in results

        # Check bid analysis
        bid_analysis = results['bids']
        assert 'mean_price' in bid_analysis
        assert 'peak_interval' in bid_analysis
        assert 'total_volume' in bid_analysis
        assert 'peak_volume' in bid_analysis
        assert bid_analysis['total_volume'] == 8.0
        assert bid_analysis['z_score'] == 1.96
        assert bid_analysis['confidence_level'] == 0.95

        # Check ask analysis
        ask_analysis = results['asks']
        assert ask_analysis['total_volume'] == 7.0
        assert ask_analysis['z_score'] == 1.96

    def test_get_volume_in_interval(self):
        """Test volume calculation within interval."""
        price_quantities = {
            100.0: 1.0,
            101.0: 2.0,
            102.0: 3.0,
            103.0: 1.0,
        }

        volume = self.analyzer._get_volume_in_interval(price_quantities, 101.0, 102.0)
        assert volume == 5.0  # 2.0 + 3.0

        volume = self.analyzer._get_volume_in_interval(price_quantities, 100.5, 101.5)
        assert volume == 2.0  # Only 101.0

        volume = self.analyzer._get_volume_in_interval(price_quantities, None, None)
        assert volume == 0.0

    def test_different_confidence_levels(self):
        """Test analyzer with different confidence levels."""
        analyzer_90 = NormalDistributionAnalyzer(confidence_level=0.90)
        analyzer_99 = NormalDistributionAnalyzer(confidence_level=0.99)

        assert analyzer_90._z_score == 1.645
        assert analyzer_99._z_score == 2.576
        assert self.analyzer._z_score == 1.96


class TestNormalDistributionPeakAnalyzer:
    """Test cases for NormalDistributionPeakAnalyzer."""

    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = NormalDistributionPeakAnalyzer(
            price_precision=1.0, confidence_level=0.95
        )

    def test_analyze_order_book_complete(self):
        """Test complete order book analysis."""
        order_book_data = {
            'bids': [(100.3, 1.5), (101.2, 3.0), (102.1, 1.0)],
            'asks': [(103.4, 2.0), (104.2, 4.0), (105.1, 1.5)]
        }

        result = self.analyzer.analyze_order_book(order_book_data)

        # Check structure
        assert 'aggregated_bids' in result
        assert 'aggregated_asks' in result
        assert 'peak_analysis' in result
        assert 'spread_analysis' in result
        assert 'market_metrics' in result

        # Check aggregated data
        assert len(result['aggregated_bids']) == 3
        assert len(result['aggregated_asks']) == 3

        # Check peak analysis
        assert 'bids' in result['peak_analysis']
        assert 'asks' in result['peak_analysis']

        # Check spread analysis
        spread = result['spread_analysis']
        assert 'best_bid' in spread
        assert 'best_ask' in spread
        assert 'spread' in spread
        assert spread['best_bid'] > spread['best_ask'] - 10  # Reasonable spread

        # Check market metrics
        metrics = result['market_metrics']
        assert 'total_bid_volume' in metrics
        assert 'total_ask_volume' in metrics
        assert 'total_volume' in metrics

    def test_analyze_order_book_empty(self):
        """Test analysis with empty order book."""
        order_book_data = {'bids': [], 'asks': []}

        result = self.analyzer.analyze_order_book(order_book_data)

        assert result['aggregated_bids'] == {}
        assert result['aggregated_asks'] == {}
        assert result['spread_analysis'] == {}
        assert result['market_metrics']['total_volume'] == 0

    def test_analyze_spread(self):
        """Test spread analysis functionality."""
        bids = {100.0: 1.0, 99.0: 2.0}
        asks = {102.0: 1.5, 103.0: 1.0}

        spread = self.analyzer._analyze_spread(bids, asks)

        assert spread['best_bid'] == 100.0
        assert spread['best_ask'] == 102.0
        assert spread['spread'] == 2.0
        assert spread['mid_price'] == 101.0
        assert spread['spread_percentage'] == (2.0 / 102.0) * 100

    def test_calculate_market_metrics(self):
        """Test market metrics calculation."""
        bids = {100.0: 1.0, 99.0: 2.0}
        asks = {102.0: 1.5, 103.0: 1.0}

        metrics = self.analyzer._calculate_market_metrics(bids, asks)

        assert metrics['total_bid_volume'] == 3.0
        assert metrics['total_ask_volume'] == 2.5
        assert metrics['total_volume'] == 5.5
        assert metrics['bid_ask_ratio'] == 3.0 / 2.5
        assert metrics['volume_imbalance'] == (3.0 - 2.5) / 5.5
        assert metrics['price_levels_count']['bid_levels'] == 2
        assert metrics['price_levels_count']['ask_levels'] == 2
        assert metrics['price_levels_count']['total_levels'] == 4

    def test_error_handling(self):
        """Test error handling in analysis."""
        # Test with malformed data that might cause errors
        malformed_data = {
            'bids': [(float('inf'), 1.0)],  # Invalid price
            'asks': [(103.0, 1.0)]
        }

        result = self.analyzer.analyze_order_book(malformed_data)

        # Should not crash, should return error info
        assert 'error' in result


class TestConvertToDecimalFormat:
    """Test cases for decimal format conversion."""

    def test_convert_basic_analysis(self):
        """Test basic conversion to decimal format."""
        analysis_result = {
            'aggregated_bids': {100.0: 1.5, 101.0: 2.0},
            'aggregated_asks': {102.0: 1.0, 103.0: 2.5},
            'peak_analysis': {
                'bids': {
                    'mean_price': 100.5,
                    'peak_interval': (99.5, 101.5),
                    'total_volume': 3.5,
                    'peak_volume': 2.0
                }
            },
            'spread_analysis': {
                'best_bid': 101.0,
                'best_ask': 102.0,
                'spread': 1.0,
                'mid_price': 101.5,
                'spread_percentage': 0.98
            }
        }

        decimal_result = convert_to_decimal_format(analysis_result)

        # Check aggregated data conversion
        assert Decimal('100.0') in decimal_result['aggregated_bids']
        assert Decimal('101.0') in decimal_result['aggregated_bids']
        assert decimal_result['aggregated_bids'][Decimal('100.0')] == Decimal('1.5')

        # Check spread analysis conversion
        spread = decimal_result['spread_analysis']
        assert spread['best_bid'] == Decimal('101.0')
        assert spread['best_ask'] == Decimal('102.0')
        assert spread['spread'] == Decimal('1.0')
        assert spread['mid_price'] == Decimal('101.5')

        # Check peak analysis conversion
        bids = decimal_result['peak_analysis']['bids']
        assert bids['mean_price'] == Decimal('100.5')
        assert bids['peak_interval'][0] == Decimal('99.5')
        assert bids['peak_interval'][1] == Decimal('101.5')
        assert bids['total_volume'] == Decimal('3.5')

    def test_convert_error_result(self):
        """Test conversion with error result."""
        error_result = {'error': 'Test error'}

        decimal_result = convert_to_decimal_format(error_result)

        assert decimal_result == error_result

    def test_convert_empty_result(self):
        """Test conversion with empty result."""
        empty_result = {}

        decimal_result = convert_to_decimal_format(empty_result)

        assert decimal_result['aggregated_bids'] == {}
        assert decimal_result['aggregated_asks'] == {}
        # Empty result should have empty dictionaries for all fields
        assert decimal_result.get('spread_analysis') == {}
        assert decimal_result.get('peak_analysis') == {}


class TestIntegration:
    """Integration tests for the complete normal distribution analysis."""

    def test_complete_analysis_workflow(self):
        """Test the complete analysis workflow."""
        analyzer = NormalDistributionPeakAnalyzer(confidence_level=0.95)

        # Simulate realistic order book data
        order_book_data = {
            'bids': [
                (99850.5, 0.5),
                (99851.2, 1.2),
                (99852.8, 2.1),
                (99853.1, 3.5),  # Peak around 99853
                (99854.6, 2.8),
                (99855.3, 1.1),
            ],
            'asks': [
                (99856.7, 1.3),
                (99857.4, 2.4),
                (99858.9, 4.2),  # Peak around 99859
                (99859.2, 3.1),
                (99860.5, 1.8),
                (99861.1, 0.9),
            ]
        }

        result = analyzer.analyze_order_book(order_book_data)

        # Verify the complete analysis
        assert 'aggregated_bids' in result
        assert 'aggregated_asks' in result
        assert 'peak_analysis' in result
        assert 'spread_analysis' in result
        assert 'market_metrics' in result

        # Check that peaks were identified
        assert 'bids' in result['peak_analysis']
        assert 'asks' in result['peak_analysis']

        bid_peak = result['peak_analysis']['bids']
        ask_peak = result['peak_analysis']['asks']

        # Verify bid peak is reasonable
        assert bid_peak['mean_price'] is not None
        assert bid_peak['peak_interval'] is not None
        assert bid_peak['confidence_level'] == 0.95

        # Verify ask peak is reasonable
        assert ask_peak['mean_price'] is not None
        assert ask_peak['peak_interval'] is not None
        assert ask_peak['confidence_level'] == 0.95

        # Verify spread analysis
        spread = result['spread_analysis']
        assert spread['best_bid'] < spread['best_ask']
        assert spread['spread'] > 0

        # Verify market metrics
        metrics = result['market_metrics']
        assert metrics['total_volume'] > 0
        assert metrics['total_bid_volume'] > 0
        assert metrics['total_ask_volume'] > 0

    def test_analysis_with_insufficient_data(self):
        """Test analysis with insufficient data."""
        analyzer = NormalDistributionPeakAnalyzer()

        # Test with minimal data
        minimal_data = {
            'bids': [(100.0, 0.1)],
            'asks': [(101.0, 0.1)]
        }

        result = analyzer.analyze_order_book(minimal_data)

        # Should still produce valid structure
        assert 'peak_analysis' in result
        assert 'spread_analysis' in result
        assert 'market_metrics' in result

        # But with limited statistical significance
        assert result['market_metrics']['total_volume'] == 0.2