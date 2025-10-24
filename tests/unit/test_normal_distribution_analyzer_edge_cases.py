"""Additional edge case tests for normal distribution analyzer."""

import pytest
from decimal import Decimal
from src.core.normal_distribution_analyzer import (
    OrderBookAggregator,
    NormalDistributionAnalyzer,
    NormalDistributionPeakAnalyzer,
    convert_to_decimal_format,
)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = NormalDistributionPeakAnalyzer(
            price_precision=1.0, confidence_level=0.95
        )

    def test_extreme_price_values(self):
        """Test with extreme price values."""
        # Test with very high prices
        high_price_data = {
            'bids': [(1_000_000.0, 1.0), (1_001_000.0, 2.0)],
            'asks': [(1_002_000.0, 1.5), (1_003_000.0, 0.8)]
        }

        result = self.analyzer.analyze_order_book(high_price_data)

        # Should handle extreme values without overflow
        assert result['aggregated_bids']
        assert result['aggregated_asks']
        assert 'peak_analysis' in result
        assert 'spread_analysis' in result

        # Check that spread is reasonable
        spread = result['spread_analysis']
        assert spread['spread'] > 0

    def test_zero_volume_data(self):
        """Test with zero volume data."""
        zero_volume_data = {
            'bids': [(100.0, 0.0), (101.0, 0.0)],
            'asks': [(102.0, 0.0), (103.0, 0.0)]
        }

        result = self.analyzer.analyze_order_book(zero_volume_data)

        # Should handle zero volumes gracefully
        assert result['aggregated_bids'] == {100.0: 0.0, 101.0: 0.0}
        assert result['aggregated_asks'] == {102.0: 0.0, 103.0: 0.0}
        assert result['market_metrics']['total_volume'] == 0.0

    def test_single_sided_order_book(self):
        """Test with only bids or only asks."""
        # Only bids
        bids_only = {'bids': [(100.0, 1.0), (101.0, 2.0)], 'asks': []}
        result_bids = self.analyzer.analyze_order_book(bids_only)

        assert result_bids['aggregated_bids']
        assert result_bids['aggregated_asks'] == {}
        assert 'bids' in result_bids['peak_analysis']
        assert 'asks' not in result_bids['peak_analysis']

        # Only asks
        asks_only = {'bids': [], 'asks': [(102.0, 1.5), (103.0, 2.0)]}
        result_asks = self.analyzer.analyze_order_book(asks_only)

        assert result_asks['aggregated_bids'] == {}
        assert result_asks['aggregated_asks']
        assert 'bids' not in result_asks['peak_analysis']
        assert 'asks' in result_asks['peak_analysis']

    def test_duplicate_prices(self):
        """Test with duplicate price levels."""
        duplicate_data = {
            'bids': [(100.0, 1.0), (100.5, 2.0), (100.2, 1.5)],  # All aggregate to 100.0
            'asks': [(102.0, 2.0), (102.8, 1.0), (102.3, 0.8)]  # All aggregate to 102.0
        }

        result = self.analyzer.analyze_order_book(duplicate_data)

        # Should aggregate duplicate prices
        assert result['aggregated_bids'][100.0] == 4.5  # 1.0 + 2.0 + 1.5
        assert result['aggregated_asks'][102.0] == 3.8  # 2.0 + 1.0 + 0.8
        assert len(result['aggregated_bids']) == 1
        assert len(result['aggregated_asks']) == 1

    def test_very_small_quantities(self):
        """Test with very small quantities."""
        small_quantity_data = {
            'bids': [(100.0, 1e-10), (101.0, 1e-8)],
            'asks': [(102.0, 1e-12), (103.0, 1e-6)]
        }

        result = self.analyzer.analyze_order_book(small_quantity_data)

        # Should handle very small quantities
        assert result['aggregated_bids']
        assert result['aggregated_asks']
        assert result['market_metrics']['total_volume'] > 0

    def test_large_order_book(self):
        """Test with large order book (many price levels)."""
        large_data = {
            'bids': [(float(50000 + i), 0.1 * (i + 1)) for i in range(1000)],
            'asks': [(float(50100 + i), 0.1 * (1001 - i)) for i in range(1000)]
        }

        result = self.analyzer.analyze_order_book(large_data)

        # Should handle large datasets
        assert len(result['aggregated_bids']) == 1000
        assert len(result['aggregated_asks']) == 1000
        assert result['market_metrics']['price_levels_count']['total_levels'] == 2000

    def test_negative_prices(self):
        """Test with negative prices (should be handled gracefully)."""
        negative_data = {
            'bids': [(-100.0, 1.0), (-101.0, 2.0)],
            'asks': [(-98.0, 1.5), (-97.0, 0.8)]
        }

        result = self.analyzer.analyze_order_book(negative_data)

        # Should handle negative prices
        assert result['aggregated_bids']
        assert result['aggregated_asks']
        assert result['spread_analysis']['best_bid'] < result['spread_analysis']['best_ask']

    def test_malformed_data_handling(self):
        """Test handling of malformed data structures."""
        # Test with non-numeric values
        malformed_data = {
            'bids': [('invalid', 1.0), (100.0, 'invalid')],
            'asks': [(102.0, 1.5)]
        }

        # Should handle gracefully and return error result
        result = self.analyzer.analyze_order_book(malformed_data)

        # Should contain error information instead of crashing
        assert 'error' in result
        assert result['aggregated_bids'] == {}
        assert result['aggregated_asks'] == {}

    def test_infinite_values(self):
        """Test with infinite values."""
        infinite_data = {
            'bids': [(float('inf'), 1.0), (100.0, 2.0)],
            'asks': [(102.0, 1.5), (float('nan'), 0.8)]
        }

        # Should handle infinite/NaN values gracefully
        result = self.analyzer.analyze_order_book(infinite_data)

        # Should contain error information instead of crashing
        assert 'error' in result
        assert result['aggregated_bids'] == {}
        assert result['aggregated_asks'] == {}

    def test_convert_decimal_format_edge_cases(self):
        """Test decimal format conversion with edge cases."""
        # Test with None values in intervals
        analysis_with_none = {
            'aggregated_bids': {100.0: 1.0},
            'aggregated_asks': {102.0: 2.0},
            'peak_analysis': {
                'bids': {
                    'mean_price': 100.5,
                    'peak_interval': (None, 101.0),  # None lower bound
                    'total_volume': 1.0
                }
            },
            'spread_analysis': {
                'best_bid': 100.0,
                'best_ask': 102.0
            }
        }

        result = convert_to_decimal_format(analysis_with_none)

        # Should handle None in intervals
        bid_peak = result['peak_analysis']['bids']
        assert bid_peak['mean_price'] == Decimal('100.5')
        assert bid_peak['peak_interval'] == (None, Decimal('101.0'))

    def test_convert_decimal_format_missing_fields(self):
        """Test decimal format conversion with missing fields."""
        minimal_result = {
            'aggregated_bids': {100.0: 1.0}
            # Missing other fields
        }

        result = convert_to_decimal_format(minimal_result)

        # Should add missing fields
        assert 'aggregated_bids' in result
        assert 'aggregated_asks' in result
        assert 'spread_analysis' in result
        assert 'peak_analysis' in result
        assert result['aggregated_bids'] == {Decimal('100.0'): Decimal('1.0')}
        assert result['aggregated_asks'] == {}

    def test_confidence_level_extremes(self):
        """Test with extreme confidence levels."""
        # Test with very high confidence
        analyzer_999 = NormalDistributionAnalyzer(confidence_level=0.999)
        price_data = {100.0: 1.0, 101.0: 2.0, 102.0: 5.0}

        mean_999, lower_999, upper_999 = analyzer_999.find_peak_interval(price_data)

        # Should produce wide intervals
        assert mean_999 is not None
        assert lower_999 is not None
        assert upper_999 is not None
        assert (upper_999 - lower_999) > (mean_999 - lower_999) * 1.5  # Wide interval

        # Test with very low confidence
        analyzer_50 = NormalDistributionAnalyzer(confidence_level=0.5)
        mean_50, lower_50, upper_50 = analyzer_50.find_peak_interval(price_data)

        # Should produce narrow intervals (zero variance case)
        assert mean_50 is not None
        assert lower_50 is not None
        assert upper_50 is not None
        assert (upper_50 - lower_50) <= (upper_999 - lower_999) * 0.3  # Much narrower

    def test_memory_efficiency(self):
        """Test memory efficiency with large datasets."""
        import sys

        # Create a large dataset
        large_data = {
            'bids': [(float(i), 0.1) for i in range(10000)],
            'asks': [(float(10000 + i), 0.1) for i in range(10000)]
        }

        # Measure memory before
        result = self.analyzer.analyze_order_book(large_data)

        # Verify structure is maintained
        assert 'aggregated_bids' in result
        assert 'aggregated_asks' in result
        assert 'peak_analysis' in result
        assert 'market_metrics' in result

        # Memory usage should be reasonable (basic check)
        assert sys.getsizeof(result) < 10 * 1024 * 1024  # Less than 10MB

    def test_concurrent_analysis_safety(self):
        """Test that analysis is thread-safe (basic check)."""
        import threading
        import time

        results = []
        errors = []

        def analyze_data():
            try:
                test_data = {
                    'bids': [(100.0 + i, 1.0) for i in range(10)],
                    'asks': [(102.0 + i, 1.0) for i in range(10)]
                }
                result = self.analyzer.analyze_order_book(test_data)
                results.append(result)
            except Exception as e:
                errors.append(e)

        # Run multiple analyses in parallel
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=analyze_data)
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Should complete without errors
        assert len(errors) == 0
        assert len(results) == 5
        assert all('peak_analysis' in result for result in results)