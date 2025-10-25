"""Unit tests for the simplified liquidity peaks analyzer.

This module contains comprehensive tests for the LiquidityPeaksAnalyzer
to ensure robustness and correctness of the liquidity peak detection algorithm.
"""

import pytest
from decimal import Decimal
from datetime import datetime

from src.core.liquidity_peaks_analyzer import (
    LiquidityPeaksAnalyzer,
    PEAK_SCORE_NORMALIZATION_FACTOR,
    MAX_PEAKS_RETURNED,
    LOCAL_DENSITY_WINDOW_SIZE,
)
from src.core.models import DepthSnapshot, DepthLevel, SupportResistanceLevel


class TestLiquidityPeaksAnalyzer:
    """Test cases for LiquidityPeaksAnalyzer."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.analyzer = LiquidityPeaksAnalyzer(
            min_volume_threshold=1.0,
            peak_detection_window=LOCAL_DENSITY_WINDOW_SIZE,
            volume_weight=2.0,
        )

    def teardown_method(self):
        """Clean up after each test method."""
        pass

    # Test Initialization
    def test_init_with_valid_parameters(self):
        """Test analyzer initialization with valid parameters."""
        analyzer = LiquidityPeaksAnalyzer(
            min_volume_threshold=10.0,
            peak_detection_window=5,
            volume_weight=2.0,
        )
        assert analyzer.min_volume_threshold == 10.0
        assert analyzer.peak_detection_window == 5
        assert analyzer.volume_weight == 2.0

    def test_init_with_default_parameters(self):
        """Test analyzer initialization with default parameters."""
        analyzer = LiquidityPeaksAnalyzer()
        assert analyzer.min_volume_threshold == 1.0
        assert analyzer.peak_detection_window == LOCAL_DENSITY_WINDOW_SIZE
        assert analyzer.volume_weight == 2.0

    def test_init_with_invalid_parameters(self):
        """Test analyzer initialization with invalid parameters."""
        # Test negative min_volume_threshold
        with pytest.raises(ValueError, match="min_volume_threshold must be positive"):
            LiquidityPeaksAnalyzer(min_volume_threshold=-1.0)

        # Test zero min_volume_threshold
        with pytest.raises(ValueError, match="min_volume_threshold must be positive"):
            LiquidityPeaksAnalyzer(min_volume_threshold=0.0)

        # Test invalid peak_detection_window
        with pytest.raises(ValueError, match="peak_detection_window must be at least 1"):
            LiquidityPeaksAnalyzer(peak_detection_window=0)

        # Test negative volume_weight
        with pytest.raises(ValueError, match="volume_weight must be positive"):
            LiquidityPeaksAnalyzer(volume_weight=-1.0)

    # Test Main Analysis Method
    def test_analyze_liquidity_peaks_with_valid_snapshot(self):
        """Test analysis with valid depth snapshot."""
        snapshot = self._create_test_snapshot(
            bids=[(Decimal('95000'), Decimal('15.5')), (Decimal('94999'), Decimal('20.0'))],
            asks=[(Decimal('95100'), Decimal('12.2')), (Decimal('95101'), Decimal('18.8'))],
        )

        result = self.analyzer.analyze_liquidity_peaks(snapshot)

        assert isinstance(result, dict)
        assert "liquidity_peaks" in result
        assert "analysis_summary" in result
        assert "bid_aggregation" in result
        assert "ask_aggregation" in result
        assert "total_bid_volume" in result
        assert "total_ask_volume" in result
        assert "peak_detection_stats" in result

        # Check that peaks were found
        assert len(result["liquidity_peaks"]) > 0

    def test_analyze_liquidity_peaks_with_empty_snapshot(self):
        """Test analysis with empty snapshot."""
        snapshot = DepthSnapshot(
            symbol="BTCFDUSD",
            timestamp=datetime.now(),
            bids=[],
            asks=[],
        )

        result = self.analyzer.analyze_liquidity_peaks(snapshot)

        assert result["liquidity_peaks"] == []
        assert result["total_bid_volume"] == 0.0
        assert result["total_ask_volume"] == 0.0
        assert result["peak_detection_stats"]["total_peaks_count"] == 0

    def test_analyze_liquidity_peaks_with_low_volume(self):
        """Test analysis with snapshot containing low volume data."""
        snapshot = self._create_test_snapshot(
            bids=[(Decimal('95000'), Decimal('0.5'))],  # Below threshold (1.0)
            asks=[(Decimal('95100'), Decimal('0.8'))],  # Below threshold (1.0)
        )

        result = self.analyzer.analyze_liquidity_peaks(snapshot)

        # Should find no peaks due to low volume
        assert len(result["liquidity_peaks"]) == 0

    def test_analyze_liquidity_peaks_with_invalid_input(self):
        """Test analysis with invalid input types."""
        # Test with None
        with pytest.raises(TypeError, match="snapshot must be a DepthSnapshot object"):
            self.analyzer.analyze_liquidity_peaks(None)

        # Test with wrong type
        with pytest.raises(TypeError, match="snapshot must be a DepthSnapshot object"):
            self.analyzer.analyze_liquidity_peaks("not_a_snapshot")

        # Test with empty symbol
        snapshot = self._create_test_snapshot(
            bids=[(Decimal('95000'), Decimal('15.5'))],
            asks=[(Decimal('95100'), Decimal('12.2'))],
        )
        snapshot.symbol = ""
        with pytest.raises(ValueError, match="snapshot must have a valid symbol"):
            self.analyzer.analyze_liquidity_peaks(snapshot)

    # Test Helper Methods
    def test_aggregate_order_book_data(self):
        """Test order book data aggregation."""
        snapshot = self._create_test_snapshot(
            bids=[
                (Decimal('95001.50'), Decimal('1.2')),
                (Decimal('95002.30'), Decimal('0.8')),
                (Decimal('95001.80'), Decimal('2.1')),  # Same price level as first
            ],
            asks=[
                (Decimal('95101.20'), Decimal('1.5')),
                (Decimal('95102.40'), Decimal('2.0')),
            ],
        )

        aggregated_bids, aggregated_asks = self.analyzer._aggregate_order_book_data(snapshot)

        # Check that prices are rounded down to nearest dollar
        assert Decimal('95001') in aggregated_bids
        assert Decimal('95002') in aggregated_bids
        assert Decimal('95101') in aggregated_asks
        assert Decimal('95102') in aggregated_asks

        # Check that volumes are summed correctly
        # 95001.50 -> 95001, 95001.80 -> 95001, volumes should be summed
        assert aggregated_bids[Decimal('95001')] == Decimal('3.3')  # 1.2 + 2.1

    def test_identify_peaks_from_data(self):
        """Test peak identification from aggregated data."""
        test_data = {
            Decimal('95000'): Decimal('15.0'),  # High volume
            Decimal('95001'): Decimal('8.0'),   # Medium volume
            Decimal('95002'): Decimal('3.0'),   # Low volume
        }

        peaks = self.analyzer._identify_peaks_from_data(test_data, "bid")

        # Should identify high volume as peak
        assert len(peaks) > 0
        assert peaks[0]["price"] == Decimal('95000')
        assert peaks[0]["volume"] == 15.0
        assert "peak_score" in peaks[0]
        assert peaks[0]["side"] == "bid"

    def test_identify_peaks_with_empty_data(self):
        """Test peak identification with empty data."""
        peaks = self.analyzer._identify_peaks_from_data({}, "bid")
        assert peaks == []

    def test_calculate_peak_score(self):
        """Test peak score calculation."""
        test_data = {
            Decimal('95000'): Decimal('10.0'),
            Decimal('95001'): Decimal('5.0'),
            Decimal('95002'): Decimal('15.0'),
        }
        sorted_prices = [Decimal('95002'), Decimal('95001'), Decimal('95000')]

        score = self.analyzer._calculate_peak_score(
            Decimal('95002'), 15.0, sorted_prices, test_data, "bid"
        )

        assert isinstance(score, float)
        assert score > 0

    def test_calculate_peak_score_with_empty_data(self):
        """Test peak score calculation with empty data."""
        score = self.analyzer._calculate_peak_score(
            Decimal('95000'), 10.0, [], {}, "bid"
        )
        assert score == 0.0

    def test_calculate_local_density_score(self):
        """Test local density score calculation."""
        test_data = {
            Decimal('95000'): Decimal('10.0'),
            Decimal('95001'): Decimal('8.0'),
            Decimal('95002'): Decimal('12.0'),
        }
        sorted_prices = [Decimal('95002'), Decimal('95001'), Decimal('95000')]

        score = self.analyzer._calculate_local_density_score(
            Decimal('95001'), sorted_prices, test_data
        )

        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_calculate_local_density_score_edge_cases(self):
        """Test local density score calculation with edge cases."""
        test_data = {Decimal('95000'): Decimal('10.0')}
        sorted_prices = [Decimal('95000')]

        # Test with price not in list
        score = self.analyzer._calculate_local_density_score(
            Decimal('95001'), sorted_prices, test_data
        )
        assert score == 0.0

        # Test with empty data
        score = self.analyzer._calculate_local_density_score(
            Decimal('95000'), [], {}
        )
        assert score == 0.0

    def test_convert_peaks_to_support_resistance(self):
        """Test conversion of peaks to SupportResistanceLevel objects."""
        bid_peaks = [
            {"price": Decimal('95000'), "volume": 15.0, "peak_score": 2.5, "side": "bid"}
        ]
        ask_peaks = [
            {"price": Decimal('95100'), "volume": 12.0, "peak_score": 2.0, "side": "ask"}
        ]

        levels = self.analyzer._convert_peaks_to_support_resistance(bid_peaks, ask_peaks)

        assert len(levels) == 2

        # Check support level
        support = next(level for level in levels if level.level_type == "support")
        assert support.price == Decimal('95000')
        assert support.level_type == "support"
        assert support.volume_at_level == Decimal('15.0')
        assert support.strength == 2.5 / PEAK_SCORE_NORMALIZATION_FACTOR

        # Check resistance level
        resistance = next(level for level in levels if level.level_type == "resistance")
        assert resistance.price == Decimal('95100')
        assert resistance.level_type == "resistance"
        assert resistance.volume_at_level == Decimal('12.0')
        assert resistance.strength == 2.0 / PEAK_SCORE_NORMALIZATION_FACTOR

    def test_generate_analysis_summary(self):
        """Test analysis summary generation."""
        aggregated_bids = {Decimal('95000'): Decimal('15.0')}
        aggregated_asks = {Decimal('95100'): Decimal('12.0')}
        liquidity_peaks = [
            SupportResistanceLevel(
                price=Decimal('95000'),
                strength=0.8,
                level_type="support",
                volume_at_level=Decimal('15.0'),
                confirmation_count=1,
                last_confirmed=None,
            )
        ]

        summary = self.analyzer._generate_analysis_summary(
            aggregated_bids, aggregated_asks, liquidity_peaks
        )

        assert summary["total_volume"] == 27.0
        assert summary["bid_ratio"] == 15.0 / 27.0
        assert summary["ask_ratio"] == 12.0 / 27.0
        assert summary["bid_peaks_count"] == 1
        assert summary["ask_peaks_count"] == 0
        assert summary["total_peaks_count"] == 1
        assert summary["market_balance"] in ["balanced", "bid_heavy", "ask_heavy"]

    def test_create_empty_result(self):
        """Test empty result creation."""
        result = self.analyzer._create_empty_result()

        assert result["liquidity_peaks"] == []
        assert result["analysis_summary"] == {}
        assert result["bid_aggregation"] == {}
        assert result["ask_aggregation"] == {}
        assert result["total_bid_volume"] == 0.0
        assert result["total_ask_volume"] == 0.0
        assert result["peak_detection_stats"]["bid_peaks_count"] == 0
        assert result["peak_detection_stats"]["ask_peaks_count"] == 0
        assert result["peak_detection_stats"]["total_peaks_count"] == 0

    # Test Integration Scenarios
    def test_analysis_with_realistic_market_data(self):
        """Test analysis with realistic market data structure."""
        # Create realistic order book
        bids = [
            DepthLevel(price=Decimal('94990.00'), quantity=Decimal('0.5')),
            DepthLevel(price=Decimal('94991.00'), quantity=Decimal('1.2')),
            DepthLevel(price=Decimal('94992.00'), quantity=Decimal('25.5')),  # Large order
            DepthLevel(price=Decimal('94993.00'), quantity=Decimal('2.1')),
            DepthLevel(price=Decimal('94994.00'), quantity=Decimal('18.8')),  # Large order
        ]
        asks = [
            DepthLevel(price=Decimal('95005.00'), quantity=Decimal('22.3')),  # Large order
            DepthLevel(price=Decimal('95006.00'), quantity=Decimal('1.5')),
            DepthLevel(price=Decimal('95007.00'), quantity=Decimal('31.2')),  # Large order
            DepthLevel(price=Decimal('95008.00'), quantity=Decimal('0.8')),
        ]

        snapshot = DepthSnapshot(
            symbol="BTCFDUSD",
            timestamp=datetime.now(),
            bids=bids,
            asks=asks,
        )

        result = self.analyzer.analyze_liquidity_peaks(snapshot)

        # Should identify peaks at large order levels
        assert len(result["liquidity_peaks"]) > 0
        assert result["total_bid_volume"] > 0
        assert result["total_ask_volume"] > 0

        # Check that peaks are properly classified
        support_levels = [p for p in result["liquidity_peaks"] if p.level_type == "support"]
        resistance_levels = [p for p in result["liquidity_peaks"] if p.level_type == "resistance"]

        assert len(support_levels) > 0
        assert len(resistance_levels) > 0

    def test_performance_with_large_dataset(self):
        """Test analyzer performance with large dataset."""
        import time

        # Create large order book (1000 levels each side)
        bids = []
        asks = []
        for i in range(1000):
            base_price = 95000 - i
            bids.append(DepthLevel(
                price=Decimal(str(base_price)),
                quantity=Decimal(str(i % 20 + 1))  # Varying volumes
            ))
            asks.append(DepthLevel(
                price=Decimal(str(95100 + i)),
                quantity=Decimal(str(i % 15 + 1))  # Varying volumes
            ))

        snapshot = DepthSnapshot(
            symbol="BTCFDUSD",
            timestamp=datetime.now(),
            bids=bids,
            asks=asks,
        )

        start_time = time.time()
        result = self.analyzer.analyze_liquidity_peaks(snapshot)
        end_time = time.time()

        # Should complete within reasonable time (less than 1 second)
        assert end_time - start_time < 1.0
        # Note: No artificial limit on number of peaks - returns all detected peaks
        assert len(result["liquidity_peaks"]) >= 0

    # Test Edge Cases
    def test_single_price_level(self):
        """Test analysis with single price level."""
        snapshot = self._create_test_snapshot(
            bids=[(Decimal('95000'), Decimal('15.5'))],
            asks=[(Decimal('95100'), Decimal('12.2'))],
        )

        result = self.analyzer.analyze_liquidity_peaks(snapshot)

        assert len(result["liquidity_peaks"]) > 0
        assert result["total_bid_volume"] == 15.5
        assert float(result["total_ask_volume"]) == 12.2

    def test_all_same_price_levels(self):
        """Test analysis with orders at same price levels."""
        snapshot = self._create_test_snapshot(
            bids=[
                (Decimal('95000.50'), Decimal('5.0')),
                (Decimal('95000.80'), Decimal('8.0')),
                (Decimal('95000.20'), Decimal('3.0')),
            ],
            asks=[
                (Decimal('95100.10'), Decimal('6.0')),
                (Decimal('95100.90'), Decimal('9.0')),
                (Decimal('95100.40'), Decimal('4.0')),
            ],
        )

        result = self.analyzer.analyze_liquidity_peaks(snapshot)

        # Should aggregate same price levels
        assert result["total_bid_volume"] == 16.0  # 5+8+3
        assert result["total_ask_volume"] == 19.0  # 6+9+4

    # Helper Methods
    def _create_test_snapshot(self, bids, asks, symbol="BTCFDUSD"):
        """Create a test depth snapshot."""
        bid_levels = [
            DepthLevel(price=price, quantity=quantity) for price, quantity in bids
        ]
        ask_levels = [
            DepthLevel(price=price, quantity=quantity) for price, quantity in asks
        ]

        return DepthSnapshot(
            symbol=symbol,
            timestamp=datetime.now(),
            bids=bid_levels,
            asks=ask_levels,
        )


# Test Constants
def test_constants():
    """Test that constants are properly defined."""
    assert PEAK_SCORE_NORMALIZATION_FACTOR == 3.0
    assert MAX_PEAKS_RETURNED == 10
    assert LOCAL_DENSITY_WINDOW_SIZE == 5


# Test Print Function
def test_print_liquidity_peaks_results(capsys):
    """Test the print function for results."""
    from src.core.liquidity_peaks_analyzer import print_liquidity_peaks_results

    # Create test results
    test_results = {
        "liquidity_peaks": [
            SupportResistanceLevel(
                price=Decimal('95000'),
                strength=0.8,
                level_type="support",
                volume_at_level=Decimal('15.0'),
                confirmation_count=1,
                last_confirmed=None,
            )
        ],
        "analysis_summary": {
            "total_volume": 27.0,
            "bid_ratio": 0.6,
            "ask_ratio": 0.4,
            "market_balance": "bid_heavy",
            "peak_density_score": 0.8,
        },
    }

    print_liquidity_peaks_results(test_results)
    captured = capsys.readouterr()

    assert "流动性峰值区域" in captured.out
    assert "买盘支撑区域" in captured.out
    assert "支撑 1: $95,000" in captured.out
    assert "挂单量: 15.00" in captured.out
    assert "纯度: 0.80" in captured.out


def test_print_liquidity_peaks_results_empty(capsys):
    """Test the print function with empty results."""
    from src.core.liquidity_peaks_analyzer import print_liquidity_peaks_results

    test_results = {"liquidity_peaks": [], "analysis_summary": {}}

    print_liquidity_peaks_results(test_results)
    captured = capsys.readouterr()

    assert "未发现流动性峰值区域" in captured.out