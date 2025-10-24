"""Integration tests for normal distribution analyzer."""

import pytest
from decimal import Decimal
from datetime import datetime

from src.core.models import DepthLevel, DepthSnapshot, MinuteTradeData
from src.core.analyzers_normal import NormalDistributionMarketAnalyzer


class TestNormalDistributionIntegration:
    """Integration tests for normal distribution analysis."""

    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = NormalDistributionMarketAnalyzer(
            min_volume_threshold=Decimal("1.0"),
            analysis_window_minutes=180,
            confidence_level=0.95
        )

    def create_test_depth_snapshot(self, symbol="BTCFDUSD"):
        """Create a realistic depth snapshot for testing."""
        timestamp = datetime.now()

        # Create bid levels (descending prices)
        bids = [
            DepthLevel(price=Decimal("99850.0"), quantity=Decimal("1.2")),
            DepthLevel(price=Decimal("99849.0"), quantity=Decimal("2.1")),
            DepthLevel(price=Decimal("99848.0"), quantity=Decimal("3.5")),  # Peak bid
            DepthLevel(price=Decimal("99847.0"), quantity=Decimal("2.8")),
            DepthLevel(price=Decimal("99846.0"), quantity=Decimal("1.5")),
        ]

        # Create ask levels (ascending prices)
        asks = [
            DepthLevel(price=Decimal("99852.0"), quantity=Decimal("1.8")),
            DepthLevel(price=Decimal("99853.0"), quantity=Decimal("2.4")),
            DepthLevel(price=Decimal("99854.0"), quantity=Decimal("4.1")),  # Peak ask
            DepthLevel(price=Decimal("99855.0"), quantity=Decimal("3.2")),
            DepthLevel(price=Decimal("99856.0"), quantity=Decimal("1.1")),
        ]

        return DepthSnapshot(
            symbol=symbol,
            timestamp=timestamp,
            bids=bids,
            asks=asks
        )

    def create_test_trade_data(self, minutes=5):
        """Create test trade data."""
        trade_data_list = []
        base_time = datetime.now()

        for i in range(minutes):
            minute_data = MinuteTradeData(timestamp=base_time)

            # Add some price levels with volume
            for price_offset in range(-2, 3):
                price = Decimal("99850") + Decimal(str(price_offset))
                volume = Decimal("0.1") * (i + 1)  # Increasing volume over time

                # Create price level data manually
                from src.core.models import PriceLevelData
                price_level_data = PriceLevelData(
                    price_level=price,
                    buy_volume=volume,
                    sell_volume=volume * Decimal("0.3"),
                    total_volume=volume,
                    delta=volume * Decimal("0.7"),
                    trade_count=i + 1
                )
                minute_data.price_levels[price] = price_level_data

            trade_data_list.append(minute_data)

        return trade_data_list

    def test_enhanced_analysis_complete_workflow(self):
        """Test complete enhanced analysis workflow."""
        snapshot = self.create_test_depth_snapshot()
        trade_data = self.create_test_trade_data()

        result = self.analyzer.analyze_market(
            snapshot=snapshot,
            trade_data_list=trade_data,
            symbol="BTCFDUSD",
            enhanced_mode=True
        )

        # Verify result structure
        assert hasattr(result, 'timestamp')
        assert result.symbol == "BTCFDUSD"
        assert hasattr(result, 'normal_distribution_peaks')
        assert hasattr(result, 'confidence_intervals')
        assert hasattr(result, 'market_metrics')
        assert hasattr(result, 'spread_analysis')

        # Check normal distribution peaks
        nd_peaks = result.normal_distribution_peaks
        assert 'bids' in nd_peaks
        assert 'asks' in nd_peaks

        # Verify bid peak analysis
        bid_peak = nd_peaks['bids']
        assert 'mean_price' in bid_peak
        assert 'peak_interval' in bid_peak
        assert 'total_volume' in bid_peak
        assert 'peak_volume' in bid_peak
        assert bid_peak['confidence_level'] == 0.95

        # Verify ask peak analysis
        ask_peak = nd_peaks['asks']
        assert 'mean_price' in ask_peak
        assert 'peak_interval' in ask_peak
        assert ask_peak['confidence_level'] == 0.95

        # Check confidence intervals
        confidence_intervals = result.confidence_intervals
        assert 'bid' in confidence_intervals
        assert 'ask' in confidence_intervals

        # Check market metrics
        metrics = result.market_metrics
        assert 'total_bid_volume' in metrics
        assert 'total_ask_volume' in metrics
        assert 'total_volume' in metrics
        assert 'bid_ask_ratio' in metrics

        # Check spread analysis
        spread = result.spread_analysis
        assert 'best_bid' in spread
        assert 'best_ask' in spread
        assert 'spread' in spread
        assert spread['best_bid'] < spread['best_ask']

    def test_legacy_analysis_compatibility(self):
        """Test legacy analysis mode for backward compatibility."""
        snapshot = self.create_test_depth_snapshot()
        trade_data = self.create_test_trade_data()

        result = self.analyzer.analyze_market(
            snapshot=snapshot,
            trade_data_list=trade_data,
            symbol="BTCFDUSD",
            enhanced_mode=False
        )

        # Legacy mode should return basic MarketAnalysisResult
        assert hasattr(result, 'timestamp')
        assert result.symbol == "BTCFDUSD"
        assert hasattr(result, 'support_levels')
        assert hasattr(result, 'resistance_levels')
        assert hasattr(result, 'poc_levels')

        # Should have support and resistance levels derived from normal distribution
        assert len(result.support_levels) > 0
        assert len(result.resistance_levels) > 0

    def test_analysis_with_empty_data(self):
        """Test analysis with minimal data."""
        # Create minimal snapshot
        minimal_snapshot = DepthSnapshot(
            symbol="BTCFDUSD",
            timestamp=datetime.now(),
            bids=[DepthLevel(price=Decimal("100.0"), quantity=Decimal("0.1"))],
            asks=[DepthLevel(price=Decimal("101.0"), quantity=Decimal("0.1"))]
        )

        result = self.analyzer.analyze_market(
            snapshot=minimal_snapshot,
            trade_data_list=[],
            symbol="BTCFDUSD",
            enhanced_mode=True
        )

        # Should still return valid structure
        assert result.symbol == "BTCFDUSD"
        assert hasattr(result, 'normal_distribution_peaks')
        assert hasattr(result, 'confidence_intervals')

    def test_different_confidence_levels(self):
        """Test analyzer with different confidence levels."""
        # Test with 90% confidence
        analyzer_90 = NormalDistributionMarketAnalyzer(confidence_level=0.90)
        snapshot = self.create_test_depth_snapshot()
        trade_data = self.create_test_trade_data()

        result_90 = analyzer_90.analyze_market(
            snapshot=snapshot,
            trade_data_list=trade_data,
            symbol="BTCFDUSD",
            enhanced_mode=True
        )

        # Test with 99% confidence
        analyzer_99 = NormalDistributionMarketAnalyzer(confidence_level=0.99)
        result_99 = analyzer_99.analyze_market(
            snapshot=snapshot,
            trade_data_list=trade_data,
            symbol="BTCFDUSD",
            enhanced_mode=True
        )

        # Both should produce results but with different confidence levels
        assert result_90.normal_distribution_peaks['bids']['confidence_level'] == 0.90
        assert result_99.normal_distribution_peaks['bids']['confidence_level'] == 0.99

        # Higher confidence should produce wider intervals generally
        interval_90 = result_90.normal_distribution_peaks['bids']['peak_interval']
        interval_99 = result_99.normal_distribution_peaks['bids']['peak_interval']

        if interval_90 and interval_99:
            width_90 = interval_90[1] - interval_90[0]
            width_99 = interval_99[1] - interval_99[0]
            # 99% interval should be wider or equal
            assert width_99 >= width_90

    def test_large_dataset_analysis(self):
        """Test analysis with larger dataset."""
        # Create larger depth snapshot
        large_snapshot = DepthSnapshot(
            symbol="BTCFDUSD",
            timestamp=datetime.now(),
            bids=[
                DepthLevel(price=Decimal(str(100000 - i)), quantity=Decimal(str(0.1 * (i + 1))))
                for i in range(100)  # 100 bid levels
            ],
            asks=[
                DepthLevel(price=Decimal(str(100001 + i)), quantity=Decimal(str(0.1 * (101 - i))))
                for i in range(100)  # 100 ask levels
            ]
        )

        # Create more trade data
        large_trade_data = self.create_test_trade_data(minutes=30)

        result = self.analyzer.analyze_market(
            snapshot=large_snapshot,
            trade_data_list=large_trade_data,
            symbol="BTCFDUSD",
            enhanced_mode=True
        )

        # Should handle large dataset efficiently
        assert result.symbol == "BTCFDUSD"
        assert len(result.normal_distribution_peaks['bids']['peak_interval']) == 2
        assert len(result.normal_distribution_peaks['asks']['peak_interval']) == 2

        # Market metrics should reflect the large dataset
        assert result.market_metrics['total_volume'] > 0
        assert result.market_metrics['price_levels_count']['total_levels'] == 200

    def test_error_handling_and_recovery(self):
        """Test error handling and recovery."""
        # Test with None snapshot
        result = self.analyzer.analyze_market(
            snapshot=None,
            trade_data_list=[],
            symbol="BTCFDUSD",
            enhanced_mode=True
        )

        # Should return empty result without crashing
        assert hasattr(result, 'timestamp')
        assert result.symbol == "BTCFDUSD"

    def test_realistic_btc_fdUSD_scenario(self):
        """Test with realistic BTC-FDUSD price levels."""
        # Create realistic BTC price around $70,000
        realistic_snapshot = DepthSnapshot(
            symbol="BTCFDUSD",
            timestamp=datetime.now(),
            bids=[
                DepthLevel(price=Decimal("69985.0"), quantity=Decimal("0.85")),
                DepthLevel(price=Decimal("69984.0"), quantity=Decimal("1.23")),
                DepthLevel(price=Decimal("69983.0"), quantity=Decimal("2.11")),  # Peak bid
                DepthLevel(price=Decimal("69982.0"), quantity=Decimal("1.67")),
                DepthLevel(price=Decimal("69981.0"), quantity=Decimal("0.94")),
                DepthLevel(price=Decimal("69980.0"), quantity=Decimal("0.52")),
            ],
            asks=[
                DepthLevel(price=Decimal("69986.0"), quantity=Decimal("0.71")),
                DepthLevel(price=Decimal("69987.0"), quantity=Decimal("1.45")),
                DepthLevel(price=Decimal("69988.0"), quantity=Decimal("2.89")),  # Peak ask
                DepthLevel(price=Decimal("69989.0"), quantity=Decimal("2.12")),
                DepthLevel(price=Decimal("69990.0"), quantity=Decimal("1.33")),
                DepthLevel(price=Decimal("69991.0"), quantity=Decimal("0.68")),
            ]
        )

        # Create realistic trade data over several hours
        realistic_trade_data = []
        base_time = datetime.now()
        base_price = Decimal("69985")

        for hour in range(3):  # 3 hours of data
            for minute in range(60):
                timestamp = base_time.replace(hour=base_time.hour - (2 - hour))
                minute_data = MinuteTradeData(timestamp=timestamp)

                # Simulate price movement around base price
                for price_offset in range(-5, 6):
                    price = base_price + Decimal(str(price_offset + hour))
                    volume = Decimal("0.05") * (minute + 1) * (3 - hour)  # Decreasing volume

                    from src.core.models import PriceLevelData
                    price_level_data = PriceLevelData(
                        price_level=price.quantize(Decimal("1")),
                        buy_volume=volume * Decimal("0.6"),
                        sell_volume=volume * Decimal("0.4"),
                        total_volume=volume,
                        delta=volume * Decimal("0.2"),
                        trade_count=max(1, minute // 10)
                    )
                    minute_data.price_levels[price_level_data.price_level] = price_level_data

                realistic_trade_data.append(minute_data)

        result = self.analyzer.analyze_market(
            snapshot=realistic_snapshot,
            trade_data_list=realistic_trade_data,
            symbol="BTCFDUSD",
            enhanced_mode=True
        )

        # Verify realistic analysis results
        assert result.symbol == "BTCFDUSD"

        # Check that peaks are detected in realistic price ranges
        bid_peak = result.normal_distribution_peaks['bids']
        ask_peak = result.normal_distribution_peaks['asks']

        # Prices should be around the $70,000 range
        assert 69000 < bid_peak['mean_price'] < 71000
        assert 69000 < ask_peak['mean_price'] < 71000
        assert bid_peak['mean_price'] < ask_peak['mean_price']

        # Spread should be reasonable for BTC
        spread = result.spread_analysis
        assert 0 < spread['spread'] < 100  # Less than $100 spread

        # Market metrics should show volume
        metrics = result.market_metrics
        assert metrics['total_volume'] > 10  # Some total volume
        assert 0.1 < metrics['bid_ask_ratio'] < 10  # Reasonable ratio

    def test_to_dict_serialization(self):
        """Test result serialization to dictionary."""
        snapshot = self.create_test_depth_snapshot()
        trade_data = self.create_test_trade_data()

        result = self.analyzer.analyze_market(
            snapshot=snapshot,
            trade_data_list=trade_data,
            symbol="BTCFDUSD",
            enhanced_mode=True
        )

        # Test serialization
        result_dict = result.to_dict()

        # Verify structure
        assert 'timestamp' in result_dict
        assert 'symbol' in result_dict
        assert 'normal_distribution_peaks' in result_dict
        assert 'confidence_intervals' in result_dict
        assert 'market_metrics' in result_dict
        assert 'spread_analysis' in result_dict

        # Verify data types are serializable
        assert isinstance(result_dict['normal_distribution_peaks'], dict)
        assert isinstance(result_dict['confidence_intervals'], dict)
        assert isinstance(result_dict['market_metrics'], dict)
        assert isinstance(result_dict['spread_analysis'], dict)

        # Verify nested structure integrity
        assert 'bids' in result_dict['normal_distribution_peaks']
        assert 'asks' in result_dict['normal_distribution_peaks']
        assert 'bid' in result_dict['confidence_intervals']
        assert 'ask' in result_dict['confidence_intervals']