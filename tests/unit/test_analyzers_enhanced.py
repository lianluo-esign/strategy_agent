"""Unit tests for analyzers_enhanced module.

Tests the enhanced market analyzer with 1-dollar precision aggregation and wave peak detection.
"""

import pytest
from datetime import datetime
from decimal import Decimal
from typing import List, Dict

from src.core.analyzers_enhanced import EnhancedMarketAnalyzer
from src.core.models import (
    DepthSnapshot,
    DepthLevel,
    MinuteTradeData,
    Trade,
    MarketAnalysisResult,
    EnhancedMarketAnalysisResult,
    WavePeak,
    PriceZone,
    SupportResistanceLevel,
)


class TestEnhancedMarketAnalyzer:
    """Test cases for EnhancedMarketAnalyzer class."""

    @pytest.fixture
    def analyzer(self):
        """Create EnhancedMarketAnalyzer instance for testing."""
        return EnhancedMarketAnalyzer(
            min_volume_threshold=Decimal('1.0'),
            analysis_window_minutes=180
        )

    @pytest.fixture
    def sample_depth_snapshot(self) -> DepthSnapshot:
        """Create sample depth snapshot for testing."""
        bids = [
            DepthLevel(price=Decimal('99.50'), quantity=Decimal('10.0')),
            DepthLevel(price=Decimal('99.20'), quantity=Decimal('15.0')),
            DepthLevel(price=Decimal('98.80'), quantity=Decimal('8.0')),
            DepthLevel(price=Decimal('98.50'), quantity=Decimal('12.0')),
            DepthLevel(price=Decimal('98.20'), quantity=Decimal('5.0')),
        ]

        asks = [
            DepthLevel(price=Decimal('100.50'), quantity=Decimal('8.0')),
            DepthLevel(price=Decimal('100.80'), quantity=Decimal('12.0')),
            DepthLevel(price=Decimal('101.20'), quantity=Decimal('6.0')),
            DepthLevel(price=Decimal('101.50'), quantity=Decimal('10.0')),
            DepthLevel(price=Decimal('101.80'), quantity=Decimal('4.0')),
        ]

        return DepthSnapshot(
            symbol='BTCFDUSD',
            timestamp=datetime.now(),
            bids=bids,
            asks=asks
        )

    @pytest.fixture
    def sample_trade_data(self) -> List[MinuteTradeData]:
        """Create sample trade data for testing."""
        trade_data_list = []

        for i in range(10):
            minute_data = MinuteTradeData(timestamp=datetime.now())

            # Add trades with different price levels
            for price_offset in range(-2, 3):
                price = Decimal('100') + Decimal(str(price_offset))
                for j in range(5):
                    trade = Trade(
                        symbol='BTCFDUSD',
                        price=price,
                        quantity=Decimal('1.0'),
                        is_buyer_maker=j % 2 == 0,
                        timestamp=minute_data.timestamp,
                        trade_id=f'trade_{i}_{price_offset}_{j}'
                    )
                    minute_data.add_trade(trade)

            trade_data_list.append(minute_data)

        return trade_data_list

    def test_analyzer_initialization(self):
        """Test EnhancedMarketAnalyzer initialization."""
        analyzer = EnhancedMarketAnalyzer(
            min_volume_threshold=Decimal('5.0'),
            analysis_window_minutes=240
        )

        assert analyzer.min_volume_threshold == Decimal('5.0')
        assert analyzer.analysis_window_minutes == 240

    def test_analyze_market_basic_functionality(self, analyzer, sample_depth_snapshot, sample_trade_data):
        """Test basic market analysis functionality."""
        result = analyzer.analyze_market(
            snapshot=sample_depth_snapshot,
            trade_data_list=sample_trade_data,
            symbol='BTCFDUSD',
            enhanced_mode=True
        )

        assert isinstance(result, EnhancedMarketAnalysisResult)
        assert result.symbol == 'BTCFDUSD'
        assert isinstance(result.timestamp, datetime)
        assert len(result.aggregated_bids) > 0
        assert len(result.aggregated_asks) > 0

    def test_analyze_market_legacy_mode(self, analyzer, sample_depth_snapshot, sample_trade_data):
        """Test market analysis in legacy mode (returns MarketAnalysisResult)."""
        result = analyzer.analyze_market(
            snapshot=sample_depth_snapshot,
            trade_data_list=sample_trade_data,
            symbol='BTCFDUSD',
            enhanced_mode=False
        )

        assert isinstance(result, MarketAnalysisResult)
        assert result.symbol == 'BTCFDUSD'
        assert isinstance(result.timestamp, datetime)

    def test_analyze_market_empty_data(self, analyzer):
        """Test analysis with empty data."""
        empty_snapshot = DepthSnapshot(
            symbol='BTCFDUSD',
            timestamp=datetime.now(),
            bids=[],
            asks=[]
        )
        empty_trade_data = []

        result = analyzer.analyze_market(
            snapshot=empty_snapshot,
            trade_data_list=empty_trade_data,
            symbol='BTCFDUSD',
            enhanced_mode=True
        )

        assert isinstance(result, EnhancedMarketAnalysisResult)
        assert result.symbol == 'BTCFDUSD'
        # Should handle empty data gracefully
        assert len(result.aggregated_bids) == 0
        assert len(result.aggregated_asks) == 0

    def test_depth_snapshot_aggregation(self, analyzer, sample_depth_snapshot):
        """Test depth snapshot aggregation to 1-dollar precision."""
        # Mock the aggregation process
        aggregated_bids, aggregated_asks = analyzer._aggregate_depth_snapshot(sample_depth_snapshot)

        assert isinstance(aggregated_bids, dict)
        assert isinstance(aggregated_asks, dict)

        # Check that prices are rounded to dollar precision
        for price in aggregated_bids.keys():
            assert price == price.quantize(Decimal('1'))

        for price in aggregated_asks.keys():
            assert price == price.quantize(Decimal('1'))

    def test_trade_data_aggregation(self, analyzer, sample_trade_data):
        """Test trade data aggregation by 1-dollar precision."""
        aggregated_trades = analyzer._aggregate_trade_data(sample_trade_data)

        assert isinstance(aggregated_trades, dict)
        assert len(aggregated_trades) > 0

        # Check that prices are aggregated to 1-dollar precision
        for price in aggregated_trades.keys():
            assert price == price.quantize(Decimal('1'))

    def test_wave_peak_detection_integration(self, analyzer):
        """Test wave peak detection integration."""
        # Create data with clear peaks
        price_volume_data = {
            Decimal('95'): Decimal('5.0'),
            Decimal('96'): Decimal('8.0'),
            Decimal('97'): Decimal('20.0'),  # Peak
            Decimal('98'): Decimal('30.0'),  # Peak
            Decimal('99'): Decimal('25.0'),  # Peak
            Decimal('100'): Decimal('10.0'),
            Decimal('101'): Decimal('15.0'),  # Peak
            Decimal('102'): Decimal('22.0'),  # Peak
            Decimal('103'): Decimal('18.0'),  # Peak
            Decimal('104'): Decimal('7.0'),
        }

        wave_peaks = analyzer._detect_wave_peaks(price_volume_data)

        assert isinstance(wave_peaks, list)
        assert len(wave_peaks) > 0

        # Verify peak properties
        for peak in wave_peaks:
            assert isinstance(peak, WavePeak)
            assert peak.center_price in price_volume_data
            assert peak.confidence > 0

    def test_price_zone_formation(self, analyzer):
        """Test price zone formation from wave peaks."""
        # Create sample wave peaks
        wave_peaks = [
            WavePeak(Decimal('100.0'), Decimal('50.0'), Decimal('2.0'), 1.5, 0.8),
            WavePeak(Decimal('102.0'), Decimal('40.0'), Decimal('2.0'), 1.2, 0.7),
            WavePeak(Decimal('110.0'), Decimal('60.0'), Decimal('3.0'), 1.8, 0.9),
            WavePeak(Decimal('112.0'), Decimal('45.0'), Decimal('2.0'), 1.3, 0.75),
        ]

        support_zones, resistance_zones = analyzer._analyze_price_formation(wave_peaks)

        assert isinstance(support_zones, list)
        assert isinstance(resistance_zones, list)

        # Verify zone properties
        for zone in support_zones + resistance_zones:
            assert isinstance(zone, PriceZone)
            assert zone.lower_price <= zone.upper_price
            assert zone.confidence > 0
            assert zone.zone_type in ['support', 'resistance']

    def test_support_resistance_level_generation(self, analyzer):
        """Test generation of traditional support/resistance levels."""
        # Create sample wave peaks and zones
        wave_peaks = [
            WavePeak(Decimal('100.0'), Decimal('50.0'), Decimal('2.0'), 1.5, 0.8),
            WavePeak(Decimal('105.0'), Decimal('30.0'), Decimal('2.0'), 1.2, 0.6),
        ]

        support_zones = [
            PriceZone(Decimal('99.0'), Decimal('101.0'), 'support', 0.8, Decimal('100.0'))
        ]

        resistance_zones = [
            PriceZone(Decimal('104.0'), Decimal('106.0'), 'resistance', 0.7, Decimal('80.0'))
        ]

        support_levels, resistance_levels = analyzer._generate_support_resistance_levels(
            wave_peaks, support_zones, resistance_zones
        )

        assert isinstance(support_levels, list)
        assert isinstance(resistance_levels, list)

        # Verify level properties
        for level in support_levels:
            assert isinstance(level, SupportResistanceLevel)
            assert level.level_type == 'support'

        for level in resistance_levels:
            assert isinstance(level, SupportResistanceLevel)
            assert level.level_type == 'resistance'

    def test_quality_metrics_calculation(self, analyzer):
        """Test quality metrics calculation for analysis results."""
        # Create sample data
        original_bids = [
            DepthLevel(price=Decimal('99.50'), quantity=Decimal('10.0')),
            DepthLevel(price=Decimal('99.80'), quantity=Decimal('15.0')),
        ]
        original_asks = [
            DepthLevel(price=Decimal('100.20'), quantity=Decimal('8.0')),
            DepthLevel(price=Decimal('100.70'), quantity=Decimal('12.0')),
        ]

        aggregated_bids = {Decimal('99'): Decimal('25.0')}
        aggregated_asks = {Decimal('100'): Decimal('20.0')}

        wave_peaks = [
            WavePeak(Decimal('99.0'), Decimal('25.0'), Decimal('2.0'), 1.5, 0.8)
        ]

        # Test depth statistics
        depth_stats = analyzer._calculate_depth_statistics(
            original_bids, original_asks, aggregated_bids, aggregated_asks
        )

        assert isinstance(depth_stats, dict)
        assert 'bid_compression_ratio' in depth_stats
        assert 'ask_compression_ratio' in depth_stats
        assert 'total_volume_preservation' in depth_stats

        # Test peak detection quality
        peak_quality = analyzer._calculate_peak_detection_quality(wave_peaks, aggregated_bids, aggregated_asks)

        assert isinstance(peak_quality, dict)
        assert 'peak_count' in peak_quality
        assert 'avg_confidence' in peak_quality
        assert 'coverage_rate' in peak_quality

    def test_comprehensive_analysis_pipeline(self, analyzer, sample_depth_snapshot, sample_trade_data):
        """Test the complete analysis pipeline."""
        result = analyzer.analyze_market(
            snapshot=sample_depth_snapshot,
            trade_data_list=sample_trade_data,
            symbol='BTCFDUSD',
            enhanced_mode=True
        )

        # Verify all components are populated
        assert isinstance(result, EnhancedMarketAnalysisResult)
        assert result.symbol == 'BTCFDUSD'

        # Aggregated data
        assert len(result.aggregated_bids) > 0
        assert len(result.aggregated_asks) > 0

        # Wave analysis
        assert isinstance(result.wave_peaks, list)
        assert isinstance(result.support_zones, list)
        assert isinstance(result.resistance_zones, list)

        # Traditional analysis (backward compatibility)
        assert isinstance(result.support_levels, list)
        assert isinstance(result.resistance_levels, list)
        assert isinstance(result.poc_levels, list)
        assert isinstance(result.liquidity_vacuum_zones, list)
        assert isinstance(result.resonance_zones, list)

        # Quality metrics
        assert isinstance(result.depth_statistics, dict)
        assert isinstance(result.peak_detection_quality, dict)

    def test_analysis_with_realistic_data(self, analyzer):
        """Test analysis with realistic market data."""
        # Create more realistic depth snapshot
        realistic_bids = [
            DepthLevel(price=Decimal(f'{100 - i*0.1}'), quantity=Decimal(f'{10 + i*2}'))
            for i in range(50)
        ]
        realistic_asks = [
            DepthLevel(price=Decimal(f'{100 + i*0.1}'), quantity=Decimal(f'{8 + i*1.5}'))
            for i in range(50)
        ]

        realistic_snapshot = DepthSnapshot(
            symbol='BTCFDUSD',
            timestamp=datetime.now(),
            bids=realistic_bids,
            asks=realistic_asks
        )

        # Create realistic trade data
        realistic_trade_data = []
        for minute in range(60):  # 1 hour of data
            minute_data = MinuteTradeData(timestamp=datetime.now())

            for trade in range(20):  # 20 trades per minute
                price = Decimal('100') + Decimal(str(trade % 10 - 5))  # Prices from 95 to 104
                trade_obj = Trade(
                    symbol='BTCFDUSD',
                    price=price,
                    quantity=Decimal('1.5'),
                    is_buyer_maker=trade % 2 == 0,
                    timestamp=minute_data.timestamp,
                    trade_id=f'trade_{minute}_{trade}'
                )
                minute_data.add_trade(trade_obj)

            realistic_trade_data.append(minute_data)

        result = analyzer.analyze_market(
            snapshot=realistic_snapshot,
            trade_data_list=realistic_trade_data,
            symbol='BTCFDUSD',
            enhanced_mode=True
        )

        # Should handle realistic data well
        assert isinstance(result, EnhancedMarketAnalysisResult)
        assert len(result.aggregated_bids) > 1
        assert len(result.aggregated_asks) > 1
        assert len(result.wave_peaks) >= 0

    def test_performance_with_large_dataset(self, analyzer):
        """Test performance with large datasets."""
        import time

        # Create large dataset
        large_bids = [
            DepthLevel(price=Decimal(f'{100 - i*0.01}'), quantity=Decimal('10.0'))
            for i in range(1000)
        ]
        large_asks = [
            DepthLevel(price=Decimal(f'{100 + i*0.01}'), quantity=Decimal('10.0'))
            for i in range(1000)
        ]

        large_snapshot = DepthSnapshot(
            symbol='BTCFDUSD',
            timestamp=datetime.now(),
            bids=large_bids,
            asks=large_asks
        )

        large_trade_data = []
        for minute in range(180):  # 3 hours
            minute_data = MinuteTradeData(timestamp=datetime.now())

            for trade in range(100):  # 100 trades per minute
                price = Decimal('100') + Decimal(str(trade % 20 - 10))
                trade_obj = Trade(
                    symbol='BTCFDUSD',
                    price=price,
                    quantity=Decimal('2.0'),
                    is_buyer_maker=trade % 2 == 0,
                    timestamp=minute_data.timestamp,
                    trade_id=f'trade_{minute}_{trade}'
                )
                minute_data.add_trade(trade_obj)

            large_trade_data.append(minute_data)

        # Measure performance
        start_time = time.time()
        result = analyzer.analyze_market(
            snapshot=large_snapshot,
            trade_data_list=large_trade_data,
            symbol='BTCFDUSD',
            enhanced_mode=True
        )
        end_time = time.time()

        processing_time = end_time - start_time

        # Should complete within reasonable time (e.g., 5 seconds)
        assert processing_time < 5.0
        assert isinstance(result, EnhancedMarketAnalysisResult)


class TestEnhancedAnalyzerEdgeCases:
    """Test edge cases and error handling for EnhancedMarketAnalyzer."""

    @pytest.fixture
    def analyzer(self):
        """Create EnhancedMarketAnalyzer instance for testing."""
        return EnhancedMarketAnalyzer()

    def test_analysis_with_none_inputs(self, analyzer):
        """Test analysis with None inputs."""
        with pytest.raises((AttributeError, TypeError)):
            analyzer.analyze_market(
                snapshot=None,
                trade_data_list=[],
                symbol='BTCFDUSD',
                enhanced_mode=True
            )

    def test_analysis_with_corrupted_data(self, analyzer):
        """Test analysis with corrupted or inconsistent data."""
        # Create corrupted snapshot
        corrupted_snapshot = DepthSnapshot(
            symbol='',
            timestamp=None,
            bids=[DepthLevel(price=Decimal('100'), quantity=Decimal('-5'))],  # Negative quantity
            asks=[]
        )

        result = analyzer.analyze_market(
            snapshot=corrupted_snapshot,
            trade_data_list=[],
            symbol='BTCFDUSD',
            enhanced_mode=True
        )

        # Should handle gracefully
        assert isinstance(result, EnhancedMarketAnalysisResult)

    def test_analysis_with_extreme_values(self, analyzer):
        """Test analysis with extreme price/volume values."""
        extreme_snapshot = DepthSnapshot(
            symbol='BTCFDUSD',
            timestamp=datetime.now(),
            bids=[
                DepthLevel(price=Decimal('0.01'), quantity=Decimal('999999999')),
                DepthLevel(price=Decimal('999999999'), quantity=Decimal('0.01')),
            ],
            asks=[
                DepthLevel(price=Decimal('0.01'), quantity=Decimal('999999999')),
                DepthLevel(price=Decimal('999999999'), quantity=Decimal('0.01')),
            ]
        )

        result = analyzer.analyze_market(
            snapshot=extreme_snapshot,
            trade_data_list=[],
            symbol='BTCFDUSD',
            enhanced_mode=True
        )

        # Should handle extreme values without crashing
        assert isinstance(result, EnhancedMarketAnalysisResult)

    def test_analysis_memory_usage(self, analyzer):
        """Test that analysis doesn't leak memory."""
        import gc
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # Run multiple analyses
        for i in range(10):
            snapshot = DepthSnapshot(
                symbol='BTCFDUSD',
                timestamp=datetime.now(),
                bids=[
                    DepthLevel(price=Decimal(f'{100 + j}'), quantity=Decimal('10'))
                    for j in range(100)
                ],
                asks=[
                    DepthLevel(price=Decimal(f'{100 + j}'), quantity=Decimal('10'))
                    for j in range(100)
                ]
            )

            result = analyzer.analyze_market(
                snapshot=snapshot,
                trade_data_list=[],
                symbol='BTCFDUSD',
                enhanced_mode=True
            )

            del result

        gc.collect()
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable (e.g., less than 50MB)
        assert memory_increase < 50 * 1024 * 1024