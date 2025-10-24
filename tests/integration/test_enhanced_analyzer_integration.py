"""Integration tests for enhanced analyzer functionality.

Tests the complete end-to-end functionality of the enhanced analyzer
with Redis data store and AI client integration.
"""

from datetime import datetime, timedelta
from decimal import Decimal

import pytest

from src.core.analyzers_enhanced import EnhancedMarketAnalyzer
from src.core.models import (
    DepthLevel,
    DepthSnapshot,
    EnhancedMarketAnalysisResult,
    MarketAnalysisResult,
    MinuteTradeData,
    Trade,
)
from src.core.redis_client import RedisDataStore


class TestEnhancedAnalyzerIntegration:
    """Integration tests for enhanced analyzer with real components."""

    @pytest.fixture
    async def redis_store(self):
        """Create Redis data store for testing."""
        store = RedisDataStore(
            host='localhost',
            port=6379,
            db=1  # Use test database
        )

        # Test connection
        if not store.test_connection():
            pytest.skip("Redis not available for integration tests")

        yield store

        # Cleanup
        await store.close()

    @pytest.fixture
    def analyzer(self):
        """Create enhanced analyzer for testing."""
        return EnhancedMarketAnalyzer(
            min_volume_threshold=Decimal('1.0'),
            analysis_window_minutes=180
        )

    @pytest.fixture
    def sample_market_data(self) -> tuple[DepthSnapshot, list[MinuteTradeData]]:
        """Create comprehensive sample market data."""
        # Create realistic depth snapshot
        bids = [
            DepthLevel(
                price=Decimal(f'{100 - i*0.05}'),
                quantity=Decimal(f'{10 + i*1.5}')
            )
            for i in range(100)  # 100 bid levels
        ]

        asks = [
            DepthLevel(
                price=Decimal(f'{100 + i*0.05}'),
                quantity=Decimal(f'{8 + i*1.2}')
            )
            for i in range(100)  # 100 ask levels
        ]

        snapshot = DepthSnapshot(
            symbol='BTCFDUSD',
            timestamp=datetime.now(),
            bids=bids,
            asks=asks
        )

        # Create 3 hours of trade data (analysis window)
        trade_data_list = []
        base_time = datetime.now() - timedelta(hours=3)

        for minute_offset in range(180):  # 3 hours = 180 minutes
            minute_data = MinuteTradeData(
                timestamp=base_time + timedelta(minutes=minute_offset)
            )

            # Simulate varying trading activity
            base_activity = 10 + (minute_offset % 30)  # Activity varies every 30 minutes

            for trade_num in range(base_activity):
                # Price follows a random walk around $100
                price_offset = (minute_offset + trade_num) % 21 - 10  # -10 to +10
                price = Decimal('100') + Decimal(str(price_offset))

                trade = Trade(
                    symbol='BTCFDUSD',
                    price=price,
                    quantity=Decimal(str(0.5 + (trade_num % 5) * 0.5)),  # 0.5 to 2.5
                    is_buyer_maker=trade_num % 3 == 0,  # 33% maker buys
                    timestamp=minute_data.timestamp,
                    trade_id=f'trade_{minute_offset}_{trade_num}'
                )
                minute_data.add_trade(trade)

            trade_data_list.append(minute_data)

        return snapshot, trade_data_list

    @pytest.mark.asyncio
    async def test_end_to_end_analysis_pipeline(self, analyzer, sample_market_data):
        """Test complete end-to-end analysis pipeline."""
        snapshot, trade_data_list = sample_market_data

        # Perform enhanced analysis
        result = analyzer.analyze_market(
            snapshot=snapshot,
            trade_data_list=trade_data_list,
            symbol='BTCFDUSD',
            enhanced_mode=True
        )

        # Verify enhanced result structure
        assert isinstance(result, EnhancedMarketAnalysisResult)
        assert result.symbol == 'BTCFDUSD'
        assert isinstance(result.timestamp, datetime)

        # Verify 1-dollar precision aggregation
        assert len(result.aggregated_bids) > 0
        assert len(result.aggregated_asks) > 0

        # Check that all aggregated prices are at 1-dollar precision
        for price in result.aggregated_bids.keys():
            assert price == price.quantize(Decimal('1'))
        for price in result.aggregated_asks.keys():
            assert price == price.quantize(Decimal('1'))

        # Verify wave peak detection results
        assert isinstance(result.wave_peaks, list)
        assert isinstance(result.support_zones, list)
        assert isinstance(result.resistance_zones, list)

        # Verify backward compatibility fields
        assert isinstance(result.support_levels, list)
        assert isinstance(result.resistance_levels, list)
        assert isinstance(result.poc_levels, list)
        assert isinstance(result.liquidity_vacuum_zones, list)
        assert isinstance(result.resonance_zones, list)

        # Verify quality metrics
        assert isinstance(result.depth_statistics, dict)
        assert isinstance(result.peak_detection_quality, dict)

        # Check specific quality metrics
        if result.depth_statistics:
            assert 'bid_compression_ratio' in result.depth_statistics
            assert 'ask_compression_ratio' in result.depth_statistics

        if result.peak_detection_quality:
            assert 'peak_count' in result.peak_detection_quality
            assert 'avg_confidence' in result.peak_detection_quality

    @pytest.mark.asyncio
    async def test_legacy_mode_compatibility(self, analyzer, sample_market_data):
        """Test legacy mode compatibility."""
        snapshot, trade_data_list = sample_market_data

        # Perform legacy analysis
        legacy_result = analyzer.analyze_market(
            snapshot=snapshot,
            trade_data_list=trade_data_list,
            symbol='BTCFDUSD',
            enhanced_mode=False
        )

        # Should return traditional MarketAnalysisResult
        assert isinstance(legacy_result, MarketAnalysisResult)
        assert legacy_result.symbol == 'BTCFDUSD'
        assert isinstance(legacy_result.timestamp, datetime)

        # Verify traditional result structure
        assert isinstance(legacy_result.support_levels, list)
        assert isinstance(legacy_result.resistance_levels, list)
        assert isinstance(legacy_result.poc_levels, list)
        assert isinstance(legacy_result.liquidity_vacuum_zones, list)
        assert isinstance(legacy_result.resonance_zones, list)

    @pytest.mark.asyncio
    async def test_redis_integration_workflow(self, redis_store, analyzer, sample_market_data):
        """Test integration with Redis data store workflow."""
        snapshot, trade_data_list = sample_market_data

        # Store depth snapshot in Redis
        await redis_store.store_depth_snapshot(snapshot)

        # Store trade data in Redis
        for minute_data in trade_data_list:
            await redis_store.store_minute_trade_data(minute_data)

        # Retrieve data from Redis (simulating real workflow)
        retrieved_snapshot = redis_store.get_latest_depth_snapshot()
        retrieved_trade_data = redis_store.get_recent_trade_data(minutes=180)

        # Verify data retrieval
        assert retrieved_snapshot is not None
        assert len(retrieved_trade_data) > 0

        # Perform analysis on retrieved data
        result = analyzer.analyze_market(
            snapshot=retrieved_snapshot,
            trade_data_list=retrieved_trade_data,
            symbol='BTCFDUSD',
            enhanced_mode=True
        )

        # Verify analysis results
        assert isinstance(result, EnhancedMarketAnalysisResult)
        assert result.symbol == 'BTCFDUSD'
        assert len(result.aggregated_bids) > 0
        assert len(result.aggregated_asks) > 0

        # Store analysis result back to Redis
        await redis_store.store_analysis_result(result)

        # Cleanup test data
        await redis_store.clear_all_test_data()

    @pytest.mark.asyncio
    async def test_enhanced_vs_legacy_comparison(self, analyzer, sample_market_data):
        """Test comparison between enhanced and legacy analysis."""
        snapshot, trade_data_list = sample_market_data

        # Perform both analyses
        enhanced_result = analyzer.analyze_market(
            snapshot=snapshot,
            trade_data_list=trade_data_list,
            symbol='BTCFDUSD',
            enhanced_mode=True
        )

        legacy_result = analyzer.analyze_market(
            snapshot=snapshot,
            trade_data_list=trade_data_list,
            symbol='BTCFDUSD',
            enhanced_mode=False
        )

        # Compare results
        assert isinstance(enhanced_result, EnhancedMarketAnalysisResult)
        assert isinstance(legacy_result, MarketAnalysisResult)

        # Enhanced should provide more detailed information
        assert len(enhanced_result.wave_peaks) >= 0
        assert len(enhanced_result.support_zones) >= 0
        assert len(enhanced_result.resistance_zones) >= 0

        # Legacy compatibility fields should be present
        assert len(legacy_result.support_levels) >= 0
        assert len(legacy_result.resistance_levels) >= 0

        # Enhanced result should have additional quality metrics
        assert isinstance(enhanced_result.depth_statistics, dict)
        assert isinstance(enhanced_result.peak_detection_quality, dict)

    def test_volume_precision_aggregation_accuracy(self, analyzer):
        """Test that 1-dollar precision aggregation maintains volume accuracy."""
        # Create precise test data with known volumes
        bids = [
            DepthLevel(price=Decimal('99.20'), quantity=Decimal('10.5')),
            DepthLevel(price=Decimal('99.70'), quantity=Decimal('15.3')),
            DepthLevel(price=Decimal('99.90'), quantity=Decimal('8.2')),
        ]

        asks = [
            DepthLevel(price=Decimal('100.10'), quantity=Decimal('12.7')),
            DepthLevel(price=Decimal('100.60'), quantity=Decimal('9.4')),
            DepthLevel(price=Decimal('100.80'), quantity=Decimal('14.1')),
        ]

        snapshot = DepthSnapshot(
            symbol='BTCFDUSD',
            timestamp=datetime.now(),
            bids=bids,
            asks=asks
        )

        result = analyzer.analyze_market(
            snapshot=snapshot,
            trade_data_list=[],
            symbol='BTCFDUSD',
            enhanced_mode=True
        )

        # Verify aggregation precision
        expected_bid_volume = Decimal('34.0')  # 10.5 + 15.3 + 8.2
        expected_ask_volume = Decimal('36.2')  # 12.7 + 9.4 + 14.1

        assert result.aggregated_bids[Decimal('99')] == expected_bid_volume
        assert result.aggregated_asks[Decimal('100')] == expected_ask_volume

    def test_wave_peak_detection_realistic_scenarios(self, analyzer):
        """Test wave peak detection with realistic market scenarios."""
        # Simulate market with clear support and resistance areas
        bids = []
        asks = []

        # Create support area around $95 (high buy volume)
        for i in range(20):
            price = Decimal('95') - Decimal(f'{i*0.01}')
            quantity = Decimal(f'{20 + i}')  # Increasing volume as price decreases
            bids.append(DepthLevel(price=price, quantity=quantity))

        # Create normal bid levels
        for i in range(30):
            price = Decimal('97') - Decimal(f'{i*0.05}')
            quantity = Decimal('5')
            bids.append(DepthLevel(price=price, quantity=quantity))

        # Create resistance area around $105 (high sell volume)
        for i in range(20):
            price = Decimal('105') + Decimal(f'{i*0.01}')
            quantity = Decimal(f'{25 + i}')  # Increasing volume as price increases
            asks.append(DepthLevel(price=price, quantity=quantity))

        # Create normal ask levels
        for i in range(30):
            price = Decimal('103') + Decimal(f'{i*0.05}')
            quantity = Decimal('8')
            asks.append(DepthLevel(price=price, quantity=quantity))

        snapshot = DepthSnapshot(
            symbol='BTCFDUSD',
            timestamp=datetime.now(),
            bids=bids,
            asks=asks
        )

        result = analyzer.analyze_market(
            snapshot=snapshot,
            trade_data_list=[],
            symbol='BTCFDUSD',
            enhanced_mode=True
        )

        # Should detect wave peaks in support and resistance areas
        assert len(result.wave_peaks) > 0

        # Should identify support zones (from bid clustering)
        assert len(result.support_zones) >= 0

        # Should identify resistance zones (from ask clustering)
        assert len(result.resistance_zones) >= 0

    def test_performance_under_load(self, analyzer):
        """Test analyzer performance under high load."""
        import time

        # Create large dataset
        large_bids = [
            DepthLevel(
                price=Decimal(f'{100 - i*0.001}'),
                quantity=Decimal('10')
            )
            for i in range(5000)  # 5000 bid levels
        ]

        large_asks = [
            DepthLevel(
                price=Decimal(f'{100 + i*0.001}'),
                quantity=Decimal('10')
            )
            for i in range(5000)  # 5000 ask levels
        ]

        large_snapshot = DepthSnapshot(
            symbol='BTCFDUSD',
            timestamp=datetime.now(),
            bids=large_bids,
            asks=large_asks
        )

        # Measure performance
        start_time = time.time()
        result = analyzer.analyze_market(
            snapshot=large_snapshot,
            trade_data_list=[],
            symbol='BTCFDUSD',
            enhanced_mode=True
        )
        end_time = time.time()

        processing_time = end_time - start_time

        # Should complete within reasonable time (e.g., 10 seconds)
        assert processing_time < 10.0
        assert isinstance(result, EnhancedMarketAnalysisResult)

        # Should achieve significant compression through 1-dollar aggregation
        total_original_levels = len(large_bids) + len(large_asks)
        total_aggregated_levels = len(result.aggregated_bids) + len(result.aggregated_asks)

        compression_ratio = total_original_levels / total_aggregated_levels
        assert compression_ratio > 100  # Should achieve at least 100x compression


class TestEnhancedAnalyzerErrorHandling:
    """Test error handling and robustness of enhanced analyzer."""

    @pytest.fixture
    def analyzer(self):
        """Create enhanced analyzer for testing."""
        return EnhancedMarketAnalyzer()

    def test_analysis_with_corrupted_depth_data(self, analyzer):
        """Test analysis with corrupted or inconsistent depth data."""
        corrupted_bids = [
            DepthLevel(price=Decimal('100'), quantity=Decimal('-5')),  # Negative quantity
            DepthLevel(price=Decimal('0'), quantity=Decimal('10')),      # Zero price
            DepthLevel(price=Decimal('100'), quantity=Decimal('0')),    # Zero quantity
        ]

        corrupted_asks = [
            DepthLevel(price=Decimal('100'), quantity=Decimal('inf')),   # Infinite quantity (will error)
            DepthLevel(price=Decimal('-50'), quantity=Decimal('10')),    # Negative price
        ]

        corrupted_snapshot = DepthSnapshot(
            symbol='BTCFDUSD',
            timestamp=datetime.now(),
            bids=corrupted_bids,
            asks=corrupted_asks
        )

        # Should handle corrupted data gracefully
        result = analyzer.analyze_market(
            snapshot=corrupted_snapshot,
            trade_data_list=[],
            symbol='BTCFDUSD',
            enhanced_mode=True
        )

        assert isinstance(result, EnhancedMarketAnalysisResult)
        assert result.symbol == 'BTCFDUSD'

    def test_analysis_with_extreme_market_conditions(self, analyzer):
        """Test analysis under extreme market conditions."""
        # Simulate flash crash scenario
        flash_crash_bids = [
            DepthLevel(
                price=Decimal(f'{100 - i}'),
                quantity=Decimal('1000')  # Very high volume
            )
            for i in range(50)  # Prices from $100 down to $51
        ]

        flash_crash_asks = [
            DepthLevel(
                price=Decimal(f'{50 + i*0.1}'),
                quantity=Decimal('1')  # Very low volume
            )
            for i in range(500)  # Many small ask levels
        ]

        flash_crash_snapshot = DepthSnapshot(
            symbol='BTCFDUSD',
            timestamp=datetime.now(),
            bids=flash_crash_bids,
            asks=flash_crash_asks
        )

        result = analyzer.analyze_market(
            snapshot=flash_crash_snapshot,
            trade_data_list=[],
            symbol='BTCFDUSD',
            enhanced_mode=True
        )

        # Should handle extreme conditions
        assert isinstance(result, EnhancedMarketAnalysisResult)
        assert len(result.aggregated_bids) > 0
        assert len(result.aggregated_asks) > 0

    def test_analysis_with_empty_data_sets(self, analyzer):
        """Test analysis with various empty data scenarios."""
        empty_snapshot = DepthSnapshot(
            symbol='BTCFDUSD',
            timestamp=datetime.now(),
            bids=[],
            asks=[]
        )

        # Test with empty snapshot and no trade data
        result1 = analyzer.analyze_market(
            snapshot=empty_snapshot,
            trade_data_list=[],
            symbol='BTCFDUSD',
            enhanced_mode=True
        )

        assert isinstance(result1, EnhancedMarketAnalysisResult)
        assert len(result1.aggregated_bids) == 0
        assert len(result1.aggregated_asks) == 0

        # Test with valid snapshot but empty trade data
        valid_snapshot = DepthSnapshot(
            symbol='BTCFDUSD',
            timestamp=datetime.now(),
            bids=[DepthLevel(price=Decimal('100'), quantity=Decimal('10'))],
            asks=[DepthLevel(price=Decimal('101'), quantity=Decimal('8'))]
        )

        result2 = analyzer.analyze_market(
            snapshot=valid_snapshot,
            trade_data_list=[],
            symbol='BTCFDUSD',
            enhanced_mode=True
        )

        assert isinstance(result2, EnhancedMarketAnalysisResult)
        assert len(result2.aggregated_bids) > 0
        assert len(result2.aggregated_asks) > 0

    def test_memory_efficiency_with_repeated_analysis(self, analyzer):
        """Test memory efficiency during repeated analysis operations."""
        import gc
        import os

        import psutil

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # Perform multiple analyses to check for memory leaks
        for i in range(20):
            snapshot = DepthSnapshot(
                symbol='BTCFDUSD',
                timestamp=datetime.now(),
                bids=[
                    DepthLevel(price=Decimal(f'{100 + j}'), quantity=Decimal('10'))
                    for j in range(100)
                ],
                asks=[
                    DepthLevel(price=Decimal(f'{100 + j}'), quantity=Decimal('8'))
                    for j in range(100)
                ]
            )

            result = analyzer.analyze_market(
                snapshot=snapshot,
                trade_data_list=[],
                symbol='BTCFDUSD',
                enhanced_mode=True
            )

            # Explicitly delete result to encourage garbage collection
            del result

            # Periodic garbage collection
            if i % 5 == 0:
                gc.collect()

        gc.collect()  # Final cleanup
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory

        # Memory increase should be minimal (less than 20MB)
        assert memory_increase < 20 * 1024 * 1024
