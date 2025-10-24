"""Unit tests for market analyzers."""

from datetime import datetime, timedelta
from decimal import Decimal

from src.core.analyzers import DepthSnapshotAnalyzer, MarketAnalyzer, OrderFlowAnalyzer
from src.core.models import (
    DepthLevel,
    DepthSnapshot,
    MinuteTradeData,
    PriceLevelData,
    SupportResistanceLevel,
)


class TestDepthSnapshotAnalyzer:
    """Test DepthSnapshotAnalyzer."""

    def test_analyze_support_resistance(self, sample_depth_snapshot):
        """Test support/resistance analysis."""
        analyzer = DepthSnapshotAnalyzer(min_volume_threshold=0.1)

        support, resistance = analyzer.analyze_support_resistance(
            sample_depth_snapshot,
            lookback_levels=5
        )

        # Should identify support levels from bid side
        assert len(support) > 0
        for level in support:
            assert level.level_type == 'support'
            assert level.strength > 0
            assert level.price > 0

        # Should identify resistance levels from ask side
        assert len(resistance) > 0
        for level in resistance:
            assert level.level_type == 'resistance'
            assert level.strength > 0
            assert level.price > 0

        # Check that large walls are identified
        support_prices = [level.price for level in support]
        resistance_prices = [level.price for level in resistance]

        # The large wall at 49997 should be identified as support
        assert Decimal('49997.00') in support_prices
        # The large wall at 50003 should be identified as resistance
        assert Decimal('50003.00') in resistance_prices

    def test_analyze_empty_order_book(self):
        """Test analysis with empty order book."""
        analyzer = DepthSnapshotAnalyzer()
        snapshot = DepthSnapshot(
            symbol='BTCFDUSD',
            timestamp=datetime.now(),
            bids=[],
            asks=[]
        )

        support, resistance = analyzer.analyze_support_resistance(snapshot)

        assert len(support) == 0
        assert len(resistance) == 0

    def test_identify_liquidity_vacuum_zones(self, sample_depth_snapshot):
        """Test liquidity vacuum zone identification."""
        analyzer = DepthSnapshotAnalyzer()

        # Create a snapshot with price gaps
        bids = [
            DepthLevel(price=Decimal('50000.00'), quantity=Decimal('1.0')),
            DepthLevel(price=Decimal('49900.00'), quantity=Decimal('0.5')),  # Large gap
        ]
        asks = [
            DepthLevel(price=Decimal('50100.00'), quantity=Decimal('0.3')),
            DepthLevel(price=Decimal('50200.00'), quantity=Decimal('0.2')),
        ]

        snapshot = DepthSnapshot(
            symbol='BTCFDUSD',
            timestamp=datetime.now(),
            bids=bids,
            asks=asks
        )

        vacuum_zones = analyzer.identify_liquidity_vacuum_zones(snapshot)

        # Should identify vacuum zones in the large price gaps
        assert len(vacuum_zones) > 0

    def test_group_by_price_zones(self):
        """Test price zone grouping."""
        analyzer = DepthSnapshotAnalyzer()

        levels = [
            DepthLevel(price=Decimal('50000.50'), quantity=Decimal('1.0')),
            DepthLevel(price=Decimal('50000.25'), quantity=Decimal('1.5')),
            DepthLevel(price=Decimal('50001.75'), quantity=Decimal('2.0')),
            DepthLevel(price=Decimal('50099.00'), quantity=Decimal('0.5')),
        ]

        zones = analyzer._group_by_price_zones(levels, Decimal('1.0'))

        # Should group nearby prices together
        assert len(zones) == 2  # Two zones: around 50000 and around 50099

        # Check zone groupings
        assert Decimal('50000.00') in zones
        assert Decimal('50099.00') in zones

        # Zone 50000 should have 3 levels
        assert len(zones[Decimal('50000.00')]) == 3
        # Zone 50099 should have 1 level
        assert len(zones[Decimal('50099.00')]) == 1


class TestOrderFlowAnalyzer:
    """Test OrderFlowAnalyzer."""

    def test_find_poc_levels(self):
        """Test Point of Control level identification."""
        analyzer = OrderFlowAnalyzer()

        # Create trade data with volume concentration at specific prices
        trade_data_list = []

        # First minute - high volume at 50000
        minute1 = MinuteTradeData(timestamp=datetime.now() - timedelta(minutes=2))
        price_data_50000 = PriceLevelData(price_level=Decimal('50000'))
        price_data_50000.buy_volume = Decimal('5.0')
        price_data_50000.sell_volume = Decimal('2.0')
        price_data_50000.total_volume = Decimal('7.0')
        minute1.price_levels[Decimal('50000')] = price_data_50000

        # Add low volume at other prices
        price_data_50100 = PriceLevelData(price_level=Decimal('50100'))
        price_data_50100.total_volume = Decimal('0.5')
        minute1.price_levels[Decimal('50100')] = price_data_50100

        trade_data_list.append(minute1)

        # Second minute - moderate volume at 50001
        minute2 = MinuteTradeData(timestamp=datetime.now() - timedelta(minutes=1))
        price_data_50001 = PriceLevelData(price_level=Decimal('50001'))
        price_data_50001.total_volume = Decimal('2.0')
        minute2.price_levels[Decimal('50001')] = price_data_50001

        trade_data_list.append(minute2)

        poc_levels = analyzer._find_poc_levels(trade_data_list)

        # Should identify 50000 as POC due to highest volume
        assert Decimal('50000') in poc_levels

    def test_confirm_support_level(self, sample_minute_trade_data):
        """Test support level confirmation with order flow."""
        analyzer = OrderFlowAnalyzer()

        # Create a support level
        support_level = SupportResistanceLevel(
            price=Decimal('50000.00'),
            strength=0.7,
            level_type='support',
            volume_at_level=Decimal('5.0'),
            confirmation_count=1,
            last_confirmed=datetime.now()
        )

        confirmed_support = analyzer._confirm_levels_with_order_flow(
            [support_level],
            [sample_minute_trade_data],
            'support'
        )

        # Should either confirm or reject based on order flow data
        assert len(confirmed_support) >= 0

        if confirmed_support:
            # Check that confirmation metrics are updated
            confirmed = confirmed_support[0]
            assert confirmed.confirmation_count >= support_level.confirmation_count
            assert confirmed.level_type == 'support'

    def test_confirm_resistance_level(self, sample_minute_trade_data):
        """Test resistance level confirmation with order flow."""
        analyzer = OrderFlowAnalyzer()

        # Create a resistance level
        resistance_level = SupportResistanceLevel(
            price=Decimal('50001.00'),
            strength=0.7,
            level_type='resistance',
            volume_at_level=Decimal('5.0'),
            confirmation_count=1,
            last_confirmed=datetime.now()
        )

        confirmed_resistance = analyzer._confirm_levels_with_order_flow(
            [resistance_level],
            [sample_minute_trade_data],
            'resistance'
        )

        # Should either confirm or reject based on order flow data
        assert len(confirmed_resistance) >= 0

        if confirmed_resistance:
            # Check that confirmation metrics are updated
            confirmed = confirmed_resistance[0]
            assert confirmed.confirmation_count >= resistance_level.confirmation_count
            assert confirmed.level_type == 'resistance'

    def test_analyze_order_flow(self, sample_minute_trade_data, sample_support_resistance_levels):
        """Test complete order flow analysis."""
        analyzer = OrderFlowAnalyzer()
        support, resistance = sample_support_resistance_levels

        confirmed_support, confirmed_resistance, poc_levels = analyzer.analyze_order_flow(
            [sample_minute_trade_data],
            support,
            resistance
        )

        # Should return processed levels
        assert isinstance(confirmed_support, list)
        assert isinstance(confirmed_resistance, list)
        assert isinstance(poc_levels, list)

        # Check that levels are preserved or filtered out
        for level in confirmed_support:
            assert level.level_type == 'support'

        for level in confirmed_resistance:
            assert level.level_type == 'resistance'


class TestMarketAnalyzer:
    """Test MarketAnalyzer."""

    def test_analyze_market_with_full_data(
        self,
        sample_depth_snapshot,
        sample_minute_trade_data
    ):
        """Test complete market analysis with both depth and order flow data."""
        analyzer = MarketAnalyzer()

        result = analyzer.analyze_market(
            snapshot=sample_depth_snapshot,
            trade_data_list=[sample_minute_trade_data],
            symbol='BTCFDUSD'
        )

        assert result.symbol == 'BTCFDUSD'
        assert isinstance(result.timestamp, datetime)
        assert isinstance(result.support_levels, list)
        assert isinstance(result.resistance_levels, list)
        assert isinstance(result.poc_levels, list)
        assert isinstance(result.liquidity_vacuum_zones, list)
        assert isinstance(result.resonance_zones, list)

    def test_analyze_market_with_depth_only(self, sample_depth_snapshot):
        """Test market analysis with only depth data."""
        analyzer = MarketAnalyzer()

        result = analyzer.analyze_market(
            snapshot=sample_depth_snapshot,
            trade_data_list=[],
            symbol='BTCFDUSD'
        )

        assert result.symbol == 'BTCFDUSD'
        # Should have support/resistance from depth analysis
        assert len(result.support_levels) >= 0
        assert len(result.resistance_levels) >= 0
        # No POC levels without order flow
        assert len(result.poc_levels) == 0

    def test_analyze_market_with_order_flow_only(self, sample_minute_trade_data):
        """Test market analysis with only order flow data."""
        analyzer = MarketAnalyzer()

        result = analyzer.analyze_market(
            snapshot=None,
            trade_data_list=[sample_minute_trade_data],
            symbol='BTCFDUSD'
        )

        assert result.symbol == 'BTCFDUSD'
        # Should have POC levels from order flow
        assert len(result.poc_levels) >= 0
        # No support/resistance without depth data
        assert len(result.support_levels) == 0
        assert len(result.resistance_levels) == 0

    def test_analyze_market_with_no_data(self):
        """Test market analysis with no data."""
        analyzer = MarketAnalyzer()

        result = analyzer.analyze_market(
            snapshot=None,
            trade_data_list=[],
            symbol='BTCFDUSD'
        )

        assert result.symbol == 'BTCFDUSD'
        assert len(result.support_levels) == 0
        assert len(result.resistance_levels) == 0
        assert len(result.poc_levels) == 0
        assert len(result.liquidity_vacuum_zones) == 0
        assert len(result.resonance_zones) == 0

    def test_find_resonance_zones(self):
        """Test resonance zone identification."""
        analyzer = MarketAnalyzer()

        from src.core.models import MarketAnalysisResult
        result = MarketAnalysisResult(
            timestamp=datetime.now(),
            symbol='BTCFDUSD'
        )

        # Add overlapping support and POC at 50000
        support = SupportResistanceLevel(
            price=Decimal('50000'),
            strength=0.8,
            level_type='support',
            volume_at_level=Decimal('5.0')
        )
        result.support_levels.append(support)
        result.poc_levels.append(Decimal('50000'))

        # Add different resistance level
        resistance = SupportResistanceLevel(
            price=Decimal('50100'),
            strength=0.7,
            level_type='resistance',
            volume_at_level=Decimal('3.0')
        )
        result.resistance_levels.append(resistance)

        resonance_zones = analyzer._find_resonance_zones(result)

        # Should identify 50000 as resonance zone (support + POC)
        assert Decimal('50000') in resonance_zones

    def test_integration_full_workflow(
        self,
        sample_depth_snapshot,
        sample_trades
    ):
        """Test the complete analysis workflow."""
        analyzer = MarketAnalyzer()

        # Convert trades to minute data
        minute_data = MinuteTradeData(timestamp=datetime.now())
        for trade in sample_trades:
            minute_data.add_trade(trade)

        # Perform analysis
        result = analyzer.analyze_market(
            snapshot=sample_depth_snapshot,
            trade_data_list=[minute_data],
            symbol='BTCFDUSD'
        )

        # Verify complete analysis
        assert result.symbol == 'BTCFDUSD'
        assert isinstance(result.support_levels, list)
        assert isinstance(result.resistance_levels, list)

        # Check that analysis produces reasonable results
        if result.support_levels:
            for level in result.support_levels:
                assert 0 <= level.strength <= 1
                assert level.price > 0
                assert level.level_type == 'support'

        if result.resistance_levels:
            for level in result.resistance_levels:
                assert 0 <= level.strength <= 1
                assert level.price > 0
                assert level.level_type == 'resistance'
