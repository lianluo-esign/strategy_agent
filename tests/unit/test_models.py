"""Unit tests for data models."""

from datetime import datetime
from decimal import Decimal

from src.core.models import (
    DepthLevel,
    DepthSnapshot,
    MarketAnalysisResult,
    MinuteTradeData,
    PriceLevelData,
    SupportResistanceLevel,
    Trade,
    TradingRecommendation,
)


class TestDepthLevel:
    """Test DepthLevel model."""

    def test_depth_level_creation(self):
        """Test depth level creation."""
        level = DepthLevel(price=Decimal('50000.00'), quantity=Decimal('1.5'))

        assert level.price == Decimal('50000.00')
        assert level.quantity == Decimal('1.5')

    def test_depth_level_string_conversion(self):
        """Test depth level converts strings to Decimal."""
        level = DepthLevel(price='50000.00', quantity='1.5')

        assert isinstance(level.price, Decimal)
        assert isinstance(level.quantity, Decimal)
        assert level.price == Decimal('50000.00')
        assert level.quantity == Decimal('1.5')


class TestDepthSnapshot:
    """Test DepthSnapshot model."""

    def test_depth_snapshot_creation(self):
        """Test depth snapshot creation."""
        bids = [DepthLevel(price=Decimal('50000.00'), quantity=Decimal('1.0'))]
        asks = [DepthLevel(price=Decimal('50001.00'), quantity=Decimal('1.0'))]

        snapshot = DepthSnapshot(
            symbol='BTCFDUSD',
            timestamp=datetime.now(),
            bids=bids,
            asks=asks
        )

        assert snapshot.symbol == 'BTCFDUSD'
        assert len(snapshot.bids) == 1
        assert len(snapshot.asks) == 1

    def test_get_best_bid(self):
        """Test getting best bid price."""
        bids = [
            DepthLevel(price=Decimal('49999.00'), quantity=Decimal('1.0')),
            DepthLevel(price=Decimal('50000.00'), quantity=Decimal('1.5')),
            DepthLevel(price=Decimal('49998.00'), quantity=Decimal('2.0'))
        ]

        snapshot = DepthSnapshot(
            symbol='BTCFDUSD',
            timestamp=datetime.now(),
            bids=bids
        )

        assert snapshot.get_best_bid() == Decimal('50000.00')

    def test_get_best_ask(self):
        """Test getting best ask price."""
        asks = [
            DepthLevel(price=Decimal('50002.00'), quantity=Decimal('1.0')),
            DepthLevel(price=Decimal('50001.00'), quantity=Decimal('1.5')),
            DepthLevel(price=Decimal('50003.00'), quantity=Decimal('2.0'))
        ]

        snapshot = DepthSnapshot(
            symbol='BTCFDUSD',
            timestamp=datetime.now(),
            asks=asks
        )

        assert snapshot.get_best_ask() == Decimal('50001.00')

    def test_empty_order_book(self):
        """Test empty order book handling."""
        snapshot = DepthSnapshot(
            symbol='BTCFDUSD',
            timestamp=datetime.now()
        )

        assert snapshot.get_best_bid() is None
        assert snapshot.get_best_ask() is None


class TestTrade:
    """Test Trade model."""

    def test_trade_creation(self):
        """Test trade creation."""
        timestamp = datetime.now()
        trade = Trade(
            symbol='BTCFDUSD',
            price=Decimal('50000.50'),
            quantity=Decimal('0.1'),
            is_buyer_maker=False,
            timestamp=timestamp,
            trade_id='12345'
        )

        assert trade.symbol == 'BTCFDUSD'
        assert trade.price == Decimal('50000.50')
        assert trade.quantity == Decimal('0.1')
        assert trade.is_buyer_maker is False
        assert trade.timestamp == timestamp
        assert trade.trade_id == '12345'

    def test_trade_string_conversion(self):
        """Test trade converts strings to Decimal."""
        timestamp = datetime.now()
        trade = Trade(
            symbol='BTCFDUSD',
            price='50000.50',
            quantity='0.1',
            is_buyer_maker=False,
            timestamp=timestamp,
            trade_id='12345'
        )

        assert isinstance(trade.price, Decimal)
        assert isinstance(trade.quantity, Decimal)


class TestPriceLevelData:
    """Test PriceLevelData model."""

    def test_price_level_data_creation(self):
        """Test price level data creation."""
        price = Decimal('50000.00')
        data = PriceLevelData(price_level=price)

        assert data.price_level == price
        assert data.buy_volume == Decimal('0')
        assert data.sell_volume == Decimal('0')
        assert data.total_volume == Decimal('0')
        assert data.delta == Decimal('0')
        assert data.trade_count == 0

    def test_add_buy_trade(self):
        """Test adding a buy trade."""
        price = Decimal('50000.00')
        data = PriceLevelData(price_level=price)

        trade = Trade(
            symbol='BTCFDUSD',
            price=price,
            quantity=Decimal('0.1'),
            is_buyer_maker=False,  # Aggressive buyer
            timestamp=datetime.now(),
            trade_id='1'
        )

        data.add_trade(trade)

        assert data.buy_volume == Decimal('0.1')
        assert data.sell_volume == Decimal('0')
        assert data.total_volume == Decimal('0.1')
        assert data.delta == Decimal('0.1')
        assert data.trade_count == 1

    def test_add_sell_trade(self):
        """Test adding a sell trade."""
        price = Decimal('50000.00')
        data = PriceLevelData(price_level=price)

        trade = Trade(
            symbol='BTCFDUSD',
            price=price,
            quantity=Decimal('0.1'),
            is_buyer_maker=True,  # Aggressive seller
            timestamp=datetime.now(),
            trade_id='1'
        )

        data.add_trade(trade)

        assert data.buy_volume == Decimal('0')
        assert data.sell_volume == Decimal('0.1')
        assert data.total_volume == Decimal('0.1')
        assert data.delta == Decimal('-0.1')
        assert data.trade_count == 1

    def test_multiple_trades(self):
        """Test adding multiple trades."""
        price = Decimal('50000.00')
        data = PriceLevelData(price_level=price)

        # Add buy trade
        buy_trade = Trade(
            symbol='BTCFDUSD',
            price=price,
            quantity=Decimal('0.1'),
            is_buyer_maker=False,
            timestamp=datetime.now(),
            trade_id='1'
        )

        # Add sell trade
        sell_trade = Trade(
            symbol='BTCFDUSD',
            price=price,
            quantity=Decimal('0.2'),
            is_buyer_maker=True,
            timestamp=datetime.now(),
            trade_id='2'
        )

        data.add_trade(buy_trade)
        data.add_trade(sell_trade)

        assert data.buy_volume == Decimal('0.1')
        assert data.sell_volume == Decimal('0.2')
        assert data.total_volume == Decimal('0.3')
        assert data.delta == Decimal('-0.1')
        assert data.trade_count == 2

    def test_to_dict(self):
        """Test converting to dictionary."""
        price = Decimal('50000.00')
        data = PriceLevelData(price_level=price)

        trade = Trade(
            symbol='BTCFDUSD',
            price=price,
            quantity=Decimal('0.1'),
            is_buyer_maker=False,
            timestamp=datetime.now(),
            trade_id='1'
        )

        data.add_trade(trade)
        result = data.to_dict()

        assert result['price_level'] == 50000.0
        assert result['buy_volume'] == 0.1
        assert result['sell_volume'] == 0.0
        assert result['total_volume'] == 0.1
        assert result['delta'] == 0.1
        assert result['trade_count'] == 1


class TestMinuteTradeData:
    """Test MinuteTradeData model."""

    def test_minute_trade_data_creation(self):
        """Test minute trade data creation."""
        timestamp = datetime.now()
        data = MinuteTradeData(timestamp=timestamp)

        assert data.timestamp == timestamp
        assert len(data.price_levels) == 0

    def test_add_trade_aggregation(self):
        """Test trade aggregation by price level."""
        timestamp = datetime.now()
        data = MinuteTradeData(timestamp=timestamp)

        # Add trades at different price levels
        trade1 = Trade(
            symbol='BTCFDUSD',
            price=Decimal('50000.00'),
            quantity=Decimal('0.1'),
            is_buyer_maker=False,
            timestamp=timestamp,
            trade_id='1'
        )

        trade2 = Trade(
            symbol='BTCFDUSD',
            price=Decimal('50001.00'),
            quantity=Decimal('0.2'),
            is_buyer_maker=True,
            timestamp=timestamp,
            trade_id='2'
        )

        trade3 = Trade(
            symbol='BTCFDUSD',
            price=Decimal('50000.00'),  # Same price as trade1
            quantity=Decimal('0.15'),
            is_buyer_maker=False,
            timestamp=timestamp,
            trade_id='3'
        )

        data.add_trade(trade1)
        data.add_trade(trade2)
        data.add_trade(trade3)

        # Check aggregation
        assert len(data.price_levels) == 2

        # Price 50000 should have aggregated data
        price_50000 = data.price_levels[Decimal('50000.00')]
        assert price_50000.total_volume == Decimal('0.25')  # 0.1 + 0.15
        assert price_50000.trade_count == 2

        # Price 50001 should have single trade
        price_50001 = data.price_levels[Decimal('50001.00')]
        assert price_50001.total_volume == Decimal('0.2')
        assert price_50001.trade_count == 1

    def test_price_precision_rounding(self):
        """Test price rounding to $1 precision."""
        timestamp = datetime.now()
        data = MinuteTradeData(timestamp=timestamp)

        # Add trades with fractional prices
        trade1 = Trade(
            symbol='BTCFDUSD',
            price=Decimal('50000.75'),
            quantity=Decimal('0.1'),
            is_buyer_maker=False,
            timestamp=timestamp,
            trade_id='1'
        )

        trade2 = Trade(
            symbol='BTCFDUSD',
            price=Decimal('50000.25'),
            quantity=Decimal('0.1'),
            is_buyer_maker=False,
            timestamp=timestamp,
            trade_id='2'
        )

        data.add_trade(trade1)
        data.add_trade(trade2)

        # Both should be rounded to 50000
        assert len(data.price_levels) == 1
        assert Decimal('50000.00') in data.price_levels
        assert data.price_levels[Decimal('50000.00')].total_volume == Decimal('0.2')


class TestSupportResistanceLevel:
    """Test SupportResistanceLevel model."""

    def test_support_resistance_creation(self):
        """Test support/resistance level creation."""
        timestamp = datetime.now()
        level = SupportResistanceLevel(
            price=Decimal('50000.00'),
            strength=0.8,
            level_type='support',
            volume_at_level=Decimal('5.0'),
            confirmation_count=2,
            last_confirmed=timestamp
        )

        assert level.price == Decimal('50000.00')
        assert level.strength == 0.8
        assert level.level_type == 'support'
        assert level.volume_at_level == Decimal('5.0')
        assert level.confirmation_count == 2
        assert level.last_confirmed == timestamp

    def test_to_dict(self):
        """Test converting to dictionary."""
        timestamp = datetime.now()
        level = SupportResistanceLevel(
            price=Decimal('50000.00'),
            strength=0.8,
            level_type='support',
            volume_at_level=Decimal('5.0'),
            confirmation_count=2,
            last_confirmed=timestamp
        )

        result = level.to_dict()

        assert result['price'] == 50000.0
        assert result['strength'] == 0.8
        assert result['level_type'] == 'support'
        assert result['volume_at_level'] == 5.0
        assert result['confirmation_count'] == 2
        assert result['last_confirmed'] == timestamp.isoformat()


class TestMarketAnalysisResult:
    """Test MarketAnalysisResult model."""

    def test_market_analysis_result_creation(self):
        """Test market analysis result creation."""
        timestamp = datetime.now()
        result = MarketAnalysisResult(
            timestamp=timestamp,
            symbol='BTCFDUSD'
        )

        assert result.timestamp == timestamp
        assert result.symbol == 'BTCFDUSD'
        assert len(result.support_levels) == 0
        assert len(result.resistance_levels) == 0
        assert len(result.poc_levels) == 0
        assert len(result.liquidity_vacuum_zones) == 0
        assert len(result.resonance_zones) == 0

    def test_to_dict(self):
        """Test converting to dictionary."""
        timestamp = datetime.now()
        result = MarketAnalysisResult(
            timestamp=timestamp,
            symbol='BTCFDUSD',
            poc_levels=[Decimal('50000.00'), Decimal('50100.00')],
            liquidity_vacuum_zones=[Decimal('50050.00')],
            resonance_zones=[Decimal('50000.00')]
        )

        dict_result = result.to_dict()

        assert dict_result['timestamp'] == timestamp.isoformat()
        assert dict_result['symbol'] == 'BTCFDUSD'
        assert dict_result['poc_levels'] == [50000.0, 50100.0]
        assert dict_result['liquidity_vacuum_zones'] == [50050.0]
        assert dict_result['resonance_zones'] == [50000.0]


class TestTradingRecommendation:
    """Test TradingRecommendation model."""

    def test_trading_recommendation_creation(self):
        """Test trading recommendation creation."""
        timestamp = datetime.now()
        recommendation = TradingRecommendation(
            timestamp=timestamp,
            symbol='BTCFDUSD',
            action='BUY',
            price_range=(Decimal('49999.00'), Decimal('50000.00')),
            confidence=0.8,
            reasoning='Strong support at $50,000 with high volume',
            risk_level='LOW'
        )

        assert recommendation.timestamp == timestamp
        assert recommendation.symbol == 'BTCFDUSD'
        assert recommendation.action == 'BUY'
        assert recommendation.price_range == (Decimal('49999.00'), Decimal('50000.00'))
        assert recommendation.confidence == 0.8
        assert recommendation.reasoning == 'Strong support at $50,000 with high volume'
        assert recommendation.risk_level == 'LOW'

    def test_to_dict(self):
        """Test converting to dictionary."""
        timestamp = datetime.now()
        recommendation = TradingRecommendation(
            timestamp=timestamp,
            symbol='BTCFDUSD',
            action='BUY',
            price_range=(Decimal('49999.00'), Decimal('50000.00')),
            confidence=0.8,
            reasoning='Strong support at $50,000 with high volume',
            risk_level='LOW'
        )

        result = recommendation.to_dict()

        assert result['timestamp'] == timestamp.isoformat()
        assert result['symbol'] == 'BTCFDUSD'
        assert result['action'] == 'BUY'
        assert result['price_range'] == [49999.0, 50000.0]
        assert result['confidence'] == 0.8
        assert result['reasoning'] == 'Strong support at $50,000 with high volume'
        assert result['risk_level'] == 'LOW'
