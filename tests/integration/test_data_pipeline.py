"""Integration tests for the complete data pipeline."""

import asyncio
import pytest
import json
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, Mock, patch

from src.core.redis_client import RedisDataStore
from src.core.models import DepthSnapshot, DepthLevel, Trade, MinuteTradeData
from src.agents.data_collector import DataCollectorAgent
from src.agents.analyzer import AnalyzerAgent
from src.utils.config import Settings


@pytest.mark.asyncio
class TestDataPipelineIntegration:
    """Test the complete data pipeline integration."""

    async def test_depth_snapshot_storage_flow(self, test_settings, mock_redis):
        """Test depth snapshot collection and storage flow."""
        # Setup mock Redis
        mock_redis.ping.return_value = True
        mock_redis.lpush.return_value = 1
        mock_redis.ltrim.return_value = True

        with patch('src.core.redis_client.redis.Redis', return_value=mock_redis):
            redis_store = RedisDataStore(
                host=test_settings.redis.host,
                port=test_settings.redis.port,
                db=test_settings.redis.db
            )

            # Create test depth snapshot
            snapshot = DepthSnapshot(
                symbol='BTCFDUSD',
                timestamp=datetime.now(),
                bids=[
                    DepthLevel(price=Decimal('50000.00'), quantity=Decimal('1.5')),
                    DepthLevel(price=Decimal('49999.00'), quantity=Decimal('2.0')),
                ],
                asks=[
                    DepthLevel(price=Decimal('50001.00'), quantity=Decimal('1.2')),
                    DepthLevel(price=Decimal('50002.00'), quantity=Decimal('0.8')),
                ]
            )

            # Store snapshot
            await redis_store.store_depth_snapshot(snapshot)

            # Verify Redis calls
            mock_redis.lpush.assert_called_once()
            mock_redis.ltrim.assert_called_once()

            # Verify the data format
            call_args = mock_redis.lpush.call_args[0]
            key = call_args[0]
            data = json.loads(call_args[1])

            assert key == "depth_snapshot_5000"
            assert data['symbol'] == 'BTCFDUSD'
            assert len(data['bids']) == 2
            assert len(data['asks']) == 2

    async def test_trade_data_aggregation_flow(self, test_settings, mock_redis):
        """Test trade data aggregation and storage flow."""
        # Setup mock Redis
        mock_redis.ping.return_value = True
        mock_redis.lpush.return_value = 1
        mock_redis.ltrim.return_value = True

        with patch('src.core.redis_client.redis.Redis', return_value=mock_redis):
            redis_store = RedisDataStore(
                host=test_settings.redis.host,
                port=test_settings.redis.port,
                db=test_settings.redis.db
            )

            # Create test minute trade data
            minute_data = MinuteTradeData(timestamp=datetime.now())

            # Add some trades
            trades = [
                Trade(
                    symbol='BTCFDUSD',
                    price=Decimal('50000.75'),
                    quantity=Decimal('0.1'),
                    is_buyer_maker=False,
                    timestamp=datetime.now(),
                    trade_id='1'
                ),
                Trade(
                    symbol='BTCFDUSD',
                    price=Decimal('50000.25'),
                    quantity=Decimal('0.2'),
                    is_buyer_maker=True,
                    timestamp=datetime.now(),
                    trade_id='2'
                ),
                Trade(
                    symbol='BTCFDUSD',
                    price=Decimal('50001.50'),
                    quantity=Decimal('0.15'),
                    is_buyer_maker=False,
                    timestamp=datetime.now(),
                    trade_id='3'
                ),
            ]

            for trade in trades:
                minute_data.add_trade(trade)

            # Store minute data
            await redis_store.store_minute_trade_data(minute_data)

            # Verify Redis calls
            mock_redis.lpush.assert_called_once()
            mock_redis.ltrim.assert_called_once()

            # Verify the data format
            call_args = mock_redis.lpush.call_args[0]
            key = call_args[0]
            data = json.loads(call_args[1])

            assert key == "trades_window"
            assert 'timestamp' in data
            assert 'price_levels' in data

            # Check price aggregation (should round to $1 precision)
            price_levels = data['price_levels']
            assert '50000.0' in price_levels  # 50000.75 and 50000.25 should aggregate
            assert '50001.0' in price_levels  # 50001.50 should round

            # Check aggregated volumes for 50000 level
            level_50000 = price_levels['50000.0']
            assert level_50000['total_volume'] == 0.3  # 0.1 + 0.2
            assert level_50000['trade_count'] == 2

    async def test_data_collector_initialization(self, test_settings):
        """Test data collector agent initialization."""
        with patch('src.core.redis_client.redis.Redis') as mock_redis:
            mock_redis.ping.return_value = True

            with patch('src.utils.binance_client.BinanceAPIClient') as mock_api:
                with patch('src.utils.binance_client.BinanceWebSocketClient') as mock_ws:
                    agent = DataCollectorAgent(test_settings)

                    # Check that components are initialized
                    assert agent.settings == test_settings
                    assert agent.redis_store is not None
                    assert agent.api_client is not None
                    assert agent.websocket_client is not None
                    assert agent.is_running is False

    async def test_analyzer_initialization(self, test_settings):
        """Test analyzer agent initialization."""
        # Mock DeepSeek client to avoid API calls
        with patch('src.core.redis_client.redis.Redis') as mock_redis:
            mock_redis.ping.return_value = True

            with patch('src.utils.ai_client.httpx.AsyncClient'):
                agent = AnalyzerAgent(test_settings)

                # Check that components are initialized
                assert agent.settings == test_settings
                assert agent.redis_store is not None
                assert agent.market_analyzer is not None
                assert agent.ai_client is not None
                assert agent.is_running is False

    @pytest.mark.skip(reason="Integration test requires Redis instance")
    async def test_end_to_end_data_flow(self, test_settings):
        """Test complete end-to-end data flow (requires Redis)."""
        # This test would require a running Redis instance
        # It's skipped by default but can be enabled for integration testing

        # Initialize real Redis connection
        redis_store = RedisDataStore(
            host=test_settings.redis.host,
            port=test_settings.redis.port,
            db=test_settings.redis.db
        )

        if not redis_store.test_connection():
            pytest.skip("Redis not available for integration testing")

        try:
            # Clear any existing data
            redis_store.clear_all_data()

            # Store test depth snapshot
            snapshot = DepthSnapshot(
                symbol='BTCFDUSD',
                timestamp=datetime.now(),
                bids=[
                    DepthLevel(price=Decimal('50000.00'), quantity=Decimal('1.5')),
                ],
                asks=[
                    DepthLevel(price=Decimal('50001.00'), quantity=Decimal('1.2')),
                ]
            )
            await redis_store.store_depth_snapshot(snapshot)

            # Store test trade data
            minute_data = MinuteTradeData(timestamp=datetime.now())
            trade = Trade(
                symbol='BTCFDUSD',
                price=Decimal('50000.50'),
                quantity=Decimal('0.1'),
                is_buyer_maker=False,
                timestamp=datetime.now(),
                trade_id='1'
            )
            minute_data.add_trade(trade)
            await redis_store.store_minute_trade_data(minute_data)

            # Retrieve and verify data
            retrieved_snapshot = redis_store.get_latest_depth_snapshot()
            assert retrieved_snapshot is not None
            assert retrieved_snapshot.symbol == 'BTCFDUSD'

            retrieved_trades = redis_store.get_recent_trade_data(minutes=60)
            assert len(retrieved_trades) > 0

        finally:
            # Cleanup
            redis_store.clear_all_data()
            await redis_store.close()

    async def test_data_flow_with_failures(self, test_settings, mock_redis):
        """Test data flow handling of failures."""
        # Setup Redis to simulate failures
        mock_redis.ping.side_effect = [True, False]  # Second call fails
        mock_redis.lpush.side_effect = Exception("Redis connection failed")

        with patch('src.core.redis_client.redis.Redis', return_value=mock_redis):
            redis_store = RedisDataStore(
                host=test_settings.redis.host,
                port=test_settings.redis.port,
                db=test_settings.redis.db
            )

            # Test connection failure handling
            assert redis_store.test_connection() == True  # First call succeeds

            # Test storage failure handling
            snapshot = DepthSnapshot(
                symbol='BTCFDUSD',
                timestamp=datetime.now(),
                bids=[],
                asks=[]
            )

            with pytest.raises(Exception):
                await redis_store.store_depth_snapshot(snapshot)

    async def test_data_retrieval_consistency(self, test_settings, mock_redis):
        """Test data retrieval consistency."""
        # Setup mock Redis with test data
        test_snapshot_data = json.dumps({
            'symbol': 'BTCFDUSD',
            'timestamp': datetime.now().isoformat(),
            'bids': [['50000.00', '1.5'], ['49999.00', '2.0']],
            'asks': [['50001.00', '1.2'], ['50002.00', '0.8']]
        })

        mock_redis.ping.return_value = True
        mock_redis.lindex.return_value = test_snapshot_data
        mock_redis.lrange.return_value = [test_snapshot_data]
        mock_redis.llen.return_value = 1

        with patch('src.core.redis_client.redis.Redis', return_value=mock_redis):
            redis_store = RedisDataStore(
                host=test_settings.redis.host,
                port=test_settings.redis.port,
                db=test_settings.redis.db
            )

            # Test depth snapshot retrieval
            snapshot = redis_store.get_latest_depth_snapshot()
            assert snapshot is not None
            assert snapshot.symbol == 'BTCFDUSD'
            assert len(snapshot.bids) == 2
            assert len(snapshot.asks) == 2

            # Test trade data retrieval
            trade_data = redis_store.get_recent_trade_data(minutes=60)
            assert isinstance(trade_data, list)

            # Test counters
            assert redis_store.get_depth_snapshot_count() == 1
            assert redis_store.get_trade_window_count() == 1

    async def test_data_format_validation(self, test_settings, mock_redis):
        """Test that stored data format is valid and can be retrieved."""
        mock_redis.ping.return_value = True
        mock_redis.lpush.return_value = 1
        mock_redis.ltrim.return_value = True

        with patch('src.core.redis_client.redis.Redis', return_value=mock_redis):
            redis_store = RedisDataStore(
                host=test_settings.redis.host,
                port=test_settings.redis.port,
                db=test_settings.redis.db
            )

            # Create and store complex data
            snapshot = DepthSnapshot(
                symbol='BTCFDUSD',
                timestamp=datetime.now(),
                bids=[
                    DepthLevel(price=Decimal('50000.123456'), quantity=Decimal('1.987654')),
                    DepthLevel(price=Decimal('49999.987654'), quantity=Decimal('2.123456')),
                ],
                asks=[
                    DepthLevel(price=Decimal('50001.456789'), quantity=Decimal('1.345678')),
                    DepthLevel(price=Decimal('50002.012345'), quantity=Decimal('0.876543')),
                ]
            )

            await redis_store.store_depth_snapshot(snapshot)

            # Verify the stored data can be properly serialized
            call_args = mock_redis.lpush.call_args[0]
            data_str = call_args[1]

            # Should be valid JSON
            parsed_data = json.loads(data_str)
            assert isinstance(parsed_data, dict)
            assert 'symbol' in parsed_data
            assert 'timestamp' in parsed_data
            assert 'bids' in parsed_data
            assert 'asks' in parsed_data

            # Verify precision handling
            assert isinstance(parsed_data['bids'][0][0], float)  # Price converted to float
            assert isinstance(parsed_data['bids'][0][1], float)  # Quantity converted to float