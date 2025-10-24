"""Pytest configuration and fixtures."""

import asyncio
from datetime import datetime, timedelta
from decimal import Decimal

import pytest

from src.core.models import (
    DepthLevel,
    DepthSnapshot,
    MinuteTradeData,
    SupportResistanceLevel,
    Trade,
)


@pytest.fixture
def sample_depth_snapshot():
    """Create a sample depth snapshot for testing."""
    bids = [
        DepthLevel(price=Decimal('50000.00'), quantity=Decimal('1.5')),
        DepthLevel(price=Decimal('49999.00'), quantity=Decimal('2.0')),
        DepthLevel(price=Decimal('49998.00'), quantity=Decimal('0.8')),
        DepthLevel(price=Decimal('49997.00'), quantity=Decimal('5.0')),  # Large wall
        DepthLevel(price=Decimal('49996.00'), quantity=Decimal('0.3')),
    ]

    asks = [
        DepthLevel(price=Decimal('50001.00'), quantity=Decimal('0.7')),
        DepthLevel(price=Decimal('50002.00'), quantity=Decimal('1.2')),
        DepthLevel(price=Decimal('50003.00'), quantity=Decimal('3.0')),  # Large wall
        DepthLevel(price=Decimal('50004.00'), quantity=Decimal('0.9')),
        DepthLevel(price=Decimal('50005.00'), quantity=Decimal('2.5')),
    ]

    return DepthSnapshot(
        symbol='BTCFDUSD',
        timestamp=datetime.now(),
        bids=bids,
        asks=asks
    )


@pytest.fixture
def sample_trades():
    """Create sample trades for testing."""
    base_time = datetime.now() - timedelta(minutes=5)

    trades = [
        Trade(
            symbol='BTCFDUSD',
            price=Decimal('50000.50'),
            quantity=Decimal('0.1'),
            is_buyer_maker=False,  # Aggressive buyer
            timestamp=base_time,
            trade_id='1'
        ),
        Trade(
            symbol='BTCFDUSD',
            price=Decimal('50000.25'),
            quantity=Decimal('0.2'),
            is_buyer_maker=True,   # Aggressive seller
            timestamp=base_time + timedelta(seconds=10),
            trade_id='2'
        ),
        Trade(
            symbol='BTCFDUSD',
            price=Decimal('50000.00'),
            quantity=Decimal('0.5'),
            is_buyer_maker=False,  # Aggressive buyer
            timestamp=base_time + timedelta(seconds=20),
            trade_id='3'
        ),
        Trade(
            symbol='BTCFDUSD',
            price=Decimal('50001.00'),
            quantity=Decimal('0.15'),
            is_buyer_maker=True,   # Aggressive seller
            timestamp=base_time + timedelta(seconds=30),
            trade_id='4'
        ),
        Trade(
            symbol='BTCFDUSD',
            price=Decimal('50000.75'),
            quantity=Decimal('0.3'),
            is_buyer_maker=False,  # Aggressive buyer
            timestamp=base_time + timedelta(seconds=40),
            trade_id='5'
        ),
    ]

    return trades


@pytest.fixture
def sample_minute_trade_data(sample_trades):
    """Create sample minute trade data for testing."""
    minute_data = MinuteTradeData(timestamp=datetime.now() - timedelta(minutes=1))

    for trade in sample_trades:
        minute_data.add_trade(trade)

    return minute_data


@pytest.fixture
def sample_support_resistance_levels():
    """Create sample support and resistance levels."""
    support = [
        SupportResistanceLevel(
            price=Decimal('49997.00'),
            strength=0.8,
            level_type='support',
            volume_at_level=Decimal('5.0'),
            confirmation_count=2,
            last_confirmed=datetime.now()
        ),
        SupportResistanceLevel(
            price=Decimal('49995.00'),
            strength=0.6,
            level_type='support',
            volume_at_level=Decimal('2.0'),
            confirmation_count=1,
            last_confirmed=datetime.now() - timedelta(minutes=5)
        )
    ]

    resistance = [
        SupportResistanceLevel(
            price=Decimal('50003.00'),
            strength=0.9,
            level_type='resistance',
            volume_at_level=Decimal('3.0'),
            confirmation_count=3,
            last_confirmed=datetime.now()
        ),
        SupportResistanceLevel(
            price=Decimal('50005.00'),
            strength=0.5,
            level_type='resistance',
            volume_at_level=Decimal('2.5'),
            confirmation_count=1,
            last_confirmed=datetime.now() - timedelta(minutes=3)
        )
    ]

    return support, resistance


@pytest.fixture
def mock_redis():
    """Create a mock Redis client for testing."""
    from unittest.mock import Mock

    mock_client = Mock()
    mock_client.ping.return_value = True
    mock_client.lpush.return_value = 1
    mock_client.ltrim.return_value = True
    mock_client.lindex.return_value = None
    mock_client.lrange.return_value = []
    mock_client.llen.return_value = 0
    mock_client.setex.return_value = True
    mock_client.get.return_value = None
    mock_client.keys.return_value = []
    mock_client.delete.return_value = 1

    return mock_client


@pytest.fixture
def test_settings():
    """Create test settings."""
    from src.utils.config import Settings

    return Settings(
        app={
            'name': 'test-agent',
            'environment': 'test',
            'log_level': 'DEBUG'
        },
        redis={
            'host': 'localhost',
            'port': 6379,
            'db': 15,  # Use test DB
            'decode_responses': True,
            'socket_timeout': 5,
            'socket_connect_timeout': 5
        },
        binance={
            'rest_api_base': 'https://api.binance.com',
            'websocket_base': 'wss://stream.binance.com:9443',
            'symbol': 'BTCFDUSD',
            'rate_limit_requests_per_minute': 1200,
            'timeout': 30
        },
        analyzer={
            'deepseek': {
                'api_key': 'test-key',
                'base_url': 'https://api.deepseek.com/v1',
                'model': 'deepseek-chat',
                'max_tokens': 4000,
                'temperature': 0.1
            },
            'analysis': {
                'interval_seconds': 60,
                'min_order_volume_threshold': 0.01,
                'support_resistance_threshold': 0.1
            }
        },
        logging={
            'level': 'DEBUG',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'file_path': 'logs/test.log',
            'max_file_size_mb': 10,
            'backup_count': 2
        }
    )


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()
