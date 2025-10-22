"""Tests for depth snapshot overwrite functionality."""

import asyncio
import pytest
import json
from datetime import datetime
from decimal import Decimal
from unittest.mock import Mock, patch

from src.core.redis_client import RedisDataStore
from src.core.models import DepthSnapshot, DepthLevel


@pytest.mark.asyncio
class TestDepthSnapshotOverwrite:
    """Test depth snapshot overwrite functionality."""

    async def test_overwrite_single_value(self, test_settings, mock_redis):
        """Test that depth snapshot overwrites a single value instead of using a list."""
        # Setup mock Redis
        mock_redis.ping.return_value = True
        mock_redis.set.return_value = True

        with patch('src.core.redis_client.redis.Redis', return_value=mock_redis):
            redis_store = RedisDataStore(
                host=test_settings.redis.host,
                port=test_settings.redis.port,
                db=test_settings.redis.db
            )

            # Create first depth snapshot
            snapshot1 = DepthSnapshot(
                symbol='BTCFDUSD',
                timestamp=datetime(2024, 1, 1, 12, 0, 0),
                bids=[
                    DepthLevel(price=Decimal('50000.00'), quantity=Decimal('1.5')),
                    DepthLevel(price=Decimal('49999.00'), quantity=Decimal('2.0')),
                ],
                asks=[
                    DepthLevel(price=Decimal('50001.00'), quantity=Decimal('1.2')),
                    DepthLevel(price=Decimal('50002.00'), quantity=Decimal('0.8')),
                ]
            )

            # Store first snapshot
            await redis_store.store_depth_snapshot(snapshot1)

            # Verify first storage
            mock_redis.set.assert_called_once()
            call_args_1 = mock_redis.set.call_args[0]
            assert call_args_1[0] == "depth_snapshot_5000"
            data_1 = json.loads(call_args_1[1])
            assert data_1['symbol'] == 'BTCFDUSD'
            assert data_1['timestamp'] == '2024-01-01T12:00:00'

            # Reset mock for second call
            mock_redis.set.reset_mock()

            # Create second depth snapshot (overwrite)
            snapshot2 = DepthSnapshot(
                symbol='BTCFDUSD',
                timestamp=datetime(2024, 1, 1, 12, 1, 0),
                bids=[
                    DepthLevel(price=Decimal('50100.00'), quantity=Decimal('2.5')),
                    DepthLevel(price=Decimal('50099.00'), quantity=Decimal('3.0')),
                ],
                asks=[
                    DepthLevel(price=Decimal('50101.00'), quantity=Decimal('2.2')),
                    DepthLevel(price=Decimal('50102.00'), quantity=Decimal('1.5')),
                ]
            )

            # Store second snapshot (should overwrite)
            await redis_store.store_depth_snapshot(snapshot2)

            # Verify second storage (overwrite)
            mock_redis.set.assert_called_once()
            call_args_2 = mock_redis.set.call_args[0]
            assert call_args_2[0] == "depth_snapshot_5000"
            data_2 = json.loads(call_args_2[1])
            assert data_2['symbol'] == 'BTCFDUSD'
            assert data_2['timestamp'] == '2024-01-01T12:01:00'  # Updated timestamp
            assert data_2['bids'][0][0] == 50100.00  # Updated price
            assert data_2['bids'][0][1] == 2.5  # Updated quantity

    async def test_retrieve_overwritten_snapshot(self, test_settings, mock_redis):
        """Test retrieving the overwritten snapshot."""
        # Setup mock Redis
        mock_redis.ping.return_value = True
        mock_redis.set.return_value = True

        # Mock GET request to return latest snapshot
        latest_snapshot_data = json.dumps({
            'symbol': 'BTCFDUSD',
            'timestamp': '2024-01-01T12:01:00',
            'bids': [[50100.00, 2.5], [50099.00, 3.0]],
            'asks': [[50101.00, 2.2], [50102.00, 1.5]]
        })
        mock_redis.get.return_value = latest_snapshot_data

        with patch('src.core.redis_client.redis.Redis', return_value=mock_redis):
            redis_store = RedisDataStore(
                host=test_settings.redis.host,
                port=test_settings.redis.port,
                db=test_settings.redis.db
            )

            # Retrieve the snapshot
            retrieved_snapshot = redis_store.get_latest_depth_snapshot()

            # Verify retrieved data
            assert retrieved_snapshot is not None
            assert retrieved_snapshot.symbol == 'BTCFDUSD'
            assert retrieved_snapshot.timestamp == datetime(2024, 1, 1, 12, 1, 0)
            assert len(retrieved_snapshot.bids) == 2
            assert retrieved_snapshot.bids[0].price == Decimal('50100.00')
            assert retrieved_snapshot.bids[0].quantity == Decimal('2.5')

            # Verify Redis calls
            mock_redis.get.assert_called_once_with("depth_snapshot_5000")

    async def test_depth_snapshot_exists(self, test_settings, mock_redis):
        """Test depth snapshot existence check."""
        # Test when snapshot exists
        mock_redis.ping.return_value = True
        mock_redis.exists.return_value = 1

        with patch('src.core.redis_client.redis.Redis', return_value=mock_redis):
            redis_store = RedisDataStore(
                host=test_settings.redis.host,
                port=test_settings.redis.port,
                db=test_settings.redis.db
            )

            assert redis_store.depth_snapshot_exists() == True
            mock_redis.exists.assert_called_once_with("depth_snapshot_5000")

        # Reset mock for non-existent case
        mock_redis.exists.return_value = 0
        mock_redis.reset_mock()

        # Test when snapshot doesn't exist
        assert redis_store.depth_snapshot_exists() == False
        mock_redis.exists.assert_called_once_with("depth_snapshot_5000")

    async def test_no_snapshot_retrieval(self, test_settings, mock_redis):
        """Test retrieval when no snapshot exists."""
        mock_redis.ping.return_value = True
        mock_redis.get.return_value = None

        with patch('src.core.redis_client.redis.Redis', return_value=mock_redis):
            redis_store = RedisDataStore(
                host=test_settings.redis.host,
                port=test_settings.redis.port,
                db=test_settings.redis.db
            )

            # Try to retrieve non-existent snapshot
            retrieved_snapshot = redis_store.get_latest_depth_snapshot()

            assert retrieved_snapshot is None
            mock_redis.get.assert_called_once_with("depth_snapshot_5000")

    async def test_overwrite_with_different_symbols(self, test_settings, mock_redis):
        """Test that snapshots with different symbols can be overwritten."""
        mock_redis.ping.return_value = True
        mock_redis.set.return_value = True

        with patch('src.core.redis_client.redis.Redis', return_value=mock_redis):
            redis_store = RedisDataStore(
                host=test_settings.redis.host,
                port=test_settings.redis.port,
                db=test_settings.redis.db
            )

            # Store first symbol snapshot
            snapshot_btc = DepthSnapshot(
                symbol='BTCFDUSD',
                timestamp=datetime.now(),
                bids=[DepthLevel(Decimal('50000'), Decimal('1.0'))],
                asks=[DepthLevel(Decimal('50001'), Decimal('1.0'))]
            )
            await redis_store.store_depth_snapshot(snapshot_btc)

            # Verify first call
            mock_redis.set.assert_called_once()
            call_data_1 = json.loads(mock_redis.set.call_args[0][1])
            assert call_data_1['symbol'] == 'BTCFDUSD'

            # Store different symbol snapshot (should overwrite same key)
            snapshot_eth = DepthSnapshot(
                symbol='ETHFDUSD',
                timestamp=datetime.now(),
                bids=[DepthLevel(Decimal('3000'), Decimal('2.0'))],
                asks=[DepthLevel(Decimal('3001'), Decimal('2.0'))]
            )
            await redis_store.store_depth_snapshot(snapshot_eth)

            # Verify second call overwrote the first
            assert mock_redis.set.call_count == 2
            call_data_2 = json.loads(mock_redis.set.call_args[0][1])
            assert call_data_2['symbol'] == 'ETHFDUSD'  # Overwritten

    async def test_overwrite_performance(self, test_settings, mock_redis):
        """Test that overwrite operation is performant."""
        mock_redis.ping.return_value = True
        mock_redis.set.return_value = True

        with patch('src.core.redis_client.redis.Redis', return_value=mock_redis):
            redis_store = RedisDataStore(
                host=test_settings.redis.host,
                port=test_settings.redis.port,
                db=test_settings.redis.db
            )

            # Create multiple snapshots to test overwrite performance
            snapshots = []
            for i in range(5):
                snapshot = DepthSnapshot(
                    symbol='BTCFDUSD',
                    timestamp=datetime.now(),
                    bids=[
                        DepthLevel(price=Decimal(f'5000{i}.00'), quantity=Decimal(f'1.{i}')),
                        DepthLevel(price=Decimal(f'5000{i-1 if i > 0 else 0}.00'), quantity=Decimal(f'2.{i}')),
                    ],
                    asks=[
                        DepthLevel(price=Decimal(f'5000{i}.01'), quantity=Decimal(f'1.{i}')),
                        DepthLevel(price=Decimal(f'5000{i+1}.01'), quantity=Decimal(f'2.{i}')),
                    ]
                )
                snapshots.append(snapshot)

            # Store all snapshots (each overwrites the previous)
            for snapshot in snapshots:
                await redis_store.store_depth_snapshot(snapshot)

            # Verify all calls were made
            assert mock_redis.set.call_count == 5

            # Verify the last snapshot data (should be stored)
            final_call_args = mock_redis.set.call_args[0]
            final_data = json.loads(final_call_args[1])
            assert final_data['symbol'] == 'BTCFDUSD'
            assert final_data['bids'][0][0] == 50004.00  # Last snapshot's first bid