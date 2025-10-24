"""Tests for trade data persistence functionality."""

import asyncio
import json
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.constants import REDIS_TRADES_WINDOW_KEY, TRADES_WINDOW_SIZE_MINUTES
from src.core.models import MinuteTradeData, PriceLevelData
from src.core.redis_client import RedisDataStore


class TestTradeDataPersistence:
    """Test suite for trade data persistence to disk."""

    @pytest.fixture
    def redis_store(self):
        """Create RedisDataStore instance with test storage directory."""
        test_storage = Path("test_storage")
        test_storage.mkdir(exist_ok=True)

        store = RedisDataStore(
            host="localhost",
            port=6379,
            db=0,
            storage_dir=str(test_storage)
        )

        # Mock Redis client
        store.redis = MagicMock()
        store.async_redis = MagicMock()

        yield store

        # Cleanup
        import shutil
        shutil.rmtree(test_storage, ignore_errors=True)

    @pytest.fixture
    def sample_trade_data(self):
        """Create sample trade data for testing."""
        timestamp = datetime.now()
        trade_data = MinuteTradeData(timestamp=timestamp)

        # Add some price levels
        trade_data.price_levels[Decimal("60000.00")] = PriceLevelData(
            price_level=Decimal("60000.00"),
            total_volume=Decimal("1.5"),
            trade_count=10,
            buy_volume=Decimal("0.8"),
            sell_volume=Decimal("0.7")
        )

        trade_data.price_levels[Decimal("60001.00")] = PriceLevelData(
            price_level=Decimal("60001.00"),
            total_volume=Decimal("0.7"),
            trade_count=5,
            buy_volume=Decimal("0.3"),
            sell_volume=Decimal("0.4")
        )

        return trade_data

    @pytest.fixture
    def sample_trade_data_dict(self, sample_trade_data):
        """Get dictionary representation of sample trade data."""
        return sample_trade_data.to_dict()

    @pytest.mark.asyncio
    async def test_handle_expired_trade_data_no_expiration(self, redis_store, sample_trade_data):
        """Test handling when no data is expired."""
        # Mock Redis to return count equal to window size
        redis_store.redis.llen.return_value = TRADES_WINDOW_SIZE_MINUTES

        # Should not trigger file serialization
        await redis_store._handle_expired_trade_data()

        # Verify no file operations were attempted
        redis_store.redis.lrange.assert_not_called()
        redis_store.redis.ltrim.assert_not_called()

    @pytest.mark.asyncio
    async def test_handle_expired_trade_data_with_expiration(self, redis_store, sample_trade_data_dict):
        """Test handling when data is expired and needs serialization."""
        # Mock Redis to return more than window size items
        expired_count = 5
        redis_store.redis.llen.return_value = TRADES_WINDOW_SIZE_MINUTES + expired_count

        # Mock expired items
        expired_items = [json.dumps(sample_trade_data_dict) for _ in range(expired_count)]
        redis_store.redis.lrange.return_value = expired_items

        # Mock file writing method
        redis_store._write_trade_data_file = AsyncMock()

        # Execute the method
        await redis_store._handle_expired_trade_data()

        # Verify Redis operations
        redis_store.redis.lrange.assert_called_once_with(
            REDIS_TRADES_WINDOW_KEY,
            TRADES_WINDOW_SIZE_MINUTES,
            -1
        )
        redis_store.redis.ltrim.assert_called_once_with(
            REDIS_TRADES_WINDOW_KEY,
            0,
            TRADES_WINDOW_SIZE_MINUTES - 1
        )

        # Verify file writing was called for each expired item
        assert redis_store._write_trade_data_file.call_count == expired_count

    @pytest.mark.asyncio
    async def test_serialize_trade_data_to_files_concurrent(self, redis_store, sample_trade_data_dict):
        """Test concurrent serialization of multiple trade data items."""
        expired_items = [json.dumps(sample_trade_data_dict) for _ in range(10)]

        # Mock file writing method
        redis_store._write_trade_data_file = AsyncMock()

        # Execute serialization
        await redis_store._serialize_trade_data_to_files(expired_items)

        # Verify all items were processed
        assert redis_store._write_trade_data_file.call_count == len(expired_items)

        # Verify all calls were made with correct data
        for call in redis_store._write_trade_data_file.call_args_list:
            args, kwargs = call
            assert args[0] in expired_items

    @pytest.mark.asyncio
    async def test_write_trade_data_file(self, redis_store, sample_trade_data_dict):
        """Test writing a single trade data file."""

        # Mock aiofiles.open with proper async context manager
        mock_file = AsyncMock()
        mock_file.write = AsyncMock()

        with patch('aiofiles.open') as mock_open_call:
            mock_open_call.return_value.__aenter__.return_value = mock_file

            # Execute file writing
            await redis_store._write_trade_data_file(json.dumps(sample_trade_data_dict))

            # Verify file was opened for writing
            mock_open_call.assert_called_once()
            args, kwargs = mock_open_call.call_args
            assert str(args[0]).endswith('.json')
            assert args[1] == 'w'  # mode is the second positional argument

            # Verify data was written to file
            mock_file.write.assert_called_once()
            written_data = mock_file.write.call_args[0][0]

            # Verify JSON structure
            parsed_data = json.loads(written_data)
            assert 'timestamp' in parsed_data
            assert 'price_levels' in parsed_data
            assert isinstance(parsed_data['price_levels'], dict)

    @pytest.mark.asyncio
    async def test_write_trade_data_file_error_handling(self, redis_store, sample_trade_data_dict):
        """Test error handling when writing trade data file."""
        # Mock aiofiles.open to raise an exception
        with patch('aiofiles.open', side_effect=OSError("Disk full")):
            # Should not raise exception, just log error
            await redis_store._write_trade_data_file(json.dumps(sample_trade_data_dict))
            # Method should complete without raising

    @pytest.mark.asyncio
    async def test_write_trade_data_file_invalid_json(self, redis_store):
        """Test handling of invalid JSON data."""
        invalid_data = "invalid json string"

        # Should not raise exception, just log error
        await redis_store._write_trade_data_file(invalid_data)
        # Method should complete without raising

    @pytest.mark.asyncio
    async def test_store_minute_trade_data_with_expiration(self, redis_store, sample_trade_data):
        """Test storing trade data with expired data handling."""
        # Mock Redis operations
        redis_store.redis.lpush.return_value = 1
        redis_store.redis.llen.return_value = TRADES_WINDOW_SIZE_MINUTES + 3
        redis_store.redis.lrange.return_value = ['{"test": "data"}', '{"test": "data2"}', '{"test": "data3"}']

        # Mock file writing
        redis_store._write_trade_data_file = AsyncMock()

        # Store trade data
        await redis_store.store_minute_trade_data(sample_trade_data)

        # Verify Redis operations
        redis_store.redis.lpush.assert_called_once()
        redis_store.redis.lrange.assert_called_once()
        redis_store.redis.ltrim.assert_called_once()

        # Verify file writing was called
        assert redis_store._write_trade_data_file.call_count == 3

    def test_filename_generation(self, redis_store, sample_trade_data_dict):
        """Test that filenames are generated correctly based on timestamp."""
        # Extract timestamp from sample data
        timestamp_str = sample_trade_data_dict['timestamp']
        timestamp = datetime.fromisoformat(timestamp_str)

        # Expected filename format
        expected_filename = f"trades_{timestamp.strftime('%Y%m%d_%H%M')}.json"
        expected_filepath = redis_store.storage_dir / expected_filename

        # Verify the path construction
        assert expected_filepath.name == expected_filename
        assert expected_filepath.parent == redis_store.storage_dir

    @pytest.mark.asyncio
    async def test_serialize_trade_data_to_files_empty_list(self, redis_store):
        """Test serialization with empty expired items list."""
        # Should not call file writing method
        await redis_store._serialize_trade_data_to_files([])
        # No exceptions should be raised

    @pytest.mark.asyncio
    async def test_handle_expired_trade_data_redis_error(self, redis_store):
        """Test handling Redis errors during expired data processing."""
        # Mock Redis to raise an exception
        redis_store.redis.llen.side_effect = Exception("Redis connection error")

        # Should not raise exception, just log error
        await redis_store._handle_expired_trade_data()
        # Method should complete without raising

    @pytest.mark.asyncio
    async def test_store_minute_trade_data_redis_error(self, redis_store, sample_trade_data):
        """Test handling Redis errors during trade data storage."""
        # Mock Redis to raise an exception during storage
        redis_store.redis.lpush.side_effect = Exception("Redis error")

        # Should raise exception for main storage operation
        with pytest.raises(Exception, match="Redis error"):
            await redis_store.store_minute_trade_data(sample_trade_data)

    def test_storage_dir_creation(self):
        """Test that storage directory is created during initialization."""
        import shutil
        import tempfile

        # Create a temporary directory
        temp_dir = Path(tempfile.mkdtemp())
        test_storage_dir = temp_dir / "test_storage"

        try:
            # Initialize Redis store with non-existent directory
            store = RedisDataStore(
                host="localhost",
                port=6379,
                db=0,
                storage_dir=str(test_storage_dir)
            )

            # Verify directory was created
            assert test_storage_dir.exists()
            assert test_storage_dir.is_dir()

        finally:
            # Cleanup
            shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_concurrent_file_writing(self, redis_store, sample_trade_data_dict):
        """Test that multiple files can be written concurrently."""

        # Create multiple trade data items with different timestamps
        items = []
        for i in range(5):
            data = sample_trade_data_dict.copy()
            data['timestamp'] = (datetime.now() + timedelta(minutes=i)).isoformat()
            items.append(json.dumps(data))

        # Track file creation
        created_files = []

        async def track_file_creation(data_str):
            data = json.loads(data_str)
            timestamp = datetime.fromisoformat(data['timestamp'])
            filename = f"trades_{timestamp.strftime('%Y%m%d_%H%M')}.json"
            filepath = redis_store.storage_dir / filename

            # Mock file writing
            mock_file = AsyncMock()
            with patch('aiofiles.open', return_value=mock_file.__aenter__()):
                await redis_store._write_trade_data_file(data_str)

            created_files.append(filepath.name)

        # Execute concurrent file writing
        await asyncio.gather(*(track_file_creation(item) for item in items))

        # Verify all files were created with unique names
        assert len(created_files) == len(items)
        assert len(set(created_files)) == len(items)  # All unique


if __name__ == "__main__":
    pytest.main([__file__])
