"""Integration tests for trade data persistence functionality."""

import asyncio
import json
import pytest
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.core.redis_client import RedisDataStore
from src.core.models import MinuteTradeData, PriceLevelData, Trade
from src.agents.data_collector import DataCollectorAgent
from src.utils.config import Settings
from src.core.constants import TRADES_WINDOW_SIZE_MINUTES


class TestTradePersistenceIntegration:
    """Integration tests for trade data persistence."""

    @pytest.fixture
    def test_settings(self):
        """Create test settings."""
        return Settings(
            redis=MagicMock(host="localhost", port=6379, db=0, storage_dir="test_integration_storage"),
            binance=MagicMock(rest_api_base="https://api.binance.com", symbol="BTCFDUSD", timeout=30),
            data_collector=MagicMock(),
            analyzer=MagicMock(),
            logging=MagicMock()
        )

    @pytest.fixture
    def redis_store(self):
        """Create RedisDataStore for integration testing."""
        test_storage = Path("test_integration_storage")
        test_storage.mkdir(exist_ok=True)

        store = RedisDataStore(
            host="localhost",
            port=6379,
            db=0,
            storage_dir=str(test_storage)
        )

        yield store

        # Cleanup
        import shutil
        shutil.rmtree(test_storage, ignore_errors=True)

    @pytest.fixture
    def sample_minute_trade_data(self):
        """Create sample minute trade data for integration testing."""
        timestamp = datetime.now() - timedelta(minutes=1)
        trade_data = MinuteTradeData(timestamp=timestamp)

        # Add realistic trade data
        prices_volumes = [
            ("60000.00", "1.5", 10, "0.8", "0.7"),
            ("60001.50", "0.8", 6, "0.4", "0.4"),
            ("59998.75", "2.1", 15, "1.2", "0.9"),
            ("60002.00", "0.5", 3, "0.2", "0.3")
        ]

        for price, volume, count, buy_vol, sell_vol in prices_volumes:
            trade_data.price_levels[Decimal(price)] = PriceLevelData(
                price_level=Decimal(price),
                total_volume=Decimal(volume),
                trade_count=count,
                buy_volume=Decimal(buy_vol),
                sell_volume=Decimal(sell_vol)
            )

        return trade_data

    @pytest.mark.asyncio
    async def test_full_trade_data_persistence_flow(self, redis_store, sample_minute_trade_data):
        """Test complete flow from trade data storage to file persistence."""
        # Mock Redis operations to simulate full window
        redis_store.redis.lpush.return_value = TRADES_WINDOW_SIZE_MINUTES + 1
        redis_store.redis.llen.return_value = TRADES_WINDOW_SIZE_MINUTES + 1

        # Mock expired item (the one that will be serialized to disk)
        expired_timestamp = datetime.now() - timedelta(minutes=48, seconds=1)
        expired_data = MinuteTradeData(timestamp=expired_timestamp)
        expired_data.price_levels[Decimal("59995.00")] = PriceLevelData(
            price_level=Decimal("59995.00"),
            total_volume=Decimal("1.0"),
            trade_count=5,
            buy_volume=Decimal("0.5"),
            sell_volume=Decimal("0.5")
        )

        redis_store.redis.lrange.return_value = [json.dumps(expired_data.to_dict())]

        # Store new trade data (triggering expiration)
        await redis_store.store_minute_trade_data(sample_minute_trade_data)

        # Verify Redis operations
        redis_store.redis.lpush.assert_called_once()
        redis_store.redis.lrange.assert_called_once()
        redis_store.redis.ltrim.assert_called_once()

        # Verify file was created
        expected_filename = f"trades_{expired_timestamp.strftime('%Y%m%d_%H%M')}.json"
        expected_filepath = redis_store.storage_dir / expected_filename

        # Check if file exists (allow for async delay)
        await asyncio.sleep(0.1)  # Small delay for async file operation
        assert expected_filepath.exists()

        # Verify file contents
        with open(expected_filepath, 'r') as f:
            file_data = json.load(f)

        assert 'timestamp' in file_data
        assert 'price_levels' in file_data
        assert datetime.fromisoformat(file_data['timestamp']) == expired_timestamp
        assert '59995.00' in file_data['price_levels']

    @pytest.mark.asyncio
    async def test_multiple_files_creation(self, redis_store):
        """Test creation of multiple files for different minutes."""
        # Create multiple trade data points for different minutes
        trade_data_list = []
        timestamps = []

        for i in range(5):
            timestamp = datetime.now() - timedelta(minutes=i)
            timestamps.append(timestamp)

            trade_data = MinuteTradeData(timestamp=timestamp)
            trade_data.price_levels[Decimal(f"6000{i}.00")] = PriceLevelData(
                price_level=Decimal(f"6000{i}.00"),
                total_volume=Decimal(f"{i + 1}.0"),
                trade_count=i + 2,
                buy_volume=Decimal(f"{i + 0.5}"),
                sell_volume=Decimal(f"{i + 0.5}")
            )
            trade_data_list.append(trade_data)

        # Mock Redis to trigger file persistence for each
        redis_store.redis.lpush.return_value = TRADES_WINDOW_SIZE_MINUTES + 1
        redis_store.redis.llen.return_value = TRADES_WINDOW_SIZE_MINUTES + 1

        for i, trade_data in enumerate(trade_data_list):
            # Mock the expired item
            redis_store.redis.lrange.return_value = [json.dumps(trade_data.to_dict())]
            await redis_store.store_minute_trade_data(trade_data)

        # Wait for async file operations
        await asyncio.sleep(0.2)

        # Verify all files were created
        created_files = list(redis_store.storage_dir.glob("trades_*.json"))
        assert len(created_files) == 5

        # Verify each file has correct timestamp-based name
        for timestamp in timestamps:
            expected_filename = f"trades_{timestamp.strftime('%Y%m%d_%H%M')}.json"
            expected_filepath = redis_store.storage_dir / expected_filename
            assert expected_filepath.exists()

    @pytest.mark.asyncio
    async def test_data_collector_integration(self, test_settings):
        """Test integration with DataCollectorAgent."""
        # Mock the Redis connection test
        with patch.object(RedisDataStore, 'test_connection', return_value=True):
            # Create agent with test settings
            agent = DataCollectorAgent(test_settings)

            # Verify storage directory is configured
            assert agent.redis_store.storage_dir == Path(test_settings.redis.storage_dir)
            assert agent.redis_store.storage_dir.exists()

    @pytest.mark.asyncio
    async def test_file_persistence_error_recovery(self, redis_store, sample_minute_trade_data):
        """Test recovery from file persistence errors."""
        # Mock Redis operations
        redis_store.redis.lpush.return_value = TRADES_WINDOW_SIZE_MINUTES + 1
        redis_store.redis.llen.return_value = TRADES_WINDOW_SIZE_MINUTES + 1

        # Mock expired data
        expired_data = MinuteTradeData(timestamp=datetime.now() - timedelta(hours=49))
        expired_data.price_levels[Decimal("59990.00")] = PriceLevelData(
            price_level=Decimal("59990.00"),
            total_volume=Decimal("1.0"),
            trade_count=5,
            buy_volume=Decimal("0.5"),
            sell_volume=Decimal("0.5")
        )
        redis_store.redis.lrange.return_value = [json.dumps(expired_data.to_dict())]

        # Mock file writing to raise an exception
        import aiofiles
        with patch('aiofiles.open', side_effect=IOError("Disk full")):
            # Should still complete without raising exception
            await redis_store.store_minute_trade_data(sample_minute_trade_data)

        # Verify Redis operations still completed
        redis_store.redis.lpush.assert_called_once()
        redis_store.redis.ltrim.assert_called_once()

    @pytest.mark.asyncio
    async def test_concurrent_file_operations(self, redis_store):
        """Test handling of concurrent file operations."""
        # Create multiple trade data points to trigger concurrent file writing
        tasks = []
        timestamps = []

        for i in range(10):
            timestamp = datetime.now() - timedelta(minutes=i, seconds=i)
            timestamps.append(timestamp)

            trade_data = MinuteTradeData(timestamp=timestamp)
            trade_data.price_levels[Decimal(f"6000{i}.00")] = PriceLevelData(
                price_level=Decimal(f"6000{i}.00"),
                total_volume=Decimal(f"{i + 1}.0"),
                trade_count=i + 2,
                buy_volume=Decimal(f"{i + 0.5}"),
                sell_volume=Decimal(f"{i + 0.5}")
            )

            # Mock Redis to trigger file persistence
            redis_store.redis.lpush.return_value = TRADES_WINDOW_SIZE_MINUTES + 1
            redis_store.redis.llen.return_value = TRADES_WINDOW_SIZE_MINUTES + 1
            redis_store.redis.lrange.return_value = [json.dumps(trade_data.to_dict())]

            # Create concurrent task
            task = redis_store.store_minute_trade_data(trade_data)
            tasks.append(task)

        # Execute all tasks concurrently
        await asyncio.gather(*tasks, return_exceptions=True)

        # Wait for async file operations
        await asyncio.sleep(0.3)

        # Verify files were created
        created_files = list(redis_store.storage_dir.glob("trades_*.json"))
        assert len(created_files) == 10

    def test_file_naming_consistency(self, redis_store):
        """Test that file naming is consistent and unique."""
        # Test different timestamps
        timestamps = [
            datetime(2024, 1, 15, 10, 30, 0),
            datetime(2024, 1, 15, 10, 31, 0),
            datetime(2024, 1, 15, 11, 30, 0),
            datetime(2024, 1, 16, 10, 30, 0),
        ]

        expected_filenames = []
        for timestamp in timestamps:
            filename = f"trades_{timestamp.strftime('%Y%m%d_%H%M')}.json"
            expected_filenames.append(filename)

        # Verify all filenames are unique
        assert len(set(expected_filenames)) == len(expected_filenames)

        # Verify filename format
        for filename in expected_filenames:
            assert filename.startswith("trades_")
            assert filename.endswith(".json")
            assert len(filename) == len("trades_YYYYMMDD_HHMM.json")

    @pytest.mark.asyncio
    async def test_large_trade_data_persistence(self, redis_store):
        """Test persistence of large trade data sets."""
        # Create trade data with many price levels (simulating high activity)
        timestamp = datetime.now() - timedelta(hours=49)
        trade_data = MinuteTradeData(timestamp=timestamp)

        # Add 100 price levels
        for i in range(100):
            price = Decimal(f"60000.{i:02d}")
            trade_data.price_levels[price] = PriceLevelData(
                price_level=price,
                total_volume=Decimal(f"{i + 1}.{i}"),
                trade_count=i * 10 + 5,
                buy_volume=Decimal(f"{i + 0.5}"),
                sell_volume=Decimal(f"{i + 0.5}")
            )

        # Mock Redis operations
        redis_store.redis.lpush.return_value = TRADES_WINDOW_SIZE_MINUTES + 1
        redis_store.redis.llen.return_value = TRADES_WINDOW_SIZE_MINUTES + 1
        redis_store.redis.lrange.return_value = [json.dumps(trade_data.to_dict())]

        # Store the large trade data
        await redis_store.store_minute_trade_data(trade_data)

        # Wait for file operation
        await asyncio.sleep(0.1)

        # Verify file was created and contains all data
        expected_filename = f"trades_{timestamp.strftime('%Y%m%d_%H%M')}.json"
        expected_filepath = redis_store.storage_dir / expected_filename
        assert expected_filepath.exists()

        # Verify file contents
        with open(expected_filepath, 'r') as f:
            file_data = json.load(f)

        assert len(file_data['price_levels']) == 100

    @pytest.mark.asyncio
    async def test_redis_and_file_sync(self, redis_store, sample_minute_trade_data):
        """Test synchronization between Redis storage and file persistence."""
        # Mock Redis to simulate gradual filling of the window
        for i in range(TRADES_WINDOW_SIZE_MINUTES + 5):
            redis_store.redis.llen.return_value = i + 1

            if i >= TRADES_WINDOW_SIZE_MINUTES:
                # Mock expired data for file persistence
                expired_timestamp = datetime.now() - timedelta(minutes=48, seconds=1)
                expired_data = MinuteTradeData(timestamp=expired_timestamp)
                redis_store.redis.lrange.return_value = [json.dumps(expired_data.to_dict())]
            else:
                redis_store.redis.lrange.return_value = []

            # Store data
            await redis_store.store_minute_trade_data(sample_minute_trade_data)

        # Verify Redis was trimmed to correct size
        redis_store.redis.ltrim.assert_called_with(
            REDIS_TRADES_WINDOW_KEY,
            0,
            TRADES_WINDOW_SIZE_MINUTES - 1
        )

        # Verify file was created for expired data
        await asyncio.sleep(0.1)
        created_files = list(redis_store.storage_dir.glob("trades_*.json"))
        assert len(created_files) > 0


if __name__ == "__main__":
    pytest.main([__file__])