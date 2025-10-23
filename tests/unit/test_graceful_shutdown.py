"""Tests for graceful shutdown functionality."""

import asyncio
import signal
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.agents.data_collector import DataCollectorAgent
from src.utils.config import Settings


class TestGracefulShutdown:
    """Test suite for graceful shutdown functionality."""

    @pytest.fixture
    def mock_settings(self):
        """Create mock settings for testing."""
        from unittest.mock import Mock

        # Create nested mock structure
        redis_mock = Mock()
        redis_mock.host = "localhost"
        redis_mock.port = 6379
        redis_mock.db = 0
        redis_mock.storage_dir = "test_storage"

        binance_mock = Mock()
        binance_mock.rest_api_base = "https://api.binance.com"
        binance_mock.symbol = "BTCFDUSD"
        binance_mock.timeout = 30

        data_collector_mock = Mock()
        data_collector_mock.depth_snapshot.update_interval_seconds = 60
        data_collector_mock.order_flow.aggregation_interval_seconds = 60

        settings = MagicMock(spec=Settings)
        settings.redis = redis_mock
        settings.binance = binance_mock
        settings.data_collector = data_collector_mock
        return settings

    @pytest.fixture
    def data_collector(self, mock_settings):
        """Create DataCollectorAgent instance for testing."""
        with patch('src.agents.data_collector.RedisDataStore'), \
             patch('src.agents.data_collector.BinanceAPIClient'), \
             patch('src.agents.data_collector.BinanceWebSocketClient'):
            agent = DataCollectorAgent(mock_settings)
            return agent

    def test_signal_handler_sets_shutdown_flags(self, data_collector):
        """Test that signal handler correctly sets shutdown flags."""
        # Initially should not be shutting down
        assert not data_collector.is_running
        assert not data_collector.shutdown_event.is_set()

        # Simulate signal handler call
        data_collector._signal_handler(signal.SIGINT, None)

        # Should set shutdown flags
        assert not data_collector.is_running
        assert data_collector.shutdown_event.is_set()

    @pytest.mark.asyncio
    async def test_task_cancellation_on_shutdown_signal(self, data_collector):
        """Test that tasks are cancelled when shutdown signal is received."""
        # Mock the event loop
        mock_loop = MagicMock()
        data_collector.loop = mock_loop

        # Create some mock tasks
        mock_task1 = AsyncMock()
        mock_task2 = AsyncMock()
        mock_task3 = AsyncMock()

        mock_task1.done.return_value = False
        mock_task2.done.return_value = True  # Already done
        mock_task3.done.return_value = False

        data_collector.tasks = [mock_task1, mock_task2, mock_task3]

        # Call signal handler
        data_collector._signal_handler(signal.SIGTERM, None)

        # Verify that call_soon_threadsafe was called
        mock_loop.call_soon_threadsafe.assert_called_once_with(
            data_collector._schedule_immediate_shutdown
        )

        # Manually call the scheduled function
        data_collector._schedule_immediate_shutdown()

        # Verify cancel was called on non-done tasks
        mock_task1.cancel.assert_called_once()
        mock_task2.cancel.assert_not_called()  # Already done
        mock_task3.cancel.assert_called_once()

    @pytest.mark.asyncio
    async def test_cancel_remaining_tasks_with_timeout(self, data_collector):
        """Test cancellation of remaining tasks with timeout."""
        # Create mock tasks
        async def mock_task():
            await asyncio.sleep(10)  # Long running task

        task1 = asyncio.create_task(mock_task())
        task2 = asyncio.create_task(mock_task())
        data_collector.tasks = [task1, task2]

        # Test task cancellation with timeout
        start_time = asyncio.get_event_loop().time()
        await data_collector._cancel_remaining_tasks()
        end_time = asyncio.get_event_loop().time()

        # Should complete quickly (within timeout period)
        assert end_time - start_time < 15  # Allow some margin

        # Tasks should be cancelled
        assert task1.cancelled()
        assert task2.cancelled()

    @pytest.mark.asyncio
    async def test_close_connections_with_timeout(self, data_collector):
        """Test connection closing with timeout protection."""
        # Mock the connections
        data_collector.websocket_client.disconnect = AsyncMock()
        data_collector.redis_store.close = AsyncMock()
        data_collector.api_client.close_async_session = AsyncMock()

        # Test normal closing
        await data_collector._close_connections_with_timeout()

        # Verify all close methods were called
        data_collector.websocket_client.disconnect.assert_called_once()
        data_collector.redis_store.close.assert_called_once()
        data_collector.api_client.close_async_session.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_connections_timeout_handling(self, data_collector):
        """Test timeout handling during connection closing."""
        # Mock slow connections that timeout
        async def slow_disconnect():
            await asyncio.sleep(10)  # Longer than timeout

        data_collector.websocket_client.disconnect = AsyncMock(side_effect=slow_disconnect)
        data_collector.redis_store.close = AsyncMock()
        data_collector.api_client.close_async_session = AsyncMock()

        # Should complete despite slow WebSocket disconnect
        start_time = asyncio.get_event_loop().time()
        await data_collector._close_connections_with_timeout()
        end_time = asyncio.get_event_loop().time()

        # Should complete within timeout period
        assert end_time - start_time < 10

        # Other connections should still be attempted
        data_collector.redis_store.close.assert_called_once()
        data_collector.api_client.close_async_session.assert_called_once()

    @pytest.mark.asyncio
    async def test_shutdown_saves_remaining_data(self, data_collector, mock_settings):
        """Test that shutdown saves remaining aggregated data."""
        # Mock some aggregated data
        from src.core.models import MinuteTradeData, PriceLevelData
        from decimal import Decimal

        data_collector.current_minute_data.price_levels[Decimal("60000")] = PriceLevelData(
            price_level=Decimal("60000"),
            total_volume=Decimal("1.0"),
            trade_count=5
        )

        # Mock the store method
        data_collector._aggregate_and_store_minute_data = AsyncMock()

        # Mock cleanup methods
        data_collector._cancel_remaining_tasks = AsyncMock()
        data_collector._close_connections_with_timeout = AsyncMock()

        # Test shutdown
        await data_collector._shutdown()

        # Verify data was saved
        data_collector._aggregate_and_store_minute_data.assert_called_once()

        # Verify cleanup was called
        data_collector._cancel_remaining_tasks.assert_called_once()
        data_collector._close_connections_with_timeout.assert_called_once()

    @pytest.mark.asyncio
    async def test_depth_snapshot_collector_cancellation(self, data_collector):
        """Test depth snapshot collector responds to cancellation."""
        # Mock Redis store and API client
        data_collector.redis_store.test_connection.return_value = True
        data_collector.redis_store.store_depth_snapshot = AsyncMock()
        data_collector.api_client.get_depth_snapshot = AsyncMock()

        # Start the collector task
        task = asyncio.create_task(data_collector._depth_snapshot_collector())

        # Let it run briefly
        await asyncio.sleep(0.1)

        # Cancel the task
        task.cancel()

        # Should complete without hanging
        with pytest.raises(asyncio.CancelledError):
            await task

    @pytest.mark.asyncio
    async def test_trade_aggregator_cancellation(self, data_collector):
        """Test trade aggregator responds to cancellation."""
        # Mock Redis store
        data_collector.redis_store.store_minute_trade_data = AsyncMock()

        # Start the aggregator task
        task = asyncio.create_task(data_collector._trade_aggregator())

        # Let it run briefly
        await asyncio.sleep(0.1)

        # Cancel the task
        task.cancel()

        # Should complete without hanging
        with pytest.raises(asyncio.CancelledError):
            await task

    @pytest.mark.asyncio
    async def test_websocket_cancellation_support(self, data_collector):
        """Test WebSocket collector cancellation support."""
        # Mock WebSocket client
        data_collector.websocket_client.connect = AsyncMock(return_value=True)
        data_collector.websocket_client.listen_trades = AsyncMock()
        data_collector.websocket_client.disconnect = AsyncMock()

        # Start the WebSocket collector task
        task = asyncio.create_task(data_collector._websocket_trade_collector())

        # Let it start
        await asyncio.sleep(0.1)

        # Trigger shutdown
        data_collector.shutdown_event.set()

        # Should complete without hanging
        await task

        # Verify disconnect was called
        data_collector.websocket_client.disconnect.assert_called()

    @pytest.mark.asyncio
    async def test_multiple_sigint_signals(self, data_collector):
        """Test handling multiple SIGINT signals."""
        # Mock event loop
        mock_loop = MagicMock()
        data_collector.loop = mock_loop

        # Create mock tasks
        mock_task = AsyncMock()
        mock_task.done.return_value = False
        data_collector.tasks = [mock_task]

        # Send multiple signals
        data_collector._signal_handler(signal.SIGINT, None)
        data_collector._signal_handler(signal.SIGINT, None)

        # Should handle gracefully (second signal should not cause issues)
        assert mock_loop.call_soon_threadsafe.call_count == 2

    @pytest.mark.asyncio
    async def test_shutdown_with_no_tasks(self, data_collector):
        """Test shutdown when no tasks are running."""
        data_collector.tasks = []

        # Should complete without errors
        await data_collector._cancel_remaining_tasks()

    @pytest.mark.asyncio
    async def test_graceful_shutdown_integration(self, data_collector, mock_settings):
        """Test complete graceful shutdown integration."""
        # Mock all dependencies
        data_collector.redis_store.test_connection.return_value = True
        data_collector.redis_store.store_depth_snapshot = AsyncMock()
        data_collector.api_client.get_depth_snapshot = AsyncMock()
        data_collector.websocket_client.connect = AsyncMock(return_value=True)
        data_collector.websocket_client.listen_trades = AsyncMock()
        data_collector.websocket_client.disconnect = AsyncMock()
        data_collector.redis_store.close = AsyncMock()
        data_collector.api_client.close_async_session = AsyncMock()

        # Start the agent
        start_task = asyncio.create_task(data_collector.start())

        # Let it start briefly
        await asyncio.sleep(0.1)

        # Trigger shutdown
        data_collector.shutdown_event.set()

        # Should complete gracefully
        await start_task

        # Verify cleanup was called
        data_collector.websocket_client.disconnect.assert_called()
        data_collector.redis_store.close.assert_called()
        data_collector.api_client.close_async_session.assert_called()


if __name__ == "__main__":
    pytest.main([__file__])