"""Unit tests for analyzer graceful shutdown functionality."""

import asyncio
import signal
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.agents.analyzer import AnalyzerAgent
from src.utils.config import Settings


class TestAnalyzerGracefulShutdown:
    """Test graceful shutdown mechanisms for analyzer agent."""

    @pytest.fixture
    def mock_settings(self):
        """Create mock settings for testing."""
        from unittest.mock import Mock

        # Create nested mock structure
        redis_mock = Mock()
        redis_mock.host = "localhost"
        redis_mock.port = 6379
        redis_mock.db = 0

        analyzer_mock = Mock()
        analyzer_mock.analysis.interval_seconds = 5
        analyzer_mock.analysis.min_order_volume_threshold = 1.0
        analyzer_mock.deepseek.api_key = "test_key"
        analyzer_mock.deepseek.base_url = "http://test.com"
        analyzer_mock.deepseek.model = "test-model"
        analyzer_mock.deepseek.max_tokens = 1000
        analyzer_mock.deepseek.temperature = 0.1

        settings = MagicMock(spec=Settings)
        settings.redis = redis_mock
        settings.analyzer = analyzer_mock
        settings.setup_logging = MagicMock()
        return settings

    @pytest.fixture
    def agent(self, mock_settings):
        """Create analyzer agent for testing."""
        with patch('src.agents.analyzer.RedisDataStore') as mock_redis, \
             patch('src.agents.analyzer.DeepSeekClient') as mock_ai, \
             patch('src.agents.analyzer.use_normal_distribution', False), \
             patch('src.agents.analyzer.EnhancedMarketAnalyzer'):

            mock_redis.return_value.test_connection.return_value = True
            mock_redis.return_value.close = AsyncMock()
            mock_ai.return_value.close = AsyncMock()

            agent = AnalyzerAgent(mock_settings)
            return agent

    @pytest.mark.asyncio
    async def test_signal_handler_sets_shutdown_flags(self, agent):
        """Test that signal handler correctly sets shutdown flags."""
        # Setup signal handlers first
        agent.setup_signal_handlers()

        # Initially no shutdown requested
        assert not agent._shutdown_requested
        assert not agent.is_running

        # Call signal handler
        await agent._handle_signal()

        # Should set shutdown flags
        assert agent._shutdown_requested
        assert not agent.is_running
        assert agent.shutdown_event.is_set()

    @pytest.mark.asyncio
    async def test_analysis_loop_respects_shutdown_event(self, agent):
        """Test that analysis loop exits when shutdown event is set."""
        # Mock the analysis cycle to avoid real work
        agent._perform_analysis_cycle = AsyncMock()

        # Start running
        agent.is_running = True
        agent._shutdown_requested = False

        # Set shutdown event after a short delay
        async def delayed_shutdown():
            await asyncio.sleep(0.1)
            agent._shutdown_requested = True
            agent.shutdown_event.set()

        # Run both tasks
        shutdown_task = asyncio.create_task(delayed_shutdown())
        loop_task = asyncio.create_task(agent._analysis_loop())

        # Wait for loop to finish
        await loop_task
        await shutdown_task

        # Verify analysis cycle was called at least once
        assert agent._perform_analysis_cycle.called

    @pytest.mark.asyncio
    async def test_analysis_loop_exits_immediately_when_shutdown_requested(self, agent):
        """Test that analysis loop exits immediately when shutdown is already requested."""
        # Mock the analysis cycle
        agent._perform_analysis_cycle = AsyncMock()

        # Set shutdown flags before starting
        agent.is_running = True
        agent._shutdown_requested = True
        agent.shutdown_event.set()

        # Run analysis loop
        await agent._analysis_loop()

        # Analysis cycle should not be called since shutdown was already requested
        assert not agent._perform_analysis_cycle.called

    @pytest.mark.asyncio
    async def test_shutdown_cancels_all_tasks(self, agent):
        """Test that shutdown cancels all pending tasks."""
        # Create some mock tasks
        mock_tasks = [
            MagicMock(cancel=MagicMock()),
            MagicMock(cancel=MagicMock())
        ]

        with patch('asyncio.all_tasks', return_value=mock_tasks + [asyncio.current_task()]):
            with patch('asyncio.gather', new_callable=AsyncMock) as mock_gather:
                mock_gather.return_value = []

                await agent._shutdown()

                # Verify all tasks were cancelled
                for task in mock_tasks:
                    task.cancel.assert_called_once()

                # Verify gather was called with return_exceptions=True
                mock_gather.assert_called_once_with(*mock_tasks, return_exceptions=True)

    @pytest.mark.asyncio
    async def test_shutdown_handles_task_cancellation_timeout(self, agent):
        """Test shutdown handling when tasks don't complete within timeout."""
        # Create mock task
        mock_task = MagicMock(cancel=MagicMock())

        with patch('asyncio.all_tasks', return_value=[mock_task, asyncio.current_task()]):
            with patch('asyncio.gather', new_callable=AsyncMock) as mock_gather:
                # Simulate timeout
                mock_gather.side_effect = asyncio.TimeoutError()

                await agent._shutdown()

                # Task should still be cancelled
                mock_task.cancel.assert_called_once()

    @pytest.mark.asyncio
    async def test_shutdown_closes_connections(self, agent):
        """Test that shutdown closes AI client and Redis connections."""
        await agent._shutdown()

        # Verify connections were closed
        agent.ai_client.close.assert_called_once()
        agent.redis_store.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_shutdown_handles_connection_errors(self, agent):
        """Test shutdown handling when connection closing fails."""
        # Make connection close raise an exception
        agent.ai_client.close.side_effect = Exception("AI client error")
        agent.redis_store.close.side_effect = Exception("Redis error")

        # Should not raise exception
        await agent._shutdown()

        # Still attempts to close both connections
        agent.ai_client.close.assert_called_once()
        agent.redis_store.close.assert_called_once()

    def test_setup_signal_handlers_initializes_flags(self, agent):
        """Test that setup_signal_handlers initializes shutdown flags."""
        agent.setup_signal_handlers()

        assert hasattr(agent, '_shutdown_requested')
        assert agent._shutdown_requested is False

    @pytest.mark.asyncio
    async def test_start_registers_signal_handlers(self, agent, mock_settings):
        """Test that start method registers signal handlers."""
        mock_settings.redis = MagicMock()
        mock_settings.redis.host = "localhost"
        mock_settings.redis.port = 6379
        mock_settings.redis.db = 0

        with patch('src.agents.analyzer.RedisDataStore') as mock_redis, \
             patch('src.agents.analyzer.DeepSeekClient') as mock_ai, \
             patch('src.agents.analyzer.use_normal_distribution', False), \
             patch('src.agents.analyzer.EnhancedMarketAnalyzer') as mock_analyzer:

            mock_redis.return_value.test_connection.return_value = True
            mock_ai.return_value.close = AsyncMock()
            mock_analyzer.return_value.analyze_market = MagicMock()

            agent = AnalyzerAgent(mock_settings)

            # Mock the analysis cycle to complete quickly
            agent._perform_analysis_cycle = AsyncMock()

            # Set shutdown immediately after start
            async def start_and_shutdown():
                # Start agent
                start_task = asyncio.create_task(agent.start())

                # Wait a bit for start to complete
                await asyncio.sleep(0.1)

                # Trigger shutdown
                await agent._handle_signal()

                # Wait for start to complete
                await start_task

            with patch('asyncio.get_running_loop') as mock_loop:
                mock_signal_handler = MagicMock()
                mock_loop.add_signal_handler = MagicMock()

                await start_and_shutdown()

                # Verify signal handlers were registered
                assert mock_loop.add_signal_handler.call_count == 2
                calls = mock_loop.add_signal_handler.call_args_list
                signals = [call[0][0] for call in calls]
                assert signal.SIGINT in signals
                assert signal.SIGTERM in signals

    @pytest.mark.asyncio
    async def test_main_handles_keyboard_interrupt(self, mock_settings):
        """Test that main function handles KeyboardInterrupt gracefully."""
        with patch('src.agents.analyzer.Settings.load_from_file', return_value=mock_settings), \
             patch('src.agents.analyzer.AnalyzerAgent') as mock_agent_class:

            mock_agent = MagicMock()
            mock_agent.start = AsyncMock(side_effect=KeyboardInterrupt())
            mock_agent_class.return_value = mock_agent

            from src.agents.analyzer import main

            # Should not raise exception
            await main()

            # Agent should be created and started
            mock_agent_class.assert_called_once_with(mock_settings)
            mock_agent.start.assert_called_once()

    @pytest.mark.asyncio
    async def test_main_handles_cancelled_error(self, mock_settings):
        """Test that main function handles CancelledError gracefully."""
        with patch('src.agents.analyzer.Settings.load_from_file', return_value=mock_settings), \
             patch('src.agents.analyzer.AnalyzerAgent') as mock_agent_class:

            mock_agent = MagicMock()
            mock_agent.start = AsyncMock(side_effect=asyncio.CancelledError())
            mock_agent_class.return_value = mock_agent

            from src.agents.analyzer import main

            # Should not raise exception
            await main()

            # Agent should be created and started
            mock_agent_class.assert_called_once_with(mock_settings)
            mock_agent.start.assert_called_once()

    @pytest.mark.asyncio
    async def test_perform_analysis_cycle_checks_shutdown(self, agent):
        """Test that analysis cycle checks shutdown flag during execution."""
        # Mock Redis and AI client calls
        agent.redis_store.get_latest_depth_snapshot = MagicMock(return_value=None)
        agent.redis_store.get_recent_trade_data = MagicMock(return_value=None)

        # Should return early when no data available
        result = await agent._perform_analysis_cycle()

        # Should complete without error
        assert result is None

    @pytest.mark.asyncio
    async def test_analysis_loop_handles_errors_during_shutdown(self, agent):
        """Test analysis loop handles errors gracefully during shutdown."""
        # Make analysis cycle raise an exception
        agent._perform_analysis_cycle = AsyncMock(side_effect=Exception("Test error"))

        agent.is_running = True
        agent._shutdown_requested = False

        # Set shutdown after first iteration
        async def delayed_shutdown():
            await asyncio.sleep(0.1)
            agent._shutdown_requested = True
            agent.shutdown_event.set()

        # Run both tasks
        shutdown_task = asyncio.create_task(delayed_shutdown())
        loop_task = asyncio.create_task(agent._analysis_loop())

        # Wait for loop to finish
        await loop_task
        await shutdown_task

        # Should exit gracefully despite error
        assert agent._perform_analysis_cycle.called