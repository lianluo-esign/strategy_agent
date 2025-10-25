"""Unit tests for analyzer agent visualization integration."""

import asyncio
from datetime import datetime
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.models import DepthLevel, DepthSnapshot
from src.utils.config import Settings, VisualizationConfig


class TestAnalyzerVisualizationIntegration:
    """Test cases for analyzer agent visualization integration."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Create minimal settings for testing
        self.settings = Settings(
            app=MagicMock(),
            redis=MagicMock(),
            binance=MagicMock(),
            data_collector=MagicMock(),
            analyzer=MagicMock(
                deepseek=MagicMock(enable=False),
                analysis=MagicMock(interval_seconds=60),
                visualization=VisualizationConfig(
                    enabled=True,
                    output_base_path="/tmp/test_visualizations"
                )
            ),
            logging=MagicMock()
        )

    @pytest.mark.asyncio
    async def test_analyzer_visualizer_initialization_enabled(self):
        """Test analyzer initializes visualizer when visualization is enabled."""
        with patch('src.agents.analyzer.OrderBookVisualizer') as mock_visualizer_class, \
             patch('src.agents.analyzer.RedisDataStore') as mock_redis, \
             patch('src.agents.analyzer.MarketAnalyzer') as mock_analyzer:

            mock_visualizer = MagicMock()
            mock_visualizer_class.return_value = mock_visualizer

            from src.agents.analyzer import AnalyzerAgent
            agent = AnalyzerAgent(self.settings)

            # Verify visualizer was initialized
            mock_visualizer_class.assert_called_once_with(
                config=self.settings.analyzer.visualization
            )
            assert agent.order_book_visualizer == mock_visualizer

    @pytest.mark.asyncio
    async def test_analyzer_visualizer_initialization_disabled(self):
        """Test analyzer does not initialize visualizer when disabled."""
        self.settings.analyzer.visualization.enabled = False

        with patch('src.agents.analyzer.RedisDataStore') as mock_redis, \
             patch('src.agents.analyzer.MarketAnalyzer') as mock_analyzer, \
             patch('src.agents.analyzer.OrderBookVisualizer') as mock_visualizer_class:

            from src.agents.analyzer import AnalyzerAgent
            agent = AnalyzerAgent(self.settings)

            # Verify visualizer was NOT initialized
            mock_visualizer_class.assert_not_called()
            assert agent.order_book_visualizer is None

    @pytest.mark.asyncio
    async def test_analyzer_visualizer_initialization_error(self):
        """Test analyzer handles visualizer initialization errors gracefully."""
        with patch('src.agents.analyzer.OrderBookVisualizer') as mock_visualizer_class, \
             patch('src.agents.analyzer.RedisDataStore') as mock_redis, \
             patch('src.agents.analyzer.MarketAnalyzer') as mock_analyzer:

            # Mock visualizer initialization to raise exception
            mock_visualizer_class.side_effect = Exception("Visualization initialization failed")

            from src.agents.analyzer import AnalyzerAgent
            agent = AnalyzerAgent(self.settings)

            # Visualizer should be None but agent should still be functional
            assert agent.order_book_visualizer is None

    @pytest.mark.asyncio
    async def test_create_order_book_visualization_success(self):
        """Test successful visualization creation during analysis cycle."""
        with patch('src.agents.analyzer.OrderBookVisualizer') as mock_visualizer_class, \
             patch('src.agents.analyzer.RedisDataStore') as mock_redis, \
             patch('src.agents.analyzer.MarketAnalyzer') as mock_analyzer:

            # Setup mocks
            mock_visualizer = MagicMock()
            mock_visualizer.create_order_book_distribution_chart.return_value = "/tmp/test_chart.png"
            mock_visualizer_class.return_value = mock_visualizer

            mock_redis_store = AsyncMock()
            mock_redis_store.get_latest_depth_snapshot.return_value = DepthSnapshot(
                symbol="BTCFDUSD",
                timestamp=datetime.now(),
                bids=[DepthLevel(price=Decimal("50000"), quantity=Decimal("1.0"))],
                asks=[DepthLevel(price=Decimal("50001"), quantity=Decimal("1.0"))]
            )
            mock_redis.return_value = mock_redis_store

            mock_market_analyzer = MagicMock()
            mock_market_analyzer.analyze_market.return_value = MagicMock()
            mock_analyzer.return_value = mock_market_analyzer

            from src.agents.analyzer import AnalyzerAgent
            agent = AnalyzerAgent(self.settings)

            # Mock the async event loop run_in_executor
            with patch('asyncio.get_running_loop') as mock_loop:
                mock_loop_instance = AsyncMock()
                mock_loop_instance.run_in_executor.return_value = "/tmp/test_chart.png"
                mock_loop.return_value = mock_loop_instance

                # Create test snapshot
                snapshot = DepthSnapshot(
                    symbol="BTCFDUSD",
                    timestamp=datetime.now(),
                    bids=[DepthLevel(price=Decimal("50000"), quantity=Decimal("1.0"))],
                    asks=[DepthLevel(price=Decimal("50001"), quantity=Decimal("1.0"))]
                )

                # Test visualization creation
                await agent._create_order_book_visualization(snapshot)

                # Verify visualization was created
                mock_loop_instance.run_in_executor.assert_called_once()
                call_args = mock_loop_instance.run_in_executor.call_args
                assert call_args[0][0] is None  # Executor
                assert call_args[0][1] == mock_visualizer.create_order_book_distribution_chart
                assert call_args[0][2] == snapshot

    @pytest.mark.asyncio
    async def test_create_order_book_visualization_disabled(self):
        """Test visualization creation is skipped when disabled."""
        # Disable visualization
        self.settings.analyzer.visualization.enabled = False

        with patch('src.agents.analyzer.RedisDataStore') as mock_redis, \
             patch('src.agents.analyzer.MarketAnalyzer') as mock_analyzer:

            from src.agents.analyzer import AnalyzerAgent
            agent = AnalyzerAgent(self.settings)

            # Create test snapshot
            snapshot = DepthSnapshot(
                symbol="BTCFDUSD",
                timestamp=datetime.now(),
                bids=[DepthLevel(price=Decimal("50000"), quantity=Decimal("1.0"))],
                asks=[DepthLevel(price=Decimal("50001"), quantity=Decimal("1.0"))]
            )

            # Should not raise exception, should simply return
            await agent._create_order_book_visualization(snapshot)

    @pytest.mark.asyncio
    async def test_create_order_book_visualization_error_handling(self):
        """Test error handling in visualization creation."""
        with patch('src.agents.analyzer.OrderBookVisualizer') as mock_visualizer_class, \
             patch('src.agents.analyzer.RedisDataStore') as mock_redis, \
             patch('src.agents.analyzer.MarketAnalyzer') as mock_analyzer:

            # Setup visualizer to raise exception
            mock_visualizer = MagicMock()
            mock_visualizer.create_order_book_distribution_chart.side_effect = Exception("Chart creation failed")
            mock_visualizer_class.return_value = mock_visualizer

            from src.agents.analyzer import AnalyzerAgent
            agent = AnalyzerAgent(self.settings)

            # Mock the async event loop run_in_executor
            with patch('asyncio.get_running_loop') as mock_loop:
                mock_loop_instance = AsyncMock()
                mock_loop_instance.run_in_executor.side_effect = Exception("Chart creation failed")
                mock_loop.return_value = mock_loop_instance

                # Create test snapshot
                snapshot = DepthSnapshot(
                    symbol="BTCFDUSD",
                    timestamp=datetime.now(),
                    bids=[DepthLevel(price=Decimal("50000"), quantity=Decimal("1.0"))],
                    asks=[DepthLevel(price=Decimal("50001"), quantity=Decimal("1.0"))]
                )

                # Should not raise exception, should handle error gracefully
                await agent._create_order_book_visualization(snapshot)

    @pytest.mark.asyncio
    async def test_visualization_cleanup_counter(self):
        """Test periodic cleanup functionality."""
        with patch('src.agents.analyzer.OrderBookVisualizer') as mock_visualizer_class, \
             patch('src.agents.analyzer.RedisDataStore') as mock_redis, \
             patch('src.agents.analyzer.MarketAnalyzer') as mock_analyzer:

            # Setup mocks
            mock_visualizer = MagicMock()
            mock_visualizer.cleanup_old_files.return_value = 5  # 5 files removed
            mock_visualizer_class.return_value = mock_visualizer

            from src.agents.analyzer import AnalyzerAgent
            agent = AnalyzerAgent(self.settings)

            # Mock the async event loop run_in_executor
            with patch('asyncio.get_running_loop') as mock_loop:
                mock_loop_instance = AsyncMock()
                mock_loop_instance.run_in_executor.return_value = 5
                mock_loop.return_value = mock_loop_instance

                # Create test snapshot
                snapshot = DepthSnapshot(
                    symbol="BTCFDUSD",
                    timestamp=datetime.now(),
                    bids=[DepthLevel(price=Decimal("50000"), quantity=Decimal("1.0"))],
                    asks=[DepthLevel(price=Decimal("50001"), quantity=Decimal("1.0"))]
                )

                # Simulate 10 visualization creations to trigger cleanup
                for i in range(10):
                    await agent._create_order_book_visualization(snapshot)

                # Cleanup should have been called once (on 10th call)
                cleanup_calls = [
                    call for call in mock_loop_instance.run_in_executor.call_args_list
                    if call[0][1] == mock_visualizer.cleanup_old_files
                ]
                assert len(cleanup_calls) == 1

    def test_get_status_with_visualization_enabled(self):
        """Test status reporting includes visualization information when enabled."""
        with patch('src.agents.analyzer.OrderBookVisualizer') as mock_visualizer_class, \
             patch('src.agents.analyzer.RedisDataStore') as mock_redis, \
             patch('src.agents.analyzer.MarketAnalyzer') as mock_analyzer:

            # Setup mocks
            mock_visualizer = MagicMock()
            mock_visualizer.get_visualization_stats.return_value = {
                "total_files": 10,
                "total_size_mb": 25.5,
                "output_directory": "/tmp/test"
            }
            mock_visualizer_class.return_value = mock_visualizer

            mock_redis_store = MagicMock()
            mock_redis_store.test_connection.return_value = True
            mock_redis_store.depth_snapshot_exists.return_value = True
            mock_redis_store.get_trade_window_count.return_value = 5
            mock_redis.return_value = mock_redis_store

            from src.agents.analyzer import AnalyzerAgent
            agent = AnalyzerAgent(self.settings)

            status = agent.get_status()

            # Verify visualization status is included
            assert "visualization" in status
            assert status["visualization"]["enabled"] is True
            assert "stats" in status["visualization"]
            assert status["visualization"]["stats"]["total_files"] == 10
            assert status["visualization"]["stats"]["total_size_mb"] == 25.5

    def test_get_status_with_visualization_disabled(self):
        """Test status reporting when visualization is disabled."""
        self.settings.analyzer.visualization.enabled = False

        with patch('src.agents.analyzer.RedisDataStore') as mock_redis, \
             patch('src.agents.analyzer.MarketAnalyzer') as mock_analyzer:

            mock_redis_store = MagicMock()
            mock_redis_store.test_connection.return_value = True
            mock_redis_store.depth_snapshot_exists.return_value = True
            mock_redis_store.get_trade_window_count.return_value = 5
            mock_redis.return_value = mock_redis_store

            from src.agents.analyzer import AnalyzerAgent
            agent = AnalyzerAgent(self.settings)

            status = agent.get_status()

            # Verify visualization status is disabled
            assert "visualization" in status
            assert status["visualization"]["enabled"] is False

    def test_get_status_with_visualization_error(self):
        """Test status reporting when visualization has errors."""
        with patch('src.agents.analyzer.OrderBookVisualizer') as mock_visualizer_class, \
             patch('src.agents.analyzer.RedisDataStore') as mock_redis, \
             patch('src.agents.analyzer.MarketAnalyzer') as mock_analyzer:

            # Setup visualizer to raise exception when getting stats
            mock_visualizer = MagicMock()
            mock_visualizer.get_visualization_stats.side_effect = Exception("Stats error")
            mock_visualizer_class.return_value = mock_visualizer

            mock_redis_store = MagicMock()
            mock_redis_store.test_connection.return_value = True
            mock_redis_store.depth_snapshot_exists.return_value = True
            mock_redis_store.get_trade_window_count.return_value = 5
            mock_redis.return_value = mock_redis_store

            from src.agents.analyzer import AnalyzerAgent
            agent = AnalyzerAgent(self.settings)

            status = agent.get_status()

            # Verify error is included in status
            assert "visualization" in status
            assert status["visualization"]["enabled"] is True
            assert "error" in status["visualization"]
            assert "Stats error" in status["visualization"]["error"]

    def test_ai_analysis_status_reporting(self):
        """Test AI analysis status is reported correctly."""
        # Test with AI enabled
        self.settings.analyzer.deepseek.enable = True

        with patch('src.agents.analyzer.DeepSeekClient') as mock_ai_client, \
             patch('src.agents.analyzer.OrderBookVisualizer') as mock_visualizer_class, \
             patch('src.agents.analyzer.RedisDataStore') as mock_redis, \
             patch('src.agents.analyzer.MarketAnalyzer') as mock_analyzer:

            mock_redis_store = MagicMock()
            mock_redis_store.test_connection.return_value = True
            mock_redis_store.depth_snapshot_exists.return_value = True
            mock_redis_store.get_trade_window_count.return_value = 5
            mock_redis.return_value = mock_redis_store

            from src.agents.analyzer import AnalyzerAgent
            agent = AnalyzerAgent(self.settings)

            status = agent.get_status()

            assert "ai_analysis" in status
            assert status["ai_analysis"]["enabled"] is True

        # Test with AI disabled
        self.settings.analyzer.deepseek.enable = False

        with patch('src.agents.analyzer.RedisDataStore') as mock_redis, \
             patch('src.agents.analyzer.MarketAnalyzer') as mock_analyzer:

            mock_redis_store = MagicMock()
            mock_redis_store.test_connection.return_value = True
            mock_redis_store.depth_snapshot_exists.return_value = True
            mock_redis_store.get_trade_window_count.return_value = 5
            mock_redis.return_value = mock_redis_store

            from src.agents.analyzer import AnalyzerAgent
            agent = AnalyzerAgent(self.settings)

            status = agent.get_status()

            assert "ai_analysis" in status
            assert status["ai_analysis"]["enabled"] is False