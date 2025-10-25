"""Unit tests for analyzer agent with DeepSeek enable/disable functionality."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.analyzer import AnalyzerAgent
from src.core.models import MarketAnalysisResult
from src.utils.config import Settings


class TestAnalyzerAgentDeepSeekEnable:
    """Test analyzer agent DeepSeek enable/disable functionality."""

    @pytest.fixture
    def mock_redis_store(self):
        """Mock Redis store."""
        store = AsyncMock()
        store.test_connection.return_value = True

        # Create a proper mock snapshot
        mock_snapshot = MagicMock()
        mock_snapshot.symbol = "BTCFDUSD"
        mock_snapshot.timestamp = "2024-01-01T00:00:00Z"
        store.get_latest_depth_snapshot.return_value = mock_snapshot

        store.depth_snapshot_exists.return_value = True
        store.get_trade_window_count.return_value = 0
        store.store_analysis_result = AsyncMock()
        return store

    @pytest.fixture
    def mock_market_analyzer(self):
        """Mock market analyzer."""
        analyzer = MagicMock()
        analyzer.analyze_market.return_value = MarketAnalysisResult(
            timestamp="2024-01-01T00:00:00Z",
            symbol="BTCFDUSD",
            support_levels=[],
            resistance_levels=[],
            resonance_zones=[],
            poc_levels=[],
            liquidity_vacuum_zones=[],
        )
        return analyzer

    @pytest.fixture
    def config_with_deepseek_enabled(self):
        """Create configuration with DeepSeek enabled."""
        return {
            "analyzer": {
                "deepseek": {
                    "enable": True,
                    "api_key": "test_key",
                    "base_url": "https://api.deepseek.com/v1",
                    "model": "deepseek-chat",
                    "max_tokens": 4000,
                    "temperature": 0.1,
                },
                "analysis": {
                    "interval_seconds": 60,
                    "min_order_volume_threshold": 0.01,
                    "support_resistance_threshold": 0.1,
                },
            },
            "redis": {"host": "localhost", "port": 6379, "db": 0},
            "binance": {"symbol": "BTCFDUSD"},
            "logging": {
                "level": "INFO",
                "file_path": "test.log",
                "max_file_size_mb": 10,
                "backup_count": 3,
            },
        }

    @pytest.fixture
    def config_with_deepseek_disabled(self):
        """Create configuration with DeepSeek disabled."""
        return {
            "analyzer": {
                "deepseek": {
                    "enable": False,
                    "api_key": "",
                    "base_url": "https://api.deepseek.com/v1",
                    "model": "deepseek-chat",
                    "max_tokens": 4000,
                    "temperature": 0.1,
                },
                "analysis": {
                    "interval_seconds": 60,
                    "min_order_volume_threshold": 0.01,
                    "support_resistance_threshold": 0.1,
                },
            },
            "redis": {"host": "localhost", "port": 6379, "db": 0},
            "binance": {"symbol": "BTCFDUSD"},
            "logging": {
                "level": "INFO",
                "file_path": "test.log",
                "max_file_size_mb": 10,
                "backup_count": 3,
            },
        }

    def test_analyzer_initialization_with_deepseek_enabled(
        self, config_with_deepseek_enabled, mock_redis_store, mock_market_analyzer
    ):
        """Test analyzer initialization when DeepSeek is enabled."""
        settings = Settings(**config_with_deepseek_enabled)

        with (
            patch("src.agents.analyzer.RedisDataStore", return_value=mock_redis_store),
            patch(
                "src.agents.analyzer.NormalDistributionMarketAnalyzer",
                return_value=mock_market_analyzer,
            ),
            patch("src.agents.analyzer.DeepSeekClient") as mock_deepseek_client,
        ):
            agent = AnalyzerAgent(settings)

            # Verify AI client is initialized
            assert agent.ai_client is not None
            mock_deepseek_client.assert_called_once()

    def test_analyzer_initialization_with_deepseek_disabled(
        self, config_with_deepseek_disabled, mock_redis_store, mock_market_analyzer
    ):
        """Test analyzer initialization when DeepSeek is disabled."""
        settings = Settings(**config_with_deepseek_disabled)

        with (
            patch("src.agents.analyzer.RedisDataStore", return_value=mock_redis_store),
            patch(
                "src.agents.analyzer.NormalDistributionMarketAnalyzer",
                return_value=mock_market_analyzer,
            ),
            patch("src.agents.analyzer.DeepSeekClient") as mock_deepseek_client,
        ):
            agent = AnalyzerAgent(settings)

            # Verify AI client is not initialized
            assert agent.ai_client is None
            mock_deepseek_client.assert_not_called()

    @pytest.mark.asyncio
    async def test_analysis_cycle_with_deepseek_enabled(
        self, config_with_deepseek_enabled, mock_redis_store, mock_market_analyzer
    ):
        """Test analysis cycle when DeepSeek is enabled."""
        settings = Settings(**config_with_deepseek_enabled)

        # Mock AI client
        mock_ai_client = AsyncMock()
        mock_ai_client.analyze_market_data.return_value = MagicMock()

        with (
            patch("src.agents.analyzer.RedisDataStore", return_value=mock_redis_store),
            patch(
                "src.agents.analyzer.NormalDistributionMarketAnalyzer",
                return_value=mock_market_analyzer,
            ),
            patch("src.agents.analyzer.DeepSeekClient", return_value=mock_ai_client),
        ):
            agent = AnalyzerAgent(settings)

            # Perform analysis cycle
            await agent._perform_analysis_cycle()

            # Verify AI analysis was called
            mock_ai_client.analyze_market_data.assert_called_once()
            mock_redis_store.store_analysis_result.assert_called_once()

    @pytest.mark.asyncio
    async def test_analysis_cycle_with_deepseek_disabled(
        self, config_with_deepseek_disabled, mock_redis_store, mock_market_analyzer
    ):
        """Test analysis cycle when DeepSeek is disabled."""
        settings = Settings(**config_with_deepseek_disabled)

        with (
            patch("src.agents.analyzer.RedisDataStore", return_value=mock_redis_store),
            patch(
                "src.agents.analyzer.NormalDistributionMarketAnalyzer",
                return_value=mock_market_analyzer,
            ),
        ):
            agent = AnalyzerAgent(settings)

            # Perform analysis cycle
            await agent._perform_analysis_cycle()

            # Verify analysis result is stored even without AI
            mock_redis_store.store_analysis_result.assert_called_once()

    @pytest.mark.asyncio
    async def test_shutdown_with_deepseek_enabled(
        self, config_with_deepseek_enabled, mock_redis_store, mock_market_analyzer
    ):
        """Test shutdown process when DeepSeek is enabled."""
        settings = Settings(**config_with_deepseek_enabled)

        mock_ai_client = AsyncMock()

        with (
            patch("src.agents.analyzer.RedisDataStore", return_value=mock_redis_store),
            patch(
                "src.agents.analyzer.NormalDistributionMarketAnalyzer",
                return_value=mock_market_analyzer,
            ),
            patch("src.agents.analyzer.DeepSeekClient", return_value=mock_ai_client),
        ):
            agent = AnalyzerAgent(settings)
            await agent._shutdown()

            # Verify AI client is closed
            mock_ai_client.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_shutdown_with_deepseek_disabled(
        self, config_with_deepseek_disabled, mock_redis_store, mock_market_analyzer
    ):
        """Test shutdown process when DeepSeek is disabled."""
        settings = Settings(**config_with_deepseek_disabled)

        with (
            patch("src.agents.analyzer.RedisDataStore", return_value=mock_redis_store),
            patch(
                "src.agents.analyzer.NormalDistributionMarketAnalyzer",
                return_value=mock_market_analyzer,
            ),
        ):
            agent = AnalyzerAgent(settings)
            await agent._shutdown()

            # Verify no error occurs when AI client is None
            assert agent.ai_client is None

    def test_get_status_with_deepseek_disabled(
        self, config_with_deepseek_disabled, mock_redis_store, mock_market_analyzer
    ):
        """Test get_status method when DeepSeek is disabled."""
        settings = Settings(**config_with_deepseek_disabled)

        with (
            patch("src.agents.analyzer.RedisDataStore", return_value=mock_redis_store),
            patch(
                "src.agents.analyzer.NormalDistributionMarketAnalyzer",
                return_value=mock_market_analyzer,
            ),
        ):
            agent = AnalyzerAgent(settings)
            status = agent.get_status()

            # Verify status includes expected fields
            assert "is_running" in status
            assert "redis_connected" in status
            assert status["is_running"] is False

    def test_backward_compatibility_config_without_enable_field(
        self, mock_redis_store, mock_market_analyzer
    ):
        """Test that configurations without enable field work (backward compatibility)."""
        # Config without enable field (old format)
        config = {
            "analyzer": {
                "deepseek": {
                    "api_key": "test_key",
                    "base_url": "https://api.deepseek.com/v1",
                    "model": "deepseek-chat",
                    "max_tokens": 4000,
                    "temperature": 0.1,
                },
                "analysis": {
                    "interval_seconds": 60,
                    "min_order_volume_threshold": 0.01,
                    "support_resistance_threshold": 0.1,
                },
            },
            "redis": {"host": "localhost", "port": 6379, "db": 0},
            "binance": {"symbol": "BTCFDUSD"},
            "logging": {
                "level": "INFO",
                "file_path": "test.log",
                "max_file_size_mb": 10,
                "backup_count": 3,
            },
        }

        settings = Settings(**config)

        with (
            patch("src.agents.analyzer.RedisDataStore", return_value=mock_redis_store),
            patch(
                "src.agents.analyzer.NormalDistributionMarketAnalyzer",
                return_value=mock_market_analyzer,
            ),
            patch("src.agents.analyzer.DeepSeekClient") as mock_deepseek_client,
        ):
            agent = AnalyzerAgent(settings)

            # Should default to enabled for backward compatibility
            assert agent.ai_client is not None
            assert settings.analyzer.deepseek.enable is True
            mock_deepseek_client.assert_called_once()
