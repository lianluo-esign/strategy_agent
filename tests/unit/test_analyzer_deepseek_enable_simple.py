"""Simplified unit tests for analyzer agent with DeepSeek enable/disable functionality."""

from unittest.mock import AsyncMock, patch

import pytest

from src.agents.analyzer import AnalyzerAgent
from src.utils.config import Settings


class TestAnalyzerDeepSeekEnable:
    """Test analyzer agent DeepSeek enable/disable functionality."""

    def test_analyzer_initialization_with_deepseek_enabled(self):
        """Test analyzer initialization when DeepSeek is enabled."""
        config = {
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

        settings = Settings(**config)

        with (
            patch("src.agents.analyzer.RedisDataStore") as mock_redis_store,
            patch(
                "src.agents.analyzer.NormalDistributionMarketAnalyzer"
            ),
            patch("src.agents.analyzer.DeepSeekClient") as mock_deepseek_client,
        ):
            mock_redis_store.return_value.test_connection.return_value = True
            agent = AnalyzerAgent(settings)

            # Verify AI client is initialized
            assert agent.ai_client is not None
            mock_deepseek_client.assert_called_once()

    def test_analyzer_initialization_with_deepseek_disabled(self):
        """Test analyzer initialization when DeepSeek is disabled."""
        config = {
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

        settings = Settings(**config)

        with (
            patch("src.agents.analyzer.RedisDataStore") as mock_redis_store,
            patch(
                "src.agents.analyzer.NormalDistributionMarketAnalyzer"
            ),
            patch("src.agents.analyzer.DeepSeekClient") as mock_deepseek_client,
        ):
            mock_redis_store.return_value.test_connection.return_value = True
            agent = AnalyzerAgent(settings)

            # Verify AI client is not initialized
            assert agent.ai_client is None
            mock_deepseek_client.assert_not_called()

    def test_backward_compatibility_config_without_enable_field(self):
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
            patch("src.agents.analyzer.RedisDataStore") as mock_redis_store,
            patch(
                "src.agents.analyzer.NormalDistributionMarketAnalyzer"
            ),
            patch("src.agents.analyzer.DeepSeekClient") as mock_deepseek_client,
        ):
            mock_redis_store.return_value.test_connection.return_value = True
            agent = AnalyzerAgent(settings)

            # Should default to enabled for backward compatibility
            assert agent.ai_client is not None
            assert settings.analyzer.deepseek.enable is True
            mock_deepseek_client.assert_called_once()

    @pytest.mark.asyncio
    async def test_shutdown_with_deepseek_enabled(self):
        """Test shutdown process when DeepSeek is enabled."""
        config = {
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

        settings = Settings(**config)

        mock_ai_client = AsyncMock()

        with (
            patch("src.agents.analyzer.RedisDataStore") as mock_redis_store,
            patch(
                "src.agents.analyzer.NormalDistributionMarketAnalyzer"
            ),
            patch("src.agents.analyzer.DeepSeekClient", return_value=mock_ai_client),
        ):
            mock_redis_store.return_value.test_connection.return_value = True
            mock_redis_store.return_value.close = AsyncMock()
            agent = AnalyzerAgent(settings)
            await agent._shutdown()

            # Verify AI client is closed
            mock_ai_client.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_shutdown_with_deepseek_disabled(self):
        """Test shutdown process when DeepSeek is disabled."""
        config = {
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

        settings = Settings(**config)

        with (
            patch("src.agents.analyzer.RedisDataStore") as mock_redis_store,
            patch(
                "src.agents.analyzer.NormalDistributionMarketAnalyzer"
            ),
        ):
            mock_redis_store.return_value.test_connection.return_value = True
            mock_redis_store.return_value.close = AsyncMock()
            agent = AnalyzerAgent(settings)
            await agent._shutdown()

            # Verify no error occurs when AI client is None
            assert agent.ai_client is None

    def test_get_status_method(self):
        """Test get_status method functionality."""
        config = {
            "analyzer": {
                "deepseek": {
                    "enable": False,
                    "api_key": "",
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
            patch("src.agents.analyzer.RedisDataStore") as mock_redis_store,
            patch(
                "src.agents.analyzer.NormalDistributionMarketAnalyzer"
            ),
        ):
            mock_redis_store.return_value.test_connection.return_value = True
            mock_redis_store.return_value.depth_snapshot_exists.return_value = True
            mock_redis_store.return_value.get_trade_window_count.return_value = 0
            agent = AnalyzerAgent(settings)
            status = agent.get_status()

            # Verify status includes expected fields
            assert "is_running" in status
            assert "redis_connected" in status
            assert status["is_running"] is False
            assert status["redis_connected"] is True
