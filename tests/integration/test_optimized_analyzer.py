"""优化版分析器集成测试。"""

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, Mock, patch

from src.core.agent_analyzer_optimized import OptimizedAgentAnalyzer


class TestOptimizedAnalyzerIntegration:
    """优化版分析器集成测试类。"""

    @pytest.fixture
    def mock_redis_store(self):
        """模拟Redis存储。"""
        redis_store = Mock()
        redis_store.test_connection.return_value = True
        redis_store.get_trade_window_count.return_value = 100
        redis_store.get_recent_trade_data.return_value = self._create_mock_trades_data()
        redis_store.close = AsyncMock()
        return redis_store

    @pytest.fixture
    def deepseek_config(self):
        """Deepseek配置。"""
        return {
            "api_key": "test_api_key",
            "base_url": "https://api.deepseek.com/v1",
            "model": "deepseek-chat",
            "max_tokens": 4000,
            "temperature": 0.1,
            "timeout": 90,
            "max_retries": 3
        }

    @pytest.fixture
    def analyzer(self, mock_redis_store, deepseek_config):
        """创建优化版分析器实例。"""
        return OptimizedAgentAnalyzer(
            redis_store=mock_redis_store,
            deepseek_config=deepseek_config,
            discord_webhook_url=None,  # 集成测试中不使用Discord
            aggregation_precision=10.0,
            min_volume_threshold=0.1,
            analysis_window_minutes=1440
        )

    def _create_mock_trades_data(self):
        """创建模拟交易数据。"""
        mock_data = []
        for i in range(10):
            minute_data = Mock()
            minute_data.timestamp = f"2025-01-01T12:{i:02d}:00"
            minute_data.price_levels = {
                str(1000 + i * 10): {"total_volume": 1.0 + i * 0.1},
                str(2000 + i * 10): {"total_volume": 0.5 + i * 0.05},
            }
            mock_data.append(minute_data)
        return mock_data

    @pytest.mark.asyncio
    async def test_analyze_market_success(self, analyzer):
        """测试成功市场分析。"""
        # 模拟Deepseek API响应
        mock_ai_response = {
            "trend": "看涨",
            "strength_levels": {
                "strong_support": 0.8,
                "weak_support": 0.6,
                "strong_resistance": 0.4,
                "weak_resistance": 0.3
            },
            "reason": "基于成交量分析和价格动量识别出的看涨趋势",
            "confidence": 0.85
        }

        with patch.object(analyzer.deepseek_analyzer, 'analyze_trend') as mock_analyze:
            # 创建模拟的趋势结果
            from src.core.agent_analyzer_optimized.deepseek_client import TrendAnalysisResult
            from datetime import datetime

            mock_trend_result = TrendAnalysisResult(
                timestamp=datetime.now(),
                trend=mock_ai_response["trend"],
                strength_levels=mock_ai_response["strength_levels"],
                reason=mock_ai_response["reason"],
                confidence=mock_ai_response["confidence"]
            )
            mock_analyze.return_value = mock_trend_result

            # 执行分析
            result = await analyzer.analyze_market("BTCFDUSD")

            # 验证结果
            assert result["status"] == "success"
            assert result["symbol"] == "BTCFDUSD"
            assert "analysis_result" in result
            assert "aggregated_data" in result
            assert result["processing_time"] > 0
            assert not result["discord_notification_sent"]  # 未启用Discord

            # 验证分析结果
            analysis_result = result["analysis_result"]
            assert analysis_result["trend"] == "看涨"
            assert analysis_result["confidence"] == 0.85

            # 验证聚合数据
            aggregated_data = result["aggregated_data"]
            assert aggregated_data["total_volume"] > 0
            assert aggregated_data["trade_count"] > 0
            assert len(aggregated_data["price_levels"]) > 0

    @pytest.mark.asyncio
    async def test_analyze_market_no_data(self, analyzer):
        """测试无数据情况。"""
        # 模拟Redis返回空数据
        analyzer.redis_store.get_recent_trade_data.return_value = []

        result = await analyzer.analyze_market("BTCFDUSD")

        assert result["status"] == "error"
        assert "没有可用的交易数据" in result["error"]

    @pytest.mark.asyncio
    async def test_analyze_market_deepseek_error(self, analyzer):
        """测试Deepseek API错误。"""
        with patch.object(analyzer.deepseek_analyzer, 'analyze_trend') as mock_analyze:
            mock_analyze.side_effect = Exception("Deepseek API错误")

            result = await analyzer.analyze_market("BTCFDUSD")

            assert result["status"] == "error"
            assert "Deepseek API错误" in result["error"]

    @pytest.mark.asyncio
    async def test_analyze_single_cycle(self, analyzer):
        """测试单次分析周期。"""
        # 模拟Deepseek响应
        mock_ai_response = {
            "trend": "震荡",
            "strength_levels": {
                "strong_support": 0.5,
                "weak_support": 0.5,
                "strong_resistance": 0.5,
                "weak_resistance": 0.5
            },
            "reason": "市场处于震荡状态",
            "confidence": 0.6
        }

        with patch.object(analyzer.deepseek_analyzer, 'analyze_trend') as mock_analyze:
            from src.core.agent_analyzer_optimized.deepseek_client import TrendAnalysisResult
            from datetime import datetime

            mock_trend_result = TrendAnalysisResult(
                timestamp=datetime.now(),
                trend=mock_ai_response["trend"],
                strength_levels=mock_ai_response["strength_levels"],
                reason=mock_ai_response["reason"],
                confidence=mock_ai_response["confidence"]
            )
            mock_analyze.return_value = mock_trend_result

            # 执行单次分析
            json_result = await analyzer.analyze_single_cycle("BTCFDUSD")

            # 验证返回的是有效JSON字符串
            parsed_result = json.loads(json_result)
            assert parsed_result["trend"] == "震荡"
            assert parsed_result["confidence"] == 0.6
            assert "timestamp" in parsed_result

    def test_get_status(self, analyzer):
        """测试状态获取。"""
        status = analyzer.get_status()

        assert status["analyzer_type"] == "optimized_agent_analyzer"
        assert status["symbol"] == "BTCFDUSD"
        assert status["redis_connected"] is True
        assert status["trades_window_available"] is True
        assert "statistics" in status
        assert "components" in status

        # 验证组件状态
        components = status["components"]
        assert "trades_aggregator" in components
        assert "deepseek_analyzer" in components
        assert "response_formatter" in components

    @pytest.mark.asyncio
    async def test_health_check(self, analyzer):
        """测试健康检查。"""
        with patch.object(analyzer, 'test_discord_connection', return_value=True):
            health = await analyzer.health_check()

            assert health["overall_status"] == "healthy"
            assert "timestamp" in health
            assert "checks" in health

            # 验证各项检查
            checks = health["checks"]
            assert "redis_connection" in checks
            assert "data_availability" in checks
            assert checks["redis_connection"]["status"] == "pass"
            assert checks["data_availability"]["status"] == "pass"

    @pytest.mark.asyncio
    async def test_health_check_unhealthy(self, analyzer):
        """测试健康检查失败情况。"""
        # 模拟Redis连接失败
        analyzer.redis_store.test_connection.return_value = False

        health = await analyzer.health_check()

        assert health["overall_status"] == "unhealthy"
        assert "failed_checks" in health

    @pytest.mark.asyncio
    async def test_close(self, analyzer):
        """测试资源关闭。"""
        with patch.object(analyzer.deepseek_analyzer, 'close') as mock_close_deepseek:
            await analyzer.close()

            # 验证Deepseek分析器被关闭
            mock_close_deepseek.assert_called_once()

            # 验证Redis连接被关闭
            analyzer.redis_store.close.assert_called_once()

    def test_statistics_update(self, analyzer):
        """测试统计信息更新。"""
        initial_stats = analyzer.stats.copy()

        # 模拟成功的分析
        analyzer._update_stats(1.5, success=True)
        assert analyzer.stats["total_analyses"] == initial_stats["total_analyses"] + 1
        assert analyzer.stats["successful_analyses"] == initial_stats["successful_analyses"] + 1

        # 模拟失败的分析
        analyzer._update_stats(0.8, success=False)
        assert analyzer.stats["total_analyses"] == initial_stats["total_analyses"] + 2
        assert analyzer.stats["failed_analyses"] == initial_stats["failed_analyses"] + 1

    @pytest.mark.asyncio
    async def test_discord_integration(self, mock_redis_store, deepseek_config):
        """测试Discord集成功能。"""
        # 使用Discord webhook URL创建分析器
        discord_url = "https://discord.com/api/webhooks/test/webhook"
        analyzer_with_discord = OptimizedAgentAnalyzer(
            redis_store=mock_redis_store,
            deepseek_config=deepseek_config,
            discord_webhook_url=discord_url
        )

        assert analyzer_with_discord.discord_manager is not None

        # 测试Discord连接测试
        with patch.object(analyzer_with_discord.discord_manager, 'get_notifier') as mock_get_notifier:
            mock_notifier = Mock()
            mock_notifier.test_connection = AsyncMock(return_value=True)
            mock_get_notifier.return_value = mock_notifier

            result = await analyzer_with_discord.test_discord_connection()
            assert result is True

        await analyzer_with_discord.close()

    @pytest.mark.asyncio
    async def test_quality_validation(self, analyzer):
        """测试分析质量验证。"""
        # 创建质量较差的模拟结果
        mock_ai_response = {
            "trend": "看涨",
            "strength_levels": {
                "strong_support": 0.9,  # 与趋势不一致的高支撑
                "weak_support": 0.7,
                "strong_resistance": 0.1,
                "weak_resistance": 0.05
            },
            "reason": "短",  # 过短的原因
            "confidence": 0.2  # 低置信度
        }

        with patch.object(analyzer.deepseek_analyzer, 'analyze_trend') as mock_analyze:
            from src.core.agent_analyzer_optimized.deepseek_client import TrendAnalysisResult
            from datetime import datetime

            mock_trend_result = TrendAnalysisResult(
                timestamp=datetime.now(),
                trend=mock_ai_response["trend"],
                strength_levels=mock_ai_response["strength_levels"],
                reason=mock_ai_response["reason"],
                confidence=mock_ai_response["confidence"]
            )
            mock_analyze.return_value = mock_trend_result

            result = await analyzer.analyze_market("BTCFDUSD")

            # 验证质量检查结果
            quality_check = result["quality_check"]
            assert quality_check["is_valid"] is True  # 非严格模式下仍然有效
            assert len(quality_check["warnings"]) > 0

            # 应该有置信度警告
            confidence_warnings = [w for w in quality_check["warnings"] if "置信度" in w]
            assert len(confidence_warnings) > 0

            # 应该有原因长度警告
            reason_warnings = [w for w in quality_check["warnings"] if "原因" in w]
            assert len(reason_warnings) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])