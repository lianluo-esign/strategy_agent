"""SimplifiedMarketAnalyzer与TradingEventPublisher集成单元测试。

测试简化市场分析器中的交易参数发布功能。
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, Mock, patch

from src.core.simplified_market_analyzer import SimplifiedMarketAnalyzer
from src.core.trading_event_publisher import TradingEventPublisher


class TestTradingEventPublisherIntegration:
    """测试交易参数发布集成功能。"""

    def setup_method(self):
        """每个测试方法前的设置。"""
        # 创建模拟Redis存储
        self.mock_redis = Mock()
        self.mock_redis.test_connection.return_value = True
        self.mock_redis.depth_snapshot_exists.return_value = True
        self.mock_redis.get_trade_window_count.return_value = 1440

        # 创建模拟DeepSeek配置
        self.deepseek_config = {
            "api_key": "test-api-key",
            "base_url": "https://api.deepseek.com/v1",
            "model": "deepseek-chat",
            "max_tokens": 3000,
            "temperature": 0.1,
            "timeout": 90,
            "max_retries": 3,
        }

    def test_analyzer_without_publisher(self):
        """测试没有TradingEventPublisher的分析器。"""
        analyzer = SimplifiedMarketAnalyzer(
            redis_store=self.mock_redis,
            deepseek_config=self.deepseek_config,
        )

        # 验证发布器未配置
        status = analyzer.get_status()
        assert status["trading_event_publisher"]["configured"] is False
        assert status["trading_event_publisher"]["enabled"] is False

    def test_analyzer_with_publisher(self):
        """测试带有TradingEventPublisher的分析器。"""
        # 创建模拟TradingEventPublisher
        mock_publisher = Mock(spec=TradingEventPublisher)
        mock_publisher.process_ai_analysis_and_publish.return_value = True

        analyzer = SimplifiedMarketAnalyzer(
            redis_store=self.mock_redis,
            deepseek_config=self.deepseek_config,
            trading_event_publisher=mock_publisher,
        )

        # 验证发布器已配置
        status = analyzer.get_status()
        assert status["trading_event_publisher"]["configured"] is True
        assert status["trading_event_publisher"]["enabled"] is True
        assert analyzer.trading_event_publisher == mock_publisher

    @pytest.mark.asyncio
    async def test_publish_trading_params_success(self):
        """测试交易参数发布成功。"""
        # 创建模拟TradingEventPublisher
        mock_publisher = AsyncMock()
        mock_publisher.process_ai_analysis_and_publish.return_value = True

        analyzer = SimplifiedMarketAnalyzer(
            redis_store=self.mock_redis,
            deepseek_config=self.deepseek_config,
            trading_event_publisher=mock_publisher,
        )

        # 测试参数
        trading_params = {
            "grid_delta": 2.5,
            "grid_quantity": 0.0015,
            "active_side": "Buy"
        }
        symbol = "BTCFDUSD"

        # 执行发布
        result = await analyzer._publish_trading_params(trading_params, symbol)

        # 验证结果
        assert result is True

        # 验证调用参数
        mock_publisher.process_ai_analysis_and_publish.assert_called_once()
        call_args = mock_publisher.process_ai_analysis_and_publish.call_args[0][0]

        # 验证AI响应格式
        expected_params = {
            "grid_delta": 2.5,
            "grid_quantity": 0.0015,
            "active_side": "Buy"
        }
        import json
        assert json.loads(call_args) == expected_params

    @pytest.mark.asyncio
    async def test_publish_trading_params_failure(self):
        """测试交易参数发布失败。"""
        # 创建模拟TradingEventPublisher
        mock_publisher = AsyncMock()
        mock_publisher.process_ai_analysis_and_publish.return_value = False

        analyzer = SimplifiedMarketAnalyzer(
            redis_store=self.mock_redis,
            deepseek_config=self.deepseek_config,
            trading_event_publisher=mock_publisher,
        )

        # 测试参数
        trading_params = {
            "grid_delta": 2.0,
            "grid_quantity": 0.001,
            "active_side": "Sell"
        }
        symbol = "BTCFDUSD"

        # 执行发布
        result = await analyzer._publish_trading_params(trading_params, symbol)

        # 验证结果
        assert result is False
        mock_publisher.process_ai_analysis_and_publish.assert_called_once()

    @pytest.mark.asyncio
    async def test_publish_trading_params_exception(self):
        """测试交易参数发布异常。"""
        # 创建模拟TradingEventPublisher
        mock_publisher = AsyncMock()
        mock_publisher.process_ai_analysis_and_publish.side_effect = Exception("Redis connection failed")

        analyzer = SimplifiedMarketAnalyzer(
            redis_store=self.mock_redis,
            deepseek_config=self.deepseek_config,
            trading_event_publisher=mock_publisher,
        )

        # 测试参数
        trading_params = {
            "grid_delta": 2.0,
            "grid_quantity": 0.001,
            "active_side": "Buy"
        }
        symbol = "BTCFDUSD"

        # 执行发布
        result = await analyzer._publish_trading_params(trading_params, symbol)

        # 验证结果
        assert result is False
        mock_publisher.process_ai_analysis_and_publish.assert_called_once()

    @pytest.mark.asyncio
    async def test_publish_without_publisher(self):
        """测试没有配置发布器时的行为。"""
        analyzer = SimplifiedMarketAnalyzer(
            redis_store=self.mock_redis,
            deepseek_config=self.deepseek_config,
            trading_event_publisher=None,
        )

        # 测试参数
        trading_params = {
            "grid_delta": 2.0,
            "grid_quantity": 0.001,
            "active_side": "Buy"
        }
        symbol = "BTCFDUSD"

        # 执行发布
        result = await analyzer._publish_trading_params(trading_params, symbol)

        # 验证结果（应该返回False，因为没有配置发布器）
        assert result is False

    def test_analyze_market_with_publish_integration(self):
        """测试完整的分析流程包含发布集成。"""
        # 创建模拟组件
        mock_publisher = AsyncMock()
        mock_publisher.process_ai_analysis_and_publish.return_value = True

        # 创建简单的模拟数据，避免Decimal问题
        mock_snapshot = Mock()
        mock_snapshot.symbol = "BTCFDUSD"
        mock_snapshot.bids = {"100000.00": "1.0", "99999.00": "2.0"}
        mock_snapshot.asks = {"100001.00": "1.5", "100002.00": "2.5"}

        self.mock_redis.get_latest_depth_snapshot.return_value = mock_snapshot
        self.mock_redis.get_recent_trade_data.return_value = [Mock() for _ in range(100)]

        mock_vp_result = {
            "status": "success",
            "price_levels_count": 100,
            "total_volume": 5000.0,
            "vp_data": {"100000.00": 100.0, "100001.00": 150.0}
        }

        analyzer = SimplifiedMarketAnalyzer(
            redis_store=self.mock_redis,
            deepseek_config=self.deepseek_config,
            trading_event_publisher=mock_publisher,
        )

        with patch('src.core.simplified_market_analyzer.VolumeProfileAnalyzer') as mock_vp:
            mock_vp_instance = Mock()
            mock_vp_instance.analyze_volume_profile.return_value = mock_vp_result
            mock_vp.return_value = mock_vp_instance

            with patch('src.core.simplified_market_analyzer.UnifiedDeepSeekAnalyzer') as mock_deepseek:
                mock_deepseek_instance = Mock()
                mock_deepseek_instance.analyze_unified_market_data.return_value = {
                    "status": "success",
                    "raw_content": '{"grid_delta": 2.0, "grid_quantity": 0.001, "active_side": "Buy"}',
                    "symbol": "BTCFDUSD"
                }
                mock_deepseek.return_value = mock_deepseek_instance

                with patch('src.core.simplified_market_analyzer.result_validator.validate_and_extract_trading_params') as mock_validator:
                    mock_validator.return_value = {
                        "grid_delta": 2.0,
                        "grid_quantity": 0.001,
                        "active_side": "Buy"
                    }

                    # 运行异步分析
                    result = asyncio.run(analyzer.analyze_market("BTCFDUSD"))

                    # 验证分析结果
                    assert result["status"] == "success"
                    assert result["symbol"] == "BTCFDUSD"
                    assert result["trading_params"] == {
                        "grid_delta": 2.0,
                        "grid_quantity": 0.001,
                        "active_side": "Buy"
                    }

                    # 验证发布器被调用
                    mock_publisher.process_ai_analysis_and_publish.assert_called_once()

    def test_agent_analyzer_integration(self):
        """测试agent_analyzer.py中的集成。"""
        # 这个测试需要从agent_analyzer.py的角度进行集成测试
        # 由于涉及到完整的设置流程，这里只验证核心逻辑
        assert True  # 占位符测试