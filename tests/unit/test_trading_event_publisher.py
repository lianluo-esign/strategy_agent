"""单元测试 - TradingEventPublisher交易事件发布器。

测试交易事件发布器的核心功能：
- 事件提取和验证
- Redis连接和发布
- 错误处理和重试机制
- 配置验证
"""

import asyncio
import json
import pytest
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

from src.core.trading_event_publisher import (
    TradingEventPublisher,
    TradingEvent,
    EventValidationError,
    RedisPublishError,
    RedisConnectionError,
)
from src.utils.config import TradingEventPublisherConfig


class TestTradingEvent:
    """测试TradingEvent数据模型。"""

    def test_valid_trading_event_creation(self):
        """测试有效交易事件创建。"""
        event = TradingEvent(
            grid_delta=2.0,
            grid_quantity=0.001,
            active_side="Buy"
        )

        assert event.grid_delta == 2.0
        assert event.grid_quantity == 0.001
        assert event.active_side == "Buy"

    def test_invalid_grid_delta(self):
        """测试无效grid_delta验证。"""
        with pytest.raises(ValueError, match="grid_delta must be positive"):
            TradingEvent(
                grid_delta=-1.0,
                grid_quantity=0.001,
                active_side="Buy"
            )

    def test_invalid_grid_quantity(self):
        """测试无效grid_quantity验证。"""
        with pytest.raises(ValueError, match="grid_quantity must be positive"):
            TradingEvent(
                grid_delta=2.0,
                grid_quantity=0.0,
                active_side="Sell"
            )

    def test_invalid_active_side(self):
        """测试无效active_side验证。"""
        with pytest.raises(ValueError, match="active_side must be 'Buy' or 'Sell'"):
            TradingEvent(
                grid_delta=2.0,
                grid_quantity=0.001,
                active_side="Invalid"
            )

    def test_edge_case_valid_values(self):
        """测试边界情况的有效值。"""
        event = TradingEvent(
            grid_delta=0.1,  # 最小值
            grid_quantity=0.0001,  # 最小值
            active_side="Sell"
        )
        assert event.grid_delta == 0.1
        assert event.grid_quantity == 0.0001
        assert event.active_side == "Sell"


class TestTradingEventPublisher:
    """测试TradingEventPublisher交易事件发布器。"""

    @pytest.fixture
    def valid_config(self):
        """创建有效的交易事件发布器配置。"""
        from src.utils.config import (
            TradingEventRedisConfig,
            TradingEventValidationConfig,
            TradingEventPublisherConfig
        )

        redis_config = TradingEventRedisConfig(
            host="localhost",
            port=6379,
            db=0,
            channel="test_trading_events",
            timeout=5,
            max_retries=3
        )

        validation_config = TradingEventValidationConfig(
            min_grid_delta=0.1,
            max_grid_delta=100.0,
            min_grid_quantity=0.0001,
            max_grid_quantity=10.0
        )

        return TradingEventPublisherConfig(
            enable=True,
            redis=redis_config,
            validation=validation_config
        )

    @pytest.fixture
    def disabled_config(self, valid_config):
        """创建禁用的交易事件发布器配置。"""
        config = TradingEventPublisherConfig(
            enable=False,
            redis=valid_config.redis,
            validation=valid_config.validation
        )
        return config

    @pytest.fixture
    def publisher(self, valid_config):
        """创建交易事件发布器实例。"""
        return TradingEventPublisher(valid_config)

    @pytest.fixture
    def disabled_publisher(self, disabled_config):
        """创建禁用的交易事件发布器实例。"""
        return TradingEventPublisher(disabled_config)

    def test_publisher_initialization(self, publisher, valid_config):
        """测试发布器初始化。"""
        assert publisher.config == valid_config
        assert publisher.redis_client is None
        assert publisher._is_connected is False

    def test_disabled_publisher_initialization(self, disabled_publisher):
        """测试禁用发布器初始化。"""
        assert disabled_publisher.config.enable is False
        assert disabled_publisher.redis_client is None

    def test_validate_event_values_valid(self, publisher):
        """测试有效事件值验证。"""
        event_data = {
            "grid_delta": 2.0,
            "grid_quantity": 0.001,
            "active_side": "Buy"
        }

        # 应该不抛出异常
        publisher._validate_event_values(event_data)

    def test_validate_event_values_missing_fields(self, publisher):
        """测试缺失字段的验证。"""
        event_data = {
            "grid_delta": 2.0,
            # 缺失 grid_quantity 和 active_side
        }

        with pytest.raises(EventValidationError, match="grid_quantity is required"):
            publisher._validate_event_values(event_data)

    def test_validate_event_values_invalid_types(self, publisher):
        """测试无效类型的验证。"""
        event_data = {
            "grid_delta": "invalid",  # 应该是数字
            "grid_quantity": 0.001,
            "active_side": "Buy"
        }

        with pytest.raises(EventValidationError, match="grid_delta must be a number"):
            publisher._validate_event_values(event_data)

    def test_validate_event_values_out_of_range(self, publisher):
        """测试超出范围的值验证。"""
        # 测试grid_delta过小
        event_data = {
            "grid_delta": 0.05,  # 小于最小值0.1
            "grid_quantity": 0.001,
            "active_side": "Buy"
        }

        with pytest.raises(EventValidationError, match="grid_delta 0.05 is below minimum 0.1"):
            publisher._validate_event_values(event_data)

        # 测试grid_quantity过大
        event_data = {
            "grid_delta": 2.0,
            "grid_quantity": 20.0,  # 大于最大值10.0
            "active_side": "Buy"
        }

        with pytest.raises(EventValidationError, match="grid_quantity 20.0 is above maximum 10.0"):
            publisher._validate_event_values(event_data)

    def test_extract_trading_event_from_json_block(self, publisher):
        """测试从JSON代码块中提取交易事件。"""
        ai_response = '''
        基于市场分析，建议以下交易策略：

        ```json
        {
          "grid_delta": 2.0,
          "grid_quantity": 0.001,
          "active_side": "Buy"
        }
        ```

        这是一个很好的买入机会。
        '''

        event_data = publisher._extract_trading_event_from_ai_response(ai_response)

        assert event_data is not None
        assert event_data["grid_delta"] == 2.0
        assert event_data["grid_quantity"] == 0.001
        assert event_data["active_side"] == "Buy"

    def test_extract_trading_event_from_direct_json(self, publisher):
        """测试从直接JSON中提取交易事件。"""
        ai_response = '''
        基于分析结果，推荐交易参数：
        {"grid_delta": 1.5, "grid_quantity": 0.002, "active_side": "Sell"}
        这是一个卖出时机。
        '''

        event_data = publisher._extract_trading_event_from_ai_response(ai_response)

        assert event_data is not None
        assert event_data["grid_delta"] == 1.5
        assert event_data["grid_quantity"] == 0.002
        assert event_data["active_side"] == "Sell"

    def test_extract_trading_event_no_json_found(self, publisher):
        """测试没有找到JSON的情况。"""
        ai_response = '''
        基于市场分析，目前不建议交易。
        市场波动较大，建议等待更好的时机。
        '''

        event_data = publisher._extract_trading_event_from_ai_response(ai_response)

        assert event_data is None

    def test_extract_trading_event_invalid_json(self, publisher):
        """测试无效JSON的处理。"""
        ai_response = '''
        推荐交易参数：
        ```json
        {
          "grid_delta": 2.0,
          "grid_quantity": 0.001,
          "active_side": "Buy",
        }
        ```
        注意：这个JSON有语法错误（尾随逗号）
        '''

        with pytest.raises(EventValidationError, match="Failed to parse trading event JSON"):
            publisher._extract_trading_event_from_ai_response(ai_response)

    def test_extract_trading_event_missing_required_fields(self, publisher):
        """测试JSON中缺失必需字段的处理。"""
        ai_response = '''
        ```json
        {
          "grid_delta": 2.0,
          "active_side": "Buy"
        }
        ```
        '''

        with pytest.raises(EventValidationError, match="Missing required fields"):
            publisher._extract_trading_event_from_ai_response(ai_response)

    @pytest.mark.asyncio
    async def test_create_redis_connection_success(self, publisher):
        """测试Redis连接创建成功。"""
        mock_redis = AsyncMock()
        mock_redis.ping.return_value = True

        with patch('redis.asyncio.Redis', return_value=mock_redis):
            client = await publisher._create_redis_connection()

            assert client == mock_redis
            assert publisher._is_connected is True
            mock_redis.ping.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_redis_connection_failure(self, publisher):
        """测试Redis连接创建失败。"""
        with patch('redis.asyncio.Redis') as mock_redis_class:
            mock_redis = AsyncMock()
            mock_redis.ping.side_effect = Exception("Connection failed")
            mock_redis_class.return_value = mock_redis

            with pytest.raises(RedisConnectionError):
                await publisher._create_redis_connection()

            assert publisher._is_connected is False

    @pytest.mark.asyncio
    async def test_publish_event_success(self, publisher):
        """测试事件发布成功。"""
        publisher.redis_client = AsyncMock()
        publisher._is_connected = True
        publisher.redis_client.publish.return_value = 1  # 1个订阅者

        event_data = {
            "grid_delta": 2.0,
            "grid_quantity": 0.001,
            "active_side": "Buy"
        }

        await publisher._publish_event(event_data)

        # 验证发布调用
        publisher.redis_client.publish.assert_called_once()
        call_args = publisher.redis_client.publish.call_args
        assert call_args[0][0] == publisher.config.redis.channel  # channel

        # 验证消息格式
        message_json = call_args[0][1]
        message = json.loads(message_json)
        assert message["event_type"] == "trading_guidance"
        assert message["data"] == event_data
        assert "timestamp" in message
        assert message["source"] == "strategy_agent_unified_analysis"

    @pytest.mark.asyncio
    async def test_publish_event_no_subscribers(self, publisher):
        """测试发布事件但没有订阅者。"""
        publisher.redis_client = AsyncMock()
        publisher._is_connected = True
        publisher.redis_client.publish.return_value = 0  # 没有订阅者

        event_data = {
            "grid_delta": 2.0,
            "grid_quantity": 0.001,
            "active_side": "Buy"
        }

        await publisher._publish_event(event_data)

        publisher.redis_client.publish.assert_called_once()

    @pytest.mark.asyncio
    async def test_publish_event_not_connected(self, publisher):
        """测试未连接时自动重连并发布。"""
        mock_redis = AsyncMock()
        mock_redis.ping.return_value = True
        mock_redis.publish.return_value = 1

        with patch('redis.asyncio.Redis', return_value=mock_redis):
            event_data = {
                "grid_delta": 2.0,
                "grid_quantity": 0.001,
                "active_side": "Buy"
            }

            await publisher._publish_event(event_data)

            assert publisher._is_connected is True
            mock_redis.publish.assert_called_once()

    @pytest.mark.asyncio
    async def test_publish_with_retry_success(self, publisher):
        """测试重试机制成功。"""
        publisher.redis_client = AsyncMock()
        publisher._is_connected = True

        # 第一次失败，第二次成功
        publisher.redis_client.publish.side_effect = [
            Exception("Network error"),
            1  # 成功
        ]

        event_data = {
            "grid_delta": 2.0,
            "grid_quantity": 0.001,
            "active_side": "Buy"
        }

        result = await publisher._publish_with_retry(event_data)

        assert result is True
        assert publisher.redis_client.publish.call_count == 2

    @pytest.mark.asyncio
    async def test_publish_with_retry_exhausted(self, publisher):
        """测试重试次数用尽。"""
        publisher.redis_client = AsyncMock()
        publisher._is_connected = True
        publisher.redis_client.publish.side_effect = Exception("Persistent error")

        event_data = {
            "grid_delta": 2.0,
            "grid_quantity": 0.001,
            "active_side": "Buy"
        }

        result = await publisher._publish_with_retry(event_data)

        assert result is False
        # 应该尝试了初始调用 + 3次重试 = 4次
        assert publisher.redis_client.publish.call_count == 4

    @pytest.mark.asyncio
    async def test_process_ai_analysis_and_publish_success(self, publisher):
        """测试处理AI分析并发布交易事件成功。"""
        ai_response = '''
        基于深度分析，推荐以下交易策略：

        ```json
        {
          "grid_delta": 2.5,
          "grid_quantity": 0.002,
          "active_side": "Sell"
        }
        ```

        市场显示卖出信号。
        '''

        # Mock Redis发布
        publisher.redis_client = AsyncMock()
        publisher._is_connected = True
        publisher.redis_client.publish.return_value = 1

        result = await publisher.process_ai_analysis_and_publish(ai_response)

        assert result is True
        publisher.redis_client.publish.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_ai_analysis_and_publish_no_event(self, publisher):
        """测试AI分析中没有交易事件的情况。"""
        ai_response = '''
        基于市场分析，目前不建议交易。
        市场波动较大，建议等待更好的时机。
        '''

        result = await publisher.process_ai_analysis_and_publish(ai_response)

        assert result is False

    @pytest.mark.asyncio
    async def test_process_ai_analysis_and_publish_validation_error(self, publisher):
        """测试事件验证失败的情况。"""
        ai_response = '''
        ```json
        {
          "grid_delta": 0.05,  # 小于最小值
          "grid_quantity": 0.001,
          "active_side": "Buy"
        }
        ```
        '''

        result = await publisher.process_ai_analysis_and_publish(ai_response)

        assert result is False

    @pytest.mark.asyncio
    async def test_process_ai_analysis_and_publish_disabled(self, disabled_publisher):
        """测试禁用状态下的处理。"""
        ai_response = '''
        ```json
        {
          "grid_delta": 2.0,
          "grid_quantity": 0.001,
          "active_side": "Buy"
        }
        ```
        '''

        result = await disabled_publisher.process_ai_analysis_and_publish(ai_response)

        assert result is False

    @pytest.mark.asyncio
    async def test_test_connection_success(self, publisher):
        """测试连接测试成功。"""
        mock_redis = AsyncMock()
        mock_redis.ping.return_value = True
        mock_redis.publish.return_value = 1

        with patch('redis.asyncio.Redis', return_value=mock_redis):
            result = await publisher.test_connection()

            assert result is True
            mock_redis.ping.assert_called_once()
            mock_redis.publish.assert_called_once()

    @pytest.mark.asyncio
    async def test_test_connection_failure(self, publisher):
        """测试连接测试失败。"""
        with patch('redis.asyncio.Redis') as mock_redis_class:
            mock_redis = AsyncMock()
            mock_redis.ping.side_effect = Exception("Connection failed")
            mock_redis_class.return_value = mock_redis

            result = await publisher.test_connection()

            assert result is False

    @pytest.mark.asyncio
    async def test_test_connection_disabled(self, disabled_publisher):
        """测试禁用状态下的连接测试。"""
        result = await disabled_publisher.test_connection()

        assert result is True  # 禁用时应该返回True

    @pytest.mark.asyncio
    async def test_close(self, publisher):
        """测试关闭资源。"""
        publisher.redis_client = AsyncMock()
        publisher._is_connected = True

        await publisher.close()

        publisher.redis_client.close.assert_called_once()
        assert publisher._is_connected is False

    def test_get_status(self, publisher):
        """测试获取状态信息。"""
        status = publisher.get_status()

        assert status["enabled"] is True
        assert status["connected"] is False
        assert "redis_config" in status
        assert "validation_config" in status

        redis_config = status["redis_config"]
        assert redis_config["host"] == "localhost"
        assert redis_config["port"] == 6379
        assert redis_config["channel"] == "test_trading_events"

        validation_config = status["validation_config"]
        assert validation_config["grid_delta_range"] == (0.1, 100.0)
        assert validation_config["grid_quantity_range"] == (0.0001, 10.0)


class TestIntegration:
    """集成测试。"""

    @pytest.mark.asyncio
    async def test_end_to_end_trading_event_flow(self):
        """端到端交易事件流程测试。"""
        from src.utils.config import (
            TradingEventRedisConfig,
            TradingEventValidationConfig,
            TradingEventPublisherConfig
        )

        # 创建配置
        redis_config = TradingEventRedisConfig(
            host="localhost",
            port=6379,
            db=0,
            channel="test_integration_events",
            timeout=5,
            max_retries=2
        )

        validation_config = TradingEventValidationConfig(
            min_grid_delta=0.1,
            max_grid_delta=50.0,
            min_grid_quantity=0.0001,
            max_grid_quantity=5.0
        )

        config = TradingEventPublisherConfig(
            enable=True,
            redis=redis_config,
            validation=validation_config
        )

        # 创建发布器
        publisher = TradingEventPublisher(config)

        # Mock Redis连接
        mock_redis = AsyncMock()
        mock_redis.ping.return_value = True
        mock_redis.publish.return_value = 2  # 2个订阅者

        with patch('redis.asyncio.Redis', return_value=mock_redis):
            # 模拟完整的AI分析响应
            ai_response = '''
            综合市场分析完成。

            支撑阻力分析显示短期买入机会。

            交易指导：
            ```json
            {
              "grid_delta": 1.8,
              "grid_quantity": 0.003,
              "active_side": "Buy"
            }
            ```

            预期收益良好，建议执行。
            '''

            # 执行完整流程
            result = await publisher.process_ai_analysis_and_publish(ai_response)

            # 验证结果
            assert result is True
            assert publisher._is_connected is True

            # 验证Redis发布调用
            mock_redis.publish.assert_called_once()
            call_args = mock_redis.publish.call_args

            # 验证channel
            assert call_args[0][0] == "test_integration_events"

            # 验证消息格式
            message_json = call_args[0][1]
            message = json.loads(message_json)

            assert message["event_type"] == "trading_guidance"
            assert message["source"] == "strategy_agent_unified_analysis"
            assert message["data"]["grid_delta"] == 1.8
            assert message["data"]["grid_quantity"] == 0.003
            assert message["data"]["active_side"] == "Buy"
            assert "timestamp" in message

            # 关闭
            await publisher.close()
            mock_redis.close.assert_called_once()