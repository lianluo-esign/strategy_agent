"""集成测试 - 统一分析与交易事件发布集成。

测试统一分析工作流中交易事件发布的完整集成：
- 深度快照和Volume Profile数据准备
- 统一AI分析执行
- 交易事件提取和发布
- 错误处理和日志记录
"""

import asyncio
import json
import pytest
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

from src.core.enhanced_market_analyzer import EnhancedMarketAnalyzer
from src.core.price_aggregator import PriceAggregator
from src.core.volume_profile_analyzer import VolumeProfileAnalyzer
from src.core.unified_deepseek_analyzer import UnifiedDeepSeekAnalyzer
from src.core.trading_event_publisher import TradingEventPublisher
from src.utils.config import TradingEventPublisherConfig


class TestUnifiedAnalysisTradingEventsIntegration:
    """测试统一分析与交易事件发布的集成。"""

    @pytest.fixture
    def mock_redis_store(self):
        """创建模拟的Redis存储。"""
        mock_store = AsyncMock()
        mock_store.test_connection.return_value = True
        mock_store.depth_snapshot_exists.return_value = True
        mock_store.get_trade_window_count.return_value = 1000
        return mock_store

    @pytest.fixture
    def mock_depth_snapshot(self):
        """创建模拟的深度快照数据。"""
        snapshot = MagicMock()
        snapshot.symbol = "BTCFDUSD"
        snapshot.timestamp = "2025-01-01T12:00:00Z"

        # 模拟聚合后的订单簿数据
        snapshot.bids = {
            Decimal("100000.00"): Decimal("1.5"),
            Decimal("99999.00"): Decimal("2.0"),
            Decimal("99998.00"): Decimal("1.8"),
            Decimal("99997.00"): Decimal("2.5"),
            Decimal("99996.00"): Decimal("1.2"),
        }
        snapshot.asks = {
            Decimal("100001.00"): Decimal("1.8"),
            Decimal("100002.00"): Decimal("2.2"),
            Decimal("100003.00"): Decimal("1.5"),
            Decimal("100004.00"): Decimal("2.8"),
            Decimal("100005.00"): Decimal("1.0"),
        }
        return snapshot

    @pytest.fixture
    def mock_trade_data(self):
        """创建模拟的交易数据。"""
        return [
            {"price": Decimal("100000.50"), "volume": Decimal("0.5"), "timestamp": "2025-01-01T11:00:00Z"},
            {"price": Decimal("100001.00"), "volume": Decimal("0.8"), "timestamp": "2025-01-01T11:30:00Z"},
            {"price": Decimal("99999.50"), "volume": Decimal("1.2"), "timestamp": "2025-01-01T12:00:00Z"},
            {"price": Decimal("100002.00"), "volume": Decimal("0.7"), "timestamp": "2025-01-01T12:30:00Z"},
            {"price": Decimal("100000.00"), "volume": Decimal("1.5"), "timestamp": "2025-01-01T13:00:00Z"},
        ]

    @pytest.fixture
    def deepseek_config(self):
        """创建DeepSeek配置。"""
        return {
            "enable": True,
            "use_unified_analysis": True,
            "api_key": "test_api_key",
            "base_url": "https://api.deepseek.com/v1",
            "model": "deepseek-chat",
            "max_tokens": 6000,
            "temperature": 0.1,
            "timeout": 90,
            "max_retries": 3,
        }

    @pytest.fixture
    def trading_event_publisher_config(self):
        """创建交易事件发布器配置。"""
        from src.utils.config import (
            TradingEventRedisConfig,
            TradingEventValidationConfig,
            TradingEventPublisherConfig
        )

        redis_config = TradingEventRedisConfig(
            host="localhost",
            port=6379,
            db=0,
            channel="hft_grid_strategy_params",
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
    def enhanced_analyzer(self, mock_redis_store, deepseek_config, trading_event_publisher_config):
        """创建增强型市场分析器。"""
        return EnhancedMarketAnalyzer(
            redis_store=mock_redis_store,
            price_aggregation_precision=1.0,
            vp_aggregation_precision=10.0,
            deepseek_config=deepseek_config,
            visualizer=None,
            trading_event_publisher_config=trading_event_publisher_config
        )

    @pytest.mark.asyncio
    async def test_unified_analysis_with_trading_event_publishing_success(
        self, enhanced_analyzer, mock_redis_store, mock_depth_snapshot, mock_trade_data
    ):
        """测试统一分析与交易事件发布成功流程。"""
        # Mock Redis数据返回
        mock_redis_store.get_latest_depth_snapshot.return_value = mock_depth_snapshot
        mock_redis_store.get_recent_trade_data.return_value = mock_trade_data

        # Mock DeepSeek API响应（包含交易事件）
        mock_api_response = {
            "choices": [
                {
                    "message": {
                        "content": '''
{
  "短期支撑位": [
    {
      "价格": "99990.00",
      "可靠性评分": "85",
      "形成原因": "订单簿支撑",
      "推荐入场区间": "99990-99995",
      "特征描述": "强支撑位，大量买盘挂单"
    }
  ],
  "短期阻力位": [
    {
      "价格": "100010.00",
      "可靠性评分": "80",
      "形成原因": "历史成交压力",
      "推荐退出区间": "100005-100010",
      "特征描述": "明显阻力位，历史多次测试"
    }
  ],
  "集中流动性供应区域": {
    "最佳价格区间": "100000-100005",
    "备选区间": ["99995-100000", "100005-100010"],
    "市场特征": "高流动性区域",
    "安全性评估": "中等风险",
    "收益潜力": "良好"
  },
  "做市策略要点": {
    "主要机会": "短期波动交易",
    "风险控制": "严格止损",
    "仓位管理": "分批建仓",
    "时机把握": "等待确认信号",
    "策略总结": "温和上涨趋势，适合做多"
  }
}

```json
{
  "grid_delta": 2.0,
  "grid_quantity": 0.002,
  "active_side": "Buy"
}
```
'''
                    }
                }
            ]
        }

        # Mock Redis交易事件发布
        mock_trading_redis = AsyncMock()
        mock_trading_redis.ping.return_value = True
        mock_trading_redis.publish.return_value = 1

        with patch('src.core.unified_deepseek_analyzer.httpx.Client') as mock_httpx:
            # Mock HTTP客户端
            mock_client = MagicMock()
            mock_client.post.return_value.json.return_value = mock_api_response
            mock_client.post.return_value.raise_for_status.return_value = None
            mock_httpx.return_value = mock_client

            with patch('redis.asyncio.Redis', return_value=mock_trading_redis):
                # 执行统一分析
                result = await enhanced_analyzer.perform_dual_analysis("BTCFDUSD")

                # 验证分析结果
                assert result["status"] == "success"
                assert result["analysis_type"] == "unified_market_analysis"
                assert "depth_analysis" in result
                assert "volume_profile_analysis" in result
                assert "unified_analysis" in result

                # 验证统一分析成功
                unified_analysis = result["unified_analysis"]
                assert unified_analysis["status"] == "success"
                assert unified_analysis["symbol"] == "BTCFDUSD"
                assert "structured_analysis" in unified_analysis
                assert "raw_content" in unified_analysis

                # 验证交易事件发布
                mock_trading_redis.publish.assert_called_once()
                call_args = mock_trading_redis.publish.call_args

                # 验证channel
                assert call_args[0][0] == "hft_grid_strategy_params"

                # 验证发布的消息格式
                message_json = call_args[0][1]
                message = json.loads(message_json)

                assert message["event_type"] == "trading_guidance"
                assert message["source"] == "strategy_agent_unified_analysis"
                assert message["data"]["grid_delta"] == 2.0
                assert message["data"]["grid_quantity"] == 0.002
                assert message["data"]["active_side"] == "Buy"
                assert "timestamp" in message

    @pytest.mark.asyncio
    async def test_unified_analysis_without_trading_event(
        self, enhanced_analyzer, mock_redis_store, mock_depth_snapshot, mock_trade_data
    ):
        """测试统一分析但没有交易事件的情况。"""
        # Mock Redis数据返回
        mock_redis_store.get_latest_depth_snapshot.return_value = mock_depth_snapshot
        mock_redis_store.get_recent_trade_data.return_value = mock_trade_data

        # Mock DeepSeek API响应（不包含交易事件）
        mock_api_response = {
            "choices": [
                {
                    "message": {
                        "content": '''
{
  "短期支撑位": [
    {
      "价格": "99990.00",
      "可靠性评分": "85",
      "形成原因": "订单簿支撑",
      "推荐入场区间": "99990-99995",
      "特征描述": "强支撑位，大量买盘挂单"
    }
  ],
  "短期阻力位": [
    {
      "价格": "100010.00",
      "可靠性评分": "80",
      "形成原因": "历史成交压力",
      "推荐退出区间": "100005-100010",
      "特征描述": "明显阻力位，历史多次测试"
    }
  ],
  "集中流动性供应区域": {
    "最佳价格区间": "100000-100005",
    "备选区间": ["99995-100000", "100005-100010"],
    "市场特征": "高流动性区域",
    "安全性评估": "中等风险",
    "收益潜力": "良好"
  },
  "做市策略要点": {
    "主要机会": "观望等待",
    "风险控制": "保守策略",
    "仓位管理": "轻仓试探",
    "时机把握": "等待明确信号",
    "策略总结": "市场震荡，建议观望"
  }
}

注意：当前市场环境下不建议进行交易，等待更明确的信号。
'''
                    }
                }
            ]
        }

        with patch('src.core.unified_deepseek_analyzer.httpx.Client') as mock_httpx:
            mock_client = MagicMock()
            mock_client.post.return_value.json.return_value = mock_api_response
            mock_client.post.return_value.raise_for_status.return_value = None
            mock_httpx.return_value = mock_client

            # 执行统一分析
            result = await enhanced_analyzer.perform_dual_analysis("BTCFDUSD")

            # 验证分析成功，但没有交易事件发布
            assert result["status"] == "success"
            assert result["unified_analysis"]["status"] == "success"

            # 由于没有交易事件，应该没有Redis发布调用
            # 但由于我们没有mock trading_redis，这里主要验证分析成功

    @pytest.mark.asyncio
    async def test_unified_analysis_trading_event_validation_error(
        self, enhanced_analyzer, mock_redis_store, mock_depth_snapshot, mock_trade_data
    ):
        """测试统一分析中交易事件验证失败的情况。"""
        # Mock Redis数据返回
        mock_redis_store.get_latest_depth_snapshot.return_value = mock_depth_snapshot
        mock_redis_store.get_recent_trade_data.return_value = mock_trade_data

        # Mock DeepSeek API响应（包含无效的交易事件）
        mock_api_response = {
            "choices": [
                {
                    "message": {
                        "content": '''
{
  "短期支撑位": [
    {
      "价格": "99990.00",
      "可靠性评分": "85",
      "形成原因": "订单簿支撑",
      "推荐入场区间": "99990-99995",
      "特征描述": "强支撑位，大量买盘挂单"
    }
  ],
  "做市策略要点": {
    "主要机会": "短期交易",
    "策略总结": "适合交易"
  }
}

```json
{
  "grid_delta": 0.05,
  "grid_quantity": 0.001,
  "active_side": "Buy"
}
```
'''
                    }
                }
            ]
        }

        with patch('src.core.unified_deepseek_analyzer.httpx.Client') as mock_httpx:
            mock_client = MagicMock()
            mock_client.post.return_value.json.return_value = mock_api_response
            mock_client.post.return_value.raise_for_status.return_value = None
            mock_httpx.return_value = mock_client

            # 执行统一分析
            result = await enhanced_analyzer.perform_dual_analysis("BTCFDUSD")

            # 验证分析成功，但交易事件发布应该失败
            assert result["status"] == "success"
            assert result["unified_analysis"]["status"] == "success"

            # 交易事件由于grid_delta太小（0.05 < 0.1）应该验证失败
            # 但不影响整体分析的成功

    @pytest.mark.asyncio
    async def test_unified_analysis_trading_event_publisher_disabled(
        self, mock_redis_store, mock_depth_snapshot, mock_trade_data, deepseek_config
    ):
        """测试禁用交易事件发布器的情况。"""
        # 创建禁用的配置
        from src.utils.config import (
            TradingEventRedisConfig,
            TradingEventValidationConfig,
            TradingEventPublisherConfig
        )

        redis_config = TradingEventRedisConfig(
            host="localhost",
            port=6379,
            db=0,
            channel="hft_grid_strategy_params",
            timeout=5,
            max_retries=3
        )

        validation_config = TradingEventValidationConfig(
            min_grid_delta=0.1,
            max_grid_delta=100.0,
            min_grid_quantity=0.0001,
            max_grid_quantity=10.0
        )

        disabled_config = TradingEventPublisherConfig(
            enable=False,  # 禁用
            redis=redis_config,
            validation=validation_config
        )

        # 创建禁用交易事件发布的分析器
        analyzer = EnhancedMarketAnalyzer(
            redis_store=mock_redis_store,
            price_aggregation_precision=1.0,
            vp_aggregation_precision=10.0,
            deepseek_config=deepseek_config,
            visualizer=None,
            trading_event_publisher_config=disabled_config
        )

        # Mock Redis数据返回
        mock_redis_store.get_latest_depth_snapshot.return_value = mock_depth_snapshot
        mock_redis_store.get_recent_trade_data.return_value = mock_trade_data

        # Mock DeepSeek API响应（包含交易事件）
        mock_api_response = {
            "choices": [
                {
                    "message": {
                        "content': '''
{
  "短期支撑位": [{"价格": "99990.00"}],
  "做市策略要点": {"策略总结": "适合交易"}
}

```json
{
  "grid_delta": 2.0,
  "grid_quantity": 0.002,
  "active_side": "Buy"
}
```
'''
                    }
                }
            ]
        }

        with patch('src.core.unified_deepseek_analyzer.httpx.Client') as mock_httpx:
            mock_client = MagicMock()
            mock_client.post.return_value.json.return_value = mock_api_response
            mock_client.post.return_value.raise_for_status.return_value = None
            mock_httpx.return_value = mock_client

            # 执行统一分析
            result = await analyzer.perform_dual_analysis("BTCFDUSD")

            # 验证分析成功
            assert result["status"] == "success"
            assert result["unified_analysis"]["status"] == "success"

            # 验证交易事件发布器被禁用
            assert analyzer.trading_event_publisher is None

            # 清理
            await analyzer.close()

    @pytest.mark.asyncio
    async def test_unified_analysis_redis_publish_error(
        self, enhanced_analyzer, mock_redis_store, mock_depth_snapshot, mock_trade_data
    ):
        """测试Redis发布错误的处理。"""
        # Mock Redis数据返回
        mock_redis_store.get_latest_depth_snapshot.return_value = mock_depth_snapshot
        mock_redis_store.get_recent_trade_data.return_value = mock_trade_data

        # Mock DeepSeek API响应（包含交易事件）
        mock_api_response = {
            "choices": [
                {
                    "message": {
                        "content": '''
{
  "短期支撑位": [{"价格": "99990.00"}],
  "做市策略要点": {"策略总结": "适合交易"}
}

```json
{
  "grid_delta": 2.0,
  "grid_quantity": 0.002,
  "active_side": "Buy"
}
```
'''
                    }
                }
            ]
        }

        # Mock Redis交易事件发布（发布失败）
        mock_trading_redis = AsyncMock()
        mock_trading_redis.ping.return_value = True
        mock_trading_redis.publish.side_effect = Exception("Redis connection failed")

        with patch('src.core.unified_deepseek_analyzer.httpx.Client') as mock_httpx:
            mock_client = MagicMock()
            mock_client.post.return_value.json.return_value = mock_api_response
            mock_client.post.return_value.raise_for_status.return_value = None
            mock_httpx.return_value = mock_client

            with patch('redis.asyncio.Redis', return_value=mock_trading_redis):
                # 执行统一分析
                result = await enhanced_analyzer.perform_dual_analysis("BTCFDUSD")

                # 验证分析成功，尽管Redis发布失败
                assert result["status"] == "success"
                assert result["unified_analysis"]["status"] == "success"

                # 验证Redis发布被调用但失败
                mock_trading_redis.publish.assert_called()

    @pytest.mark.asyncio
    async def test_end_to_end_workflow_with_real_data_flow(
        self, enhanced_analyzer, mock_redis_store
    ):
        """测试端到端工作流程，模拟真实数据流。"""
        # 创建更真实的模拟数据
        mock_snapshot = MagicMock()
        mock_snapshot.symbol = "BTCFDUSD"
        mock_snapshot.timestamp = "2025-01-01T12:00:00Z"

        # 模拟更多层级的订单簿数据
        mock_snapshot.bids = {
            Decimal("100000.00"): Decimal("5.2"),
            Decimal("99999.00"): Decimal("3.8"),
            Decimal("99998.00"): Decimal("4.1"),
            Decimal("99997.00"): Decimal("2.9"),
            Decimal("99996.00"): Decimal("3.5"),
            Decimal("99995.00"): Decimal("6.2"),
            Decimal("99994.00"): Decimal("4.8"),
            Decimal("99993.00"): Decimal("5.5"),
        }
        mock_snapshot.asks = {
            Decimal("100001.00"): Decimal("4.5"),
            Decimal("100002.00"): Decimal("3.2"),
            Decimal("100003.00"): Decimal("5.8"),
            Decimal("100004.00"): Decimal("4.1"),
            Decimal("100005.00"): Decimal("3.7"),
            Decimal("100006.00"): Decimal("6.1"),
            Decimal("100007.00"): Decimal("4.3"),
            Decimal("100008.00"): Decimal("5.2"),
        }

        # 模拟更丰富的交易数据
        mock_trade_data = []
        base_price = Decimal("100000.00")
        for i in range(50):
            price = base_price + Decimal(str((i % 20) - 10))  # ±10的价格波动
            volume = Decimal(str(0.1 + (i % 10) * 0.2))  # 0.1-2.9的交易量
            mock_trade_data.append({
                "price": price,
                "volume": volume,
                "timestamp": f"2025-01-01T{i//2:02d}:{(i%2)*30:02d}:00Z"
            })

        # Mock Redis数据返回
        mock_redis_store.get_latest_depth_snapshot.return_value = mock_snapshot
        mock_redis_store.get_recent_trade_data.return_value = mock_trade_data

        # Mock DeepSeek API响应
        mock_api_response = {
            "choices": [
                {
                    "message": {
                        "content": '''
{
  "短期支撑位": [
    {
      "价格": "99995.00",
      "可靠性评分": "90",
      "形成原因": "订单簿支撑+成交量共识",
      "推荐入场区间": "99994-99996",
      "特征描述": "强支撑位，大量买盘和高成交量确认"
    },
    {
      "价格": "99990.00",
      "可靠性评分": "75",
      "形成原因": "成交量共识",
      "推荐入场区间": "99989-99991",
      "特征描述": "次要支撑位，中等成交量支撑"
    }
  ],
  "短期阻力位": [
    {
      "价格": "100005.00",
      "可靠性评分": "85",
      "形成原因": "订单簿阻力",
      "推荐退出区间": "100004-100006",
      "特征描述": "明显阻力位，大量卖盘挂单"
    }
  ],
  "集中流动性供应区域": {
    "最佳价格区间": "99998-100002",
    "备选区间": ["99995-99999", "100002-100006"],
    "市场特征": "高流动性集中区域，买卖活跃",
    "安全性评估": "较低风险，流动性充足",
    "收益潜力": "良好，波动性适中"
  },
  "做市策略要点": {
    "主要机会": "区间震荡交易，利用支撑阻力位差价",
    "风险控制": "设置在99988和100008的止损",
    "仓位管理": "分3批建仓，每批0.001",
    "时机把握": "等待价格确认支撑后入场",
    "策略总结": "温和上涨趋势，适合区间做市策略"
  }
}

```json
{
  "grid_delta": 1.5,
  "grid_quantity": 0.003,
  "active_side": "Buy"
}
```
'''
                    }
                }
            ]
        }

        # Mock Redis交易事件发布
        mock_trading_redis = AsyncMock()
        mock_trading_redis.ping.return_value = True
        mock_trading_redis.publish.return_value = 3  # 3个订阅者

        with patch('src.core.unified_deepseek_analyzer.httpx.Client') as mock_httpx:
            mock_client = MagicMock()
            mock_client.post.return_value.json.return_value = mock_api_response
            mock_client.post.return_value.raise_for_status.return_value = None
            mock_httpx.return_value = mock_client

            with patch('redis.asyncio.Redis', return_value=mock_trading_redis):
                # 执行完整的端到端工作流
                result = await enhanced_analyzer.perform_dual_analysis("BTCFDUSD")

                # 验证完整流程成功
                assert result["status"] == "success"
                assert result["symbol"] == "BTCFDUSD"
                assert result["analysis_type"] == "unified_market_analysis"

                # 验证深度分析
                depth_analysis = result["depth_analysis"]
                assert depth_analysis["status"] == "success"
                assert "aggregated_bids" in depth_analysis
                assert "aggregated_asks" in depth_analysis

                # 验证Volume Profile分析
                vp_analysis = result["volume_profile_analysis"]
                assert vp_analysis["status"] == "success"

                # 验证统一AI分析
                unified_analysis = result["unified_analysis"]
                assert unified_analysis["status"] == "success"
                assert unified_analysis["symbol"] == "BTCFDUSD"

                # 验证结构化分析内容
                structured_analysis = unified_analysis["structured_analysis"]
                assert "短期支撑位" in structured_analysis
                assert "短期阻力位" in structured_analysis
                assert "集中流动性供应区域" in structured_analysis
                assert "做市策略要点" in structured_analysis

                # 验证支撑位分析
                support_levels = structured_analysis["短期支撑位"]
                assert len(support_levels) >= 1
                assert support_levels[0]["价格"] == "99995.00"
                assert support_levels[0]["可靠性评分"] == "90"

                # 验证交易事件发布
                mock_trading_redis.publish.assert_called_once()
                call_args = mock_trading_redis.publish.call_args

                # 验证发布的交易事件
                assert call_args[0][0] == "hft_grid_strategy_params"
                message_json = call_args[0][1]
                message = json.loads(message_json)

                assert message["event_type"] == "trading_guidance"
                assert message["data"]["grid_delta"] == 1.5
                assert message["data"]["grid_quantity"] == 0.003
                assert message["data"]["active_side"] == "Buy"
                assert "timestamp" in message

                # 验证发布器状态
                status = enhanced_analyzer.get_status()
                assert status["analysis_mode"] == "unified"
                assert "unified_analysis" in status

    @pytest.mark.asyncio
    async def test_trading_event_publisher_connection_retry(
        self, enhanced_analyzer, mock_redis_store, mock_depth_snapshot, mock_trade_data
    ):
        """测试交易事件发布器连接重试机制。"""
        # Mock Redis数据返回
        mock_redis_store.get_latest_depth_snapshot.return_value = mock_depth_snapshot
        mock_redis_store.get_recent_trade_data.return_value = mock_trade_data

        # Mock DeepSeek API响应
        mock_api_response = {
            "choices": [
                {
                    "message": {
                        "content': '''
{
  "短期支撑位": [{"价格": "99990.00"}],
  "做市策略要点": {"策略总结": "适合交易"}
}

```json
{
  "grid_delta": 2.0,
  "grid_quantity": 0.002,
  "active_side": "Buy"
}
```
'''
                    }
                }
            ]
        }

        # Mock Redis连接重试场景
        mock_trading_redis = AsyncMock()
        mock_trading_redis.ping.side_effect = [
            Exception("First connection failed"),
            True  # 第二次成功
        ]
        mock_trading_redis.publish.return_value = 1

        with patch('src.core.unified_deepseek_analyzer.httpx.Client') as mock_httpx:
            mock_client = MagicMock()
            mock_client.post.return_value.json.return_value = mock_api_response
            mock_client.post.return_value.raise_for_status.return_value = None
            mock_httpx.return_value = mock_client

            with patch('redis.asyncio.Redis', return_value=mock_trading_redis):
                # 执行分析
                result = await enhanced_analyzer.perform_dual_analysis("BTCFDUSD")

                # 验证最终成功
                assert result["status"] == "success"
                assert result["unified_analysis"]["status"] == "success"

                # 验证Redis重试后成功发布
                mock_trading_redis.publish.assert_called_once()

    @pytest.mark.asyncio
    async def test_analyzer_close_resources(self, enhanced_analyzer):
        """测试分析器资源关闭。"""
        # Mock交易事件发布器的close方法
        enhanced_analyzer.trading_event_publisher = AsyncMock()

        # 执行关闭
        await enhanced_analyzer.close()

        # 验证交易事件发布器被关闭
        enhanced_analyzer.trading_event_publisher.close.assert_called_once()