"""简化市场分析器单元测试。

测试SimplifiedMarketAnalyzer类的核心功能。
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, Mock, patch
from decimal import Decimal

from src.core.simplified_market_analyzer import SimplifiedMarketAnalyzer


class TestSimplifiedMarketAnalyzer:
    """SimplifiedMarketAnalyzer测试类。"""

    def setup_method(self):
        """每个测试方法前的设置。"""
        # 创建模拟Redis存储
        self.mock_redis = Mock()
        self.mock_redis.test_connection.return_value = True
        self.mock_redis.depth_snapshot_exists.return_value = True
        self.mock_redis.get_trade_window_count.return_value = 1440

        # 创建DeepSeek配置
        self.deepseek_config = {
            "api_key": "test-api-key",
            "base_url": "https://api.deepseek.com/v1",
            "model": "deepseek-chat",
            "max_tokens": 3000,
            "temperature": 0.1,
            "timeout": 90,
            "max_retries": 3,
        }

    def test_initialization(self):
        """测试分析器初始化。"""
        with patch('src.core.simplified_market_analyzer.UnifiedDeepSeekAnalyzer'):
            analyzer = SimplifiedMarketAnalyzer(
                redis_store=self.mock_redis,
                deepseek_config=self.deepseek_config,
                price_aggregation_precision=1.0,
                vp_aggregation_precision=10.0,
            )

            assert analyzer.redis_store == self.mock_redis
            assert analyzer.price_aggregator.precision == 1.0
            assert analyzer.vp_analyzer.aggregation_precision == 10.0

    def test_get_status(self):
        """测试获取分析器状态。"""
        with patch('src.core.simplified_market_analyzer.UnifiedDeepSeekAnalyzer'):
            analyzer = SimplifiedMarketAnalyzer(
                redis_store=self.mock_redis,
                deepseek_config=self.deepseek_config,
            )

            status = analyzer.get_status()

            assert status["analyzer_type"] == "simplified_market_analyzer"
            assert "price_aggregation" in status
            assert "volume_profile" in status
            assert "unified_analyzer" in status
            assert "redis_connection" in status

    def test_read_and_aggregate_market_data_success(self):
        """测试成功读取和聚合市场数据。"""
        # 创建模拟深度快照
        mock_snapshot = Mock()
        mock_snapshot.symbol = "BTCFDUSD"
        mock_snapshot.timestamp = "2025-10-27 18:00:00"
        mock_snapshot.bids = {Decimal('115000.00'): Decimal('2.5')}
        mock_snapshot.asks = {Decimal('115010.00'): Decimal('1.9')}

        self.mock_redis.get_latest_depth_snapshot.return_value = mock_snapshot
        self.mock_redis.get_recent_trade_data.return_value = ["trade1", "trade2"]

        # 模拟Volume Profile结果
        mock_vp_result = {
            "status": "success",
            "price_levels_count": 100,
            "total_volume": 5000.0,
            "vp_data": {"115000.00": 100.0, "115010.00": 150.0}
        }

        with patch('src.core.simplified_market_analyzer.UnifiedDeepSeekAnalyzer'):
            with patch('src.core.simplified_market_analyzer.PriceAggregator') as mock_aggregator:
                with patch('src.core.simplified_market_analyzer.VolumeProfileAnalyzer') as mock_vp:
                    # 设置聚合器模拟
                    mock_aggregator_instance = Mock()
                    mock_aggregator_instance.aggregate_order_book_levels.return_value = (
                        {Decimal('115000.00'): Decimal('2.5')},
                        {Decimal('115010.00'): Decimal('1.9')}
                    )
                    mock_aggregator.return_value = mock_aggregator_instance

                    # 设置Volume Profile分析器模拟
                    mock_vp_instance = Mock()
                    mock_vp_instance.analyze_volume_profile.return_value = mock_vp_result
                    mock_vp.return_value = mock_vp_instance

                    analyzer = SimplifiedMarketAnalyzer(
                        redis_store=self.mock_redis,
                        deepseek_config=self.deepseek_config,
                    )

                    # 运行异步方法
                    result = asyncio.run(analyzer._read_and_aggregate_market_data("BTCFDUSD"))

                    assert result["status"] == "success"
                    assert "aggregated_bids" in result
                    assert "aggregated_asks" in result
                    assert "vp_result" in result

    def test_read_and_aggregate_market_data_no_depth_snapshot(self):
        """测试无深度快照数据的情况。"""
        self.mock_redis.get_latest_depth_snapshot.return_value = None

        with patch('src.core.simplified_market_analyzer.UnifiedDeepSeekAnalyzer'):
            analyzer = SimplifiedMarketAnalyzer(
                redis_store=self.mock_redis,
                deepseek_config=self.deepseek_config,
            )

            result = asyncio.run(analyzer._read_and_aggregate_market_data("BTCFDUSD"))

            assert result["status"] == "no_data"
            assert "No depth snapshot available" in result["error"]

    def test_read_and_aggregate_market_data_no_trades(self):
        """测试无交易数据的情况。"""
        # 设置深度快照但无交易数据
        mock_snapshot = Mock()
        mock_snapshot.symbol = "BTCFDUSD"
        mock_snapshot.bids = {Decimal('115000.00'): Decimal('2.5')}
        mock_snapshot.asks = {Decimal('115010.00'): Decimal('1.9')}

        self.mock_redis.get_latest_depth_snapshot.return_value = mock_snapshot
        self.mock_redis.get_recent_trade_data.return_value = None

        with patch('src.core.simplified_market_analyzer.UnifiedDeepSeekAnalyzer'):
            with patch('src.core.simplified_market_analyzer.PriceAggregator') as mock_aggregator:
                mock_aggregator_instance = Mock()
                mock_aggregator_instance.aggregate_order_book_levels.return_value = ({}, {})
                mock_aggregator.return_value = mock_aggregator_instance

                analyzer = SimplifiedMarketAnalyzer(
                    redis_store=self.mock_redis,
                    deepseek_config=self.deepseek_config,
                )

                result = asyncio.run(analyzer._read_and_aggregate_market_data("BTCFDUSD"))

                assert result["status"] == "no_data"
                assert "No trades window data available" in result["error"]

    def test_create_error_result(self):
        """测试创建错误结果。"""
        with patch('src.core.simplified_market_analyzer.UnifiedDeepSeekAnalyzer'):
            analyzer = SimplifiedMarketAnalyzer(
                redis_store=self.mock_redis,
                deepseek_config=self.deepseek_config,
            )

            error_message = "Test error message"
            result = analyzer._create_error_result("BTCFDUSD", error_message)

            assert result["symbol"] == "BTCFDUSD"
            assert result["status"] == "error"
            assert result["error"] == error_message
            assert result["trading_params"] is None
            assert "timestamp" in result

    @patch('src.core.simplified_market_analyzer.result_validator')
    @patch('src.core.simplified_market_analyzer.UnifiedDeepSeekAnalyzer')
    @patch('src.core.simplified_market_analyzer.VolumeProfileAnalyzer')
    @patch('src.core.simplified_market_analyzer.PriceAggregator')
    def test_analyze_market_success(self, mock_aggregator, mock_vp, mock_unified, mock_validator):
        """测试完整的市场分析流程成功情况。"""
        # 设置所有模拟对象
        mock_aggregator_instance = Mock()
        mock_aggregator_instance.aggregate_order_book_levels.return_value = (
            {Decimal('115000.00'): Decimal('2.5')},
            {Decimal('115010.00'): Decimal('1.9')}
        )
        mock_aggregator.return_value = mock_aggregator_instance

        mock_vp_instance = Mock()
        mock_vp_result = {
            "status": "success",
            "price_levels_count": 100,
            "total_volume": 5000.0,
        }
        mock_vp_instance.analyze_volume_profile.return_value = mock_vp_result
        mock_vp.return_value = mock_vp_instance

        mock_unified_instance = Mock()
        mock_analysis_result = {
            "status": "success",
            "raw_content": '{"grid_delta": 2.0, "grid_quantity": 0.001, "active_side": "Buy"}',
            "symbol": "BTCFDUSD"
        }
        mock_unified_instance.analyze_unified_market_data.return_value = mock_analysis_result
        mock_unified.return_value = mock_unified_instance

        # 设置验证器模拟
        mock_trading_params = {
            "grid_delta": 2.0,
            "grid_quantity": 0.001,
            "active_side": "Buy"
        }
        mock_validator.validate_and_extract_trading_params.return_value = mock_trading_params

        # 设置Redis模拟
        mock_snapshot = Mock()
        mock_snapshot.symbol = "BTCFDUSD"
        mock_snapshot.bids = {Decimal('115000.00'): Decimal('2.5')}
        mock_snapshot.asks = {Decimal('115010.00'): Decimal('1.9')}
        self.mock_redis.get_latest_depth_snapshot.return_value = mock_snapshot
        self.mock_redis.get_recent_trade_data.return_value = ["trade1", "trade2"]

        analyzer = SimplifiedMarketAnalyzer(
            redis_store=self.mock_redis,
            deepseek_config=self.deepseek_config,
        )

        result = asyncio.run(analyzer.analyze_market("BTCFDUSD"))

        assert result["status"] == "success"
        assert result["symbol"] == "BTCFDUSD"
        assert result["trading_params"] == mock_trading_params
        assert result["analysis_type"] == "simplified_market_analysis"

        # 验证市场数据摘要
        market_summary = result["market_data_summary"]
        assert market_summary["bid_levels"] == 1
        assert market_summary["ask_levels"] == 1
        assert market_summary["vp_price_levels"] == 100
        assert market_summary["total_volume"] == 5000.0

    @patch('src.core.simplified_market_analyzer.result_validator')
    @patch('src.core.simplified_market_analyzer.UnifiedDeepSeekAnalyzer')
    @patch('src.core.simplified_market_analyzer.VolumeProfileAnalyzer')
    @patch('src.core.simplified_market_analyzer.PriceAggregator')
    def test_analyze_market_validation_failure(self, mock_aggregator, mock_vp, mock_unified, mock_validator):
        """测试市场分析验证失败情况。"""
        # 设置模拟对象返回数据聚合成功
        mock_aggregator_instance = Mock()
        mock_aggregator_instance.aggregate_order_book_levels.return_value = ({}, {})
        mock_aggregator.return_value = mock_aggregator_instance

        mock_vp_instance = Mock()
        mock_vp_result = {
            "status": "success",
            "price_levels_count": 100,
            "total_volume": 5000.0,
        }
        mock_vp_instance.analyze_volume_profile.return_value = mock_vp_result
        mock_vp.return_value = mock_vp_instance

        # 设置DeepSeek分析成功
        mock_unified_instance = Mock()
        mock_analysis_result = {
            "status": "success",
            "raw_content": '{"grid_delta": 2.0, "grid_quantity": 0.001, "active_side": "Buy"}',
            "symbol": "BTCFDUSD"
        }
        mock_unified_instance.analyze_unified_market_data.return_value = mock_analysis_result
        mock_unified.return_value = mock_unified_instance

        # 设置验证器抛出验证错误
        mock_validator.validate_and_extract_trading_params.side_effect = Exception("Validation failed")

        # 设置Redis模拟
        mock_snapshot = Mock()
        mock_snapshot.bids = {}
        mock_snapshot.asks = {}
        self.mock_redis.get_latest_depth_snapshot.return_value = mock_snapshot
        self.mock_redis.get_recent_trade_data.return_value = ["trade1"]

        analyzer = SimplifiedMarketAnalyzer(
            redis_store=self.mock_redis,
            deepseek_config=self.deepseek_config,
        )

        result = asyncio.run(analyzer.analyze_market("BTCFDUSD"))

        assert result["status"] == "error"
        assert result["symbol"] == "BTCFDUSD"
        assert "Validation failed" in result["error"]
        assert result["trading_params"] is None

    def test_analyze_market_data_aggregation_failure(self):
        """测试数据聚合失败情况。"""
        # 设置无深度快照
        self.mock_redis.get_latest_depth_snapshot.return_value = None

        with patch('src.core.simplified_market_analyzer.UnifiedDeepSeekAnalyzer'):
            analyzer = SimplifiedMarketAnalyzer(
                redis_store=self.mock_redis,
                deepseek_config=self.deepseek_config,
            )

            result = asyncio.run(analyzer.analyze_market("BTCFDUSD"))

            assert result["status"] == "error"
            assert "Data aggregation failed" in result["error"]
            assert result["trading_params"] is None

    @patch('src.core.simplified_market_analyzer.UnifiedDeepSeekAnalyzer')
    def test_close(self, mock_unified):
        """测试关闭分析器。"""
        mock_unified_instance = Mock()
        mock_unified.return_value = mock_unified_instance

        analyzer = SimplifiedMarketAnalyzer(
            redis_store=self.mock_redis,
            deepseek_config=self.deepseek_config,
        )

        # 运行关闭方法
        asyncio.run(analyzer.close())

        # 验证DeepSeek分析器关闭方法被调用
        mock_unified_instance.close.assert_called_once()