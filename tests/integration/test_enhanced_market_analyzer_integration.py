"""增强型市场分析器的集成测试。

这个模块测试EnhancedMarketAnalyzer的完整集成功能：
1. 统一分析模式的端到端测试
2. 传统分析模式的兼容性测试
3. 数据流和处理管道测试
4. 错误处理和恢复测试
"""

import json
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from src.core.enhanced_market_analyzer import EnhancedMarketAnalyzer
from src.core.unified_deepseek_analyzer import UnifiedDeepSeekAnalyzer


class TestEnhancedMarketAnalyzerIntegration:
    """增强型市场分析器集成测试类。"""

    def setup_method(self):
        """每个测试方法前的设置。"""
        # 模拟Redis存储
        self.mock_redis_store = MagicMock()
        self.mock_redis_store.test_connection.return_value = True
        self.mock_redis_store.depth_snapshot_exists.return_value = True
        self.mock_redis_store.get_trade_window_count.return_value = 100

        # 模拟可视化工具
        self.mock_visualizer = MagicMock()

        # DeepSeek配置
        self.deepseek_config = {
            "enable": True,
            "api_key": "test_api_key",
            "base_url": "https://api.deepseek.com/v1",
            "model": "deepseek-chat",
            "max_tokens": 6000,
            "temperature": 0.1,
            "timeout": 90,
            "max_retries": 3,
            "use_unified_analysis": True,  # 启用统一分析模式
        }

    def teardown_method(self):
        """每个测试方法后的清理。"""
        pass

    def create_test_depth_snapshot(self):
        """创建测试用深度快照数据。"""
        mock_snapshot = MagicMock()
        mock_snapshot.symbol = "BTCFDUSD"
        mock_snapshot.timestamp = "2024-01-01T12:00:00Z"
        mock_snapshot.bids = [
            (Decimal('99999.00'), Decimal('10.5')),
            (Decimal('99998.00'), Decimal('8.2')),
            (Decimal('99997.00'), Decimal('15.1')),
        ]
        mock_snapshot.asks = [
            (Decimal('100001.00'), Decimal('12.3')),
            (Decimal('100002.00'), Decimal('9.1')),
            (Decimal('100003.00'), Decimal('18.7')),
        ]
        return mock_snapshot

    def create_test_trade_data(self):
        """创建测试用交易数据。"""
        trade_data = []
        for i in range(1440):  # 24小时数据
            price = 100000 + (i % 1000) - 500  # 价格在95000-105000之间波动
            volume = 1.0 + (i % 10) * 0.5
            trade_data.append({
                'timestamp': f'2024-01-01T{i:02d}:00:00Z',
                'price': Decimal(str(price)),
                'volume': Decimal(str(volume)),
                'side': 'buy' if i % 2 == 0 else 'sell'
            })
        return trade_data

    @patch('src.core.enhanced_market_analyzer.PriceAggregator')
    @patch('src.core.enhanced_market_analyzer.VolumeProfileAnalyzer')
    @patch('src.core.enhanced_market_analyzer.UnifiedDeepSeekAnalyzer')
    def test_unified_analysis_mode_end_to_end(self, mock_unified, mock_vp, mock_aggregator):
        """测试统一分析模式的端到端流程。"""
        # 设置模拟对象
        mock_aggregator_instance = MagicMock()
        mock_aggregator_instance.precision = 1.0
        mock_aggregator_instance.aggregate_order_book_levels.return_value = (
            {Decimal('99999.00'): Decimal('10.5'), Decimal('99998.00'): Decimal('8.2')},
            {Decimal('100001.00'): Decimal('12.3'), Decimal('100002.00'): Decimal('9.1')}
        )
        mock_aggregator.return_value = mock_aggregator_instance

        mock_vp_instance = MagicMock()
        mock_vp_instance.aggregation_precision = 10.0
        mock_vp_instance.analyze_volume_profile.return_value = {
            "status": "success",
            "vp_data": {Decimal('100000.00'): Decimal('100.0'), Decimal('100100.00'): Decimal('150.0')},
            "poc_analysis": {
                "poc_price": Decimal('100000.00'),
                "poc_volume": Decimal('100.0'),
                "value_area_high": Decimal('100200.00'),
                "value_area_low": Decimal('99900.00'),
                "value_area_range": Decimal('300.00'),
            },
            "total_volume": Decimal('1000.0'),
            "price_levels_count": 20,
        }
        mock_vp.return_value = mock_vp_instance

        mock_unified_instance = MagicMock()
        mock_unified_instance.analyze_unified_market_data.return_value = {
            "status": "success",
            "symbol": "BTCFDUSD",
            "analysis_type": "unified_market_analysis",
            "structured_analysis": {
                "短期支撑位": [
                    {
                        "价格": "99900.00",
                        "可靠性评分": "85",
                        "形成原因": "成交量共识",
                        "推荐入场区间": "99850.00-99950.00",
                        "特征描述": "强支撑区域"
                    }
                ],
                "短期阻力位": [
                    {
                        "价格": "100200.00",
                        "可靠性评分": "80",
                        "形成原因": "订单簿阻力",
                        "推荐退出区间": "100150.00-100250.00",
                        "特征描述": "明显阻力"
                    }
                ],
                "集中流动性供应区域": {
                    "最佳价格区间": "99800.00-100300.00",
                    "备选区间": ["99600.00-99700.00", "100400.00-100500.00"],
                    "市场特征": "高流动性集中区",
                    "安全性评估": "中等偏高",
                    "收益潜力": "良好"
                },
                "做市策略要点": {
                    "主要机会": "支撑阻力区间明确",
                    "风险控制": "注意突破风险",
                    "仓位管理": "分批建仓",
                    "时机把握": "回调入场",
                    "策略总结": "区间做市策略"
                }
            }
        }
        mock_unified.return_value = mock_unified_instance

        # 设置Redis数据
        self.mock_redis_store.get_latest_depth_snapshot.return_value = self.create_test_depth_snapshot()
        self.mock_redis_store.get_recent_trade_data.return_value = self.create_test_trade_data()

        # 创建分析器
        analyzer = EnhancedMarketAnalyzer(
            redis_store=self.mock_redis_store,
            price_aggregation_precision=1.0,
            vp_aggregation_precision=10.0,
            deepseek_config=self.deepseek_config,
            visualizer=self.mock_visualizer,
        )

        # 执行分析
        result = analyzer.perform_dual_analysis("BTCFDUSD")

        # 验证结果
        assert result["status"] == "success"
        assert result["symbol"] == "BTCFDUSD"
        assert result["analysis_type"] == "unified_market_analysis"
        assert "depth_analysis" in result
        assert "volume_profile_analysis" in result
        assert "unified_analysis" in result

        # 验证深度分析部分
        assert result["depth_analysis"]["status"] == "success"
        assert "aggregated_bids" in result["depth_analysis"]
        assert "aggregated_asks" in result["depth_analysis"]

        # 验证Volume Profile分析部分
        assert result["volume_profile_analysis"]["status"] == "success"
        assert "vp_analysis" in result["volume_profile_analysis"]

        # 验证统一分析部分
        assert result["unified_analysis"]["status"] == "success"
        assert result["unified_analysis"]["structured_analysis"] is not None
        assert "短期支撑位" in result["unified_analysis"]["structured_analysis"]
        assert "短期阻力位" in result["unified_analysis"]["structured_analysis"]
        assert "集中流动性供应区域" in result["unified_analysis"]["structured_analysis"]

        # 验证模拟调用
        mock_unified_instance.analyze_unified_market_data.assert_called_once()
        mock_aggregator_instance.aggregate_order_book_levels.assert_called_once()
        mock_vp_instance.analyze_volume_profile.assert_called_once()

        analyzer.close()

    @patch('src.core.enhanced_market_analyzer.PriceAggregator')
    @patch('src.core.enhanced_market_analyzer.VolumeProfileAnalyzer')
    @patch('src.core.enhanced_market_analyzer.DeepSeekOrderBookAnalyzer')
    @patch('src.core.enhanced_market_analyzer.DeepSeekVPAnalyzer')
    def test_traditional_analysis_mode_compatibility(self, mock_vp_analyzer, mock_orderbook_analyzer, mock_vp, mock_aggregator):
        """测试传统分析模式的兼容性。"""
        # 使用传统模式配置
        traditional_config = self.deepseek_config.copy()
        traditional_config["use_unified_analysis"] = False

        # 设置模拟对象
        mock_aggregator_instance = MagicMock()
        mock_aggregator_instance.precision = 1.0
        mock_aggregator_instance.aggregate_order_book_levels.return_value = (
            {Decimal('99999.00'): Decimal('10.5')},
            {Decimal('100001.00'): Decimal('12.3')}
        )
        mock_aggregator.return_value = mock_aggregator_instance

        mock_vp_instance = MagicMock()
        mock_vp_instance.analyze_volume_profile.return_value = {
            "status": "success",
            "vp_data": {Decimal('100000.00'): Decimal('100.0')},
            "poc_analysis": {"poc_price": Decimal('100000.00')},
            "total_volume": Decimal('1000.0'),
        }
        mock_vp.return_value = mock_vp_instance

        mock_orderbook_instance = MagicMock()
        mock_orderbook_instance.analyze_order_book_with_llm.return_value = {
            "status": "success",
            "structured_analysis": {"支撑区域": [{"价格区间": "99900-100000"}]}
        }
        mock_orderbook_analyzer.return_value = mock_orderbook_instance

        mock_vp_analyzer_instance = MagicMock()
        mock_vp_analyzer_instance.analyze_volume_profile_with_llm.return_value = {
            "status": "success",
            "structured_analysis": {"poc分析": {"poc价格": "100000"}}
        }
        mock_vp_analyzer.return_value = mock_vp_analyzer_instance

        # 设置Redis数据
        self.mock_redis_store.get_latest_depth_snapshot.return_value = self.create_test_depth_snapshot()
        self.mock_redis_store.get_recent_trade_data.return_value = self.create_test_trade_data()

        # 创建分析器
        analyzer = EnhancedMarketAnalyzer(
            redis_store=self.mock_redis_store,
            price_aggregation_precision=1.0,
            vp_aggregation_precision=10.0,
            deepseek_config=traditional_config,
            visualizer=self.mock_visualizer,
        )

        # 执行分析
        result = analyzer.perform_dual_analysis("BTCFDUSD")

        # 验证结果
        assert result["status"] == "success"
        assert result["analysis_type"] == "traditional_dual_analysis"
        assert "depth_analysis" in result
        assert "volume_profile_analysis" in result
        assert "unified_analysis" not in result  # 传统模式不应该有统一分析

        # 验证传统分析调用
        mock_orderbook_instance.analyze_order_book_with_llm.assert_called_once()
        mock_vp_analyzer_instance.analyze_volume_profile_with_llm.assert_called_once()

        analyzer.close()

    def test_unified_analysis_data_flow_validation(self):
        """测试统一分析模式的数据流验证。"""
        with patch('src.core.enhanced_market_analyzer.PriceAggregator') as mock_aggregator, \
             patch('src.core.enhanced_market_analyzer.VolumeProfileAnalyzer') as mock_vp, \
             patch('src.core.enhanced_market_analyzer.UnifiedDeepSeekAnalyzer') as mock_unified:

            # 设置模拟对象
            mock_aggregator_instance = MagicMock()
            mock_aggregator_instance.aggregate_order_book_levels.return_value = (
                {Decimal('99999.00'): Decimal('10.5')},
                {Decimal('100001.00'): Decimal('12.3')}
            )
            mock_aggregator.return_value = mock_aggregator_instance

            mock_vp_instance = MagicMock()
            mock_vp_instance.analyze_volume_profile.return_value = {
                "status": "success",
                "vp_data": {Decimal('100000.00'): Decimal('100.0')},
                "poc_analysis": {"poc_price": Decimal('100000.00')},
                "total_volume": Decimal('1000.0'),
            }
            mock_vp.return_value = mock_vp_instance

            mock_unified_instance = MagicMock()
            mock_unified_instance.analyze_unified_market_data.return_value = {
                "status": "success",
                "structured_analysis": {"test": "data"}
            }
            mock_unified.return_value = mock_unified_instance

            # 设置Redis数据
            self.mock_redis_store.get_latest_depth_snapshot.return_value = self.create_test_depth_snapshot()
            self.mock_redis_store.get_recent_trade_data.return_value = self.create_test_trade_data()

            # 创建分析器
            analyzer = EnhancedMarketAnalyzer(
                redis_store=self.mock_redis_store,
                deepseek_config=self.deepseek_config
            )

            # 执行分析
            result = analyzer.perform_dual_analysis("BTCFDUSD")

            # 验证数据流：深度快照 -> 聚合 -> 统一分析
            expected_bids = {Decimal('99999.00'): Decimal('10.5')}
            expected_asks = {Decimal('100001.00'): Decimal('12.3')}
            expected_vp_result = {
                "status": "success",
                "vp_data": {Decimal('100000.00'): Decimal('100.0')},
                "poc_analysis": {"poc_price": Decimal('100000.00')},
                "total_volume": Decimal('1000.0'),
            }

            mock_unified_instance.analyze_unified_market_data.assert_called_once_with(
                aggregated_bids=expected_bids,
                aggregated_asks=expected_asks,
                vp_result=expected_vp_result,
                symbol="BTCFDUSD"
            )

            analyzer.close()

    def test_error_handling_depth_snapshot_failure(self):
        """测试深度快照获取失败的错误处理。"""
        with patch('src.core.enhanced_market_analyzer.PriceAggregator') as mock_aggregator, \
             patch('src.core.enhanced_market_analyzer.VolumeProfileAnalyzer') as mock_vp, \
             patch('src.core.enhanced_market_analyzer.UnifiedDeepSeekAnalyzer') as mock_unified:

            # 设置深度快照获取失败
            self.mock_redis_store.get_latest_depth_snapshot.return_value = None

            mock_aggregator.return_value = MagicMock()
            mock_vp.return_value = MagicMock()
            mock_unified.return_value = MagicMock()

            analyzer = EnhancedMarketAnalyzer(
                redis_store=self.mock_redis_store,
                deepseek_config=self.deepseek_config
            )

            result = analyzer.perform_dual_analysis("BTCFDUSD")

            # 验证错误处理
            assert result["status"] == "error"
            assert "Depth analysis failed" in result["error"]

            # 验证不会调用统一分析
            mock_unified.return_value.analyze_unified_market_data.assert_not_called()

            analyzer.close()

    def test_error_handling_volume_profile_failure(self):
        """测试Volume Profile分析失败的错误处理。"""
        with patch('src.core.enhanced_market_analyzer.PriceAggregator') as mock_aggregator, \
             patch('src.core.enhanced_market_analyzer.VolumeProfileAnalyzer') as mock_vp, \
             patch('src.core.enhanced_market_analyzer.UnifiedDeepSeekAnalyzer') as mock_unified:

            # 设置成功获取深度快照
            mock_aggregator_instance = MagicMock()
            mock_aggregator_instance.aggregate_order_book_levels.return_value = (
                {Decimal('99999.00'): Decimal('10.5')},
                {Decimal('100001.00'): Decimal('12.3')}
            )
            mock_aggregator.return_value = mock_aggregator_instance

            # 设置Volume Profile分析失败
            mock_vp_instance = MagicMock()
            mock_vp_instance.analyze_volume_profile.return_value = {
                "status": "error",
                "error": "VP analysis failed"
            }
            mock_vp.return_value = mock_vp_instance

            self.mock_redis_store.get_latest_depth_snapshot.return_value = self.create_test_depth_snapshot()
            self.mock_redis_store.get_recent_trade_data.return_value = self.create_test_trade_data()

            mock_unified.return_value = MagicMock()

            analyzer = EnhancedMarketAnalyzer(
                redis_store=self.mock_redis_store,
                deepseek_config=self.deepseek_config
            )

            result = analyzer.perform_dual_analysis("BTCFDUSD")

            # 验证错误处理
            assert result["status"] == "error"
            assert "Volume Profile analysis failed" in result["error"]

            # 验证不会调用统一分析
            mock_unified.return_value.analyze_unified_market_data.assert_not_called()

            analyzer.close()

    def test_unified_analysis_api_failure_handling(self):
        """测试统一分析API失败的错误处理。"""
        with patch('src.core.enhanced_market_analyzer.PriceAggregator') as mock_aggregator, \
             patch('src.core.enhanced_market_analyzer.VolumeProfileAnalyzer') as mock_vp, \
             patch('src.core.enhanced_market_analyzer.UnifiedDeepSeekAnalyzer') as mock_unified:

            # 设置成功的数据处理
            mock_aggregator_instance = MagicMock()
            mock_aggregator_instance.aggregate_order_book_levels.return_value = (
                {Decimal('99999.00'): Decimal('10.5')},
                {Decimal('100001.00'): Decimal('12.3')}
            )
            mock_aggregator.return_value = mock_aggregator_instance

            mock_vp_instance = MagicMock()
            mock_vp_instance.analyze_volume_profile.return_value = {
                "status": "success",
                "vp_data": {Decimal('100000.00'): Decimal('100.0')},
                "poc_analysis": {"poc_price": Decimal('100000.00')},
                "total_volume": Decimal('1000.0'),
            }
            mock_vp.return_value = mock_vp_instance

            # 设置统一分析API失败
            mock_unified_instance = MagicMock()
            mock_unified_instance.analyze_unified_market_data.return_value = {
                "status": "error",
                "error": "API request failed"
            }
            mock_unified.return_value = mock_unified_instance

            self.mock_redis_store.get_latest_depth_snapshot.return_value = self.create_test_depth_snapshot()
            self.mock_redis_store.get_recent_trade_data.return_value = self.create_test_trade_data()

            analyzer = EnhancedMarketAnalyzer(
                redis_store=self.mock_redis_store,
                deepseek_config=self.deepseek_config
            )

            result = analyzer.perform_dual_analysis("BTCFDUSD")

            # 验证API错误被正确处理，整体分析仍然成功
            assert result["status"] == "success"
            assert result["analysis_type"] == "unified_market_analysis"
            assert result["unified_analysis"]["status"] == "error"
            assert "API request failed" in result["unified_analysis"]["error"]

            analyzer.close()

    def test_get_status_unified_mode(self):
        """测试统一模式下的状态获取。"""
        with patch('src.core.enhanced_market_analyzer.UnifiedDeepSeekAnalyzer') as mock_unified:
            mock_unified_instance = MagicMock()
            mock_unified_instance.model = "deepseek-chat"
            mock_unified_instance.max_tokens = 6000
            mock_unified_instance.timeout = 90
            mock_unified.return_value = mock_unified_instance

            analyzer = EnhancedMarketAnalyzer(
                redis_store=self.mock_redis_store,
                deepseek_config=self.deepseek_config
            )

            status = analyzer.get_status()

            # 验证状态信息
            assert status["analysis_mode"] == "unified"
            assert "unified_analysis" in status
            assert status["unified_analysis"]["enabled"] is True
            assert status["unified_analysis"]["model"] == "deepseek-chat"
            assert status["unified_analysis"]["max_tokens"] == 6000
            assert status["unified_analysis"]["timeout"] == 90

            analyzer.close()

    def test_get_status_traditional_mode(self):
        """测试传统模式下的状态获取。"""
        traditional_config = self.deepseek_config.copy()
        traditional_config["use_unified_analysis"] = False

        with patch('src.core.enhanced_market_analyzer.DeepSeekOrderBookAnalyzer') as mock_orderbook, \
             patch('src.core.enhanced_market_analyzer.DeepSeekVPAnalyzer') as mock_vp:

            mock_orderbook_instance = MagicMock()
            mock_orderbook_instance.model = "deepseek-chat"
            mock_orderbook.return_value = mock_orderbook_instance

            mock_vp_instance = MagicMock()
            mock_vp_instance.model = "deepseek-chat"
            mock_vp.return_value = mock_vp_instance

            analyzer = EnhancedMarketAnalyzer(
                redis_store=self.mock_redis_store,
                deepseek_config=traditional_config
            )

            status = analyzer.get_status()

            # 验证状态信息
            assert status["analysis_mode"] == "traditional"
            assert "unified_analysis" not in status
            assert status["depth_analysis"]["deepseek_analysis"]["enabled"] is True
            assert status["volume_profile_analysis"]["deepseek_analysis"]["enabled"] is True

            analyzer.close()

    def test_visualization_integration(self):
        """测试可视化功能集成。"""
        with patch('src.core.enhanced_market_analyzer.PriceAggregator') as mock_aggregator, \
             patch('src.core.enhanced_market_analyzer.VolumeProfileAnalyzer') as mock_vp, \
             patch('src.core.enhanced_market_analyzer.UnifiedDeepSeekAnalyzer') as mock_unified:

            # 设置模拟对象
            mock_aggregator_instance = MagicMock()
            mock_aggregator_instance.aggregate_order_book_levels.return_value = (
                {Decimal('99999.00'): Decimal('10.5')},
                {Decimal('100001.00'): Decimal('12.3')}
            )
            mock_aggregator.return_value = mock_aggregator_instance

            mock_vp_instance = MagicMock()
            mock_vp_instance.analyze_volume_profile.return_value = {
                "status": "success",
                "vp_data": {Decimal('100000.00'): Decimal('100.0')},
                "poc_analysis": {"poc_price": Decimal('100000.00')},
                "total_volume": Decimal('1000.0'),
            }
            mock_vp.return_value = mock_vp_instance

            mock_unified_instance = MagicMock()
            mock_unified_instance.analyze_unified_market_data.return_value = {
                "status": "success",
                "structured_analysis": {"test": "data"}
            }
            mock_unified.return_value = mock_unified_instance

            # 设置可视化和Redis数据
            self.mock_visualizer.create_order_book_distribution_chart.return_value = "test_chart.png"
            self.mock_redis_store.get_latest_depth_snapshot.return_value = self.create_test_depth_snapshot()
            self.mock_redis_store.get_recent_trade_data.return_value = self.create_test_trade_data()

            analyzer = EnhancedMarketAnalyzer(
                redis_store=self.mock_redis_store,
                deepseek_config=self.deepseek_config,
                visualizer=self.mock_visualizer
            )

            result = analyzer.perform_dual_analysis("BTCFDUSD")

            # 验证可视化结果
            assert result["visualization"]["status"] == "success"
            assert result["visualization"]["output_file"] == "test_chart.png"

            # 验证可视化调用
            self.mock_visualizer.create_order_book_distribution_chart.assert_called_once()

            analyzer.close()

    def test_deepseek_disabled_mode(self):
        """测试DeepSeek禁用模式。"""
        disabled_config = {
            "enable": False,
        }

        with patch('src.core.enhanced_market_analyzer.PriceAggregator') as mock_aggregator, \
             patch('src.core.enhanced_market_analyzer.VolumeProfileAnalyzer') as mock_vp:

            mock_aggregator.return_value = MagicMock()
            mock_vp.return_value = MagicMock()

            analyzer = EnhancedMarketAnalyzer(
                redis_store=self.mock_redis_store,
                deepseek_config=disabled_config
            )

            status = analyzer.get_status()

            # 验证DeepSeek禁用状态
            assert status["analysis_mode"] == "disabled"
            assert status["ai_analysis"]["enabled"] is False

            analyzer.close()