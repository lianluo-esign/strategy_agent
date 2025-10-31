"""交易聚合器单元测试。"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from src.core.agent_analyzer_optimized.trades_aggregator import (
    AggregatedTradesData,
    TradesAggregator
)


class TestTradesAggregator:
    """交易聚合器测试类。"""

    def setup_method(self):
        """每个测试方法前的设置。"""
        self.aggregator = TradesAggregator(
            aggregation_precision=10.0,
            min_volume_threshold=0.1,
            minutes_to_analyze=60
        )

    def test_initialization_valid_parameters(self):
        """测试有效参数的初始化。"""
        aggregator = TradesAggregator(
            aggregation_precision=5.0,
            min_volume_threshold=0.5,
            minutes_to_analyze=720
        )
        assert aggregator.aggregation_precision == 5.0
        assert aggregator.min_volume_threshold == 0.5
        assert aggregator.minutes_to_analyze == 720

    def test_initialization_invalid_aggregation_precision(self):
        """测试无效聚合精度参数。"""
        with pytest.raises(ValueError, match="聚合精度必须为正数"):
            TradesAggregator(aggregation_precision=0)

        with pytest.raises(ValueError, match="聚合精度必须为正数"):
            TradesAggregator(aggregation_precision=-1.0)

    def test_initialization_invalid_min_volume_threshold(self):
        """测试无效最小成交量阈值参数。"""
        with pytest.raises(ValueError, match="最小成交量阈值不能为负数"):
            TradesAggregator(min_volume_threshold=-0.1)

    def test_initialization_invalid_minutes_to_analyze(self):
        """测试无效分析时间窗口参数。"""
        with pytest.raises(ValueError, match="分析时间窗口必须为正数"):
            TradesAggregator(minutes_to_analyze=0)

    def test_aggregate_trades_window_empty_data(self):
        """测试空数据处理。"""
        with pytest.raises(ValueError, match="trades_window_data不能为空"):
            self.aggregator.aggregate_trades_window([])

    def test_aggregate_trades_window_valid_data(self):
        """测试有效数据聚合。"""
        # 创建模拟数据
        mock_data = self._create_mock_trades_data()

        result = self.aggregator.aggregate_trades_window(mock_data, "BTCFDUSD")

        # 验证结果类型
        assert isinstance(result, AggregatedTradesData)
        assert result.symbol == "BTCFDUSD"
        assert result.total_volume > 0
        assert result.trade_count > 0
        assert len(result.price_levels) > 0
        assert len(result.price_range) == 2
        assert result.price_range[0] <= result.price_range[1]

    def test_aggregate_trades_window_filter_invalid_data(self):
        """测试过滤无效数据。"""
        # 创建包含无效数据的混合列表
        mixed_data = [
            self._create_valid_minute_data(),
            None,  # 无效数据
            "invalid",  # 无效数据
            self._create_valid_minute_data(),
        ]

        result = self.aggregator.aggregate_trades_window(mixed_data, "BTCFDUSD")

        # 应该只处理有效数据
        assert result.trade_count == 2
        assert result.total_volume > 0

    def test_align_price_to_precision(self):
        """测试价格对齐功能。"""
        # 测试对齐到10的精度
        assert self.aggregator._align_price_to_precision(100.5) == 100.0
        assert self.aggregator._align_price_to_precision(109.9) == 100.0
        assert self.aggregator._align_price_to_precision(110.0) == 110.0
        assert self.aggregator._align_price_to_precision(115.5) == 110.0

    def test_get_market_summary(self):
        """测试市场摘要生成。"""
        mock_data = self._create_mock_trades_data()
        aggregated_data = self.aggregator.aggregate_trades_window(mock_data, "BTCFDUSD")

        summary = self.aggregator.get_market_summary(aggregated_data)

        # 验证摘要结构
        assert "analysis_timestamp" in summary
        assert "symbol" in summary
        assert "data_summary" in summary
        assert "volume_analysis" in summary
        assert "poc_analysis" in summary

        assert summary["symbol"] == "BTCFDUSD"
        assert summary["data_summary"]["total_volume"] > 0
        assert summary["data_summary"]["price_levels_count"] > 0

    def test_get_market_summary_empty_data(self):
        """测试空数据的市场摘要。"""
        empty_data = AggregatedTradesData(
            timestamp=datetime.now(),
            symbol="BTCFDUSD",
            price_levels={},
            total_volume=0,
            trade_count=0,
            price_range=(0, 0)
        )

        summary = self.aggregator.get_market_summary(empty_data)
        assert "error" in summary

    def _create_mock_trades_data(self):
        """创建模拟交易数据。"""
        return [
            self._create_valid_minute_data(),
            self._create_valid_minute_data(),
            self._create_valid_minute_data(),
        ]

    def _create_valid_minute_data(self):
        """创建有效的分钟数据。"""
        mock_data = Mock()
        mock_data.timestamp = datetime.now()
        mock_data.price_levels = {
            "1000": {"total_volume": 1.5},
            "1010": {"total_volume": 2.0},
            "1020": {"total_volume": 1.2},
        }
        return mock_data


class TestAggregatedTradesData:
    """聚合交易数据模型测试类。"""

    def test_to_dict_conversion(self):
        """测试转换为字典格式。"""
        timestamp = datetime.now()
        price_levels = {1000.0: 1.5, 1010.0: 2.0}
        price_range = (1000.0, 1010.0)

        data = AggregatedTradesData(
            timestamp=timestamp,
            symbol="BTCFDUSD",
            price_levels=price_levels,
            total_volume=3.5,
            trade_count=100,
            price_range=price_range
        )

        result = data.to_dict()

        assert result["timestamp"] == timestamp.isoformat()
        assert result["symbol"] == "BTCFDUSD"
        assert result["price_levels"] == price_levels
        assert result["total_volume"] == 3.5
        assert result["trade_count"] == 100
        assert result["price_range"] == price_range
        assert result["price_levels_count"] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])