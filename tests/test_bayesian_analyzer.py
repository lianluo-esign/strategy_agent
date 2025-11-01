"""贝叶斯分析器测试。

测试贝叶斯趋势分析器的核心功能：
1. 静态订单簿分析测试
2. 动态数据分析测试
3. 贝叶斯概率计算测试
4. 响应格式化测试
5. 集成功能测试
"""

import pytest
from decimal import Decimal
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch

from src.core.agent_analyzer_optimized.static_orderbook_analyzer import StaticOrderBookAnalyzer
from src.core.agent_analyzer_optimized.bayesian_analyzer import BayesianAnalyzer
from src.core.agent_analyzer_optimized.bayesian_response_formatter import BayesianResponseFormatter


class TestStaticOrderBookAnalyzer:
    """静态订单簿分析器测试。"""

    def setup_method(self):
        """测试前设置。"""
        self.analyzer = StaticOrderBookAnalyzer(aggregation_precision=10.0)

    def test_initialization(self):
        """测试初始化。"""
        assert self.analyzer.aggregation_precision == Decimal("10.0")

    def test_aggregate_order_book_depth(self):
        """测试订单簿深度聚合。"""
        # 模拟原始订单簿数据
        raw_bids = [
            ("50005.5", "1.5"),
            ("50012.3", "2.0"),
            ("50008.7", "1.0"),
        ]
        raw_asks = [
            ("50015.2", "1.2"),
            ("50021.8", "2.5"),
            ("50018.4", "0.8"),
        ]

        # 执行聚合
        aggregated_bids, aggregated_asks = self.analyzer._aggregate_order_book_depth(
            raw_bids, raw_asks
        )

        # 验证结果
        assert len(aggregated_bids) == 2  # 50000和50010两个价位
        assert len(aggregated_asks) == 3  # 50010, 50020两个价位

        # 验证聚合精度
        assert Decimal("50000") in aggregated_bids
        assert Decimal("50010") in aggregated_bids
        assert Decimal("50010") in aggregated_asks
        assert Decimal("50020") in aggregated_asks

    def test_analyze_order_book_with_valid_data(self):
        """测试使用有效数据进行订单簿分析。"""
        # 创建模拟深度快照
        mock_depth_snapshot = Mock()
        mock_depth_snapshot.symbol = "BTCFDUSD"
        mock_depth_snapshot.timestamp = datetime.now()
        mock_depth_snapshot.bids = [
            ("50005.5", "1.5"),
            ("50012.3", "2.0"),
        ]
        mock_depth_snapshot.asks = [
            ("50015.2", "1.2"),
            ("50021.8", "2.5"),
        ]

        # 执行分析
        result = self.analyzer.analyze_order_book(mock_depth_snapshot, "BTCFDUSD")

        # 验证结果结构
        assert result["status"] == "success"
        assert result["symbol"] == "BTCFDUSD"
        assert "aggregated_bids" in result
        assert "aggregated_asks" in result
        assert "liquidity_analysis" in result
        assert "key_levels" in result
        assert "imbalance_metrics" in result

    def test_analyze_order_book_with_empty_data(self):
        """测试使用空数据进行订单簿分析。"""
        result = self.analyzer.analyze_order_book(None, "BTCFDUSD")

        assert result["status"] == "no_data"
        assert result["symbol"] == "BTCFDUSD"

    def test_calculate_order_book_imbalance(self):
        """测试订单簿不平衡度计算。"""
        aggregated_bids = {"50000": Decimal("10.0"), "50010": Decimal("5.0")}
        aggregated_asks = {"50020": Decimal("8.0"), "50030": Decimal("4.0")}

        result = self.analyzer._calculate_order_book_imbalance(aggregated_bids, aggregated_asks)

        # 验证计算结果
        assert result["bid_percentage"] == 15.0 / 27.0  # 15/(15+12)
        assert result["ask_percentage"] == 12.0 / 27.0
        assert result["direction"] == "bullish"  # 买方更多


class TestBayesianAnalyzer:
    """贝叶斯分析器测试。"""

    def setup_method(self):
        """测试前设置。"""
        self.analyzer = BayesianAnalyzer()

    def test_initialization(self):
        """测试初始化。"""
        assert "震荡" in self.analyzer.prior_probabilities
        assert "看涨" in self.analyzer.prior_probabilities
        assert sum(self.analyzer.prior_probabilities.values()) == pytest.approx(1.0)

    def test_extract_static_liquidity_evidence(self):
        """测试静态流动性证据提取。"""
        static_data = {
            "liquidity_analysis": {
                "total_liquidity": 1500.0,
                "bid_ask_ratio": 1.2,
            },
            "key_levels": {
                "strongest_support": {"volume": 100.0},
                "strongest_resistance": {"volume": 80.0},
            }
        }

        evidence = self.analyzer._extract_static_liquidity_evidence(static_data)

        assert evidence["total_liquidity"] == 1500.0
        assert evidence["bid_ask_ratio"] == 1.2
        assert evidence["liquidity_strength"] == "high"

    def test_extract_dynamic_volume_evidence(self):
        """测试动态成交量证据提取。"""
        dynamic_data = {
            "minute_data_points": [
                {
                    "price_levels": {
                        "50000": {"total_volume": 10.0},
                        "50010": {"total_volume": 5.0},
                    }
                }
            ] * 20  # 20个数据点
        }

        evidence = self.analyzer._extract_dynamic_volume_evidence(dynamic_data)

        assert evidence["total_volume"] > 0
        assert evidence["data_points_count"] == 20
        assert evidence["recent_activity"] in ["high", "medium", "low"]

    def test_calculate_likelihoods(self):
        """测试似然函数计算。"""
        evidences = {
            "static_liquidity": {
                "bid_ask_ratio": 1.5,
                "liquidity_strength": "high"
            },
            "dynamic_volume": {
                "volume_trend": "increasing",
                "recent_activity": "high"
            }
        }

        likelihoods = self.analyzer._calculate_likelihoods(evidences)

        # 验证所有趋势类型都有似然值
        for trend in ["震荡", "看涨", "看跌", "微弱看涨", "微弱看跌", "强力看涨", "强力看跌"]:
            assert trend in likelihoods
            assert "static_liquidity" in likelihoods[trend]
            assert "dynamic_volume" in likelihoods[trend]

    def test_bayesian_update(self):
        """测试贝叶斯更新计算。"""
        likelihoods = {
            "看涨": {"static_liquidity": 0.8, "dynamic_volume": 0.7},
            "看跌": {"static_liquidity": 0.3, "dynamic_volume": 0.4},
            "震荡": {"static_liquidity": 0.5, "dynamic_volume": 0.6},
        }

        posteriors = self.analyzer._bayesian_update(likelihoods)

        # 验证后验概率
        assert sum(posteriors.values()) == pytest.approx(1.0)
        assert posteriors["看涨"] > posteriors["看跌"]  # 看涨的似然更高

    def test_analyze_bayesian_trend_integration(self):
        """测试完整的贝叶斯趋势分析集成。"""
        static_data = {
            "status": "success",
            "liquidity_analysis": {
                "total_liquidity": 1500.0,
                "bid_ask_ratio": 1.3,
            },
            "imbalance_metrics": {
                "direction": "bullish",
                "imbalance_strength": 0.6,
            }
        }

        dynamic_data = {
            "data_points_count": 100,
            "minute_data_points": [
                {
                    "price_levels": {
                        "50000": {"total_volume": 10.0},
                    }
                }
            ] * 50
        }

        result = self.analyzer.analyze_bayesian_trend(static_data, dynamic_data, "BTCFDUSD")

        # 验证结果结构
        assert result["status"] == "success"
        assert result["symbol"] == "BTCFDUSD"
        assert "posterior_probabilities" in result
        assert "analysis_result" in result

        # 验证分析结果
        analysis_result = result["analysis_result"]
        assert "most_likely_trend" in analysis_result
        assert "confidence" in analysis_result
        assert "uncertainty" in analysis_result
        assert 0 <= analysis_result["confidence"] <= 1


class TestBayesianResponseFormatter:
    """贝叶斯响应格式化器测试。"""

    def setup_method(self):
        """测试前设置。"""
        self.formatter = BayesianResponseFormatter(include_metadata=True, pretty_print=False)

    def test_format_probability_distribution(self):
        """测试概率分布格式化。"""
        probabilities = {
            "看涨": 0.4,
            "震荡": 0.3,
            "看跌": 0.2,
            "微弱看涨": 0.1,
        }

        result = self.formatter._format_probability_distribution(probabilities)

        # 验证结果结构
        assert "full_distribution" in result
        assert "top_three_trends" in result
        assert "entropy" in result
        assert "distribution_type" in result

        # 验证Top3趋势
        top_three = result["top_three_trends"]
        assert len(top_three) == 3
        assert top_three[0]["trend"] == "看涨"
        assert top_three[0]["probability"] == 0.4

    def test_get_confidence_level(self):
        """测试置信度等级分类。"""
        assert self.formatter._get_confidence_level(0.9) == "very_high"
        assert self.formatter._get_confidence_level(0.7) == "high"
        assert self.formatter._get_confidence_level(0.5) == "medium"
        assert self.formatter._get_confidence_level(0.3) == "low"
        assert self.formatter._get_confidence_level(0.1) == "very_low"

    def test_get_risk_level(self):
        """测试风险等级分类。"""
        assert self.formatter._get_risk_level(0.1) == "low_risk"
        assert self.formatter._get_risk_level(0.3) == "moderate_risk"
        assert self.formatter._get_risk_level(0.5) == "high_risk"
        assert self.formatter._get_risk_level(0.8) == "very_high_risk"

    def test_validate_bayesian_response(self):
        """测试贝叶斯响应验证。"""
        # 有效响应
        valid_response = """{
            "timestamp": "2024-01-01T00:00:00",
            "symbol": "BTCFDUSD",
            "analysis_type": "bayesian_trend_analysis",
            "trend_analysis": {
                "most_likely_trend": "看涨",
                "confidence": 0.7,
                "uncertainty": 0.3
            },
            "probability_distribution": {
                "full_distribution": {
                    "看涨": 0.7,
                    "震荡": 0.3
                }
            }
        }"""

        assert self.formatter.validate_bayesian_response(valid_response) == True

        # 无效响应（缺少字段）
        invalid_response = """{
            "timestamp": "2024-01-01T00:00:00",
            "symbol": "BTCFDUSD"
        }"""

        assert self.formatter.validate_bayesian_response(invalid_response) == False


class TestIntegration:
    """集成测试。"""

    @pytest.mark.asyncio
    async def test_static_analyzer_integration(self):
        """测试静态分析器集成。"""
        analyzer = StaticOrderBookAnalyzer()

        # 创建模拟深度快照
        mock_depth_snapshot = Mock()
        mock_depth_snapshot.symbol = "BTCFDUSD"
        mock_depth_snapshot.timestamp = datetime.now()
        mock_depth_snapshot.bids = [
            ("50005.5", "1.5"),
            ("50012.3", "2.0"),
        ]
        mock_depth_snapshot.asks = [
            ("50015.2", "1.2"),
            ("50021.8", "2.5"),
        ]

        result = analyzer.analyze_order_book(mock_depth_snapshot, "BTCFDUSD")

        # 验证集成结果
        assert result["status"] == "success"
        assert result["symbol"] == "BTCFDUSD"
        assert len(result["aggregated_bids"]) > 0
        assert len(result["aggregated_asks"]) > 0

    def test_bayesian_analyzer_integration(self):
        """测试贝叶斯分析器集成。"""
        analyzer = BayesianAnalyzer()

        static_data = {
            "status": "success",
            "liquidity_analysis": {
                "total_liquidity": 1000.0,
                "bid_ask_ratio": 1.2,
            },
            "imbalance_metrics": {
                "direction": "bullish",
                "imbalance_strength": 0.5,
            }
        }

        dynamic_data = {
            "data_points_count": 50,
            "minute_data_points": [
                {
                    "price_levels": {
                        "50000": {"total_volume": 5.0},
                    }
                }
            ] * 30
        }

        result = analyzer.analyze_bayesian_trend(static_data, dynamic_data, "BTCFDUSD")

        # 验证集成结果
        assert result["status"] == "success"
        assert "posterior_probabilities" in result
        assert "analysis_result" in result

        # 验证概率分布总和
        posteriors = result["posterior_probabilities"]
        total_prob = sum(posteriors.values())
        assert abs(total_prob - 1.0) < 0.01

    def test_error_handling_integration(self):
        """测试错误处理集成。"""
        analyzer = BayesianAnalyzer()

        # 测试空数据
        result = analyzer.analyze_bayesian_trend({}, {}, "BTCFDUSD")
        assert result["status"] in ["error", "no_data"]

        # 测试无效数据
        static_data = {"status": "error", "error": "Test error"}
        dynamic_data = {"data_points_count": 0}

        result = analyzer.analyze_bayesian_trend(static_data, dynamic_data, "BTCFDUSD")
        assert result["status"] in ["error", "no_data"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])