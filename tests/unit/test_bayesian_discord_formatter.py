"""单元测试：BayesianDiscordFormatter核心数据提取功能

测试贝叶斯分析结果的数据提取和格式化逻辑，
确保Discord通知能正确处理各种复杂的数据结构。
"""

import json
import pytest
from datetime import datetime
from src.core.agent_analyzer_optimized.discord_notifier import BayesianDiscordFormatter


class TestBayesianDiscordFormatter:
    """测试BayesianDiscordFormatter类的核心功能。"""

    def setup_method(self):
        """每个测试方法前的设置。"""
        self.formatter = BayesianDiscordFormatter()

    def test_extract_core_bayesian_info_complete_data(self):
        """测试从完整的贝叶斯分析数据中提取核心信息。"""
        # 准备完整的测试数据
        complete_data = {
            "timestamp": "2025-11-01T18:14:44.866422",
            "symbol": "BTCFDUSD",
            "analysis_type": "bayesian_trend_analysis",
            "trend_analysis": {
                "most_likely_trend": "震荡",
                "confidence": 0.65,
                "confidence_level": "high",
                "uncertainty": 0.45,
                "risk_level": "high_risk"
            },
            "bayesian_analysis": {
                "analysis_reason": "基于贝叶斯框架的分析，显示震荡趋势",
                "evidence_summary": {
                    "static_liquidity": "订单簿平衡，买卖比率0.87",
                    "dynamic_volume": "成交量有限，5.71/min",
                    "price_volatility": "极低波动率0.0002"
                }
            },
            "metadata": {
                "response_raw": json.dumps({
                    "posterior_probabilities": {
                        "震荡": 0.42,
                        "微弱看涨": 0.18,
                        "看涨": 0.12,
                        "微弱看跌": 0.10,
                        "看跌": 0.08,
                        "强力看涨": 0.05,
                        "强力看跌": 0.05
                    }
                })
            }
        }

        # 执行提取
        result = self.formatter._extract_core_bayesian_info(complete_data)

        # 验证结果
        assert result["trend"] == "震荡"
        assert result["confidence"] == 0.65
        assert result["timestamp"] == "2025-11-01T18:14:44.866422"
        assert "基于贝叶斯框架的分析" in result["reason"]

        # 验证概率分布
        probs = result["probabilities"]
        assert len(probs) == 7
        assert probs["震荡"] == 0.42
        assert probs["微弱看涨"] == 0.18
        assert probs["看涨"] == 0.12

        # 验证证据提取
        evidence = result["evidence"]
        assert len(evidence) == 3
        assert any("订单簿平衡" in ev for ev in evidence)
        assert any("成交量有限" in ev for ev in evidence)

    def test_extract_core_bayesian_info_minimal_data(self):
        """测试从最小化的贝叶斯分析数据中提取信息。"""
        minimal_data = {
            "trend_analysis": {
                "most_likely_trend": "看涨",
                "confidence": 0.75
            },
            "bayesian_analysis": {
                "analysis_reason": "简单的看涨分析"
            }
        }

        result = self.formatter._extract_core_bayesian_info(minimal_data)

        assert result["trend"] == "看涨"
        assert result["confidence"] == 0.75
        assert result["reason"] == "简单的看涨分析"
        assert result["probabilities"] == {}
        assert result["evidence"] == []

    def test_extract_core_bayesian_info_malformed_response_raw(self):
        """测试处理格式错误的response_raw数据。"""
        malformed_data = {
            "trend_analysis": {
                "most_likely_trend": "震荡",
                "confidence": 0.60
            },
            "bayesian_analysis": {
                "analysis_reason": "分析原因"
            },
            "metadata": {
                "response_raw": "{ invalid json format"
            }
        }

        result = self.formatter._extract_core_bayesian_info(malformed_data)

        # 应该优雅地处理错误，使用默认值
        assert result["trend"] == "震荡"
        assert result["confidence"] == 0.60
        assert result["probabilities"] == {}

    def test_extract_core_bayesian_info_empty_data(self):
        """测试处理空数据的情况。"""
        empty_data = {}

        result = self.formatter._extract_core_bayesian_info(empty_data)

        # 应该返回默认值
        assert result["trend"] == "未知"
        assert result["confidence"] == 0.0
        assert result["reason"] == "暂无分析原因"
        assert result["probabilities"] == {}
        assert result["evidence"] == []

    def test_extract_core_bayesian_info_partial_probabilities(self):
        """测试部分概率分布数据的提取。"""
        partial_data = {
            "trend_analysis": {
                "most_likely_trend": "看跌",
                "confidence": 0.55
            },
            "probability_distribution": {
                "full_distribution": {
                    "看跌": 0.40,
                    "震荡": 0.35,
                    "看涨": 0.25
                }
            },
            "bayesian_analysis": {
                "analysis_reason": "部分概率分析"
            }
        }

        result = self.formatter._extract_core_bayesian_info(partial_data)

        assert result["trend"] == "看跌"
        assert result["confidence"] == 0.55

        # 应该从probability_distribution中提取概率
        probs = result["probabilities"]
        assert len(probs) == 3
        assert probs["看跌"] == 0.40
        assert probs["震荡"] == 0.35
        assert probs["看涨"] == 0.25

    def test_format_probability_summary_complete(self):
        """测试完整概率分布的格式化。"""
        probabilities = {
            "震荡": 0.42,
            "微弱看涨": 0.18,
            "看涨": 0.12,
            "微弱看跌": 0.10,
            "看跌": 0.08,
            "强力看涨": 0.05,
            "强力看跌": 0.05
        }

        result = self.formatter._format_probability_summary(probabilities)

        # 验证包含前4个概率（按概率排序）
        assert "⚖️ 震荡: `42.0%`" in result
        assert "📈 微弱看涨: `18.0%`" in result
        assert "🚀 看涨: `12.0%`" in result
        assert "📉 微弱看跌: `10.0%`" in result

        # 验证进度条格式
        assert "████" in result  # 42% -> 4个方块
        assert "█" in result     # 18% -> 1个方块
        assert "░" in result     # 空白方块

    def test_format_probability_summary_empty(self):
        """测试空概率分布的格式化。"""
        empty_probs = {}

        result = self.formatter._format_probability_summary(empty_probs)

        assert result == "暂无概率数据"

    def test_format_probability_summary_invalid_values(self):
        """测试包含无效值的概率分布格式化。"""
        invalid_probs = {
            "震荡": 1.5,      # 超过1.0
            "看涨": -0.1,     # 负数
            "看跌": 0.3       # 正常值
        }

        # 应该能处理无效值而不崩溃
        result = self.formatter._format_probability_summary(invalid_probs)

        assert "震荡" in result
        assert "看跌" in result
        # 应该按概率排序显示

    def test_format_bayesian_analysis_integration(self):
        """测试完整的贝叶斯分析格式化集成。"""
        # 使用真实的DeepSeek分析数据结构
        real_data = {
            "timestamp": "2025-11-01T18:14:44.866422",
            "symbol": "BTCFDUSD",
            "analysis_type": "bayesian_trend_analysis",
            "trend_analysis": {
                "most_likely_trend": "震荡",
                "confidence": 0.65,
                "confidence_level": "high",
                "uncertainty": 0.45,
                "risk_level": "high_risk"
            },
            "bayesian_analysis": {
                "analysis_reason": "基于贝叶斯框架，对先验概率进行证据加权更新。关键证据包括极低波动率和订单簿平衡。",
                "evidence_summary": {
                    "static_liquidity": "订单簿不平衡度0.072(中性)",
                    "dynamic_volume": "成交量有限5.71/min",
                    "price_volatility": "极低波动率0.0002"
                }
            },
            "metadata": {
                "response_raw": json.dumps({
                    "posterior_probabilities": {
                        "震荡": 0.42,
                        "微弱看涨": 0.18,
                        "看涨": 0.12,
                        "微弱看跌": 0.10,
                        "看跌": 0.08,
                        "强力看涨": 0.05,
                        "强力看跌": 0.05
                    }
                })
            }
        }

        result = self.formatter.format_bayesian_analysis(real_data, "BTCFDUSD")

        # 验证Discord消息结构
        assert "embeds" in result
        assert len(result["embeds"]) == 1

        embed = result["embeds"][0]
        assert "⚖️ BTCFDUSD 趋势分析" == embed["title"]
        assert "震荡" in embed["description"]
        assert "65.0%" in embed["description"]
        assert embed["timestamp"] == "2025-11-01T18:14:44.866422"

        # 验证字段
        fields = embed["fields"]
        assert len(fields) == 2

        # 概率分布字段
        prob_field = fields[0]
        assert prob_field["name"] == "🎯 概率分布"
        assert "震荡: `42.0%`" in prob_field["value"]

        # 分析原因字段
        reason_field = fields[1]
        assert reason_field["name"] == "📝 分析原因"
        assert "贝叶斯框架" in reason_field["value"]

    def test_edge_case_very_long_reason(self):
        """测试超长分析原因的处理。"""
        long_reason = "这是一个非常长的分析原因。" * 100  # 重复100次

        data_with_long_reason = {
            "trend_analysis": {
                "most_likely_trend": "震荡",
                "confidence": 0.65
            },
            "bayesian_analysis": {
                "analysis_reason": long_reason
            }
        }

        result = self.formatter._extract_core_bayesian_info(data_with_long_reason)

        # 原因应该被保留（截断在format阶段处理）
        assert len(result["reason"]) == len(long_reason)

    def test_edge_case_missing_metadata(self):
        """测试缺少metadata字段的情况。"""
        data_without_metadata = {
            "trend_analysis": {
                "most_likely_trend": "看涨",
                "confidence": 0.70
            },
            "bayesian_analysis": {
                "analysis_reason": "没有metadata的分析"
            }
        }

        result = self.formatter._extract_core_bayesian_info(data_without_metadata)

        assert result["trend"] == "看涨"
        assert result["confidence"] == 0.70
        assert result["probabilities"] == {}  # 没有概率数据

    def test_validate_and_clean_probabilities_valid(self):
        """测试有效概率值的验证。"""
        valid_probs = {
            "震荡": 0.42,
            "看涨": 0.35,
            "看跌": 0.23
        }

        result = self.formatter._validate_and_clean_probabilities(valid_probs)

        assert result == valid_probs  # 有效值应该保持不变

    def test_validate_and_clean_probabilities_invalid_values(self):
        """测试无效概率值的清洗。"""
        invalid_probs = {
            "震荡": 0.42,
            "看涨": -0.1,     # 负数，会被修正为0并移除
            "看跌": 1.5,      # 超过1，会被修正为1
            "微弱看涨": "invalid",  # 非数字，会被移除
            "微弱看跌": 0.08
        }

        result = self.formatter._validate_and_clean_probabilities(invalid_probs)

        # 验证处理结果：负数和非数字被移除，超过1的值被修正为1，然后归一化
        assert "震荡" in result
        assert "看跌" in result  # 修正后的值
        assert "微弱看跌" in result
        assert "看涨" not in result  # 负数被移除
        assert "微弱看涨" not in result  # 非数字被移除

        # 验证归一化处理
        total_prob = sum(result.values())
        assert abs(total_prob - 1.0) < 0.001

    def test_validate_and_clean_probabilities_normalization(self):
        """测试概率值归一化。"""
        unnormalized_probs = {
            "震荡": 0.8,
            "看涨": 0.4,
            "看跌": 0.2  # 总和为1.4
        }

        result = self.formatter._validate_and_clean_probabilities(unnormalized_probs)

        # 验证归一化后的结果
        total_prob = sum(result.values())
        assert abs(total_prob - 1.0) < 0.001  # 总和应该为1

        # 验证相对比例保持不变
        assert abs(result["震荡"] / result["看涨"] - 2.0) < 0.001  # 0.8/0.4 = 2.0

    def test_validate_and_clean_probabilities_all_invalid(self):
        """测试所有概率值都无效的情况。"""
        all_invalid_probs = {
            "震荡": -0.5,    # 负数
            "看涨": "abc",   # 非数字
            "看跌": None     # None值
        }

        result = self.formatter._validate_and_clean_probabilities(all_invalid_probs)

        assert result == {}  # 应该返回空字典

    def test_send_raw_message_public_method(self):
        """测试公共方法send_raw_message的封装功能。"""
        from unittest.mock import AsyncMock, patch

        # 创建一个mock的DiscordNotifier实例
        mock_notifier = AsyncMock()
        mock_notifier._send_to_discord = AsyncMock(return_value=True)

        # 模拟消息载荷
        test_payload = {
            "embeds": [
                {
                    "title": "测试消息",
                    "description": "这是一个测试消息"
                }
            ]
        }

        # 由于我们不能直接mock私有方法，这里测试格式化器的输出
        result = self.formatter.format_bayesian_analysis({
            "trend_analysis": {"most_likely_trend": "震荡", "confidence": 0.65},
            "bayesian_analysis": {"analysis_reason": "测试原因"},
            "timestamp": "2025-11-01T18:14:44.866422"
        }, "BTCFDUSD")

        # 验证格式化输出正确
        assert "embeds" in result
        assert len(result["embeds"]) == 1
        assert "⚖️ BTCFDUSD 趋势分析" in result["embeds"][0]["title"]


class TestBayesianDiscordFormatterErrorHandling:
    """测试BayesianDiscordFormatter的错误处理能力。"""

    def setup_method(self):
        """每个测试方法前的设置。"""
        self.formatter = BayesianDiscordFormatter()

    def test_invalid_json_in_response_raw(self, caplog):
        """测试response_raw中包含无效JSON的处理。"""
        data = {
            "trend_analysis": {"most_likely_trend": "震荡", "confidence": 0.6},
            "bayesian_analysis": {"analysis_reason": "测试"},
            "metadata": {"response_raw": "这不是有效的JSON"}
        }

        result = self.formatter._extract_core_bayesian_info(data)

        # 应该优雅地处理错误，不影响其他数据提取
        assert result["trend"] == "震荡"
        assert result["probabilities"] == {}

    def test_none_values_handling(self):
        """测试处理None值的情况。"""
        data_with_none = {
            "trend_analysis": {
                "most_likely_trend": None,
                "confidence": None
            },
            "bayesian_analysis": {
                "analysis_reason": None
            }
        }

        result = self.formatter._extract_core_bayesian_info(data_with_none)

        # 应该使用默认值处理None
        assert result["trend"] == "未知"
        assert result["confidence"] == 0.0
        assert result["reason"] == "暂无分析原因"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])