#!/usr/bin/env python3
"""Discord价格分析增强功能单元测试。

测试Discord消息格式从"🎯 关键价格位置"更改为"📊 价格分析"
并验证data_statistics数据的正确格式化显示。
"""

import pytest
from unittest.mock import Mock, patch

from src.core.agent_analyzer_optimized.discord_notifier import (
    DiscordNotifier,
    BayesianDiscordFormatter,
    DiscordNotificationManager
)


class TestDiscordPriceAnalysisEnhancement:
    """测试Discord价格分析增强功能。"""

    def setup_method(self):
        """每个测试方法前的设置。"""
        self.webhook_url = "https://discord.com/api/webhooks/test/123456789"
        self.discord_notifier = DiscordNotifier(self.webhook_url)
        self.bayesian_formatter = BayesianDiscordFormatter()

    def test_format_data_statistics_for_discord_with_full_data(self):
        """测试完整数据统计信息的Discord格式化。"""
        # 准备测试数据
        analysis_result = {
            "trend": "看涨",
            "confidence": 0.75,
            "reason": "测试分析原因",
            "metadata": {
                "data_statistics": {
                    "total_volume": 1234.56,
                    "trade_count": 156,
                    "price_levels_count": 89,
                    "price_range": [67234.0, 68912.0]
                }
            }
        }

        # 执行格式化
        result = self.discord_notifier._format_data_statistics_for_discord(analysis_result)

        # 验证结果
        assert "• 总成交量: 1.23K BTC" in result
        assert "• 交易数量: 156 笔" in result
        assert "• 价格档位: 89 个" in result
        assert "• 价格区间: $67,234 - $68,912" in result

    def test_format_data_statistics_for_discord_with_small_volume(self):
        """测试小成交量数据的格式化。"""
        analysis_result = {
            "trend": "震荡",
            "confidence": 0.5,
            "metadata": {
                "data_statistics": {
                    "total_volume": 123.45,
                    "trade_count": 45,
                    "price_levels_count": 23,
                    "price_range": [65000.0, 66000.0]
                }
            }
        }

        result = self.discord_notifier._format_data_statistics_for_discord(analysis_result)

        assert "• 总成交量: 123.4500 BTC" in result
        assert "• 交易数量: 45 笔" in result
        assert "• 价格档位: 23 个" in result

    def test_format_data_statistics_for_discord_with_zero_data(self):
        """测试零数据的格式化。"""
        analysis_result = {
            "trend": "看跌",
            "confidence": 0.3,
            "metadata": {
                "data_statistics": {
                    "total_volume": 0,
                    "trade_count": 0,
                    "price_levels_count": 0,
                    "price_range": [0, 0]
                }
            }
        }

        result = self.discord_notifier._format_data_statistics_for_discord(analysis_result)

        assert "• 总成交量: 0 BTC" in result
        assert "• 交易数量: 0 笔" in result
        assert "• 价格档位: 0 个" in result
        assert "• 价格区间: 数据异常" in result

    def test_format_data_statistics_for_discord_missing_metadata(self):
        """测试缺少metadata的情况。"""
        analysis_result = {
            "trend": "震荡",
            "confidence": 0.5
        }

        result = self.discord_notifier._format_data_statistics_for_discord(analysis_result)

        assert result == "暂无统计数据"

    def test_format_data_statistics_for_discord_missing_data_statistics(self):
        """测试缺少data_statistics的情况。"""
        analysis_result = {
            "trend": "看涨",
            "confidence": 0.7,
            "metadata": {}
        }

        result = self.discord_notifier._format_data_statistics_for_discord(analysis_result)

        assert result == "暂无统计数据"

    def test_format_data_statistics_for_discord_invalid_price_range(self):
        """测试无效价格范围的处理。"""
        analysis_result = {
            "trend": "微弱看涨",
            "confidence": 0.6,
            "metadata": {
                "data_statistics": {
                    "total_volume": 500.0,
                    "trade_count": 78,
                    "price_levels_count": 45,
                    "price_range": [0, 0]  # 无效价格范围
                }
            }
        }

        result = self.discord_notifier._format_data_statistics_for_discord(analysis_result)

        assert "• 价格区间: 数据异常" in result

    def test_format_data_statistics_text_format(self):
        """测试文本格式的数据统计格式化。"""
        analysis_result = {
            "trend": "看涨",
            "confidence": 0.8,
            "metadata": {
                "data_statistics": {
                    "total_volume": 2500.0,
                    "trade_count": 234,
                    "price_levels_count": 112,
                    "price_range": [67000.0, 69000.0]
                }
            }
        }

        result = self.discord_notifier._format_data_statistics_text(analysis_result)

        assert "  • 总成交量: 2.50K BTC" in result
        assert "  • 交易数量: 234 笔" in result
        assert "  • 价格档位: 112 个" in result
        assert "  • 价格区间: $67,000 - $69,000" in result

    def test_discord_message_field_name_change(self):
        """测试Discord消息字段名称从'关键价格位置'更改为'价格分析'。"""
        # 准备完整的分析结果
        analysis_result = {
            "trend": "看涨",
            "confidence": 0.75,
            "reason": "市场表现强劲",
            "metadata": {
                "data_statistics": {
                    "total_volume": 1500.0,
                    "trade_count": 180,
                    "price_levels_count": 95,
                    "price_range": [67500.0, 68800.0]
                }
            }
        }

        # 格式化Discord消息
        message_payload = self.discord_notifier._format_discord_message(analysis_result, "BTCFDUSD")

        # 验证字段名称已更改
        embeds = message_payload["embeds"]
        fields = embeds[0]["fields"]

        # 查找价格分析字段
        price_analysis_field = None
        for field in fields:
            if field["name"] == "📊 价格分析":
                price_analysis_field = field
                break

        assert price_analysis_field is not None, "未找到'📊 价格分析'字段"
        assert "📊 价格分析" in price_analysis_field["name"]

        # 验证字段内容包含数据统计信息
        field_value = price_analysis_field["value"]
        assert "总成交量" in field_value
        assert "交易数量" in field_value
        assert "价格档位" in field_value
        assert "价格区间" in field_value

    def test_bayesian_format_data_statistics(self):
        """测试贝叶斯格式化器的数据统计格式化。"""
        analysis_data = {
            "trend_analysis": {
                "most_likely_trend": "看涨",
                "confidence": 0.82
            },
            "metadata": {
                "data_statistics": {
                    "total_volume": 3200.0,
                    "trade_count": 412,
                    "price_levels_count": 156,
                    "price_range": [67200.0, 68900.0]
                }
            }
        }

        result = self.bayesian_formatter._format_data_statistics_for_bayesian(analysis_data)

        assert "**成交量**: 3.2K BTC" in result
        assert "**交易数**: 412" in result
        assert "**档位数**: 156" in result
        assert "**区间**: $67,200-$68,900" in result

    def test_bayesian_format_data_statistics_small_values(self):
        """测试贝叶斯格式化器小数值的处理。"""
        analysis_data = {
            "metadata": {
                "data_statistics": {
                    "total_volume": 567.89,
                    "trade_count": 89,
                    "price_levels_count": 34,
                    "price_range": [66100.0, 66500.0]
                }
            }
        }

        result = self.bayesian_formatter._format_data_statistics_for_bayesian(analysis_data)

        assert "**成交量**: 567.89 BTC" in result
        assert "**交易数**: 89" in result
        assert "**档位数**: 34" in result

    def test_bayesian_discord_message_field_name_change(self):
        """测试贝叶斯Discord消息字段名称更改。"""
        analysis_data = {
            "trend_analysis": {
                "most_likely_trend": "强力看涨",
                "confidence": 0.91
            },
            "metadata": {
                "data_statistics": {
                    "total_volume": 4500.0,
                    "trade_count": 520,
                    "price_levels_count": 201,
                    "price_range": [68000.0, 69500.0]
                }
            }
        }

        # 格式化贝叶斯分析消息
        message_payload = self.bayesian_formatter.format_bayesian_analysis(analysis_data, "BTCFDUSD")

        # 验证字段名称已更改
        embeds = message_payload["embeds"]
        fields = embeds[0]["fields"]

        # 查找价格分析字段
        price_analysis_field = None
        for field in fields:
            if field["name"] == "📊 价格分析":
                price_analysis_field = field
                break

        assert price_analysis_field is not None, "贝叶斯消息中未找到'📊 价格分析'字段"
        assert "📊 价格分析" in price_analysis_field["name"]

        # 验证字段内容包含贝叶斯格式的数据统计信息
        field_value = price_analysis_field["value"]
        assert "**成交量**" in field_value
        assert "**交易数**" in field_value
        assert "**档位数**" in field_value
        assert "**区间**" in field_value

    def test_error_handling_malformed_data(self):
        """测试异常数据处理。"""
        # 测试非数值的total_volume
        analysis_result = {
            "metadata": {
                "data_statistics": {
                    "total_volume": "invalid",  # 非数值
                    "trade_count": 100,
                    "price_levels_count": 50,
                    "price_range": [65000, 66000]
                }
            }
        }

        result = self.discord_notifier._format_data_statistics_for_discord(analysis_result)

        # 应该有降级处理，不抛出异常
        assert isinstance(result, str)
        assert len(result) > 0

    def test_discord_notification_manager_integration(self):
        """测试DiscordNotificationManager集成。"""
        manager = DiscordNotificationManager(self.webhook_url)

        analysis_result = {
            "trend_analysis": {
                "most_likely_trend": "看涨",
                "confidence": 0.78
            },
            "metadata": {
                "data_statistics": {
                    "total_volume": 1800.0,
                    "trade_count": 245,
                    "price_levels_count": 123,
                    "price_range": [67300.0, 68700.0]
                }
            }
        }

        # 测试_is_bayesian_analysis方法
        is_bayesian = manager._is_bayesian_analysis(analysis_result)
        assert is_bayesian == True

    def test_bayesian_analysis_message_format_structure(self):
        """测试贝叶斯分析消息格式结构（简化版端到端测试）。"""
        analysis_result = {
            "trend_analysis": {
                "most_likely_trend": "看涨",
                "confidence": 0.83
            },
            "metadata": {
                "data_statistics": {
                    "total_volume": 2100.0,
                    "trade_count": 289,
                    "price_levels_count": 145,
                    "price_range": [67400.0, 68600.0]
                }
            }
        }

        # 创建管理器
        manager = DiscordNotificationManager(self.webhook_url)

        # 测试贝叶斯分析识别
        is_bayesian = manager._is_bayesian_analysis(analysis_result)
        assert is_bayesian == True

        # 测试格式化结果（不实际发送）
        formatted_message = manager.bayesian_formatter.format_bayesian_analysis(analysis_result, "BTCFDUSD")

        # 验证消息结构
        assert "embeds" in formatted_message
        embeds = formatted_message["embeds"]
        assert len(embeds) > 0

        # 验证字段包含价格分析
        fields = embeds[0]["fields"]
        price_analysis_field = None
        for field in fields:
            if field["name"] == "📊 价格分析":
                price_analysis_field = field
                break

        assert price_analysis_field is not None
        assert "**成交量**" in price_analysis_field["value"]
        assert "**交易数**" in price_analysis_field["value"]


class TestPerformanceRequirements:
    """测试性能要求。"""

    def setup_method(self):
        """设置测试环境。"""
        self.discord_notifier = DiscordNotifier("https://discord.com/api/webhooks/test/123")
        self.bayesian_formatter = BayesianDiscordFormatter()

    def test_formatting_performance_under_50ms(self):
        """测试格式化性能在50ms以内。"""
        import time

        analysis_result = {
            "trend": "看涨",
            "confidence": 0.75,
            "metadata": {
                "data_statistics": {
                    "total_volume": 1500.0,
                    "trade_count": 200,
                    "price_levels_count": 100,
                    "price_range": [67000.0, 69000.0]
                }
            }
        }

        # 测试DiscordNotifier格式化性能
        start_time = time.time()
        result1 = self.discord_notifier._format_data_statistics_for_discord(analysis_result)
        discord_time = time.time() - start_time

        # 测试BayesianDiscordFormatter格式化性能
        start_time = time.time()
        result2 = self.bayesian_formatter._format_data_statistics_for_bayesian(analysis_result)
        bayesian_time = time.time() - start_time

        # 验证性能要求（<50ms）
        assert discord_time < 0.05, f"Discord格式化耗时 {discord_time*1000:.1f}ms，超过50ms限制"
        assert bayesian_time < 0.05, f"贝叶斯格式化耗时 {bayesian_time*1000:.1f}ms，超过50ms限制"

        # 验证结果不为空
        assert len(result1) > 0
        assert len(result2) > 0


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v", "--tb=short"])