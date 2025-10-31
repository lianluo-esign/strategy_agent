"""响应格式化器单元测试。"""

import json
import pytest
from datetime import datetime

from src.core.agent_analyzer_optimized.response_formatter import (
    ResponseFormatter,
    ResponseValidator
)


class TestResponseFormatter:
    """响应格式化器测试类。"""

    def setup_method(self):
        """每个测试方法前的设置。"""
        self.formatter = ResponseFormatter(
            include_metadata=True,
            pretty_print=False
        )

    def test_initialization(self):
        """测试初始化。"""
        formatter = ResponseFormatter(
            include_metadata=False,
            pretty_print=True
        )
        assert not formatter.include_metadata
        assert formatter.pretty_print

    def test_format_analysis_response_valid_data(self):
        """测试有效分析结果格式化。"""
        trend_result = {
            "timestamp": "2025-01-01T12:00:00",
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

        aggregated_data = {
            "total_volume": 1000.0,
            "trade_count": 500,
            "price_levels_count": 20
        }

        result = self.formatter.format_analysis_response(
            trend_result, aggregated_data, "BTCFDUSD"
        )

        # 验证返回的是有效JSON
        parsed = json.loads(result)
        assert parsed["trend"] == "看涨"
        assert parsed["confidence"] == 0.85
        assert "metadata" in parsed

    def test_format_analysis_response_empty_data(self):
        """测试空数据分析结果格式化。"""
        with pytest.raises(ValueError, match="趋势分析结果不能为空"):
            self.formatter.format_analysis_response(None)

        with pytest.raises(ValueError, match="趋势分析结果不能为空"):
            self.formatter.format_analysis_response({})

    def test_format_analysis_response_invalid_trend(self):
        """测试无效趋势类型处理。"""
        trend_result = {
            "trend": "invalid_trend",
            "strength_levels": {},
            "reason": "test",
            "confidence": 0.5
        }

        result = self.formatter.format_analysis_response(trend_result)
        parsed = json.loads(result)
        # 应该使用默认值"震荡"
        assert parsed["trend"] == "震荡"

    def test_format_analysis_response_invalid_strength_levels(self):
        """测试无效强度等级处理。"""
        trend_result = {
            "trend": "震荡",
            "strength_levels": {
                "strong_support": "invalid",  # 无效值
                "weak_support": 1.5,         # 超出范围
                # 缺少其他必需等级
            },
            "reason": "test",
            "confidence": 0.5
        }

        result = self.formatter.format_analysis_response(trend_result)
        parsed = json.loads(result)

        # 验证强度等级被正确处理
        strength_levels = parsed["strength_levels"]
        assert strength_levels["strong_support"] == 0.0  # 无效值设为0
        assert strength_levels["weak_support"] == 1.0  # 超出范围的值被限制
        assert strength_levels["strong_resistance"] == 0.0  # 缺失的等级设为0
        assert strength_levels["weak_resistance"] == 0.0   # 缺失的等级设为0

    def test_format_analysis_response_invalid_confidence(self):
        """测试无效置信度处理。"""
        trend_result = {
            "trend": "震荡",
            "strength_levels": {},
            "reason": "test",
            "confidence": "invalid"  # 无效置信度
        }

        result = self.formatter.format_analysis_response(trend_result)
        parsed = json.loads(result)
        # 应该使用默认值0.5
        assert parsed["confidence"] == 0.5

    def test_format_analysis_response_without_metadata(self):
        """测试不包含元数据的格式化。"""
        formatter = ResponseFormatter(include_metadata=False)

        trend_result = {
            "trend": "看涨",
            "strength_levels": {},
            "reason": "test",
            "confidence": 0.8
        }

        result = formatter.format_analysis_response(trend_result)
        parsed = json.loads(result)
        assert "metadata" not in parsed

    def test_format_compact_response(self):
        """测试紧凑响应格式化。"""
        trend_result = {
            "trend": "看跌",
            "strength_levels": {
                "strong_support": 0.7,
                "weak_support": 0.5,
                "strong_resistance": 0.8,
                "weak_resistance": 0.6
            },
            "reason": "测试原因",
            "confidence": 0.75
        }

        result = self.formatter.format_compact_response(trend_result, "BTCFDUSD")
        parsed = json.loads(result)

        # 紧凑格式应该包含核心字段但不包含元数据
        assert parsed["trend"] == "看跌"
        assert parsed["confidence"] == 0.75
        assert "metadata" not in parsed

    def test_validate_response_schema_valid(self):
        """测试有效响应Schema验证。"""
        valid_response = json.dumps({
            "timestamp": "2025-01-01T12:00:00",
            "trend": "看涨",
            "strength_levels": {
                "strong_support": 0.8,
                "weak_support": 0.6,
                "strong_resistance": 0.4,
                "weak_resistance": 0.3
            },
            "reason": "有效的分析原因",
            "confidence": 0.85
        })

        assert self.formatter.validate_response_schema(valid_response)

    def test_validate_response_schema_missing_fields(self):
        """测试缺少必需字段的响应验证。"""
        invalid_response = json.dumps({
            "trend": "看涨",
            # 缺少其他必需字段
        })

        assert not self.formatter.validate_response_schema(invalid_response)

    def test_validate_response_schema_invalid_json(self):
        """测试无效JSON格式验证。"""
        invalid_json = "invalid json string"
        assert not self.formatter.validate_response_schema(invalid_json)

    def test_validate_response_schema_invalid_trend(self):
        """测试无效趋势类型验证。"""
        invalid_response = json.dumps({
            "timestamp": "2025-01-01T12:00:00",
            "trend": "invalid_trend",
            "strength_levels": {
                "strong_support": 0.5,
                "weak_support": 0.5,
                "strong_resistance": 0.5,
                "weak_resistance": 0.5
            },
            "reason": "test",
            "confidence": 0.5
        })

        assert not self.formatter.validate_response_schema(invalid_response)

    def test_validate_response_schema_invalid_strength_values(self):
        """测试无效强度值验证。"""
        invalid_response = json.dumps({
            "timestamp": "2025-01-01T12:00:00",
            "trend": "看涨",
            "strength_levels": {
                "strong_support": 1.5,  # 超出范围
                "weak_support": 0.5,
                "strong_resistance": 0.5,
                "weak_resistance": 0.5
            },
            "reason": "test",
            "confidence": 0.5
        })

        assert not self.formatter.validate_response_schema(invalid_response)


class TestResponseValidator:
    """响应验证器测试类。"""

    def setup_method(self):
        """每个测试方法前的设置。"""
        self.validator = ResponseValidator(strict_mode=False)

    def test_validate_analysis_quality_high_confidence(self):
        """测试高置信度分析质量验证。"""
        analysis_result = {
            "trend": "看涨",
            "strength_levels": {
                "strong_support": 0.8,
                "weak_support": 0.6,
                "strong_resistance": 0.3,
                "weak_resistance": 0.2
            },
            "reason": "这是一个详细的分析原因，解释了为什么市场呈现看涨趋势，基于成交量分析、价格动量以及市场情绪等多个因素的综合判断。",
            "confidence": 0.85
        }

        result = self.validator.validate_analysis_quality(analysis_result)
        assert result["is_valid"]
        assert len(result["errors"]) == 0

    def test_validate_analysis_quality_low_confidence_strict_mode(self):
        """测试严格模式下低置信度验证。"""
        strict_validator = ResponseValidator(strict_mode=True)

        analysis_result = {
            "trend": "看涨",
            "strength_levels": {},
            "reason": "test",
            "confidence": 0.2  # 低置信度
        }

        result = strict_validator.validate_analysis_quality(analysis_result)
        assert not result["is_valid"]
        assert any("置信度过低" in error for error in result["errors"])

    def test_validate_analysis_quality_low_confidence_normal_mode(self):
        """测试普通模式下低置信度验证。"""
        analysis_result = {
            "trend": "看涨",
            "strength_levels": {},
            "reason": "test",
            "confidence": 0.2  # 低置信度
        }

        result = self.validator.validate_analysis_quality(analysis_result)
        assert result["is_valid"]  # 普通模式下仍然有效
        assert any("置信度较低" in warning for warning in result["warnings"])

    def test_validate_analysis_quality_trend_consistency_warning(self):
        """测试趋势一致性警告。"""
        analysis_result = {
            "trend": "看涨",
            "strength_levels": {
                "strong_support": 0.3,  # 支撑强度低于阻力（应该相反）
                "strong_resistance": 0.9,  # 阻力强度高于支撑（与看涨趋势不一致）
            },
            "reason": "test reason",
            "confidence": 0.8
        }

        result = self.validator.validate_analysis_quality(analysis_result)
        # 应该产生一致性警告
        assert any("阻力强度高于支撑强度" in warning for warning in result["warnings"])

    def test_validate_analysis_quality_short_reason_warning(self):
        """测试分析原因过短警告。"""
        analysis_result = {
            "trend": "看涨",
            "strength_levels": {},
            "reason": "短",  # 过短的原因
            "confidence": 0.8
        }

        result = self.validator.validate_analysis_quality(analysis_result)
        assert any("分析原因过于简短" in warning for warning in result["warnings"])

    def test_validate_analysis_quality_all_zero_strengths(self):
        """测试所有强度为零的情况。"""
        analysis_result = {
            "trend": "震荡",
            "strength_levels": {
                "strong_support": 0.0,
                "weak_support": 0.0,
                "strong_resistance": 0.0,
                "weak_resistance": 0.0
            },
            "reason": "test reason",
            "confidence": 0.5
        }

        result = self.validator.validate_analysis_quality(analysis_result)
        assert any("强度等级都为0" in warning for warning in result["warnings"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])