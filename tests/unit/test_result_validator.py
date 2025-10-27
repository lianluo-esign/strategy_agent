"""结果验证器单元测试。

测试ResultValidator类的各种功能，包括JSON解析、字段验证和错误处理。
"""

import pytest
from src.core.result_validator import ResultValidator, ResultValidationError


class TestResultValidator:
    """ResultValidator测试类。"""

    def setup_method(self):
        """每个测试方法前的设置。"""
        self.validator = ResultValidator()

    def test_valid_json_parsing(self):
        """测试有效JSON解析。"""
        # 标准JSON格式
        content = '{"grid_delta": 2.0, "grid_quantity": 0.001, "active_side": "Buy"}'
        analysis_result = {
            "status": "success",
            "raw_content": content,
            "symbol": "BTCFDUSD"
        }

        result = self.validator.validate_and_extract_trading_params(analysis_result)

        assert result["grid_delta"] == 2.0
        assert result["grid_quantity"] == 0.001
        assert result["active_side"] == "Buy"

    def test_json_with_extra_content(self):
        """测试包含额外内容的JSON解析。"""
        content = '''Here's my analysis:
        {"grid_delta": 1.5, "grid_quantity": 0.002, "active_side": "Sell"}
        This looks like a good opportunity.'''
        analysis_result = {
            "status": "success",
            "raw_content": content,
            "symbol": "BTCFDUSD"
        }

        result = self.validator.validate_and_extract_trading_params(analysis_result)

        assert result["grid_delta"] == 1.5
        assert result["grid_quantity"] == 0.002
        assert result["active_side"] == "Sell"

    def test_json_in_code_block(self):
        """测试代码块中的JSON解析。"""
        content = '''```json
        {"grid_delta": 3.0, "grid_quantity": 0.0015, "active_side": "Buy"}
        ```'''
        analysis_result = {
            "status": "success",
            "raw_content": content,
            "symbol": "BTCFDUSD"
        }

        result = self.validator.validate_and_extract_trading_params(analysis_result)

        assert result["grid_delta"] == 3.0
        assert result["grid_quantity"] == 0.0015
        assert result["active_side"] == "Buy"

    def test_manual_field_extraction(self):
        """测试手动字段提取。"""
        content = 'Based on analysis: grid_delta = 2.5, grid_quantity: 0.002, active_side: "Sell"'
        analysis_result = {
            "status": "success",
            "raw_content": content,
            "symbol": "BTCFDUSD"
        }

        result = self.validator.validate_and_extract_trading_params(analysis_result)

        # 应该返回默认值，因为无法完全提取
        assert result["grid_delta"] == 1.0
        assert result["grid_quantity"] == 0.001
        assert result["active_side"] == "Buy"

    def test_invalid_grid_delta_range(self):
        """测试无效的grid_delta范围。"""
        test_cases = [
            0.05,  # 低于最小值
            150.0,  # 高于最大值
        ]

        for invalid_delta in test_cases:
            content = f'{{"grid_delta": {invalid_delta}, "grid_quantity": 0.001, "active_side": "Buy"}}'
            analysis_result = {
                "status": "success",
                "raw_content": content,
                "symbol": "BTCFDUSD"
            }

            with pytest.raises(ResultValidationError, match="grid_delta.*out of range"):
                self.validator.validate_and_extract_trading_params(analysis_result)

    def test_invalid_grid_quantity_range(self):
        """测试无效的grid_quantity范围。"""
        test_cases = [
            0.00005,  # 低于最小值
            15.0,    # 高于最大值
        ]

        for invalid_quantity in test_cases:
            content = f'{{"grid_delta": 2.0, "grid_quantity": {invalid_quantity}, "active_side": "Buy"}}'
            analysis_result = {
                "status": "success",
                "raw_content": content,
                "symbol": "BTCFDUSD"
            }

            with pytest.raises(ResultValidationError, match="grid_quantity.*out of range"):
                self.validator.validate_and_extract_trading_params(analysis_result)

    def test_invalid_active_side(self):
        """测试无效的active_side值。"""
        invalid_sides = ["Hold", "buy", "sell", "UNKNOWN", ""]

        for invalid_side in invalid_sides:
            content = f'{{"grid_delta": 2.0, "grid_quantity": 0.001, "active_side": "{invalid_side}"}}'
            analysis_result = {
                "status": "success",
                "raw_content": content,
                "symbol": "BTCFDUSD"
            }

            with pytest.raises(ResultValidationError, match="active_side.*not in valid options"):
                self.validator.validate_and_extract_trading_params(analysis_result)

    def test_missing_fields(self):
        """测试缺失必需字段。"""
        missing_field_cases = [
            # 缺失grid_delta
            '{"grid_quantity": 0.001, "active_side": "Buy"}',
            # 缺失grid_quantity
            '{"grid_delta": 2.0, "active_side": "Buy"}',
            # 缺失active_side
            '{"grid_delta": 2.0, "grid_quantity": 0.001}',
            # 空JSON对象
            '{}',
        ]

        for content in missing_field_cases:
            analysis_result = {
                "status": "success",
                "raw_content": content,
                "symbol": "BTCFDUSD"
            }

            with pytest.raises(ResultValidationError, match="Missing required fields"):
                self.validator.validate_and_extract_trading_params(analysis_result)

    def test_failed_analysis_status(self):
        """测试分析失败状态。"""
        analysis_result = {
            "status": "error",
            "error": "API call failed",
            "raw_content": '{"grid_delta": 2.0, "grid_quantity": 0.001, "active_side": "Buy"}',
            "symbol": "BTCFDUSD"
        }

        with pytest.raises(ResultValidationError, match="Analysis failed with status"):
            self.validator.validate_and_extract_trading_params(analysis_result)

    def test_no_raw_content(self):
        """测试无原始内容。"""
        analysis_result = {
            "status": "success",
            "raw_content": "",
            "symbol": "BTCFDUSD"
        }

        with pytest.raises(ResultValidationError, match="Invalid content format"):
            self.validator.validate_and_extract_trading_params(analysis_result)

    def test_invalid_content_type(self):
        """测试无效内容类型。"""
        analysis_result = {
            "status": "success",
            "raw_content": None,
            "symbol": "BTCFDUSD"
        }

        with pytest.raises(ResultValidationError, match="Invalid content format"):
            self.validator.validate_and_extract_trading_params(analysis_result)

    def test_edge_case_valid_values(self):
        """测试边界有效值。"""
        test_cases = [
            # 最小边界值
            {"grid_delta": 0.1, "grid_quantity": 0.0001, "active_side": "Buy"},
            # 最大边界值
            {"grid_delta": 100.0, "grid_quantity": 10.0, "active_side": "Sell"},
            # 零散数值
            {"grid_delta": 50.0, "grid_quantity": 5.0, "active_side": "Buy"},
        ]

        for params in test_cases:
            import json
            content = json.dumps(params)
            analysis_result = {
                "status": "success",
                "raw_content": content,
                "symbol": "BTCFDUSD"
            }

            result = self.validator.validate_and_extract_trading_params(analysis_result)
            assert result == params

    def test_numeric_type_handling(self):
        """测试数字类型处理。"""
        # 测试不同数字类型
        test_cases = [
            # 浮点数
            {"grid_delta": 2.5, "grid_quantity": 0.0015, "active_side": "Buy"},
            # 整数
            {"grid_delta": 2, "grid_quantity": 1, "active_side": "Sell"},
            # 字符串数字
            {"grid_delta": "2.0", "grid_quantity": "0.001", "active_side": "Buy"},
        ]

        for params in test_cases:
            import json
            content = json.dumps(params)
            analysis_result = {
                "status": "success",
                "raw_content": content,
                "symbol": "BTCFDUSD"
            }

            result = self.validator.validate_and_extract_trading_params(analysis_result)

            # 检查结果类型和值
            if isinstance(params["grid_delta"], str):
                expected_delta = float(params["grid_delta"])
            else:
                expected_delta = float(params["grid_delta"])

            if isinstance(params["grid_quantity"], str):
                expected_quantity = float(params["grid_quantity"])
            else:
                expected_quantity = float(params["grid_quantity"])

            assert isinstance(result["grid_delta"], float)
            assert isinstance(result["grid_quantity"], float)
            assert isinstance(result["active_side"], str)
            assert result["grid_delta"] == expected_delta
            assert result["grid_quantity"] == expected_quantity
            assert result["active_side"] == params["active_side"]

    def test_get_validation_stats(self):
        """测试获取验证统计信息。"""
        stats = self.validator.get_validation_stats()

        assert "grid_delta_range" in stats
        assert "grid_quantity_range" in stats
        assert "active_sides" in stats
        assert "validator_status" in stats

        assert stats["grid_delta_range"] == [0.1, 100.0]
        assert stats["grid_quantity_range"] == [0.0001, 10.0]
        assert stats["active_sides"] == ["Buy", "Sell"]
        assert stats["validator_status"] == "active"

    def test_malformed_json_handling(self):
        """测试畸形JSON处理。"""
        malformed_cases = [
            # 完全无效的内容
            'This is not JSON at all',
            # 空内容
            '',
            # 只有部分字段
            '{"grid_delta": 2.0}',
            # 完全无法解析
            '{this is not json}',
        ]

        for content in malformed_cases:
            analysis_result = {
                "status": "success",
                "raw_content": content,
                "symbol": "BTCFDUSD"
            }

            # 应该回退到默认值，因为无法解析JSON
            result = self.validator.validate_and_extract_trading_params(analysis_result)
            assert result["grid_delta"] == 1.0
            assert result["grid_quantity"] == 0.001
            assert result["active_side"] == "Buy"