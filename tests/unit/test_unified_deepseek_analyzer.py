"""统一DeepSeek分析器的单元测试。

这个模块测试UnifiedDeepSeekAnalyzer的核心功能：
1. 初始化和配置
2. 数据格式化和提示词生成
3. API请求处理
4. 响应解析
"""

import json
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from src.core.unified_deepseek_analyzer import UnifiedDeepSeekAnalyzer


class TestUnifiedDeepSeekAnalyzer:
    """统一DeepSeek分析器测试类。"""

    def setup_method(self):
        """每个测试方法前的设置。"""
        self.api_key = "test_api_key"
        self.analyzer = UnifiedDeepSeekAnalyzer(
            api_key=self.api_key,
            model="deepseek-chat",
            max_tokens=4000,
            temperature=0.1,
            timeout=60,
        )

    def teardown_method(self):
        """每个测试方法后的清理。"""
        if self.analyzer:
            self.analyzer.close()

    def test_initialization_with_valid_config(self):
        """测试使用有效配置的初始化。"""
        analyzer = UnifiedDeepSeekAnalyzer(
            api_key="valid_key",
            base_url="https://api.test.com/v1",
            model="test-model",
            max_tokens=2000,
            temperature=0.5,
            timeout=30,
            max_retries=5,
        )

        assert analyzer.api_key == "valid_key"
        assert analyzer.base_url == "https://api.test.com/v1"
        assert analyzer.model == "test-model"
        assert analyzer.max_tokens == 2000
        assert analyzer.temperature == 0.5
        assert analyzer.timeout == 30
        assert analyzer.max_retries == 5
        assert analyzer.client is not None

        analyzer.close()

    def test_initialization_with_empty_api_key_raises_error(self):
        """测试空API密钥应引发错误。"""
        with pytest.raises(ValueError, match="DeepSeek API密钥是必需的"):
            UnifiedDeepSeekAnalyzer(api_key="")

        with pytest.raises(ValueError, match="DeepSeek API密钥是必需的"):
            UnifiedDeepSeekAnalyzer(api_key=None)

    def test_get_price_level_classification(self):
        """测试价格等级分类功能。"""
        # 测试不同价格等级
        assert self.analyzer._get_price_level(150000) == "十万价位"
        assert self.analyzer._get_price_level(50000) == "万价位"
        assert self.analyzer._get_price_level(5000) == "千价位"
        assert self.analyzer._get_price_level(500) == "百价位"
        assert self.analyzer._get_price_level(50) == "十价位以下"

    def test_create_unified_analysis_prompt_structure(self):
        """测试统一分析提示词的创建结构。"""
        # 准备测试数据
        aggregated_bids = {Decimal('100000.00'): Decimal('10.5'), Decimal('99000.00'): Decimal('8.2')}
        aggregated_asks = {Decimal('101000.00'): Decimal('12.3'), Decimal('102000.00'): Decimal('9.1')}
        vp_result = {
            "vp_data": {Decimal('100500.00'): Decimal('100.0'), Decimal('101000.00'): Decimal('150.0')},
            "poc_analysis": {
                "poc_price": Decimal('100500.00'),
                "poc_volume": Decimal('100.0'),
                "value_area_high": Decimal('101000.00'),
                "value_area_low": Decimal('100000.00'),
                "value_area_range": Decimal('1000.00'),
            },
            "total_volume": Decimal('500.0'),
        }

        # 生成提示词
        prompt = self.analyzer._create_unified_analysis_prompt(
            aggregated_bids, aggregated_asks, vp_result, "BTCFDUSD"
        )

        # 验证提示词包含必要元素
        assert "BTCFDUSD" in prompt
        assert "深度快照数据" in prompt
        assert "Volume Profile数据" in prompt
        assert "POC点价格" in prompt
        assert "短期支撑位" in prompt
        assert "短期阻力位" in prompt
        assert "集中流动性供应区域" in prompt
        assert "$100,000.00" in prompt or "100,000.00" in prompt  # 价格数据（正确格式）
        assert "JSON格式" in prompt

    def test_format_order_book_data(self):
        """测试订单簿数据格式化。"""
        aggregated_bids = {
            Decimal('100000.00'): Decimal('10.5'),
            Decimal('99000.00'): Decimal('8.2'),
            Decimal('98000.00'): Decimal('15.1'),
        }
        aggregated_asks = {
            Decimal('101000.00'): Decimal('12.3'),
            Decimal('102000.00'): Decimal('9.1'),
        }

        vp_result = {
            "vp_data": {Decimal('100500.00'): Decimal('100.0')},
            "poc_analysis": {
                "poc_price": Decimal('100500.00'),
                "poc_volume": Decimal('100.0'),
                "value_area_high": Decimal('101000.00'),
                "value_area_low": Decimal('100000.00'),
                "value_area_range": Decimal('1000.00'),
            },
            "total_volume": Decimal('500.0'),
        }

        prompt = self.analyzer._create_unified_analysis_prompt(
            aggregated_bids, aggregated_asks, vp_result, "BTCFDUSD"
        )

        # 验证数据格式化正确
        assert "$100,000.00" in prompt
        assert "$101,000.00" in prompt
        assert "10.50" in prompt
        assert "12.30" in prompt

    @patch('src.core.unified_deepseek_analyzer.httpx.Client')
    def test_make_api_request_success(self, mock_client_class):
        """测试成功的API请求。"""
        # 模拟成功响应
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": '{"test": "response"}'
                    }
                }
            ]
        }
        mock_client.post.return_value = mock_response

        # 创建新的分析器实例
        analyzer = UnifiedDeepSeekAnalyzer(api_key="test_key")
        response = analyzer._make_api_request("system prompt", "user prompt")

        assert response["choices"][0]["message"]["content"] == '{"test": "response"}'
        mock_client.post.assert_called_once()

        analyzer.close()

    @patch('src.core.unified_deepseek_analyzer.httpx.Client')
    def test_make_api_request_http_error(self, mock_client_class):
        """测试API请求HTTP错误。"""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = Exception("HTTP Error")
        mock_client.post.return_value = mock_response

        analyzer = UnifiedDeepSeekAnalyzer(api_key="test_key")

        with pytest.raises(Exception, match="HTTP Error"):
            analyzer._make_api_request("system prompt", "user prompt")

        analyzer.close()

    def test_parse_unified_analysis_response_valid_json(self):
        """测试解析有效的JSON响应。"""
        response_data = {
            "choices": [
                {
                    "message": {
                        "content": '{"短期支撑位": [{"价格": "100000", "可靠性评分": "85"}], "短期阻力位": []}'
                    }
                }
            ]
        }

        result = self.analyzer._parse_unified_analysis_response(response_data, "BTCFDUSD")

        assert result["status"] == "success"
        assert result["symbol"] == "BTCFDUSD"
        assert result["analysis_type"] == "unified_market_analysis"
        assert result["structured_analysis"] is not None
        assert "短期支撑位" in result["structured_analysis"]
        assert result["structured_analysis"]["短期支撑位"][0]["价格"] == "100000"

    def test_parse_unified_analysis_response_invalid_json(self):
        """测试解析无效JSON响应（回退到原始内容）。"""
        response_data = {
            "choices": [
                {
                    "message": {
                        "content": 'This is not valid JSON content'
                    }
                }
            ]
        }

        result = self.analyzer._parse_unified_analysis_response(response_data, "BTCFDUSD")

        assert result["status"] == "success"
        assert result["symbol"] == "BTCFDUSD"
        assert result["structured_analysis"] is None
        assert result["raw_content"] == 'This is not valid JSON content'

    def test_parse_unified_analysis_response_malformed_response(self):
        """测试解析格式错误的响应。"""
        response_data = {"invalid": "structure"}

        result = self.analyzer._parse_unified_analysis_response(response_data, "BTCFDUSD")

        assert result["status"] == "error"
        assert "解析失败" in result["error"]

    def test_create_error_analysis(self):
        """测试创建错误分析结果。"""
        result = self.analyzer._create_error_analysis("BTCFDUSD", "Test error message")

        assert result["status"] == "error"
        assert result["symbol"] == "BTCFDUSD"
        assert result["error"] == "Test error message"
        assert result["analysis_type"] == "unified_market_analysis"
        assert result["raw_content"] is None
        assert result["structured_analysis"] is None

    @patch.object(UnifiedDeepSeekAnalyzer, '_make_api_request')
    @patch.object(UnifiedDeepSeekAnalyzer, '_parse_unified_analysis_response')
    def test_analyze_unified_market_data_success(self, mock_parse, mock_request):
        """测试成功的统一市场数据分析。"""
        # 模拟API请求和解析
        mock_request.return_value = {"mock": "response"}
        mock_parse.return_value = {
            "status": "success",
            "symbol": "BTCFDUSD",
            "structured_analysis": {
                "短期支撑位": [{"价格": "100000", "可靠性评分": "85"}],
                "短期阻力位": [{"价格": "102000", "可靠性评分": "80"}],
            }
        }

        # 准备测试数据
        aggregated_bids = {Decimal('100000.00'): Decimal('10.5')}
        aggregated_asks = {Decimal('101000.00'): Decimal('12.3')}
        vp_result = {
            "vp_data": {Decimal('100500.00'): Decimal('100.0')},
            "poc_analysis": {"poc_price": Decimal('100500.00')},
            "total_volume": Decimal('500.0'),
        }

        result = self.analyzer.analyze_unified_market_data(
            aggregated_bids, aggregated_asks, vp_result, "BTCFDUSD"
        )

        assert result["status"] == "success"
        assert result["symbol"] == "BTCFDUSD"
        assert result["structured_analysis"] is not None
        mock_request.assert_called_once()
        mock_parse.assert_called_once()

    @patch.object(UnifiedDeepSeekAnalyzer, '_make_api_request')
    def test_analyze_unified_market_data_api_failure(self, mock_request):
        """测试统一市场数据分析API失败。"""
        mock_request.side_effect = Exception("API Error")

        aggregated_bids = {Decimal('100000.00'): Decimal('10.5')}
        aggregated_asks = {Decimal('101000.00'): Decimal('12.3')}
        vp_result = {
            "vp_data": {Decimal('100500.00'): Decimal('100.0')},
            "poc_analysis": {"poc_price": Decimal('100500.00')},
            "total_volume": Decimal('500.0'),
        }

        result = self.analyzer.analyze_unified_market_data(
            aggregated_bids, aggregated_asks, vp_result, "BTCFDUSD"
        )

        assert result["status"] == "error"
        assert "API Error" in result["error"]

    def test_get_unified_analysis_system_prompt_content(self):
        """测试统一分析系统提示词内容。"""
        prompt = self.analyzer._get_unified_analysis_system_prompt()

        # 验证提示词包含关键内容
        assert "高频做市策略" in prompt
        assert "支撑阻力位分析" in prompt
        assert "深度快照数据" in prompt
        assert "Volume Profile数据" in prompt
        assert "短期支撑位" in prompt
        assert "短期阻力位" in prompt
        assert "集中流动性供应区域" in prompt
        assert "输出要求" in prompt  # 修改为实际存在的内容

    def test_close_method(self):
        """测试关闭方法。"""
        # 创建一个客户端并验证关闭
        analyzer = UnifiedDeepSeekAnalyzer(api_key="test_key")
        client = analyzer.client
        client.close = MagicMock()

        analyzer.close()

        client.close.assert_called_once()


class TestUnifiedAnalysisPromptFormatting:
    """统一分析提示词格式化专项测试。"""

    def setup_method(self):
        """设置测试环境。"""
        self.analyzer = UnifiedDeepSeekAnalyzer(api_key="test_key")

    def teardown_method(self):
        """清理测试环境。"""
        self.analyzer.close()

    def test_large_dataset_formatting(self):
        """测试大数据集的格式化。"""
        # 创建大型测试数据集
        large_bids = {Decimal(str(100000 + i * 100)): Decimal(str(10 + i * 0.5)) for i in range(20)}
        large_asks = {Decimal(str(101000 + i * 100)): Decimal(str(12 + i * 0.3)) for i in range(20)}
        large_vp_data = {Decimal(str(100500 + i * 50)): Decimal(str(100 + i * 10)) for i in range(30)}

        vp_result = {
            "vp_data": large_vp_data,
            "poc_analysis": {
                "poc_price": Decimal('100500.00'),
                "poc_volume": Decimal('500.0'),
                "value_area_high": Decimal('102000.00'),
                "value_area_low": Decimal('100000.00'),
                "value_area_range": Decimal('2000.00'),
            },
            "total_volume": Decimal('10000.0'),
        }

        prompt = self.analyzer._create_unified_analysis_prompt(
            large_bids, large_asks, vp_result, "BTCFDUSD"
        )

        # 验证大数据集被正确处理（只取前N个）
        assert len(prompt) > 1000  # 确保有足够的内容
        assert "BTCFDUSD" in prompt

    def test_empty_data_handling(self):
        """测试空数据处理。"""
        empty_bids = {}
        empty_asks = {}
        empty_vp_result = {
            "vp_data": {},
            "poc_analysis": {
                "poc_price": Decimal('0.00'),
                "poc_volume": Decimal('0.00'),
                "value_area_high": Decimal('0.00'),
                "value_area_low": Decimal('0.00'),
                "value_area_range": Decimal('0.00'),
            },
            "total_volume": Decimal('0.0'),
        }

        prompt = self.analyzer._create_unified_analysis_prompt(
            empty_bids, empty_asks, empty_vp_result, "BTCFDUSD"
        )

        # 验证空数据不会导致错误
        assert "BTCFDUSD" in prompt
        assert "0.00" in prompt

    def test_extreme_price_values(self):
        """测试极端价格值的处理。"""
        extreme_bids = {
            Decimal('999999.99'): Decimal('999999.99'),  # 极大值
            Decimal('0.01'): Decimal('0.01'),            # 极小值
        }
        extreme_asks = {
            Decimal('1000000.00'): Decimal('1000000.00'),
            Decimal('0.02'): Decimal('0.02'),
        }

        vp_result = {
            "vp_data": {
                Decimal('500000.00'): Decimal('500000.00'),
            },
            "poc_analysis": {
                "poc_price": Decimal('500000.00'),
                "poc_volume": Decimal('500000.00'),
                "value_area_high": Decimal('1000000.00'),
                "value_area_low": Decimal('0.01'),
                "value_area_range": Decimal('999999.99'),
            },
            "total_volume": Decimal('1000000.0'),
        }

        # 这应该不会引发异常
        prompt = self.analyzer._create_unified_analysis_prompt(
            extreme_bids, extreme_asks, vp_result, "BTCFDUSD"
        )

        assert "BTCFDUSD" in prompt
        assert "999,999.99" in prompt or "999999.99" in prompt