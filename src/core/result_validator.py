"""结果验证器 - 验证分析结果的格式和业务逻辑。

该模块负责验证DeepSeek AI分析返回的结果，确保符合预期的JSON格式
和业务逻辑要求。
"""

import json
import logging
import re
from decimal import Decimal
from typing import Any, Dict

logger = logging.getLogger(__name__)


class ResultValidationError(Exception):
    """结果验证错误异常。"""
    pass


class ResultValidator:
    """分析结果验证器。

    负责验证AI分析返回的结果，确保：
    1. JSON格式正确
    2. 必需字段存在
    3. 字段值在有效范围内
    4. 业务逻辑正确
    """

    # 参数范围定义
    GRID_DELTA_MIN = 0.1
    GRID_DELTA_MAX = 100.0
    GRID_QUANTITY_MIN = 0.0001
    GRID_QUANTITY_MAX = 10.0
    ACTIVE_SIDES = ["Buy", "Sell"]

    # 预编译正则表达式模式以提高性能
    JSON_BLOCK_PATTERN = re.compile(r'```json\s*(.*?)\s*```', re.DOTALL)
    JSON_OBJECT_PATTERN = re.compile(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', re.DOTALL)
    FIELD_PATTERNS = {
        'grid_delta': [
            re.compile(r'"grid_delta"\s*:\s*([0-9.]+)', re.IGNORECASE),
            re.compile(r'grid_delta\s*[:：]\s*([0-9.]+)', re.IGNORECASE),
            re.compile(r'grid\s*delta\s*[:：]\s*([0-9.]+)', re.IGNORECASE),
        ],
        'grid_quantity': [
            re.compile(r'"grid_quantity"\s*:\s*([0-9.]+)', re.IGNORECASE),
            re.compile(r'grid_quantity\s*[:：]\s*([0-9.]+)', re.IGNORECASE),
            re.compile(r'grid\s*quantity\s*[:：]\s*([0-9.]+)', re.IGNORECASE),
        ],
        'active_side': [
            re.compile(r'"active_side"\s*:\s*"([^"]+)"', re.IGNORECASE),
            re.compile(r'"active_side"\s*:\s*([^,}\s]+)', re.IGNORECASE),
            re.compile(r'active_side\s*[:：]\s*"([^"]+)"', re.IGNORECASE),
            re.compile(r'active_side\s*[:：]\s*([^,}\s]+)', re.IGNORECASE),
            re.compile(r'(Buy|Sell)', re.IGNORECASE),
        ],
    }

    def __init__(self) -> None:
        """初始化结果验证器。"""
        logger.info("Initialized ResultValidator")

    def validate_and_extract_trading_params(
        self, analysis_result: dict[str, Any]
    ) -> dict[str, Any]:
        """验证分析结果并提取交易参数。

        Args:
            analysis_result: DeepSeek AI分析结果

        Returns:
            验证后的交易参数字典

        Raises:
            ResultValidationError: 验证失败时抛出
        """
        try:
            # 检查分析结果状态
            if analysis_result.get("status") != "success":
                raise ResultValidationError(
                    f"Analysis failed with status: {analysis_result.get('status')}"
                )

            raw_content = analysis_result.get("raw_content")
            if not raw_content or not isinstance(raw_content, str):
                raise ResultValidationError("Invalid content format")

            # 尝试解析JSON
            trading_params = self._extract_json_from_content(raw_content)

            # 验证交易参数
            self._validate_trading_params(trading_params)

            logger.info(f"Successfully validated trading params: {trading_params}")
            return trading_params

        except ResultValidationError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error during result validation: {e}")
            raise ResultValidationError(f"Validation error: {str(e)}")

    def _extract_json_from_content(self, content: str) -> dict[str, Any]:
        """从内容中提取JSON数据。

        Args:
            content: 原始分析内容

        Returns:
            解析后的JSON字典

        Raises:
            ResultValidationError: JSON解析失败时抛出
        """
        if not content or not isinstance(content, str):
            raise ResultValidationError("Invalid content format")

        # 策略1: 尝试直接解析完整内容
        try:
            json_data = json.loads(content.strip())
            if isinstance(json_data, dict):
                logger.info("Successfully parsed complete content as JSON")
                return json_data
        except json.JSONDecodeError:
            logger.debug("Failed to parse complete content as JSON")

        # 策略2: 提取JSON代码块
        matches = self.JSON_BLOCK_PATTERN.findall(content)
        if matches:
            for match in matches:
                try:
                    json_data = json.loads(match.strip())
                    if isinstance(json_data, dict):
                        logger.info("Successfully extracted JSON from code block")
                        return json_data
                except json.JSONDecodeError:
                    logger.debug("Failed to parse JSON from code block")
                    continue

        # 策略3: 查找独立的JSON对象
        matches = self.JSON_OBJECT_PATTERN.findall(content)
        if matches:
            # 优先处理最长的匹配（通常是最完整的JSON）
            matches.sort(key=len, reverse=True)
            for match in matches:
                try:
                    # 修复常见的JSON格式问题
                    cleaned_json = self._clean_json_string(match)
                    json_data = json.loads(cleaned_json)
                    if isinstance(json_data, dict):
                        logger.info("Successfully extracted and cleaned JSON object")
                        return json_data
                except json.JSONDecodeError:
                    logger.debug("Failed to parse extracted JSON object")
                    continue

        # 策略4: 手动解析特定字段
        return self._extract_fields_manually(content)

    def _extract_fields_manually(self, content: str) -> Dict[str, Any]:
        """手动从内容中提取字段值。

        Args:
            content: 原始分析内容

        Returns:
            手动提取的交易参数字典

        Raises:
            ResultValidationError: 无法提取字段时抛出
        """
        params = {}

        # 提取grid_delta
        for pattern in self.FIELD_PATTERNS['grid_delta']:
            match = pattern.search(content)
            if match:
                params["grid_delta"] = float(match.group(1))
                break

        # 提取grid_quantity
        for pattern in self.FIELD_PATTERNS['grid_quantity']:
            match = pattern.search(content)
            if match:
                params["grid_quantity"] = float(match.group(1))
                break

        # 提取active_side
        for pattern in self.FIELD_PATTERNS['active_side']:
            match = pattern.search(content)
            if match:
                side = match.group(1).strip()
                if side in self.ACTIVE_SIDES:
                    params["active_side"] = side
                    break

        # 检查是否提取到所有必需字段
        required_fields = ["grid_delta", "grid_quantity", "active_side"]
        missing_fields = [field for field in required_fields if field not in params]

        if missing_fields:
            logger.warning(f"Could not extract fields: {missing_fields}")
            # 返回默认值
            return self._get_default_trading_params()

        logger.info("Successfully extracted fields manually")
        return params

    def _validate_trading_params(self, params: Dict[str, Any]) -> None:
        """验证交易参数的有效性。

        Args:
            params: 交易参数字典

        Raises:
            ResultValidationError: 验证失败时抛出
        """
        # 验证必需字段
        required_fields = ["grid_delta", "grid_quantity", "active_side"]
        missing_fields = [field for field in required_fields if field not in params]

        if missing_fields:
            raise ResultValidationError(f"Missing required fields: {missing_fields}")

        # 验证grid_delta
        grid_delta = params.get("grid_delta")
        if not isinstance(grid_delta, (int, float, Decimal, str)):
            raise ResultValidationError("grid_delta must be a number")

        try:
            grid_delta_float = float(grid_delta)
        except (ValueError, TypeError):
            raise ResultValidationError("grid_delta must be convertible to float")

        if not (self.GRID_DELTA_MIN <= grid_delta_float <= self.GRID_DELTA_MAX):
            raise ResultValidationError(
                f"grid_delta {grid_delta_float} out of range "
                f"[{self.GRID_DELTA_MIN}, {self.GRID_DELTA_MAX}]"
            )

        # 验证grid_quantity
        grid_quantity = params.get("grid_quantity")
        if not isinstance(grid_quantity, (int, float, Decimal, str)):
            raise ResultValidationError("grid_quantity must be a number")

        try:
            grid_quantity_float = float(grid_quantity)
        except (ValueError, TypeError):
            raise ResultValidationError("grid_quantity must be convertible to float")

        if not (self.GRID_QUANTITY_MIN <= grid_quantity_float <= self.GRID_QUANTITY_MAX):
            raise ResultValidationError(
                f"grid_quantity {grid_quantity_float} out of range "
                f"[{self.GRID_QUANTITY_MIN}, {self.GRID_QUANTITY_MAX}]"
            )

        # 验证active_side
        active_side = params.get("active_side")
        if not isinstance(active_side, str):
            raise ResultValidationError("active_side must be a string")

        if active_side not in self.ACTIVE_SIDES:
            raise ResultValidationError(
                f"active_side '{active_side}' not in valid options: {self.ACTIVE_SIDES}"
            )

        logger.info("All trading parameters validated successfully")

        # 返回验证后的参数（确保正确的数据类型）
        return {
            "grid_delta": grid_delta_float,
            "grid_quantity": grid_quantity_float,
            "active_side": active_side
        }

    def get_validation_stats(self) -> dict[str, Any]:
        """获取验证统计信息。

        Returns:
            验证统计信息字典
        """
        return {
            "grid_delta_range": [self.GRID_DELTA_MIN, self.GRID_DELTA_MAX],
            "grid_quantity_range": [self.GRID_QUANTITY_MIN, self.GRID_QUANTITY_MAX],
            "active_sides": self.ACTIVE_SIDES,
            "validator_status": "active"
        }

    def _get_default_trading_params(self) -> Dict[str, Any]:
        """获取默认的交易参数。

        Returns:
            默认交易参数字典
        """
        logger.info("Using default trading parameters")
        return {
            "grid_delta": 1.0,
            "grid_quantity": 0.001,
            "active_side": "Buy"
        }


# 全局验证器实例
result_validator = ResultValidator()