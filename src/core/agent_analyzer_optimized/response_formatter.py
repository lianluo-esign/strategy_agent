"""响应格式化器 - 统一的JSON输出格式处理。

该模块负责将AI分析结果格式化为标准化的JSON响应，
确保输出格式的一致性和可解析性。
"""

import json
import logging
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


class ResponseFormatter:
    """响应格式化器，提供统一的JSON输出格式。

    该类确保：
    1. 输出格式的一致性
    2. 数据验证和清理
    3. 错误处理和降级
    4. 多种输出格式支持
    """

    def __init__(self, include_metadata: bool = True, pretty_print: bool = False):
        """初始化响应格式化器。

        Args:
            include_metadata: 是否包含元数据
            pretty_print: 是否美化输出
        """
        self.include_metadata = include_metadata
        self.pretty_print = pretty_print

        logger.info(
            f"Initialized ResponseFormatter: include_metadata={include_metadata}, "
            f"pretty_print={pretty_print}"
        )

    def format_analysis_response(
        self,
        trend_result: dict[str, Any],
        aggregated_data: dict[str, Any] | None = None,
        symbol: str = "BTCFDUSD"
    ) -> str:
        """格式化分析响应为标准JSON格式。

        Args:
            trend_result: AI趋势分析结果
            aggregated_data: 聚合数据（可选）
            symbol: 交易符号

        Returns:
            格式化的JSON字符串

        Raises:
            ValueError: 当输入数据无效时
        """
        # 验证输入数据
        if not trend_result or not isinstance(trend_result, dict):
            raise ValueError("趋势分析结果不能为空且必须是字典格式")

        try:

            # 提取并验证关键信息
            timestamp = self._extract_timestamp(trend_result)
            trend = self._validate_trend(trend_result.get("trend", "震荡"))
            strength_levels = self._validate_strength_levels(
                trend_result.get("strength_levels", {})
            )
            reason = trend_result.get("reason", "暂无分析原因")
            confidence = self._validate_confidence(trend_result.get("confidence", 0.5))

            # 构建标准响应
            response = {
                "timestamp": timestamp,
                "symbol": symbol,
                "trend": trend,
                "strength_levels": strength_levels,
                "reason": reason,
                "confidence": confidence
            }

            # 添加元数据
            if self.include_metadata:
                response["metadata"] = self._build_metadata(
                    trend_result, aggregated_data, symbol
                )

            # 序列化为JSON
            json_string = json.dumps(
                response,
                ensure_ascii=False,
                indent=2 if self.pretty_print else None,
                separators=(',', ': ') if self.pretty_print else (',', ':')
            )

            logger.info(f"响应格式化完成: trend={trend}, confidence={confidence:.2f}")
            return json_string

        except Exception as e:
            logger.error(f"响应格式化失败: {e}")
            # 返回错误响应
            return self._format_error_response(str(e), symbol)

    def _extract_timestamp(self, trend_result: dict[str, Any]) -> str:
        """提取时间戳。

        Args:
            trend_result: 趋势分析结果

        Returns:
            ISO格式的时间戳字符串
        """
        timestamp = trend_result.get("timestamp")
        if timestamp:
            if isinstance(timestamp, str):
                return timestamp
            elif isinstance(timestamp, datetime):
                return timestamp.isoformat()
            else:
                logger.warning(f"未知的时间戳格式: {type(timestamp)}")

        # 默认使用当前时间
        return datetime.now().isoformat()

    def _validate_trend(self, trend: str) -> str:
        """验证趋势类型。

        Args:
            trend: 趋势字符串

        Returns:
            验证后的趋势字符串
        """
        valid_trends = [
            "震荡", "微弱看涨", "看涨", "强力看涨",
            "微弱看跌", "看跌", "强力看跌"
        ]

        if trend not in valid_trends:
            logger.warning(f"无效的趋势类型: {trend}，使用默认值")
            return "震荡"

        return trend

    def _validate_strength_levels(self, strength_levels: dict[str, Any]) -> dict[str, float]:
        """验证强度等级。

        Args:
            strength_levels: 强度等级字典

        Returns:
            验证后的强度等级字典
        """
        valid_levels = [
            "strong_support", "weak_support",
            "strong_resistance", "weak_resistance"
        ]

        validated_levels = {}
        for level in valid_levels:
            value = strength_levels.get(level, 0.0)
            try:
                float_value = float(value)
                # 确保值在0-1范围内
                validated_levels[level] = max(0.0, min(1.0, float_value))
            except (ValueError, TypeError):
                logger.warning(f"无效的强度值: {level}={value}，使用默认值")
                validated_levels[level] = 0.0

        return validated_levels

    def _validate_confidence(self, confidence: Any) -> float:
        """验证置信度。

        Args:
            confidence: 置信度值

        Returns:
            验证后的置信度
        """
        try:
            float_confidence = float(confidence)
            # 确保值在0-1范围内
            return max(0.0, min(1.0, float_confidence))
        except (ValueError, TypeError):
            logger.warning(f"无效的置信度值: {confidence}，使用默认值")
            return 0.5

    def _build_metadata(
        self,
        trend_result: dict[str, Any],
        aggregated_data: dict[str, Any] | None,
        symbol: str
    ) -> dict[str, Any]:
        """构建元数据。

        Args:
            trend_result: 趋势分析结果
            aggregated_data: 聚合数据
            symbol: 交易符号

        Returns:
            元数据字典
        """
        metadata = {
            "analysis_method": "trades_window_aggregation",
            "formatter_version": "1.0",
            "generated_at": datetime.now().isoformat()
        }

        # 添加AI分析元数据
        if "analysis_metadata" in trend_result:
            ai_metadata = trend_result["analysis_metadata"]
            metadata.update({
                "ai_model": ai_metadata.get("model", "unknown"),
                "analysis_symbols": ai_metadata.get("symbol", symbol),
                "response_truncated": ai_metadata.get("response_truncated", False)
            })

        # 添加聚合数据统计
        if aggregated_data:
            metadata["data_statistics"] = {
                "total_volume": aggregated_data.get("total_volume", 0),
                "trade_count": aggregated_data.get("trade_count", 0),
                "price_levels_count": aggregated_data.get("price_levels_count", 0),
                "price_range": aggregated_data.get("price_range", [0, 0])
            }

        return metadata

    def _format_error_response(self, error_message: str, symbol: str) -> str:
        """格式化错误响应。

        Args:
            error_message: 错误消息
            symbol: 交易符号

        Returns:
            错误响应JSON字符串
        """
        error_response = {
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "status": "error",
            "error": error_message,
            "trend": "震荡",
            "strength_levels": {
                "strong_support": 0.0,
                "weak_support": 0.0,
                "strong_resistance": 0.0,
                "weak_resistance": 0.0
            },
            "reason": f"分析过程中发生错误: {error_message}",
            "confidence": 0.0
        }

        if self.include_metadata:
            error_response["metadata"] = {
                "error_occurred": True,
                "formatter_version": "1.0",
                "generated_at": datetime.now().isoformat()
            }

        return json.dumps(
            error_response,
            ensure_ascii=False,
            indent=2 if self.pretty_print else None
        )

    def format_compact_response(
        self,
        trend_result: dict[str, Any],
        symbol: str = "BTCFDUSD"
    ) -> str:
        """格式化紧凑响应（不包含元数据）。

        Args:
            trend_result: AI趋势分析结果
            symbol: 交易符号

        Returns:
            紧凑的JSON字符串
        """
        try:
            # 提取核心信息
            timestamp = self._extract_timestamp(trend_result)
            trend = self._validate_trend(trend_result.get("trend", "震荡"))
            strength_levels = self._validate_strength_levels(
                trend_result.get("strength_levels", {})
            )
            reason = trend_result.get("reason", "暂无分析原因")
            confidence = self._validate_confidence(trend_result.get("confidence", 0.5))

            # 构建紧凑响应
            response = {
                "timestamp": timestamp,
                "trend": trend,
                "strength_levels": strength_levels,
                "reason": reason,
                "confidence": confidence
            }

            return json.dumps(response, ensure_ascii=False, separators=(',', ':'))

        except Exception as e:
            logger.error(f"紧凑响应格式化失败: {e}")
            return json.dumps({
                "timestamp": datetime.now().isoformat(),
                "trend": "震荡",
                "strength_levels": {
                    "strong_support": 0.0, "weak_support": 0.0,
                    "strong_resistance": 0.0, "weak_resistance": 0.0
                },
                "reason": f"格式化错误: {str(e)}",
                "confidence": 0.0
            }, ensure_ascii=False, separators=(',', ':'))

    def validate_response_schema(self, response_json: str) -> bool:
        """验证响应JSON的Schema。

        Args:
            response_json: JSON响应字符串

        Returns:
            Schema是否有效
        """
        try:
            data = json.loads(response_json)

            # 检查必需字段
            required_fields = ["timestamp", "trend", "strength_levels", "reason", "confidence"]
            for field in required_fields:
                if field not in data:
                    logger.error(f"缺少必需字段: {field}")
                    return False

            # 验证趋势类型
            trend = data["trend"]
            valid_trends = [
                "震荡", "微弱看涨", "看涨", "强力看涨",
                "微弱看跌", "看跌", "强力看跌"
            ]
            if trend not in valid_trends:
                logger.error(f"无效的趋势类型: {trend}")
                return False

            # 验证强度等级
            strength_levels = data["strength_levels"]
            required_levels = [
                "strong_support", "weak_support",
                "strong_resistance", "weak_resistance"
            ]
            for level in required_levels:
                if level not in strength_levels:
                    logger.error(f"缺少强度等级: {level}")
                    return False
                if not isinstance(strength_levels[level], (int, float)):
                    logger.error(f"强度等级类型错误: {level}={type(strength_levels[level])}")
                    return False
                if not 0 <= strength_levels[level] <= 1:
                    logger.error(f"强度等级值超出范围: {level}={strength_levels[level]}")
                    return False

            # 验证置信度
            confidence = data["confidence"]
            if not isinstance(confidence, (int, float)):
                logger.error(f"置信度类型错误: {type(confidence)}")
                return False
            if not 0 <= confidence <= 1:
                logger.error(f"置信度值超出范围: {confidence}")
                return False

            return True

        except json.JSONDecodeError as e:
            logger.error(f"JSON解析失败: {e}")
            return False
        except Exception as e:
            logger.error(f"Schema验证异常: {e}")
            return False


class ResponseValidator:
    """响应验证器，提供数据质量检查功能。"""

    def __init__(self, strict_mode: bool = False):
        """初始化响应验证器。

        Args:
            strict_mode: 是否启用严格模式
        """
        self.strict_mode = strict_mode

    def validate_analysis_quality(
        self,
        analysis_result: dict[str, Any]
    ) -> dict[str, Any]:
        """验证分析质量。

        Args:
            analysis_result: 分析结果

        Returns:
            验证结果字典
        """
        validation_result = {
            "is_valid": True,
            "warnings": [],
            "errors": []
        }

        try:
            # 检查置信度
            confidence = analysis_result.get("confidence", 0)
            if confidence < 0.3:
                if self.strict_mode:
                    validation_result["errors"].append(f"置信度过低: {confidence:.2f}")
                    validation_result["is_valid"] = False
                else:
                    validation_result["warnings"].append(f"置信度较低: {confidence:.2f}")

            # 检查趋势一致性
            trend = analysis_result.get("trend", "")
            strength_levels = analysis_result.get("strength_levels", {})

            if "看涨" in trend:
                strong_support = strength_levels.get("strong_support", 0)
                strong_resistance = strength_levels.get("strong_resistance", 0)
                if strong_resistance > strong_support:
                    validation_result["warnings"].append("看涨趋势但阻力强度高于支撑强度")

            if "看跌" in trend:
                strong_support = strength_levels.get("strong_support", 0)
                strong_resistance = strength_levels.get("strong_resistance", 0)
                if strong_support > strong_resistance:
                    validation_result["warnings"].append("看跌趋势但支撑强度高于阻力强度")

            # 检查分析原因
            reason = analysis_result.get("reason", "")
            if len(reason) < 20:
                validation_result["warnings"].append("分析原因过于简短")

            # 检查强度等级分布
            strengths = list(strength_levels.values())
            if all(strength == 0 for strength in strengths):
                if self.strict_mode:
                    validation_result["errors"].append("所有强度等级都为0")
                    validation_result["is_valid"] = False
                else:
                    validation_result["warnings"].append("所有强度等级都为0")

        except Exception as e:
            validation_result["errors"].append(f"验证过程异常: {str(e)}")
            validation_result["is_valid"] = False

        return validation_result
