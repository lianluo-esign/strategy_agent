"""优化版Agent Analyzer - 基于trades_window数据的AI趋势分析器。

该模块是原agent_analyzer的优化版本，专注于：
1. 仅使用trades_window聚合数据（移除5000层深度快照依赖）
2. 集成Deepseek AI进行趋势分析
3. 提供标准JSON输出和Discord通知
4. 优化性能和可靠性
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Any

from .deepseek_client import DeepSeekAnalyzer
from .discord_notifier import DiscordNotificationManager
from .response_formatter import ResponseFormatter, ResponseValidator
from .trades_aggregator import TradesAggregator

logger = logging.getLogger(__name__)


class OptimizedAgentAnalyzer:
    """优化版Agent Analyzer，专注基于trades_window数据的AI趋势分析。

    核心功能：
    1. 数据聚合：处理trades_window数据并生成聚合视图
    2. AI分析：使用Deepseek进行趋势判断和强度评估
    3. 结果格式化：提供标准JSON输出
    4. Discord通知：实时推送分析结果

    主要优化：
    - 移除对5000层深度快照的依赖
    - 简化数据处理流程
    - 提高AI分析的准确性
    - 增强错误处理和重试机制
    """

    def __init__(
        self,
        redis_store: Any,
        deepseek_config: dict[str, Any],
        discord_webhook_url: str | None = None,
        analysis_window_minutes: int = 240
    ):
        """初始化优化版分析器。

        Args:
            redis_store: Redis数据存储实例
            deepseek_config: Deepseek配置字典
            discord_webhook_url: Discord webhook URL（可选）
            analysis_window_minutes: 分析时间窗口（分钟）
        """
        self.redis_store = redis_store
        self.symbol = "BTCFDUSD"

        # 初始化数据收集器
        self.trades_aggregator = TradesAggregator(
            minutes_to_collect=analysis_window_minutes
        )

        # 初始化Deepseek分析器
        self.deepseek_analyzer = DeepSeekAnalyzer(**deepseek_config)

        # 初始化响应格式化器
        self.response_formatter = ResponseFormatter(
            include_metadata=True,
            pretty_print=False
        )

        # 初始化响应验证器
        self.response_validator = ResponseValidator(strict_mode=False)

        # 初始化Discord通知管理器（如果提供了webhook URL）
        self.discord_manager: DiscordNotificationManager | None = None
        if discord_webhook_url:
            try:
                self.discord_manager = DiscordNotificationManager(discord_webhook_url)
                logger.info("Discord通知功能已启用")
            except Exception as e:
                logger.warning(f"Discord通知初始化失败: {e}")

        # 统计信息
        self.stats = {
            "total_analyses": 0,
            "successful_analyses": 0,
            "failed_analyses": 0,
            "discord_notifications_sent": 0,
            "average_processing_time": 0
        }

        logger.info(
            f"OptimizedAgentAnalyzer initialized: symbol={self.symbol}, "
            f"analysis_window_minutes={analysis_window_minutes}, "
            f"discord_enabled={self.discord_manager is not None}"
        )

    async def analyze_market(self, symbol: str = "BTCFDUSD") -> dict[str, Any]:
        """执行完整的市场分析流程。

        Args:
            symbol: 交易符号

        Returns:
            分析结果字典

        流程：
        1. 从Redis读取trades_window数据
        2. 聚合数据并生成分析视图
        3. 调用Deepseek进行AI分析
        4. 格式化输出结果
        5. 发送Discord通知（如果启用）
        """
        start_time = asyncio.get_event_loop().time()
        self.stats["total_analyses"] += 1

        try:
            logger.info(f"开始市场分析: {symbol}")

            # 执行分析流程
            raw_dict = await self._collect_and_raw_data(symbol)
            trend_result = await self._perform_ai_analysis(raw_dict, symbol)
            json_response = await self._format_analysis_result(trend_result, raw_dict, symbol)

            # 处理通知和质量检查
            quality_check, discord_sent = await self._handle_notifications_and_quality(
                trend_result, symbol
            )

            # 构建最终结果
            processing_time = asyncio.get_event_loop().time() - start_time
            self._update_stats(processing_time, success=True)

            result = self._build_final_result(
                symbol, json_response, raw_dict,
                processing_time, discord_sent, quality_check
            )

            logger.info(
                f"市场分析完成: {symbol}, 耗时 {processing_time:.2f}s, "
                f"Discord通知={'已发送' if discord_sent else '未发送'}"
            )

            return result

        except Exception as e:
            return await self._handle_analysis_error(symbol, e, start_time)

    async def _collect_and_raw_data(self, symbol: str) -> dict[str, Any]:
        """收集原始数据。"""
        # 第一步：从Redis读取trades_window数据
        trades_window_data = self.redis_store.get_recent_trade_data(
            minutes=self.trades_aggregator.minutes_to_collect
        )

        if not trades_window_data:
            raise ValueError("没有可用的交易数据")

        logger.info(f"读取到 {len(trades_window_data)} 分钟的交易数据")

        # 第二步：收集原始数据
        raw_data = self.trades_aggregator.collect_raw_trades_data(
            trades_window_data, symbol
        )
        raw_dict = raw_data.to_dict()

        logger.info(
            f"原始数据收集完成: {raw_data.data_points_count} 个分钟数据点, "
            f"时间跨度 {len(raw_data.minute_data_points)} 分钟"
        )

        return raw_dict

    async def _perform_ai_analysis(self, raw_dict: dict[str, Any], symbol: str) -> dict[str, Any]:
        """执行AI分析。"""
        # 第三步：AI分析
        trend_result_dict = await self.deepseek_analyzer.analyze_trend(
            raw_dict, symbol
        )
        trend_result = trend_result_dict.to_dict()

        logger.info(
            f"AI分析完成: 趋势={trend_result['trend']}, "
            f"置信度={trend_result['confidence']:.2f}"
        )

        return trend_result

    async def _format_analysis_result(
        self,
        trend_result: dict[str, Any],
        raw_dict: dict[str, Any],
        symbol: str
    ) -> str:
        """格式化分析结果。"""
        # 第四步：格式化响应
        json_response = self.response_formatter.format_analysis_response(
            trend_result, raw_dict, symbol
        )

        # 验证响应格式
        if not self.response_formatter.validate_response_schema(json_response):
            logger.warning("响应Schema验证失败，但仍返回结果")

        return json_response

    async def _handle_notifications_and_quality(
        self,
        trend_result: dict[str, Any],
        symbol: str
    ) -> tuple:
        """处理通知和质量检查。"""
        # 验证分析质量
        quality_check = self.response_validator.validate_analysis_quality(trend_result)
        if not quality_check["is_valid"]:
            logger.error(f"分析质量验证失败: {quality_check['errors']}")
            if self.discord_manager:
                await self.discord_manager.send_error_notification(
                    f"分析质量验证失败: {quality_check['errors']}",
                    f"符号: {symbol}"
                )

        # 第五步：Discord通知
        discord_sent = False
        if self.discord_manager:
            try:
                discord_sent = await self.discord_manager.send_trend_alert(
                    trend_result, symbol
                )
                if discord_sent:
                    self.stats["discord_notifications_sent"] += 1
                    logger.info("Discord通知发送成功")
            except Exception as e:
                logger.error(f"Discord通知发送失败: {e}")

        return quality_check, discord_sent

    def _build_final_result(
        self,
        symbol: str,
        json_response: str,
        raw_dict: dict[str, Any],
        processing_time: float,
        discord_sent: bool,
        quality_check: dict[str, Any]
    ) -> dict[str, Any]:
        """构建最终结果。"""
        result = {
            "status": "success",
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "analysis_result": json.loads(json_response),
            "raw_data": raw_dict,
            "processing_time": processing_time,
            "discord_notification_sent": discord_sent,
            "quality_check": quality_check
        }

        # 添加性能统计
        result["performance_stats"] = {
            "data_collection_time": processing_time * 0.15,  # 估算
            "ai_analysis_time": processing_time * 0.75,     # 估算
            "formatting_time": processing_time * 0.1        # 估算
        }

        return result

    async def _handle_analysis_error(self, symbol: str, error: Exception, start_time: float) -> dict[str, Any]:
        """处理分析错误。"""
        processing_time = asyncio.get_event_loop().time() - start_time
        self._update_stats(processing_time, success=False)
        logger.error(f"市场分析失败: {error}")

        # 发送错误通知
        if self.discord_manager:
            try:
                await self.discord_manager.send_error_notification(
                    str(error), f"市场分析失败: {symbol}"
                )
            except Exception as discord_error:
                logger.error(f"错误通知发送失败: {discord_error}")

        return self._create_error_result(symbol, str(error))

    async def analyze_single_cycle(self, symbol: str = "BTCFDUSD") -> str:
        """执行单次分析并返回JSON响应。

        Args:
            symbol: 交易符号

        Returns:
            JSON格式的分析结果字符串
        """
        try:
            result = await self.analyze_market(symbol)

            if result["status"] == "success":
                return result["analysis_result"]
            else:
                # 返回错误响应
                return self.response_formatter._format_error_response(
                    result.get("error", "未知错误"), symbol
                )

        except Exception as e:
            logger.error(f"单次分析失败: {e}")
            return self.response_formatter._format_error_response(str(e), symbol)

    def _create_error_result(self, symbol: str, error_message: str) -> dict[str, Any]:
        """创建错误结果。

        Args:
            symbol: 交易符号
            error_message: 错误消息

        Returns:
            错误结果字典
        """
        return {
            "status": "error",
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "error": error_message,
            "analysis_result": self.response_formatter._format_error_response(
                error_message, symbol
            ),
            "aggregated_data": None,
            "processing_time": 0,
            "discord_notification_sent": False,
            "quality_check": {
                "is_valid": False,
                "errors": [error_message],
                "warnings": []
            }
        }

    def _update_stats(self, processing_time: float, success: bool) -> None:
        """更新统计信息。

        Args:
            processing_time: 处理时间
            success: 是否成功
        """
        if success:
            self.stats["successful_analyses"] += 1
        else:
            self.stats["failed_analyses"] += 1

        # 更新平均处理时间
        total_analyses = self.stats["total_analyses"]
        current_avg = self.stats["average_processing_time"]
        self.stats["average_processing_time"] = (
            (current_avg * (total_analyses - 1) + processing_time) / total_analyses
        )

    def get_status(self) -> dict[str, Any]:
        """获取分析器状态。

        Returns:
            状态信息字典
        """
        # 基础状态
        status = {
            "analyzer_type": "optimized_agent_analyzer",
            "symbol": self.symbol,
            "redis_connected": self.redis_store.test_connection(),
            "trades_window_available": self.redis_store.get_trade_window_count() > 0,
            "statistics": self.stats,
            "components": {
                "trades_aggregator": {
                    "enabled": True,
                    "minutes_to_collect": self.trades_aggregator.minutes_to_collect
                },
                "deepseek_analyzer": self.deepseek_analyzer.get_stats(),
                "response_formatter": {
                    "enabled": True,
                    "include_metadata": self.response_formatter.include_metadata
                },
                "discord_manager": {
                    "enabled": self.discord_manager is not None,
                    "stats": self.discord_manager.get_notifier().get_stats() if self.discord_manager else None
                }
            }
        }

        # 计算成功率
        total = self.stats["total_analyses"]
        if total > 0:
            status["statistics"]["success_rate"] = self.stats["successful_analyses"] / total
        else:
            status["statistics"]["success_rate"] = 0.0

        return status


    async def health_check(self) -> dict[str, Any]:
        """执行健康检查。

        Returns:
            健康检查结果
        """
        health_status = {
            "overall_status": "healthy",
            "checks": {},
            "timestamp": datetime.now().isoformat()
        }

        # Redis连接检查
        try:
            redis_ok = self.redis_store.test_connection()
            health_status["checks"]["redis_connection"] = {
                "status": "pass" if redis_ok else "fail",
                "message": "Redis连接正常" if redis_ok else "Redis连接失败"
            }
        except Exception as e:
            health_status["checks"]["redis_connection"] = {
                "status": "fail",
                "message": f"Redis检查异常: {str(e)}"
            }

        # 数据可用性检查
        try:
            trades_count = self.redis_store.get_trade_window_count()
            data_ok = trades_count > 0
            health_status["checks"]["data_availability"] = {
                "status": "pass" if data_ok else "fail",
                "message": f"可用数据: {trades_count} 分钟" if data_ok else "没有可用数据"
            }
        except Exception as e:
            health_status["checks"]["data_availability"] = {
                "status": "fail",
                "message": f"数据检查异常: {str(e)}"
            }

        # Discord功能检查（如果启用）
        if self.discord_manager:
            health_status["checks"]["discord_service"] = {
                "status": "pass",
                "message": "Discord通知服务已启用，将在分析完成后发送结果"
            }

        # 确定整体状态
        failed_checks = [
            name for name, check in health_status["checks"].items()
            if check["status"] == "fail"
        ]

        if failed_checks:
            health_status["overall_status"] = "unhealthy"
            health_status["failed_checks"] = failed_checks

        return health_status

    async def close(self) -> None:
        """关闭分析器资源。"""
        logger.info("正在关闭OptimizedAgentAnalyzer...")

        try:
            # 关闭Deepseek分析器
            if self.deepseek_analyzer:
                self.deepseek_analyzer.close()
                logger.info("Deepseek分析器已关闭")
        except Exception as e:
            logger.error(f"关闭Deepseek分析器时出错: {e}")

        try:
            # 关闭Discord通知管理器
            if self.discord_manager:
                await self.discord_manager.close()
                logger.info("Discord通知管理器已关闭")
        except Exception as e:
            logger.error(f"关闭Discord通知管理器时出错: {e}")

        try:
            # 关闭Redis连接
            if self.redis_store:
                await self.redis_store.close()
                logger.info("Redis连接已关闭")
        except Exception as e:
            logger.error(f"关闭Redis连接时出错: {e}")

        logger.info("OptimizedAgentAnalyzer已完全关闭")
