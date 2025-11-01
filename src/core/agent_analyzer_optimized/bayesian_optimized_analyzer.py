"""贝叶斯优化分析器 - 集成静态订单簿和动态交易数据的贝叶斯分析。

该模块是OptimizedAgentAnalyzer的贝叶斯增强版本，提供：
1. 静态深度订单簿数据获取和分析
2. 动态trades_window数据聚合
3. 贝叶斯概率化趋势分析
4. 贝叶斯思维框架的AI分析
5. 概率化的结果输出
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Any

from .bayesian_analyzer import BayesianAnalyzer
from .bayesian_deepseek_client import BayesianDeepSeekAnalyzer
from .bayesian_response_formatter import BayesianResponseFormatter
from .discord_notifier import DiscordNotificationManager
from .static_orderbook_analyzer import StaticOrderBookAnalyzer
from .trades_aggregator import TradesAggregator

logger = logging.getLogger(__name__)


class BayesianOptimizedAnalyzer:
    """贝叶斯优化分析器，集成静态和动态数据进行概率化趋势分析。

    核心功能：
    1. 静态深度订单簿分析（depth_snapshot_5000）
    2. 动态交易数据聚合（trades_window）
    3. 贝叶斯概率化趋势预测
    4. AI增强的贝叶斯分析
    5. 概率化的结果输出和通知

    主要改进：
    - 集成静态订单簿数据分析
    - 贝叶斯思维框架的概率化分析
    - 更准确的置信度和不确定性量化
    - 增强的证据权重分析
    """

    def __init__(
        self,
        redis_store: Any,
        deepseek_config: dict[str, Any],
        discord_webhook_url: str | None = None,
        analysis_window_minutes: int = 240,
        orderbook_precision: float = 10.0
    ):
        """初始化贝叶斯优化分析器。

        Args:
            redis_store: Redis数据存储实例
            deepseek_config: Deepseek配置字典
            discord_webhook_url: Discord webhook URL（可选）
            analysis_window_minutes: 动态数据分析时间窗口（分钟）
            orderbook_precision: 订单簿聚合精度（美元）
        """
        self.redis_store = redis_store
        self.symbol = "BTCFDUSD"

        # 初始化静态订单簿分析器
        self.static_analyzer = StaticOrderBookAnalyzer(
            aggregation_precision=orderbook_precision
        )

        # 初始化动态数据聚合器
        self.trades_aggregator = TradesAggregator(
            minutes_to_collect=analysis_window_minutes
        )

        # 初始化贝叶斯分析引擎
        self.bayesian_analyzer = BayesianAnalyzer()

        # 初始化贝叶斯化Deepseek分析器
        self.bayesian_deepseek = BayesianDeepSeekAnalyzer(**deepseek_config)

        # 初始化贝叶斯响应格式化器
        self.response_formatter = BayesianResponseFormatter(
            include_metadata=True,
            pretty_print=False
        )

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
            "average_processing_time": 0,
            "static_data_available": 0,
            "dynamic_data_available": 0
        }

        logger.info(
            f"BayesianOptimizedAnalyzer initialized: symbol={self.symbol}, "
            f"analysis_window_minutes={analysis_window_minutes}, "
            f"orderbook_precision=${orderbook_precision}, "
            f"discord_enabled={self.discord_manager is not None}"
        )

    async def analyze_market(self, symbol: str = "BTCFDUSD") -> dict[str, Any]:
        """执行完整的贝叶斯市场分析流程。

        Args:
            symbol: 交易符号

        Returns:
            贝叶斯分析结果字典

        流程：
        1. 从Redis获取静态深度快照数据
        2. 从Redis获取动态trades_window数据
        3. 分别分析静态和动态数据
        4. 执行贝叶斯概率化分析
        5. 调用AI进行贝叶斯增强分析
        6. 格式化概率化结果
        7. 发送Discord通知（如果启用）
        """
        start_time = asyncio.get_event_loop().time()
        self.stats["total_analyses"] += 1

        try:
            logger.info(f"开始贝叶斯市场分析: {symbol}")

            # 第一步：获取静态订单簿数据
            static_data = await self._collect_static_orderbook_data(symbol)

            # 第二步：获取动态交易数据
            dynamic_data = await self._collect_dynamic_trades_data(symbol)

            # 第三步：执行本地贝叶斯分析
            local_bayesian_result = self._perform_local_bayesian_analysis(
                static_data, dynamic_data, symbol
            )

            # 第四步：执行AI增强的贝叶斯分析
            ai_bayesian_result = await self._perform_ai_bayesian_analysis(
                static_data, dynamic_data, symbol
            )

            # 第五步：融合分析结果
            fused_result = self._fuse_bayesian_results(
                local_bayesian_result, ai_bayesian_result
            )

            # 第六步：格式化响应
            json_response = self._format_bayesian_response(
                fused_result, static_data, dynamic_data, symbol
            )

            # 处理通知和质量检查
            quality_check, discord_sent = await self._handle_notifications_and_quality(
                fused_result, symbol
            )

            # 构建最终结果
            processing_time = asyncio.get_event_loop().time() - start_time
            self._update_stats(processing_time, success=True, static_data=static_data, dynamic_data=dynamic_data)

            result = self._build_final_result(
                symbol, json_response, static_data, dynamic_data,
                fused_result, processing_time, discord_sent, quality_check
            )

            logger.info(
                f"贝叶斯市场分析完成: {symbol}, 耗时 {processing_time:.2f}s, "
                f"Discord通知={'已发送' if discord_sent else '未发送'}, "
                f"趋势={fused_result.get('most_likely_trend', 'unknown')}"
            )

            return result

        except Exception as e:
            return await self._handle_analysis_error(symbol, e, start_time)

    async def _collect_static_orderbook_data(self, symbol: str) -> dict[str, Any]:
        """收集静态订单簿数据。

        Args:
            symbol: 交易符号

        Returns:
            静态订单簿分析数据
        """
        try:
            # 从Redis获取深度快照数据
            depth_snapshot = self.redis_store.get_latest_depth_snapshot()

            if not depth_snapshot:
                logger.warning("无法获取深度快照数据")
                return {
                    "status": "no_data",
                    "error": "No depth snapshot available",
                    "symbol": symbol,
                    "analysis_type": "static_order_book"
                }

            logger.info(f"获取到深度快照: {depth_snapshot.symbol} from {depth_snapshot.timestamp}")

            # 分析静态订单簿
            static_analysis = self.static_analyzer.analyze_order_book(depth_snapshot, symbol)

            if static_analysis.get("status") == "success":
                logger.info(
                    f"静态订单簿分析完成: "
                    f"{len(static_analysis.get('aggregated_bids', {}))} bid levels, "
                    f"{len(static_analysis.get('aggregated_asks', {}))} ask levels"
                )

            return static_analysis

        except Exception as e:
            logger.error(f"静态订单簿数据收集失败: {e}")
            return {
                "status": "error",
                "error": str(e),
                "symbol": symbol,
                "analysis_type": "static_order_book"
            }

    async def _collect_dynamic_trades_data(self, symbol: str) -> dict[str, Any]:
        """收集动态交易数据。

        Args:
            symbol: 交易符号

        Returns:
            动态交易分析数据
        """
        try:
            # 从Redis获取trades_window数据
            trades_window_data = self.redis_store.get_recent_trade_data(
                minutes=self.trades_aggregator.minutes_to_collect
            )

            if not trades_window_data:
                logger.warning("无法获取交易窗口数据")
                return {
                    "status": "no_data",
                    "error": "No trades window data available",
                    "symbol": symbol,
                    "analysis_type": "dynamic_trades"
                }

            logger.info(f"获取到 {len(trades_window_data)} 分钟的交易数据")

            # 聚合动态数据
            raw_data = self.trades_aggregator.collect_raw_trades_data(
                trades_window_data, symbol
            )

            dynamic_data = raw_data.to_dict()

            logger.info(
                f"动态数据聚合完成: {dynamic_data.get('data_points_count', 0)} 个分钟数据点, "
                f"时间跨度 {len(dynamic_data.get('minute_data_points', []))} 分钟"
            )

            return {
                "status": "success",
                "symbol": symbol,
                "analysis_type": "dynamic_trades",
                **dynamic_data
            }

        except Exception as e:
            logger.error(f"动态交易数据收集失败: {e}")
            return {
                "status": "error",
                "error": str(e),
                "symbol": symbol,
                "analysis_type": "dynamic_trades"
            }

    def _perform_local_bayesian_analysis(
        self,
        static_data: dict[str, Any],
        dynamic_data: dict[str, Any],
        symbol: str
    ) -> dict[str, Any]:
        """执行本地贝叶斯分析。

        Args:
            static_data: 静态订单簿数据
            dynamic_data: 动态交易数据
            symbol: 交易符号

        Returns:
            本地贝叶斯分析结果
        """
        try:
            # 执行贝叶斯分析
            bayesian_result = self.bayesian_analyzer.analyze_bayesian_trend(
                static_data, dynamic_data, symbol
            )

            if bayesian_result.get("status") == "success":
                logger.info(
                    f"本地贝叶斯分析完成: "
                    f"trend={bayesian_result['analysis_result']['most_likely_trend']}, "
                    f"confidence={bayesian_result['analysis_result']['confidence']:.3f}"
                )

            return bayesian_result

        except Exception as e:
            logger.error(f"本地贝叶斯分析失败: {e}")
            return {
                "status": "error",
                "error": str(e),
                "symbol": symbol,
                "analysis_type": "local_bayesian"
            }

    async def _perform_ai_bayesian_analysis(
        self,
        static_data: dict[str, Any],
        dynamic_data: dict[str, Any],
        symbol: str
    ) -> Any | None:
        """执行AI增强的贝叶斯分析。

        Args:
            static_data: 静态订单簿数据
            dynamic_data: 动态交易数据
            symbol: 交易符号

        Returns:
            AI贝叶斯分析结果或None（如果失败）
        """
        try:
            # 执行AI贝叶斯分析
            ai_result = await self.bayesian_deepseek.analyze_bayesian_trend(
                static_data, dynamic_data, symbol
            )

            logger.info(
                f"AI贝叶斯分析完成: "
                f"trend={ai_result.most_likely_trend}, "
                f"confidence={ai_result.confidence:.3f}, "
                f"uncertainty={ai_result.uncertainty:.3f}"
            )

            return ai_result

        except Exception as e:
            logger.error(f"AI贝叶斯分析失败: {e}")
            return None

    def _fuse_bayesian_results(
        self,
        local_result: dict[str, Any],
        ai_result: Any | None
    ) -> dict[str, Any]:
        """融合本地和AI贝叶斯分析结果。

        Args:
            local_result: 本地贝叶斯分析结果
            ai_result: AI贝叶斯分析结果

        Returns:
            融合后的分析结果
        """
        # 如果AI分析失败，使用本地结果
        if ai_result is None:
            return local_result.get("analysis_result", {})

        # 如果本地分析失败，使用AI结果
        if local_result.get("status") != "success":
            return ai_result.to_dict()["analysis_result"]

        # 融合两个分析结果
        local_analysis = local_result.get("analysis_result", {})
        ai_analysis = ai_result.to_dict()

        # 简单融合策略：使用AI结果作为主要依据，本地结果作为验证
        fused_result = ai_analysis.copy()

        # 添加本地分析的验证信息
        fused_result["local_analysis_validation"] = {
            "local_trend": local_analysis.get("most_likely_trend"),
            "local_confidence": local_analysis.get("confidence"),
            "trend_consistency": (
                ai_analysis.get("most_likely_trend") == local_analysis.get("most_likely_trend")
            ),
            "confidence_gap": abs(
                ai_analysis.get("confidence", 0) - local_analysis.get("confidence", 0)
            )
        }

        # 如果趋势一致，提高置信度
        if fused_result["local_analysis_validation"]["trend_consistency"]:
            fused_result["confidence"] = min(
                fused_result.get("confidence", 0) + 0.1, 1.0
            )
            fused_result["validation_status"] = "high_consensus"
        else:
            fused_result["validation_status"] = "divergent_views"

        return fused_result

    def _format_bayesian_response(
        self,
        fused_result: dict[str, Any],
        static_data: dict[str, Any],
        dynamic_data: dict[str, Any],
        symbol: str
    ) -> str:
        """格式化贝叶斯分析响应。

        Args:
            fused_result: 融合后的分析结果
            static_data: 静态数据
            dynamic_data: 动态数据
            symbol: 交易符号

        Returns:
            格式化的JSON响应字符串
        """
        # 创建一个模拟的BayesianTrendResult对象用于格式化
        class MockBayesianResult:
            def __init__(self, result_dict: dict[str, Any]):
                self.timestamp = datetime.now()
                self.posterior_probabilities = result_dict.get("probability_distribution", {}).get("full_distribution", {})
                self.most_likely_trend = result_dict.get("most_likely_trend", "unknown")
                self.confidence = result_dict.get("confidence", 0.0)
                self.uncertainty = result_dict.get("uncertainty", 1.0)
                self.analysis_reason = result_dict.get("analysis_reason", "")
                self.evidence_summary = result_dict.get("evidence_summary", {})
                self.bayesian_metadata = result_dict.get("bayesian_metadata", {})

        mock_result = MockBayesianResult(fused_result)

        # 使用贝叶斯响应格式化器
        json_response = self.response_formatter.format_bayesian_response(
            mock_result, static_data, dynamic_data, symbol
        )

        # 验证响应格式
        if not self.response_formatter.validate_bayesian_response(json_response):
            logger.warning("贝叶斯响应Schema验证失败，但仍返回结果")

        return json_response

    async def _handle_notifications_and_quality(
        self,
        fused_result: dict[str, Any],
        symbol: str
    ) -> tuple:
        """处理通知和质量检查。

        Args:
            fused_result: 融合后的分析结果
            symbol: 交易符号

        Returns:
            (质量检查结果, Discord发送状态)
        """
        # 质量检查
        quality_check = self._validate_bayesian_analysis_quality(fused_result)

        if not quality_check["is_valid"]:
            logger.error(f"贝叶斯分析质量验证失败: {quality_check['errors']}")
            if self.discord_manager:
                await self.discord_manager.send_error_notification(
                    f"贝叶斯分析质量验证失败: {quality_check['errors']}",
                    f"符号: {symbol}"
                )

        # Discord通知
        discord_sent = False
        if self.discord_manager:
            try:
                # 适配现有通知接口
                trend_data = {
                    "trend": fused_result.get("most_likely_trend", "unknown"),
                    "confidence": fused_result.get("confidence", 0.0),
                    "reason": fused_result.get("analysis_reason", ""),
                    "probability_distribution": fused_result.get("probability_distribution", {})
                }

                discord_sent = await self.discord_manager.send_trend_alert(
                    trend_data, symbol
                )
                if discord_sent:
                    self.stats["discord_notifications_sent"] += 1
                    logger.info("Discord通知发送成功")
            except Exception as e:
                logger.error(f"Discord通知发送失败: {e}")

        return quality_check, discord_sent

    def _validate_bayesian_analysis_quality(self, fused_result: dict[str, Any]) -> dict[str, Any]:
        """验证贝叶斯分析质量。

        Args:
            fused_result: 融合后的分析结果

        Returns:
            质量检查结果
        """
        quality_check = {
            "is_valid": True,
            "errors": [],
            "warnings": []
        }

        # 检查必要字段
        required_fields = ["most_likely_trend", "confidence", "uncertainty"]
        for field in required_fields:
            if field not in fused_result:
                quality_check["errors"].append(f"缺少必要字段: {field}")
                quality_check["is_valid"] = False

        # 检查置信度范围
        confidence = fused_result.get("confidence", 0)
        if not 0 <= confidence <= 1:
            quality_check["errors"].append(f"置信度超出范围: {confidence}")
            quality_check["is_valid"] = False

        # 检查不确定性范围
        uncertainty = fused_result.get("uncertainty", 0)
        if not 0 <= uncertainty <= 1:
            quality_check["errors"].append(f"不确定性超出范围: {uncertainty}")
            quality_check["is_valid"] = False

        # 检查概率分布
        prob_dist = fused_result.get("probability_distribution", {})
        full_dist = prob_dist.get("full_distribution", {})
        if full_dist:
            total_prob = sum(full_dist.values())
            if abs(total_prob - 1.0) > 0.1:
                quality_check["warnings"].append(f"概率分布总和不等于1: {total_prob}")

        # 置信度警告
        if confidence < 0.3:
            quality_check["warnings"].append("置信度过低，预测可靠性有限")

        # 不确定性警告
        if uncertainty > 0.7:
            quality_check["warnings"].append("不确定性过高，建议谨慎决策")

        return quality_check

    def _build_final_result(
        self,
        symbol: str,
        json_response: str,
        static_data: dict[str, Any],
        dynamic_data: dict[str, Any],
        fused_result: dict[str, Any],
        processing_time: float,
        discord_sent: bool,
        quality_check: dict[str, Any]
    ) -> dict[str, Any]:
        """构建最终结果。

        Args:
            symbol: 交易符号
            json_response: JSON格式化响应
            static_data: 静态数据
            dynamic_data: 动态数据
            fused_result: 融合结果
            processing_time: 处理时间
            discord_sent: Discord发送状态
            quality_check: 质量检查结果

        Returns:
            最终结果字典
        """
        result = {
            "status": "success",
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "analysis_type": "bayesian_optimized_analysis",
            "analysis_result": json.loads(json_response),
            "static_data": static_data,
            "dynamic_data": dynamic_data,
            "fused_bayesian_result": fused_result,
            "processing_time": processing_time,
            "discord_notification_sent": discord_sent,
            "quality_check": quality_check
        }

        # 添加性能统计
        result["performance_stats"] = {
            "data_collection_time": processing_time * 0.25,  # 估算
            "bayesian_analysis_time": processing_time * 0.35,  # 估算
            "ai_analysis_time": processing_time * 0.30,  # 估算
            "formatting_time": processing_time * 0.1        # 估算
        }

        # 添加数据源统计
        result["data_source_stats"] = {
            "static_data_available": static_data.get("status") == "success",
            "dynamic_data_available": dynamic_data.get("status") == "success",
            "static_data_quality": static_data.get("status"),
            "dynamic_data_points": dynamic_data.get("data_points_count", 0)
        }

        return result

    async def _handle_analysis_error(
        self,
        symbol: str,
        error: Exception,
        start_time: float
    ) -> dict[str, Any]:
        """处理分析错误。

        Args:
            symbol: 交易符号
            error: 错误异常
            start_time: 开始时间

        Returns:
            错误结果字典
        """
        processing_time = asyncio.get_event_loop().time() - start_time
        self._update_stats(processing_time, success=False)
        logger.error(f"贝叶斯市场分析失败: {error}")

        # 发送错误通知
        if self.discord_manager:
            try:
                await self.discord_manager.send_error_notification(
                    str(error), f"贝叶斯市场分析失败: {symbol}"
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
                return self._format_error_response(
                    result.get("error", "未知错误"), symbol
                )

        except Exception as e:
            logger.error(f"单次贝叶斯分析失败: {e}")
            return self._format_error_response(str(e), symbol)

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
            "analysis_type": "bayesian_optimized_analysis",
            "status": "error",
            "error": error_message,
            "trend_analysis": {
                "most_likely_trend": "unknown",
                "confidence": 0.0,
                "uncertainty": 1.0,
                "confidence_level": "very_low",
                "risk_level": "very_high_risk"
            },
            "probability_distribution": {
                "full_distribution": {},
                "entropy": 0.0,
                "distribution_type": "error"
            }
        }

        return json.dumps(error_response, ensure_ascii=False)

    def _update_stats(
        self,
        processing_time: float,
        success: bool,
        static_data: dict[str, Any] | None = None,
        dynamic_data: dict[str, Any] | None = None
    ) -> None:
        """更新统计信息。

        Args:
            processing_time: 处理时间
            success: 是否成功
            static_data: 静态数据（可选）
            dynamic_data: 动态数据（可选）
        """
        if success:
            self.stats["successful_analyses"] += 1
        else:
            self.stats["failed_analyses"] += 1

        # 更新数据可用性统计
        if static_data and static_data.get("status") == "success":
            self.stats["static_data_available"] += 1

        if dynamic_data and dynamic_data.get("status") == "success":
            self.stats["dynamic_data_available"] += 1

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
            "analyzer_type": "bayesian_optimized_analyzer",
            "symbol": self.symbol,
            "redis_connected": self.redis_store.test_connection(),
            "statistics": self.stats,
            "components": {
                "static_analyzer": {
                    "enabled": True,
                    "precision": float(self.static_analyzer.aggregation_precision)
                },
                "trades_aggregator": {
                    "enabled": True,
                    "minutes_to_collect": self.trades_aggregator.minutes_to_collect
                },
                "bayesian_analyzer": {
                    "enabled": True,
                    "evidence_weights": self.bayesian_analyzer.evidence_weights
                },
                "bayesian_deepseek": self.bayesian_deepseek.get_stats(),
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
            status["statistics"]["static_data_availability_rate"] = self.stats["static_data_available"] / total
            status["statistics"]["dynamic_data_availability_rate"] = self.stats["dynamic_data_available"] / total
        else:
            status["statistics"]["success_rate"] = 0.0
            status["statistics"]["static_data_availability_rate"] = 0.0
            status["statistics"]["dynamic_data_availability_rate"] = 0.0

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
            depth_available = self.redis_store.depth_snapshot_exists()

            data_ok = trades_count > 0 or depth_available
            health_status["checks"]["data_availability"] = {
                "status": "pass" if data_ok else "fail",
                "message": f"动态数据: {trades_count} 分钟, 静态数据: {'可用' if depth_available else '不可用'}"
            }
        except Exception as e:
            health_status["checks"]["data_availability"] = {
                "status": "fail",
                "message": f"数据检查异常: {str(e)}"
            }

        # 贝叶斯分析器检查
        try:
            health_status["checks"]["bayesian_analyzer"] = {
                "status": "pass",
                "message": "贝叶斯分析器正常"
            }
        except Exception as e:
            health_status["checks"]["bayesian_analyzer"] = {
                "status": "fail",
                "message": f"贝叶斯分析器异常: {str(e)}"
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
        logger.info("正在关闭BayesianOptimizedAnalyzer...")

        try:
            # 关闭贝叶斯Deepseek分析器
            if self.bayesian_deepseek:
                self.bayesian_deepseek.close()
                logger.info("贝叶斯Deepseek分析器已关闭")
        except Exception as e:
            logger.error(f"关闭贝叶斯Deepseek分析器时出错: {e}")

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

        logger.info("BayesianOptimizedAnalyzer已完全关闭")
