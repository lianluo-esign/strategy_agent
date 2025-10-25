"""增强型市场分析器 - 集成静态和动态分析功能。

这个模块提供完整的市场分析功能：
1. 深度快照数据的静态支撑阻力分析
2. Volume Profile数据的动态市场分析
3. 双AI分析集成和统一输出
"""

import logging
from datetime import datetime
from typing import Any

from .deepseek_analyzer import (
    DeepSeekOrderBookAnalyzer,
)
from .deepseek_vp_analyzer import DeepSeekVPAnalyzer
from .price_aggregator import PriceAggregator
from .volume_profile_analyzer import VolumeProfileAnalyzer

logger = logging.getLogger(__name__)


class EnhancedMarketAnalyzer:
    """增强型市场分析器，集成静态和动态分析功能。

    这个分析器提供：
    1. Redis深度快照数据读取和聚合
    2. DeepSeek LLM静态支撑阻力分析
    3. 24小时交易数据的Volume Profile分析
    4. DeepSeek LLM动态市场分析
    5. 统一的分析结果输出
    """

    def __init__(
        self,
        redis_store: Any,
        price_aggregation_precision: float = 1.0,
        vp_aggregation_precision: float = 10.0,
        deepseek_config: dict[str, Any] | None = None,
        visualizer: Any | None = None,
    ):
        """初始化增强型市场分析器。

        Args:
            redis_store: Redis数据存储实例
            price_aggregation_precision: 深度快照价格聚合精度（例如：1.0表示$1精度）
            vp_aggregation_precision: Volume Profile聚合精度（例如：10.0表示$10精度）
            deepseek_config: DeepSeek配置字典
            visualizer: 可视化工具实例
        """
        self.redis_store = redis_store
        self.visualizer = visualizer

        # 初始化深度快照价格聚合器
        self.price_aggregator = PriceAggregator(
            precision=price_aggregation_precision, enabled=True, max_price_levels=5000
        )
        logger.info(
            f"Initialized depth snapshot price aggregator with precision=${price_aggregation_precision}"
        )

        # 初始化Volume Profile分析器
        self.vp_analyzer = VolumeProfileAnalyzer(
            aggregation_precision=vp_aggregation_precision, min_volume_threshold=0.1
        )
        logger.info(
            f"Initialized Volume Profile analyzer with precision=${vp_aggregation_precision}"
        )

        # 初始化DeepSeek分析器
        self.deepseek_orderbook_analyzer = None
        self.deepseek_vp_analyzer = None

        if deepseek_config and deepseek_config.get("enable", False):
            try:
                # 深度快照分析器
                self.deepseek_orderbook_analyzer = DeepSeekOrderBookAnalyzer(
                    api_key=deepseek_config["api_key"],
                    base_url=deepseek_config.get(
                        "base_url", "https://api.deepseek.com/v1"
                    ),
                    model=deepseek_config.get("model", "deepseek-chat"),
                    max_tokens=deepseek_config.get("max_tokens", 4000),
                    temperature=deepseek_config.get("temperature", 0.1),
                    timeout=deepseek_config.get("timeout", 60),
                    max_retries=deepseek_config.get("max_retries", 3),
                )
                logger.info("DeepSeek order book analyzer initialized successfully")

                # Volume Profile分析器
                self.deepseek_vp_analyzer = DeepSeekVPAnalyzer(
                    api_key=deepseek_config["api_key"],
                    base_url=deepseek_config.get(
                        "base_url", "https://api.deepseek.com/v1"
                    ),
                    model=deepseek_config.get("model", "deepseek-chat"),
                    max_tokens=deepseek_config.get("max_tokens", 4000),
                    temperature=deepseek_config.get("temperature", 0.1),
                    timeout=deepseek_config.get("timeout", 60),
                    max_retries=deepseek_config.get("max_retries", 3),
                )
                logger.info("DeepSeek Volume Profile analyzer initialized successfully")

            except Exception as e:
                logger.error(f"Failed to initialize DeepSeek analyzers: {e}")
                self.deepseek_orderbook_analyzer = None
                self.deepseek_vp_analyzer = None
        else:
            logger.info("DeepSeek LLM analysis is disabled")

    def perform_dual_analysis(self, symbol: str = "BTCFDUSD") -> dict[str, Any]:
        """执行双重分析：静态深度快照分析 + 动态Volume Profile分析。

        Args:
            symbol: 交易符号

        Returns:
            包含两种分析结果的完整字典
        """
        logger.info(f"Starting enhanced dual market analysis for {symbol}")

        try:
            # 第一部分：静态深度快照分析
            depth_analysis = self._analyze_depth_snapshot(symbol)

            # 第二部分：动态Volume Profile分析
            vp_analysis = self._analyze_volume_profile(symbol)

            # 第三部分：可视化处理
            visualization_result = None
            if self.visualizer and depth_analysis.get("snapshot"):
                try:
                    logger.info("Creating order book visualization")
                    output_file = self.visualizer.create_order_book_distribution_chart(
                        depth_analysis["snapshot"]
                    )
                    visualization_result = {
                        "status": "success",
                        "output_file": output_file,
                    }
                    logger.info(f"Order book visualization created: {output_file}")
                except Exception as e:
                    logger.error(f"Failed to create visualization: {e}")
                    visualization_result = {"status": "error", "error": str(e)}

            # 整合分析结果
            result = {
                "symbol": symbol,
                "timestamp": datetime.now(),
                "analysis_type": "enhanced_dual_analysis",
                "depth_analysis": depth_analysis,
                "volume_profile_analysis": vp_analysis,
                "visualization": visualization_result,
                "status": "success",
            }

            logger.info(f"Enhanced dual analysis completed for {symbol}")
            return result

        except Exception as e:
            logger.error(f"Enhanced dual analysis failed: {e}")
            return self._create_error_result(symbol, str(e))

    def _analyze_depth_snapshot(self, symbol: str) -> dict[str, Any]:
        """分析深度快照数据。

        Args:
            symbol: 交易符号

        Returns:
            深度快照分析结果
        """
        logger.info("Starting depth snapshot analysis")

        try:
            # 读取深度快照数据
            snapshot = self.redis_store.get_latest_depth_snapshot()
            if not snapshot:
                logger.warning("No depth snapshot available")
                return {
                    "status": "no_data",
                    "error": "No depth snapshot available",
                    "deepseek_analysis": None,
                }

            logger.info(
                f"Retrieved depth snapshot: {snapshot.symbol} from {snapshot.timestamp}"
            )

            # 聚合订单簿数据
            aggregated_bids, aggregated_asks = (
                self.price_aggregator.aggregate_order_book_levels(
                    snapshot.bids, snapshot.asks
                )
            )

            if not aggregated_bids and not aggregated_asks:
                logger.warning("No order book data after aggregation")
                return {
                    "status": "no_data",
                    "error": "No order book data after aggregation",
                    "deepseek_analysis": None,
                }

            logger.info(
                f"Aggregated order book: {len(aggregated_bids)} bid levels, "
                f"{len(aggregated_asks)} ask levels"
            )

            # DeepSeek深度快照分析
            deepseek_analysis = None
            if self.deepseek_orderbook_analyzer:
                try:
                    logger.info("Starting DeepSeek LLM depth snapshot analysis")
                    deepseek_analysis = (
                        self.deepseek_orderbook_analyzer.analyze_order_book_with_llm(
                            aggregated_bids, aggregated_asks, symbol
                        )
                    )
                    logger.info(
                        "DeepSeek depth snapshot analysis completed successfully"
                    )

                    # 在info日志中打印分析结果
                    self._log_deepseek_analysis(deepseek_analysis, "深度快照")

                except Exception as e:
                    logger.error(f"DeepSeek depth snapshot analysis failed: {e}")
                    deepseek_analysis = {"status": "error", "error": str(e)}

            return {
                "status": "success",
                "snapshot": snapshot,
                "aggregated_bids": aggregated_bids,
                "aggregated_asks": aggregated_asks,
                "aggregation_precision": self.price_aggregator.precision,
                "deepseek_analysis": deepseek_analysis,
            }

        except Exception as e:
            logger.error(f"Depth snapshot analysis failed: {e}")
            return {"status": "error", "error": str(e), "deepseek_analysis": None}

    def _analyze_volume_profile(self, symbol: str) -> dict[str, Any]:
        """分析Volume Profile数据。

        Args:
            symbol: 交易符号

        Returns:
            Volume Profile分析结果
        """
        logger.info("Starting Volume Profile analysis")

        try:
            # 获取24小时交易窗口数据
            trades_window_data = self.redis_store.get_recent_trade_data(
                minutes=1440
            )  # 24小时 = 1440分钟
            if not trades_window_data:
                logger.warning("No trades window data available")
                return {
                    "status": "no_data",
                    "error": "No trades window data available",
                    "deepseek_analysis": None,
                }

            logger.info(f"Retrieved {len(trades_window_data)} minutes of trade data")

            # Volume Profile分析
            vp_result = self.vp_analyzer.analyze_volume_profile(
                trades_window_data, symbol
            )

            if vp_result.get("status") != "success":
                logger.warning(
                    f"Volume Profile analysis failed: {vp_result.get('error')}"
                )
                return {
                    "status": "error",
                    "error": vp_result.get("error"),
                    "vp_analysis": vp_result,
                    "deepseek_analysis": None,
                }

            logger.info(
                f"Volume Profile generated: {vp_result.get('price_levels_count', 0)} price levels, "
                f"total_volume={vp_result.get('total_volume', 0):.2f}"
            )

            # DeepSeek Volume Profile分析
            deepseek_analysis = None
            if self.deepseek_vp_analyzer:
                try:
                    logger.info("Starting DeepSeek LLM Volume Profile analysis")
                    deepseek_analysis = (
                        self.deepseek_vp_analyzer.analyze_volume_profile_with_llm(
                            vp_result
                        )
                    )
                    logger.info(
                        "DeepSeek Volume Profile analysis completed successfully"
                    )

                    # 在info日志中打印分析结果
                    self._log_deepseek_analysis(deepseek_analysis, "Volume Profile")

                except Exception as e:
                    logger.error(f"DeepSeek Volume Profile analysis failed: {e}")
                    deepseek_analysis = {"status": "error", "error": str(e)}

            return {
                "status": "success",
                "vp_analysis": vp_result,
                "deepseek_analysis": deepseek_analysis,
            }

        except Exception as e:
            logger.error(f"Volume Profile analysis failed: {e}")
            return {"status": "error", "error": str(e), "deepseek_analysis": None}

    def _log_deepseek_analysis(
        self, analysis_result: dict[str, Any], analysis_type: str
    ) -> None:
        """在info日志中打印DeepSeek分析结果。

        Args:
            analysis_result: DeepSeek分析结果
            analysis_type: 分析类型（"深度快照" 或 "Volume Profile"）
        """
        if analysis_result.get("status") != "success":
            logger.info(
                f"❌ DeepSeek {analysis_type}分析失败: {analysis_result.get('error', '未知错误')}"
            )
            return

        symbol = analysis_result.get("symbol", "UNKNOWN")
        logger.info(f"=== {symbol} DeepSeek LLM {analysis_type} 分析 ===")

        structured_analysis = analysis_result.get("structured_analysis")
        if structured_analysis:
            # 根据分析类型调整输出格式
            if analysis_type == "深度快照":
                self._log_depth_snapshot_analysis(structured_analysis)
            elif analysis_type == "Volume Profile":
                self._log_volume_profile_analysis(structured_analysis)
        else:
            # 打印原始内容
            raw_content = analysis_result.get("raw_content")
            if raw_content:
                logger.info(f"📋 DeepSeek {analysis_type}分析内容:")
                for line in raw_content.split("\n")[:8]:  # 限制前8行
                    if line.strip():
                        logger.info(
                            f"   {line[:120]}{'...' if len(line) > 120 else ''}"
                        )

        logger.info("=" * 60)

    def _log_depth_snapshot_analysis(self, structured_analysis: dict[str, Any]) -> None:
        """打印深度快照分析结果。

        Args:
            structured_analysis: 结构化分析结果
        """
        if "支撑区域" in structured_analysis:
            logger.info("🟢 买盘支撑区域:")
            for i, support in enumerate(
                structured_analysis["支撑区域"][:3], 1
            ):  # 限制前3个
                logger.info(
                    f"  支撑 {i}: {support.get('价格区间', 'N/A')} | "
                    f"强度: {support.get('强度', 'N/A')} | "
                    f"特征: {support.get('特征', 'N/A')[:60]}{'...' if len(support.get('特征', '')) > 60 else ''}"
                )

        if "阻力区域" in structured_analysis:
            logger.info("🔻 卖盘阻力区域:")
            for i, resistance in enumerate(
                structured_analysis["阻力区域"][:3], 1
            ):  # 限制前3个
                logger.info(
                    f"  阻力 {i}: {resistance.get('价格区间', 'N/A')} | "
                    f"强度: {resistance.get('强度', 'N/A')} | "
                    f"特征: {resistance.get('特征', 'N/A')[:60]}{'...' if len(resistance.get('特征', '')) > 60 else ''}"
                )

        if "市场平衡" in structured_analysis:
            balance = structured_analysis["市场平衡"]
            logger.info(f"⚖️  市场平衡状态: {balance.get('状态', 'N/A')}")
            if balance.get("分析"):
                logger.info(
                    f"   分析: {balance['分析'][:80]}{'...' if len(balance['分析']) > 80 else ''}"
                )

    def _log_volume_profile_analysis(self, structured_analysis: dict[str, Any]) -> None:
        """打印Volume Profile分析结果。

        Args:
            structured_analysis: 结构化分析结果
        """
        if "poc分析" in structured_analysis:
            poc = structured_analysis["poc分析"]
            logger.info("🎯 POC点分析:")
            logger.info(f"   POC价格: {poc.get('poc价格', 'N/A')}")
            logger.info(
                f"   市场意义: {poc.get('市场意义', 'N/A')[:80]}{'...' if len(poc.get('市场意义', '')) > 80 else ''}"
            )

        if "成交密集区域" in structured_analysis:
            logger.info("📊 成交密集区域:")
            for i, area in enumerate(
                structured_analysis["成交密集区域"][:2], 1
            ):  # 限制前2个
                logger.info(
                    f"   区域 {i}: {area.get('价格区间', 'N/A')} | "
                    f"成交量: {area.get('成交量', 'N/A')} | "
                    f"做市适用性: {area.get('做市适用性', 'N/A')}"
                )

        if "流动性做市建议" in structured_analysis:
            suggestions = structured_analysis["流动性做市建议"]
            logger.info("💡 流动性做市建议:")
            logger.info(f"   最佳部署区域: {suggestions.get('最佳部署区域', 'N/A')}")
            if suggestions.get("风险控制点"):
                logger.info(f"   风险控制点: {suggestions.get('风险控制点', 'N/A')}")

    def _create_error_result(self, symbol: str, error_message: str) -> dict[str, Any]:
        """创建错误分析结果。

        Args:
            symbol: 交易符号
            error_message: 错误消息

        Returns:
            错误分析结果字典
        """
        return {
            "symbol": symbol,
            "timestamp": datetime.now(),
            "analysis_type": "enhanced_dual_analysis",
            "depth_analysis": {"status": "error", "error": error_message},
            "volume_profile_analysis": {"status": "error", "error": error_message},
            "visualization": {"status": "error", "error": error_message},
            "status": "error",
            "error": error_message,
        }

    def get_status(self) -> dict[str, Any]:
        """获取分析器状态。

        Returns:
            分析器状态字典
        """
        return {
            "depth_analysis": {
                "price_aggregation": {
                    "enabled": True,
                    "precision": float(self.price_aggregator.precision),
                    "max_levels": self.price_aggregator.max_price_levels,
                },
                "deepseek_analysis": {
                    "enabled": self.deepseek_orderbook_analyzer is not None,
                    "model": self.deepseek_orderbook_analyzer.model
                    if self.deepseek_orderbook_analyzer
                    else None,
                },
            },
            "volume_profile_analysis": {
                "vp_analyzer": {
                    "enabled": True,
                    "precision": float(self.vp_analyzer.aggregation_precision),
                },
                "deepseek_analysis": {
                    "enabled": self.deepseek_vp_analyzer is not None,
                    "model": self.deepseek_vp_analyzer.model
                    if self.deepseek_vp_analyzer
                    else None,
                },
            },
            "visualization": {
                "enabled": self.visualizer is not None,
            },
            "redis_connection": self.redis_store.test_connection(),
            "depth_snapshot_available": self.redis_store.depth_snapshot_exists(),
            "trades_window_available": self.redis_store.get_trade_window_count() > 0,
        }

    def close(self) -> None:
        """关闭分析器资源。"""
        if self.deepseek_orderbook_analyzer:
            try:
                self.deepseek_orderbook_analyzer.close()
                logger.info("DeepSeek order book analyzer closed")
            except Exception as e:
                logger.error(f"Error closing DeepSeek order book analyzer: {e}")

        if self.deepseek_vp_analyzer:
            try:
                self.deepseek_vp_analyzer.close()
                logger.info("DeepSeek Volume Profile analyzer closed")
            except Exception as e:
                logger.error(f"Error closing DeepSeek Volume Profile analyzer: {e}")

        logger.info("Enhanced market analyzer resources closed")
