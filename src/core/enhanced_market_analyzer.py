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
from .trading_event_publisher import TradingEventPublisher
from .unified_deepseek_analyzer import UnifiedDeepSeekAnalyzer
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
        trading_event_publisher_config: Any | None = None,
    ):
        """初始化增强型市场分析器。

        Args:
            redis_store: Redis数据存储实例
            price_aggregation_precision: 深度快照价格聚合精度（例如：1.0表示$1精度）
            vp_aggregation_precision: Volume Profile聚合精度（例如：10.0表示$10精度）
            deepseek_config: DeepSeek配置字典
            visualizer: 可视化工具实例
            trading_event_publisher_config: 交易事件发布器配置
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
        self.unified_deepseek_analyzer = None

        if deepseek_config and deepseek_config.get("enable", False):
            try:
                # 优先使用统一分析器
                if deepseek_config.get("use_unified_analysis", True):
                    self.unified_deepseek_analyzer = UnifiedDeepSeekAnalyzer(
                        api_key=deepseek_config["api_key"],
                        base_url=deepseek_config.get(
                            "base_url", "https://api.deepseek.com/v1"
                        ),
                        model=deepseek_config.get("model", "deepseek-chat"),
                        max_tokens=deepseek_config.get("max_tokens", 6000),
                        temperature=deepseek_config.get("temperature", 0.1),
                        timeout=deepseek_config.get("timeout", 90),
                        max_retries=deepseek_config.get("max_retries", 3),
                    )
                    logger.info("Unified DeepSeek analyzer initialized successfully")
                else:
                    # 深度快照分析器（传统分离模式）
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
                self.unified_deepseek_analyzer = None
        else:
            logger.info("DeepSeek LLM analysis is disabled")

        # 初始化交易事件发布器
        self.trading_event_publisher = None
        if trading_event_publisher_config and trading_event_publisher_config.enable:
            try:
                self.trading_event_publisher = TradingEventPublisher(
                    config=trading_event_publisher_config
                )
                logger.info("Trading event publisher initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize trading event publisher: {e}")
                self.trading_event_publisher = None
        else:
            logger.info("Trading event publisher is disabled")

    async def perform_dual_analysis(self, symbol: str = "BTCFDUSD") -> dict[str, Any]:
        """执行双重分析：静态深度快照分析 + 动态Volume Profile分析。

        Args:
            symbol: 交易符号

        Returns:
            包含两种分析结果的完整字典
        """
        logger.info(f"Starting enhanced dual market analysis for {symbol}")

        try:
            # 检查是否使用统一分析模式
            if self.unified_deepseek_analyzer:
                logger.info("Using unified analysis mode")
                return await self._perform_unified_analysis(symbol)
            else:
                logger.info("Using traditional dual analysis mode")
                return self._perform_tradual_dual_analysis(symbol)

        except Exception as e:
            logger.error(f"Enhanced dual analysis failed: {e}")
            return self._create_error_result(symbol, str(e))

    async def _perform_unified_analysis(self, symbol: str) -> dict[str, Any]:
        """执行统一分析：单次AI请求处理深度快照和Volume Profile数据。

        Args:
            symbol: 交易符号

        Returns:
            包含统一分析结果的完整字典
        """
        logger.info(f"Starting unified market analysis for {symbol}")

        try:
            # 第一步：获取并处理深度快照数据
            depth_analysis = self._get_depth_snapshot_data(symbol)
            if depth_analysis.get("status") != "success":
                return self._create_error_result(symbol, f"Depth analysis failed: {depth_analysis.get('error')}")

            # 第二步：获取并处理Volume Profile数据
            vp_analysis = self._get_volume_profile_data(symbol)
            if vp_analysis.get("status") != "success":
                return self._create_error_result(symbol, f"Volume Profile analysis failed: {vp_analysis.get('error')}")

            # 第三步：统一AI分析
            unified_analysis = None
            if self.unified_deepseek_analyzer:
                try:
                    logger.info("Starting unified DeepSeek LLM analysis")
                    unified_analysis = self.unified_deepseek_analyzer.analyze_unified_market_data(
                        aggregated_bids=depth_analysis["aggregated_bids"],
                        aggregated_asks=depth_analysis["aggregated_asks"],
                        vp_result=vp_analysis["vp_analysis"],
                        symbol=symbol
                    )
                    logger.info("Unified DeepSeek analysis completed successfully")

                    # 在info日志中打印统一分析结果
                    self._log_unified_analysis(unified_analysis)

                    # 发布交易事件（如果启用且分析成功）
                    if (self.trading_event_publisher and
                        unified_analysis.get("status") == "success" and
                        unified_analysis.get("raw_content")):
                        try:
                            logger.info("Processing trading event publication")
                            success = await self.trading_event_publisher.process_ai_analysis_and_publish(
                                unified_analysis["raw_content"]
                            )
                            if success:
                                logger.info("Trading event published successfully")
                            else:
                                logger.warning("Trading event publication failed")
                        except Exception as e:
                            logger.error(f"Trading event publication error: {e}")

                except Exception as e:
                    logger.error(f"Unified DeepSeek analysis failed: {e}")
                    unified_analysis = {"status": "error", "error": str(e)}

            # 第四步：可视化处理
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

            # 整合统一分析结果
            result = {
                "symbol": symbol,
                "timestamp": datetime.now(),
                "analysis_type": "unified_market_analysis",
                "depth_analysis": depth_analysis,
                "volume_profile_analysis": vp_analysis,
                "unified_analysis": unified_analysis,
                "visualization": visualization_result,
                "status": "success",
            }

            logger.info(f"Unified market analysis completed for {symbol}")
            return result

        except Exception as e:
            logger.error(f"Unified analysis failed: {e}")
            return self._create_error_result(symbol, str(e))

    def _perform_tradual_dual_analysis(self, symbol: str) -> dict[str, Any]:
        """执行传统的双重分析模式。

        Args:
            symbol: 交易符号

        Returns:
            包含传统双重分析结果的完整字典
        """
        logger.info(f"Starting traditional dual analysis for {symbol}")

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
                "analysis_type": "traditional_dual_analysis",
                "depth_analysis": depth_analysis,
                "volume_profile_analysis": vp_analysis,
                "visualization": visualization_result,
                "status": "success",
            }

            logger.info(f"Traditional dual analysis completed for {symbol}")
            return result

        except Exception as e:
            logger.error(f"Traditional dual analysis failed: {e}")
            return self._create_error_result(symbol, str(e))

    def _get_depth_snapshot_data(self, symbol: str) -> dict[str, Any]:
        """获取深度快照数据（不包含AI分析）。

        Args:
            symbol: 交易符号

        Returns:
            深度快照数据结果
        """
        logger.info("Getting depth snapshot data")

        try:
            # 读取深度快照数据
            snapshot = self.redis_store.get_latest_depth_snapshot()
            if not snapshot:
                logger.warning("No depth snapshot available")
                return {
                    "status": "no_data",
                    "error": "No depth snapshot available",
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
                }

            logger.info(
                f"Aggregated order book: {len(aggregated_bids)} bid levels, "
                f"{len(aggregated_asks)} ask levels"
            )

            return {
                "status": "success",
                "snapshot": snapshot,
                "aggregated_bids": aggregated_bids,
                "aggregated_asks": aggregated_asks,
                "aggregation_precision": self.price_aggregator.precision,
            }

        except Exception as e:
            logger.error(f"Depth snapshot data retrieval failed: {e}")
            return {"status": "error", "error": str(e)}

    def _analyze_depth_snapshot(self, symbol: str) -> dict[str, Any]:
        """分析深度快照数据（包含AI分析）。

        Args:
            symbol: 交易符号

        Returns:
            深度快照分析结果
        """
        logger.info("Starting depth snapshot analysis")

        # 获取基础数据
        depth_data = self._get_depth_snapshot_data(symbol)
        if depth_data.get("status") != "success":
            return depth_data

        # DeepSeek深度快照分析
        deepseek_analysis = None
        if self.deepseek_orderbook_analyzer:
            try:
                logger.info("Starting DeepSeek LLM depth snapshot analysis")
                deepseek_analysis = (
                    self.deepseek_orderbook_analyzer.analyze_order_book_with_llm(
                        depth_data["aggregated_bids"], depth_data["aggregated_asks"], symbol
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

        depth_data["deepseek_analysis"] = deepseek_analysis
        return depth_data

    def _get_volume_profile_data(self, symbol: str) -> dict[str, Any]:
        """获取Volume Profile数据（不包含AI分析）。

        Args:
            symbol: 交易符号

        Returns:
            Volume Profile数据结果
        """
        logger.info("Getting Volume Profile data")

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
                }

            logger.info(
                f"Volume Profile generated: {vp_result.get('price_levels_count', 0)} price levels, "
                f"total_volume={vp_result.get('total_volume', 0):.2f}"
            )

            return {
                "status": "success",
                "vp_analysis": vp_result,
            }

        except Exception as e:
            logger.error(f"Volume Profile data retrieval failed: {e}")
            return {"status": "error", "error": str(e)}

    def _analyze_volume_profile(self, symbol: str) -> dict[str, Any]:
        """分析Volume Profile数据（包含AI分析）。

        Args:
            symbol: 交易符号

        Returns:
            Volume Profile分析结果
        """
        logger.info("Starting Volume Profile analysis")

        # 获取基础数据
        vp_data = self._get_volume_profile_data(symbol)
        if vp_data.get("status") != "success":
            return vp_data

        # DeepSeek Volume Profile分析
        deepseek_analysis = None
        if self.deepseek_vp_analyzer:
            try:
                logger.info("Starting DeepSeek LLM Volume Profile analysis")
                deepseek_analysis = (
                    self.deepseek_vp_analyzer.analyze_volume_profile_with_llm(
                        vp_data["vp_analysis"]
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

        vp_data["deepseek_analysis"] = deepseek_analysis
        return vp_data

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

    def _log_unified_analysis(self, analysis_result: dict[str, Any]) -> None:
        """在info日志中打印统一分析结果。

        Args:
            analysis_result: 统一分析结果
        """
        if analysis_result.get("status") != "success":
            logger.info(
                f"❌ 统一AI分析失败: {analysis_result.get('error', '未知错误')}"
            )
            return

        symbol = analysis_result.get("symbol", "UNKNOWN")
        logger.info(f"=== {symbol} 统一AI分析结果 - 高频做市策略 ===")

        structured_analysis = analysis_result.get("structured_analysis")
        if structured_analysis:
            # 打印短期支撑位
            if "短期支撑位" in structured_analysis:
                logger.info("🟢 短期支撑位（入场机会）:")
                for i, support in enumerate(structured_analysis["短期支撑位"][:3], 1):
                    logger.info(
                        f"  支撑位 {i}: ${support.get('价格', 'N/A')} | "
                        f"可靠性: {support.get('可靠性评分', 'N/A')}/100 | "
                        f"入场区间: {support.get('推荐入场区间', 'N/A')}"
                    )
                    reason = support.get('形成原因', 'N/A')
                    logger.info(f"           原因: {reason[:60]}{'...' if len(reason) > 60 else ''}")

            # 打印短期阻力位
            if "短期阻力位" in structured_analysis:
                logger.info("🔻 短期阻力位（退出目标）:")
                for i, resistance in enumerate(structured_analysis["短期阻力位"][:3], 1):
                    logger.info(
                        f"  阻力位 {i}: ${resistance.get('价格', 'N/A')} | "
                        f"可靠性: {resistance.get('可靠性评分', 'N/A')}/100 | "
                        f"退出区间: {resistance.get('推荐退出区间', 'N/A')}"
                    )
                    reason = resistance.get('形成原因', 'N/A')
                    logger.info(f"           原因: {reason[:60]}{'...' if len(reason) > 60 else ''}")

            # 打印流动性供应区域
            if "集中流动性供应区域" in structured_analysis:
                liquidity = structured_analysis["集中流动性供应区域"]
                logger.info("💰 集中流动性供应区域:")
                logger.info(f"  最佳区间: {liquidity.get('最佳价格区间', 'N/A')}")
                backup_zones = liquidity.get('备选区间', [])
                if backup_zones:
                    logger.info(f"  备选区间: {', '.join(backup_zones)}")
                logger.info(f"  市场特征: {liquidity.get('市场特征', 'N/A')[:80]}{'...' if len(liquidity.get('市场特征', '')) > 80 else ''}")

            # 打印做市策略要点
            if "做市策略要点" in structured_analysis:
                strategy = structured_analysis["做市策略要点"]
                logger.info("📋 做市策略要点:")
                logger.info(f"  主要机会: {strategy.get('主要机会', 'N/A')[:80]}{'...' if len(strategy.get('主要机会', '')) > 80 else ''}")
                logger.info(f"  策略总结: {strategy.get('策略总结', 'N/A')[:100]}{'...' if len(strategy.get('策略总结', '')) > 100 else ''}")

        else:
            # 打印原始内容
            raw_content = analysis_result.get("raw_content")
            if raw_content:
                logger.info("📋 统一AI分析内容:")
                for line in raw_content.split("\n")[:8]:  # 限制前8行
                    if line.strip():
                        logger.info(f"   {line[:120]}{'...' if len(line) > 120 else ''}")

        logger.info("=" * 70)

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
        # 确定当前使用的分析模式
        analysis_mode = "unified" if self.unified_deepseek_analyzer else "traditional" if (self.deepseek_orderbook_analyzer or self.deepseek_vp_analyzer) else "disabled"

        status = {
            "analysis_mode": analysis_mode,
            "depth_analysis": {
                "price_aggregation": {
                    "enabled": True,
                    "precision": float(self.price_aggregator.precision),
                    "max_levels": self.price_aggregator.max_price_levels,
                },
            },
            "volume_profile_analysis": {
                "vp_analyzer": {
                    "enabled": True,
                    "precision": float(self.vp_analyzer.aggregation_precision),
                },
            },
            "visualization": {
                "enabled": self.visualizer is not None,
            },
            "redis_connection": self.redis_store.test_connection(),
            "depth_snapshot_available": self.redis_store.depth_snapshot_exists(),
            "trades_window_available": self.redis_store.get_trade_window_count() > 0,
        }

        # 根据分析模式添加不同的AI分析状态
        if analysis_mode == "unified":
            status["unified_analysis"] = {
                "enabled": True,
                "model": self.unified_deepseek_analyzer.model,
                "max_tokens": self.unified_deepseek_analyzer.max_tokens,
                "timeout": self.unified_deepseek_analyzer.timeout,
            }
        elif analysis_mode == "traditional":
            status["depth_analysis"]["deepseek_analysis"] = {
                "enabled": self.deepseek_orderbook_analyzer is not None,
                "model": self.deepseek_orderbook_analyzer.model
                if self.deepseek_orderbook_analyzer
                else None,
            }
            status["volume_profile_analysis"]["deepseek_analysis"] = {
                "enabled": self.deepseek_vp_analyzer is not None,
                "model": self.deepseek_vp_analyzer.model
                if self.deepseek_vp_analyzer
                else None,
            }
        else:
            status["ai_analysis"] = {"enabled": False}

        return status

    async def close(self) -> None:
        """关闭分析器资源。"""
        if self.unified_deepseek_analyzer:
            try:
                self.unified_deepseek_analyzer.close()
                logger.info("Unified DeepSeek analyzer closed")
            except Exception as e:
                logger.error(f"Error closing unified DeepSeek analyzer: {e}")

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

        if self.trading_event_publisher:
            try:
                await self.trading_event_publisher.close()
                logger.info("Trading event publisher closed")
            except Exception as e:
                logger.error(f"Error closing trading event publisher: {e}")

        logger.info("Enhanced market analyzer resources closed")
