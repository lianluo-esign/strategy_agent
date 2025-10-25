"""简化的市场分析器 - 专注于核心功能。

这个模块提供简化的市场分析流程：
1. 读取Redis缓存的深度快照数据
2. 按照聚合精度进行聚合处理
3. 将聚合数据提交给DeepSeek进行支撑阻力分析
4. 可视化挂单分布
"""

import logging
from datetime import datetime
from typing import Any

from .deepseek_analyzer import DeepSeekOrderBookAnalyzer
from .price_aggregator import PriceAggregator

logger = logging.getLogger(__name__)


class SimpleMarketAnalyzer:
    """简化的市场分析器，专注于核心功能。

    这个分析器提供：
    1. Redis深度快照数据读取和聚合
    2. DeepSeek LLM支撑阻力分析
    3. 挂单分布可视化
    """

    def __init__(
        self,
        redis_store,
        price_aggregation_precision: float = 1.0,
        deepseek_config: dict[str, Any] | None = None,
        visualizer: Any | None = None,
    ):
        """初始化简化市场分析器。

        Args:
            redis_store: Redis数据存储实例
            price_aggregation_precision: 价格聚合精度（例如：1.0表示$1精度）
            deepseek_config: DeepSeek配置字典
            visualizer: 可视化工具实例
        """
        self.redis_store = redis_store
        self.visualizer = visualizer

        # 初始化价格聚合器
        self.price_aggregator = PriceAggregator(
            precision=price_aggregation_precision, enabled=True, max_price_levels=5000
        )
        logger.info(
            f"Initialized price aggregator with precision=${price_aggregation_precision}"
        )

        # 初始化DeepSeek分析器
        self.deepseek_analyzer = None
        if deepseek_config and deepseek_config.get("enable", False):
            try:
                self.deepseek_analyzer = DeepSeekOrderBookAnalyzer(
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
                logger.info("DeepSeek LLM analyzer initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize DeepSeek analyzer: {e}")
                self.deepseek_analyzer = None
        else:
            logger.info("DeepSeek LLM analysis is disabled")

    def analyze_market(self, symbol: str = "BTCFDUSD") -> dict[str, Any]:
        """执行完整的市场分析流程。

        Args:
            symbol: 交易符号

        Returns:
            包含分析结果的字典
        """
        logger.info(f"Starting simplified market analysis for {symbol}")

        try:
            # 步骤1: 读取Redis缓存的深度快照数据
            snapshot = self.redis_store.get_latest_depth_snapshot()
            if not snapshot:
                logger.warning("No depth snapshot available in Redis")
                return self._create_empty_result(symbol)

            logger.info(
                f"Retrieved depth snapshot: {snapshot.symbol} from {snapshot.timestamp}"
            )

            # 步骤2: 按照聚合精度进行聚合处理
            aggregated_bids, aggregated_asks = (
                self.price_aggregator.aggregate_order_book_levels(
                    snapshot.bids, snapshot.asks
                )
            )

            if not aggregated_bids and not aggregated_asks:
                logger.warning("No order book data after aggregation")
                return self._create_empty_result(symbol)

            logger.info(
                f"Aggregated order book: {len(aggregated_bids)} bid levels, "
                f"{len(aggregated_asks)} ask levels"
            )

            # 步骤3: 将聚合数据提交给DeepSeek进行支撑阻力分析
            deepseek_analysis = None
            if self.deepseek_analyzer:
                try:
                    logger.info("Starting DeepSeek LLM support/resistance analysis")
                    deepseek_analysis = (
                        self.deepseek_analyzer.analyze_order_book_with_llm(
                            aggregated_bids, aggregated_asks, symbol
                        )
                    )
                    logger.info("DeepSeek LLM analysis completed successfully")

                    # 在info日志中打印分析结果
                    self._log_deepseek_analysis(deepseek_analysis)

                except Exception as e:
                    logger.error(f"DeepSeek LLM analysis failed: {e}")
            else:
                logger.info("DeepSeek LLM analysis is disabled")

            # 步骤4: 可视化挂单分布
            if self.visualizer:
                try:
                    logger.info("Creating order book visualization")
                    output_file = self.visualizer.create_order_book_distribution_chart(
                        snapshot
                    )
                    logger.info(f"Order book visualization created: {output_file}")
                except Exception as e:
                    logger.error(f"Failed to create visualization: {e}")

            # 返回分析结果
            result = {
                "symbol": symbol,
                "timestamp": datetime.now(),
                "snapshot_timestamp": snapshot.timestamp,
                "aggregated_bids": aggregated_bids,
                "aggregated_asks": aggregated_asks,
                "aggregation_precision": self.price_aggregator.precision,
                "deepseek_analysis": deepseek_analysis,
                "status": "success",
            }

            logger.info(f"Market analysis completed for {symbol}")
            return result

        except Exception as e:
            logger.error(f"Market analysis failed: {e}")
            return self._create_error_result(symbol, str(e))

    def _log_deepseek_analysis(self, analysis_result: dict[str, Any]) -> None:
        """在info日志中打印DeepSeek分析结果。

        Args:
            analysis_result: DeepSeek分析结果
        """
        if analysis_result.get("status") != "success":
            logger.info(
                f"❌ DeepSeek分析失败: {analysis_result.get('error', '未知错误')}"
            )
            return

        symbol = analysis_result.get("symbol", "UNKNOWN")
        logger.info(f"=== {symbol} DeepSeek LLM 支撑阻力分析 ===")

        structured_analysis = analysis_result.get("structured_analysis")
        if structured_analysis:
            # 打印支撑区域
            if "支撑区域" in structured_analysis:
                logger.info("🟢 买盘支撑区域:")
                for i, support in enumerate(structured_analysis["支撑区域"], 1):
                    logger.info(
                        f"  支撑 {i}: {support.get('价格区间', 'N/A')} | "
                        f"强度: {support.get('强度', 'N/A')} | "
                        f"特征: {support.get('特征', 'N/A')[:80]}{'...' if len(support.get('特征', '')) > 80 else ''}"
                    )

            # 打印阻力区域
            if "阻力区域" in structured_analysis:
                logger.info("🔻 卖盘阻力区域:")
                for i, resistance in enumerate(structured_analysis["阻力区域"], 1):
                    logger.info(
                        f"  阻力 {i}: {resistance.get('价格区间', 'N/A')} | "
                        f"强度: {resistance.get('强度', 'N/A')} | "
                        f"特征: {resistance.get('特征', 'N/A')[:80]}{'...' if len(resistance.get('特征', '')) > 80 else ''}"
                    )

            # 打印市场平衡状态
            if "市场平衡" in structured_analysis:
                balance = structured_analysis["市场平衡"]
                logger.info(f"⚖️  市场平衡状态: {balance.get('状态', 'N/A')}")
                if balance.get("分析"):
                    logger.info(
                        f"   分析: {balance['分析'][:100]}{'...' if len(balance['分析']) > 100 else ''}"
                    )

            # 打印关键价位
            if "关键价位" in structured_analysis:
                logger.info("📍 关键价位:")
                for i, key_level in enumerate(
                    structured_analysis["关键价位"][:5], 1
                ):  # 限制前5个
                    logger.info(
                        f"  关键价位 {i}: ${key_level.get('价格', 'N/A')} | "
                        f"作用: {key_level.get('作用', 'N/A')} | "
                        f"重要性: {key_level.get('重要性', 'N/A')[:50]}{'...' if len(key_level.get('重要性', '')) > 50 else ''}"
                    )

        else:
            # 打印原始内容
            raw_content = analysis_result.get("raw_content")
            if raw_content:
                logger.info("📋 DeepSeek分析内容:")
                # 分行打印，每行限制长度
                for line in raw_content.split("\n")[:10]:  # 限制前10行
                    if line.strip():
                        logger.info(
                            f"   {line[:120]}{'...' if len(line) > 120 else ''}"
                        )

        logger.info("=" * 60)

    def _create_empty_result(self, symbol: str) -> dict[str, Any]:
        """创建空的分析结果。

        Args:
            symbol: 交易符号

        Returns:
            空的分析结果字典
        """
        return {
            "symbol": symbol,
            "timestamp": datetime.now(),
            "aggregated_bids": {},
            "aggregated_asks": {},
            "aggregation_precision": self.price_aggregator.precision,
            "deepseek_analysis": None,
            "status": "no_data",
        }

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
            "aggregated_bids": {},
            "aggregated_asks": {},
            "aggregation_precision": self.price_aggregator.precision,
            "deepseek_analysis": None,
            "status": "error",
            "error": error_message,
        }

    def get_status(self) -> dict[str, Any]:
        """获取分析器状态。

        Returns:
            分析器状态字典
        """
        return {
            "price_aggregation": {
                "enabled": True,
                "precision": float(self.price_aggregator.precision),
                "max_levels": self.price_aggregator.max_price_levels,
            },
            "deepseek_analysis": {
                "enabled": self.deepseek_analyzer is not None,
                "model": self.deepseek_analyzer.model
                if self.deepseek_analyzer
                else None,
            },
            "visualization": {
                "enabled": self.visualizer is not None,
            },
            "redis_connection": self.redis_store.test_connection(),
            "depth_snapshot_available": self.redis_store.depth_snapshot_exists(),
        }
