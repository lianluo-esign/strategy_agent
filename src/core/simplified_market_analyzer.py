"""简化市场分析器 - 专注三步分析流程。

实现清晰的三步分析流程：
1. 初始化
2. 读取redis中的历史数据并按照要求进行聚合
3. 将orderbook和trades_window聚合后的数据通过prompt发送给deepseek进行分析并返回标准JSON输出
"""

import logging
from datetime import datetime
from typing import Any

from .price_aggregator import PriceAggregator
from .result_validator import result_validator
from .trading_event_publisher import TradingEventPublisher
from .unified_deepseek_analyzer import UnifiedDeepSeekAnalyzer
from .volume_profile_analyzer import VolumeProfileAnalyzer

logger = logging.getLogger(__name__)


class SimplifiedMarketAnalyzer:
    """简化市场分析器，专注三步分析流程。

    核心流程：
    1. 初始化所有依赖组件
    2. 从Redis读取历史数据并进行聚合
    3. 调用DeepSeek分析并返回标准JSON格式结果
    """

    def __init__(
        self,
        redis_store: Any,
        deepseek_config: dict[str, Any],
        price_aggregation_precision: float = 1.0,
        vp_aggregation_precision: float = 10.0,
        trading_event_publisher: TradingEventPublisher | None = None,
    ):
        """初始化简化市场分析器。

        Args:
            redis_store: Redis数据存储实例
            deepseek_config: DeepSeek配置字典
            price_aggregation_precision: 深度快照价格聚合精度
            vp_aggregation_precision: Volume Profile聚合精度
            trading_event_publisher: 交易事件发布器实例（可选）
        """
        logger.info("Initializing SimplifiedMarketAnalyzer")

        # 1. 初始化依赖组件
        self.redis_store = redis_store

        # 初始化价格聚合器
        self.price_aggregator = PriceAggregator(
            precision=price_aggregation_precision, enabled=True, max_price_levels=5000
        )

        # 初始化Volume Profile分析器
        self.vp_analyzer = VolumeProfileAnalyzer(
            aggregation_precision=vp_aggregation_precision, min_volume_threshold=0.1
        )

        # 初始化DeepSeek统一分析器
        self.unified_analyzer = UnifiedDeepSeekAnalyzer(
            api_key=deepseek_config["api_key"],
            base_url=deepseek_config.get("base_url", "https://api.deepseek.com/v1"),
            model=deepseek_config.get("model", "deepseek-chat"),
            max_tokens=deepseek_config.get("max_tokens", 4000),  # 减少令牌数以获得更简洁的响应
            temperature=deepseek_config.get("temperature", 0.1),
            timeout=deepseek_config.get("timeout", 90),
            max_retries=deepseek_config.get("max_retries", 3),
        )

        # 存储交易事件发布器实例
        self.trading_event_publisher = trading_event_publisher

        # 初始化JSON模板（优化性能）
        self._json_template = '{{"grid_delta": {}, "grid_quantity": {}, "active_side": "{}"}}'

        logger.info("SimplifiedMarketAnalyzer initialized successfully")

    async def analyze_market(
        self, symbol: str = "BTCFDUSD"
    ) -> dict[str, Any]:
        """执行完整的市场分析流程。

        Args:
            symbol: 交易符号

        Returns:
            包含标准交易参数的分析结果
        """
        logger.info(f"Starting simplified market analysis for {symbol}")

        try:
            # 第一步：初始化（已在__init__中完成）
            logger.info("Step 1: Initialization completed")

            # 第二步：读取Redis中的历史数据并聚合
            logger.info("Step 2: Reading and aggregating data from Redis")
            market_data = await self._read_and_aggregate_market_data(symbol)

            if market_data.get("status") != "success":
                return self._create_error_result(
                    symbol,
                    f"Data aggregation failed: {market_data.get('error', 'Unknown error')}"
                )

            # 第三步：发送给DeepSeek分析
            logger.info("Step 3: Sending aggregated data to DeepSeek for analysis")
            analysis_result = self.unified_analyzer.analyze_unified_market_data(
                aggregated_bids=market_data["aggregated_bids"],
                aggregated_asks=market_data["aggregated_asks"],
                vp_result=market_data["vp_result"],
                symbol=symbol
            )

            # 验证并提取交易参数
            try:
                trading_params = result_validator.validate_and_extract_trading_params(analysis_result)

                result = {
                    "symbol": symbol,
                    "timestamp": datetime.now(),
                    "analysis_type": "simplified_market_analysis",
                    "status": "success",
                    "trading_params": trading_params,
                    "market_data_summary": {
                        "bid_levels": len(market_data["aggregated_bids"]),
                        "ask_levels": len(market_data["aggregated_asks"]),
                        "vp_price_levels": market_data["vp_result"].get("price_levels_count", 0),
                        "total_volume": market_data["vp_result"].get("total_volume", 0),
                    }
                }

                logger.info(f"Simplified market analysis completed successfully: {trading_params}")

                # 发布交易参数到Redis（可选，不影响主流程）
                if self.trading_event_publisher:
                    try:
                        await self._publish_trading_params(trading_params, symbol)
                    except Exception as publish_error:
                        logger.error(f"❌ Failed to publish trading parameters: {publish_error}")
                        # 发布失败不影响主分析结果
                else:
                    logger.debug("TradingEventPublisher not configured, skipping Redis publish")

                return result

            except Exception as validation_error:
                logger.error(f"Trading parameter validation failed: {validation_error}")
                return self._create_error_result(symbol, f"Validation failed: {str(validation_error)}")

        except Exception as e:
            logger.error(f"Simplified market analysis failed: {e}")
            return self._create_error_result(symbol, str(e))

    async def _read_and_aggregate_market_data(
        self, symbol: str
    ) -> dict[str, Any]:
        """从Redis读取并聚合市场数据。

        Args:
            symbol: 交易符号

        Returns:
            聚合后的市场数据
        """
        try:
            # 读取深度快照数据
            depth_snapshot = self.redis_store.get_latest_depth_snapshot()
            if not depth_snapshot:
                return {
                    "status": "no_data",
                    "error": "No depth snapshot available in Redis"
                }

            logger.info(f"Retrieved depth snapshot: {depth_snapshot.symbol} from {depth_snapshot.timestamp}")

            # 聚合订单簿数据
            aggregated_bids, aggregated_asks = self.price_aggregator.aggregate_order_book_levels(
                depth_snapshot.bids, depth_snapshot.asks
            )

            if not aggregated_bids and not aggregated_asks:
                return {
                    "status": "no_data",
                    "error": "No order book data after aggregation"
                }

            logger.info(f"Aggregated order book: {len(aggregated_bids)} bids, {len(aggregated_asks)} asks")

            # 读取历史交易数据
            trades_window_data = self.redis_store.get_recent_trade_data(minutes=1440)  # 24小时
            if not trades_window_data:
                return {
                    "status": "no_data",
                    "error": "No trades window data available in Redis"
                }

            logger.info(f"Retrieved {len(trades_window_data)} minutes of trade data")

            # Volume Profile分析
            vp_result = self.vp_analyzer.analyze_volume_profile(trades_window_data, symbol)
            if vp_result.get("status") != "success":
                return {
                    "status": "error",
                    "error": f"Volume Profile analysis failed: {vp_result.get('error')}"
                }

            logger.info(
                f"Volume Profile generated: {vp_result.get('price_levels_count', 0)} levels, "
                f"total_volume={vp_result.get('total_volume', 0):.2f}"
            )

            return {
                "status": "success",
                "aggregated_bids": aggregated_bids,
                "aggregated_asks": aggregated_asks,
                "vp_result": vp_result,
                "depth_snapshot_time": depth_snapshot.timestamp,
                "trades_data_count": len(trades_window_data)
            }

        except Exception as e:
            logger.error(f"Failed to read and aggregate market data: {e}")
            return {"status": "error", "error": str(e)}

    def _create_error_result(self, symbol: str, error_message: str) -> dict[str, Any]:
        """创建错误分析结果。

        Args:
            symbol: 交易符号
            error_message: 错误消息

        Returns:
            错误分析结果
        """
        return {
            "symbol": symbol,
            "timestamp": datetime.now(),
            "analysis_type": "simplified_market_analysis",
            "status": "error",
            "error": error_message,
            "trading_params": None,
        }

    async def _publish_trading_params(self, trading_params: dict[str, Any], symbol: str) -> bool:
        """发布交易参数到Redis channel。

        Args:
            trading_params: 交易参数字典
            symbol: 交易符号

        Returns:
            发布是否成功
        """
        if not self.trading_event_publisher:
            logger.debug("TradingEventPublisher not configured, skipping publish")
            return False

        try:
            logger.info(f"🚀 Publishing trading parameters to Redis: {trading_params}")

            # 使用预编译的JSON模板（优化性能）
            ai_response_text = self._json_template.format(
                trading_params["grid_delta"],
                trading_params["grid_quantity"],
                trading_params["active_side"]
            )

            # 使用TradingEventPublisher发布到Redis
            publish_success = await self.trading_event_publisher.process_ai_analysis_and_publish(ai_response_text)

            if publish_success:
                logger.info(f"✅ Successfully published trading parameters for {symbol}")
                return True
            else:
                logger.warning(f"❌ Failed to publish trading parameters for {symbol}")
                return False

        except Exception as e:
            logger.error(f"❌ Error publishing trading parameters for {symbol}: {e}")
            return False

    def get_status(self) -> dict[str, Any]:
        """获取分析器状态。

        Returns:
            分析器状态字典
        """
        return {
            "analyzer_type": "simplified_market_analyzer",
            "price_aggregation": {
                "enabled": True,
                "precision": float(self.price_aggregator.precision),
                "max_levels": self.price_aggregator.max_price_levels,
            },
            "volume_profile": {
                "enabled": True,
                "precision": float(self.vp_analyzer.aggregation_precision),
            },
            "unified_analyzer": {
                "enabled": True,
                "model": self.unified_analyzer.model,
                "max_tokens": self.unified_analyzer.max_tokens,
            },
            "redis_connection": {
                "connected": self.redis_store.test_connection(),
                "depth_snapshot_available": self.redis_store.depth_snapshot_exists(),
                "trades_window_available": self.redis_store.get_trade_window_count() > 0,
            },
            "trading_event_publisher": {
                "configured": self.trading_event_publisher is not None,
                "enabled": self.trading_event_publisher is not None,
            }
        }

    async def close(self) -> None:
        """关闭分析器资源。"""
        try:
            if self.unified_analyzer:
                self.unified_analyzer.close()
                logger.info("Unified DeepSeek analyzer closed")

            # 关闭TradingEventPublisher资源
            if self.trading_event_publisher:
                try:
                    await self.trading_event_publisher.close()
                    logger.info("TradingEventPublisher closed")
                except Exception as e:
                    logger.error(f"Error closing TradingEventPublisher: {e}")

        except Exception as e:
            logger.error(f"Error closing analyzer: {e}")

        logger.info("SimplifiedMarketAnalyzer closed")