"""增强型市场分析器代理 - 集成静态和动态双AI分析功能。

这个代理提供简化的市场分析流程：
1. Redis深度快照数据读取和聚合处理
2. DeepSeek LLM深度快照支撑阻力分析
3. 24小时交易数据Volume Profile分析
4. DeepSeek LLM动态市场分析
5. 挂单分布可视化
"""

import asyncio
import logging
import signal
import sys
from datetime import datetime

from ..core.enhanced_market_analyzer import EnhancedMarketAnalyzer
from ..core.redis_client import RedisDataStore
from ..utils.config import Settings
from ..visualization.order_book_visualizer import OrderBookVisualizer

logger = logging.getLogger(__name__)

# 配置常量
SHUTDOWN_TASK_TIMEOUT = 5.0
RETRY_DELAY_ON_ERROR = 10


class AnalyzerAgent:
    """增强型分析器代理，集成静态和动态双AI分析功能。

    这个代理执行增强的市场分析流程：
    - 静态深度快照数据的支撑阻力分析
    - 动态Volume Profile数据的市场分析
    - 统一的可视化输出
    """

    def __init__(self, settings: Settings):
        """初始化增强型分析器代理。

        Args:
            settings: 配置设置
        """
        self.settings = settings

        # 初始化Redis连接
        self.redis_store = RedisDataStore(
            host=settings.redis.host, port=settings.redis.port, db=settings.redis.db
        )

        # 初始化可视化工具
        self.visualizer = None
        if settings.analyzer.visualization.enabled:
            try:
                self.visualizer = OrderBookVisualizer(
                    config=settings.analyzer.visualization
                )
                logger.info("Order book visualizer initialized")
            except Exception as e:
                logger.error(f"Failed to initialize order book visualizer: {e}")

        # 初始化DeepSeek配置
        deepseek_config = None
        if settings.analyzer.deepseek.enable and settings.analyzer.deepseek.api_key:
            deepseek_config = {
                "enable": True,
                "api_key": settings.analyzer.deepseek.api_key,
                "base_url": settings.analyzer.deepseek.base_url,
                "model": settings.analyzer.deepseek.model,
                "max_tokens": settings.analyzer.deepseek.max_tokens,
                "temperature": settings.analyzer.deepseek.temperature,
                "timeout": settings.analyzer.deepseek.timeout,
                "max_retries": 3,
            }
        else:
            logger.info("DeepSeek LLM analysis is disabled")

        # 初始化增强型市场分析器
        self.market_analyzer = EnhancedMarketAnalyzer(
            redis_store=self.redis_store,
            price_aggregation_precision=settings.analyzer.price_aggregation.precision,
            vp_aggregation_precision=getattr(
                settings.analyzer, "volume_profile_aggregation_precision", 10.0
            ),
            deepseek_config=deepseek_config,
            visualizer=self.visualizer,
        )

        logger.info(
            "Enhanced analyzer agent initialized with dual AI analysis capabilities"
        )

        # 控制标志
        self.is_running = False
        self.shutdown_event = asyncio.Event()

    def setup_signal_handlers(self) -> None:
        """设置异步兼容的信号处理器。"""
        self._shutdown_requested = False

    def _signal_handler(self) -> None:
        """处理关闭信号的直接同步处理器。"""
        logger.info("Signal received, triggering shutdown...")
        self._shutdown_requested = True
        self.is_running = False
        self.shutdown_event.set()

    async def start(self) -> None:
        """启动增强型分析流程。"""
        logger.info("Starting Enhanced Market Analyzer Agent")

        # 设置信号处理器
        self.setup_signal_handlers()

        # 添加信号处理器用于优雅关闭
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, self._signal_handler)
                logger.debug(f"Signal handler for {sig} registered")
            except Exception as e:
                logger.error(f"Failed to register signal handler for {sig}: {e}")
                raise RuntimeError(f"Signal handler registration failed: {e}") from e

        # 测试Redis连接
        if not self.redis_store.test_connection():
            logger.error("Failed to connect to Redis. Exiting...")
            return

        # 主分析循环
        try:
            self.is_running = True
            await self._analysis_loop()
        except asyncio.CancelledError:
            logger.info("Analysis loop cancelled")
        except Exception as e:
            logger.error(f"Enhanced analyzer agent error: {e}")
        finally:
            await self._shutdown()

    async def _analysis_loop(self) -> None:
        """主分析循环。"""
        interval = self.settings.analyzer.analysis.interval_seconds

        while self.is_running:
            try:
                logger.debug("Starting enhanced market analysis cycle")
                await self._perform_enhanced_analysis_cycle()

                # 等待下一个周期，支持取消
                try:
                    await asyncio.wait_for(self.shutdown_event.wait(), timeout=interval)
                    logger.info("Shutdown event triggered, exiting analysis loop")
                    break
                except TimeoutError:
                    # 正常超时，继续下一个周期
                    continue

            except Exception as e:
                logger.error(f"Enhanced analysis cycle error: {e}")
                # 重试前等待，但尊重关闭信号
                try:
                    await asyncio.wait_for(
                        self.shutdown_event.wait(), timeout=RETRY_DELAY_ON_ERROR
                    )
                except TimeoutError:
                    # 正常超时，继续重试
                    continue
                # 如果等待完成，表示请求关闭
                break

    async def _perform_enhanced_analysis_cycle(self) -> None:
        """执行增强型分析周期。"""
        try:
            # 执行双重分析：静态深度快照 + 动态Volume Profile
            symbol = self.settings.binance.symbol
            analysis_result = self.market_analyzer.perform_dual_analysis(symbol)

            if analysis_result["status"] == "success":
                logger.info(
                    f"Enhanced dual analysis completed successfully for {symbol}"
                )

                # 打印分析摘要
                await self._log_analysis_summary(analysis_result)

                # 可以在这里添加其他处理逻辑
                # 比如发送通知、更新仪表板等

            elif analysis_result["status"] == "no_data":
                logger.info("No data available for enhanced analysis")
            else:
                logger.error(
                    f"Enhanced analysis failed: {analysis_result.get('error', 'Unknown error')}"
                )

        except Exception as e:
            logger.error(f"Enhanced analysis cycle failed: {e}")

    async def _log_analysis_summary(self, analysis_result: dict) -> None:
        """记录分析摘要。

        Args:
            analysis_result: 分析结果字典
        """
        symbol = analysis_result.get("symbol", "UNKNOWN")
        analysis_type = analysis_result.get("analysis_type", "unknown")

        logger.info(f"=== {symbol} Enhanced Analysis Summary ===")

        # 根据分析类型显示不同的摘要
        if analysis_type == "unified_market_analysis":
            # 统一分析模式摘要
            logger.info("🔍 Analysis Mode: Unified AI Analysis")

            # 数据摘要
            depth_analysis = analysis_result.get("depth_analysis", {})
            vp_analysis = analysis_result.get("volume_profile_analysis", {})

            if depth_analysis.get("status") == "success":
                aggregated_bids = depth_analysis.get("aggregated_bids", {})
                aggregated_asks = depth_analysis.get("aggregated_asks", {})
                logger.info(
                    f"📊 Depth Snapshot: {len(aggregated_bids)} bid levels, "
                    f"{len(aggregated_asks)} ask levels"
                )
            else:
                logger.info(f"❌ Depth snapshot failed: {depth_analysis.get('error', 'Unknown error')}")

            if vp_analysis.get("status") == "success":
                vp_data = vp_analysis.get("vp_analysis", {})
                logger.info(
                    f"📈 Volume Profile: {vp_data.get('price_levels_count', 0)} price levels, "
                    f"total_volume={vp_data.get('total_volume', 0):.2f}"
                )
            else:
                logger.info(f"❌ Volume Profile failed: {vp_analysis.get('error', 'Unknown error')}")

            # 统一AI分析结果摘要
            unified_analysis = analysis_result.get("unified_analysis", {})
            if unified_analysis and unified_analysis.get("status") == "success":
                logger.info("✅ Unified DeepSeek analysis completed successfully")

                # 显示关键结果摘要
                structured_analysis = unified_analysis.get("structured_analysis")
                if structured_analysis:
                    support_levels = structured_analysis.get("短期支撑位", [])
                    resistance_levels = structured_analysis.get("短期阻力位", [])
                    liquidity_zone = structured_analysis.get("集中流动性供应区域", {})

                    logger.info(f"🟢 Support Levels Identified: {len(support_levels)}")
                    logger.info(f"🔻 Resistance Levels Identified: {len(resistance_levels)}")
                    if liquidity_zone.get("最佳价格区间"):
                        logger.info(f"💰 Optimal Liquidity Zone: {liquidity_zone['最佳价格区间']}")
            else:
                logger.info(f"❌ Unified AI analysis failed: {unified_analysis.get('error', 'Unknown error') if unified_analysis else 'No analysis result'}")

        elif analysis_type == "traditional_dual_analysis":
            # 传统分离分析模式摘要
            logger.info("🔍 Analysis Mode: Traditional Dual Analysis")

            # 深度快照分析摘要
            depth_analysis = analysis_result.get("depth_analysis", {})
            if depth_analysis.get("status") == "success":
                aggregated_bids = depth_analysis.get("aggregated_bids", {})
                aggregated_asks = depth_analysis.get("aggregated_asks", {})
                logger.info(
                    f"📊 Depth Snapshot: {len(aggregated_bids)} bid levels, "
                    f"{len(aggregated_asks)} ask levels"
                )

                depth_deepseek = depth_analysis.get("deepseek_analysis")
                if depth_deepseek and depth_deepseek.get("status") == "success":
                    logger.info("✅ DeepSeek depth snapshot analysis completed")
                else:
                    logger.info("❌ DeepSeek depth snapshot analysis failed")
            else:
                logger.info(
                    f"❌ Depth snapshot analysis failed: {depth_analysis.get('error', 'Unknown error')}"
                )

            # Volume Profile分析摘要
            vp_analysis = analysis_result.get("volume_profile_analysis", {})
            if vp_analysis.get("status") == "success":
                vp_data = vp_analysis.get("vp_analysis", {})
                logger.info(
                    f"📈 Volume Profile: {vp_data.get('price_levels_count', 0)} price levels, "
                    f"total_volume={vp_data.get('total_volume', 0):.2f}"
                )

                vp_deepseek = vp_analysis.get("deepseek_analysis")
                if vp_deepseek and vp_deepseek.get("status") == "success":
                    logger.info("✅ DeepSeek Volume Profile analysis completed")
                else:
                    logger.info("❌ DeepSeek Volume Profile analysis failed")
            else:
                logger.info(
                    f"❌ Volume Profile analysis failed: {vp_analysis.get('error', 'Unknown error')}"
                )
        else:
            logger.info(f"🔍 Analysis Mode: {analysis_type}")
            logger.info("❌ Unknown analysis type")

        # 可视化摘要
        visualization = analysis_result.get("visualization", {})
        if visualization and visualization.get("status") == "success":
            logger.info(
                f"📊 Visualization: {visualization.get('output_file', 'Generated')}"
            )
        elif visualization:
            logger.info(
                f"❌ Visualization failed: {visualization.get('error', 'Unknown error')}"
            )

        logger.info("=" * 50)

    async def _shutdown(self) -> None:
        """清理和关闭增强型代理。"""
        logger.info("Shutting down Enhanced Market Analyzer Agent")

        self.is_running = False
        self._shutdown_requested = True

        # 取消所有待处理的任务
        tasks = [
            task for task in asyncio.all_tasks() if task is not asyncio.current_task()
        ]
        if tasks:
            logger.info(f"Cancelling {len(tasks)} pending tasks...")
            for task in tasks:
                task.cancel()

            # 等待任务完成，带超时
            try:
                await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=SHUTDOWN_TASK_TIMEOUT,
                )
            except TimeoutError:
                logger.warning("Some tasks did not complete within timeout")

        # 关闭增强型分析器资源
        if self.market_analyzer:
            try:
                self.market_analyzer.close()
                logger.info("Enhanced market analyzer closed")
            except Exception as e:
                logger.error(f"Error closing enhanced market analyzer: {e}")

        # 关闭Redis连接
        try:
            await self.redis_store.close()
            logger.info("Redis connection closed")
        except Exception as e:
            logger.error(f"Error closing Redis connection: {e}")

        logger.info("Enhanced Market Analyzer Agent shutdown complete")

    def get_status(self) -> dict:
        """获取当前增强型代理状态。"""
        base_status = {
            "is_running": self.is_running,
            "redis_connected": self.redis_store.test_connection(),
            "last_analysis": datetime.now().isoformat(),
            "depth_snapshot_available": self.redis_store.depth_snapshot_exists(),
            "trade_window_count": self.redis_store.get_trade_window_count(),
        }

        # 添加增强型分析器状态
        if self.market_analyzer:
            analyzer_status = self.market_analyzer.get_status()
            base_status.update(analyzer_status)

        return base_status


async def main() -> None:
    """增强型分析器代理的主入口点。"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Enhanced Strategy Agent Market Analyzer"
    )
    parser.add_argument(
        "--config", default="config/development.yaml", help="Configuration file path"
    )
    args = parser.parse_args()

    # 加载设置
    settings = Settings.load_from_file(args.config)
    settings.setup_logging()

    # 验证DeepSeek API密钥（如果启用）
    if settings.analyzer.deepseek.enable and not settings.analyzer.deepseek.api_key:
        logger.error(
            "DeepSeek API key is required when DeepSeek is enabled. "
            "Please set DEEPSEEK_API_KEY environment variable or set enable: false in configuration."
        )
        sys.exit(1)

    # 创建并启动增强型代理
    agent = AnalyzerAgent(settings)

    try:
        await agent.start()
        logger.info("Enhanced agent startup completed successfully")
    except KeyboardInterrupt:
        logger.info("Interrupted by user - initiating shutdown")
    except asyncio.CancelledError:
        logger.info("Tasks cancelled - shutting down")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)
    finally:
        logger.info("Main function exiting")


if __name__ == "__main__":
    asyncio.run(main())
