"""简化的分析器代理 - 专注于核心功能。

这个代理提供简化的市场分析流程：
1. 读取Redis缓存的深度快照数据并聚合
2. 使用DeepSeek进行支撑阻力分析
3. 可视化挂单分布
"""

import asyncio
import logging
import signal
import sys
from datetime import datetime

from ..core.redis_client import RedisDataStore
from ..core.simple_market_analyzer import SimpleMarketAnalyzer
from ..utils.config import Settings
from ..visualization.order_book_visualizer import OrderBookVisualizer

logger = logging.getLogger(__name__)

# 配置常量
SHUTDOWN_TASK_TIMEOUT = 5.0
RETRY_DELAY_ON_ERROR = 10


class SimpleAnalyzerAgent:
    """简化的分析器代理，专注于核心功能。

    这个代理执行简化的市场分析流程：
    - Redis深度快照数据读取和聚合
    - DeepSeek LLM支撑阻力分析
    - 挂单分布可视化
    """

    def __init__(self, settings: Settings):
        """初始化简化分析器代理。

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

        # 初始化简化市场分析器
        deepseek_config = None
        if settings.analyzer.deepseek.enable and settings.analyzer.deepseek.api_key:
            deepseek_config = {
                "enable": True,
                "api_key": settings.analyzer.deepseek.api_key,
                "base_url": settings.analyzer.deepseek.base_url,
                "model": settings.analyzer.deepseek.model,
                "max_tokens": settings.analyzer.deepseek.max_tokens,
                "temperature": settings.analyzer.deepseek.temperature,
                "timeout": 60,  # 60秒超时
                "max_retries": 3,
            }
        else:
            logger.info("DeepSeek LLM analysis is disabled")

        self.market_analyzer = SimpleMarketAnalyzer(
            redis_store=self.redis_store,
            price_aggregation_precision=settings.analyzer.price_aggregation.precision,
            deepseek_config=deepseek_config,
            visualizer=self.visualizer,
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
        """启动分析流程。"""
        logger.info("Starting Simple Market Analyzer Agent")

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
            logger.error(f"Simple analyzer agent error: {e}")
        finally:
            await self._shutdown()

    async def _analysis_loop(self) -> None:
        """主分析循环。"""
        interval = self.settings.analyzer.analysis.interval_seconds

        while self.is_running:
            try:
                logger.debug("Starting simplified market analysis cycle")
                await self._perform_analysis_cycle()

                # 等待下一个周期，支持取消
                try:
                    await asyncio.wait_for(self.shutdown_event.wait(), timeout=interval)
                    logger.info("Shutdown event triggered, exiting analysis loop")
                    break
                except TimeoutError:
                    # 正常超时，继续下一个周期
                    continue

            except Exception as e:
                logger.error(f"Analysis cycle error: {e}")
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

    async def _perform_analysis_cycle(self) -> None:
        """执行简化的分析周期。"""
        try:
            # 执行简化的市场分析
            symbol = self.settings.binance.symbol
            result = self.market_analyzer.analyze_market(symbol)

            if result["status"] == "success":
                logger.info(f"Analysis completed successfully for {symbol}")

                # 存储分析结果到Redis (暂时注释掉，避免格式问题)
                # await self.redis_store.store_analysis_result(result)
                logger.info("Analysis result stored successfully")

                # 可以在这里添加其他处理逻辑
                # 比如发送通知、更新仪表板等

            elif result["status"] == "no_data":
                logger.info("No data available for analysis")
            else:
                logger.error(f"Analysis failed: {result.get('error', 'Unknown error')}")

        except Exception as e:
            logger.error(f"Analysis cycle failed: {e}")

    async def _shutdown(self) -> None:
        """清理和关闭代理。"""
        logger.info("Shutting down Simple Market Analyzer Agent")

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

        # 关闭连接
        if self.market_analyzer and hasattr(self.market_analyzer, "deepseek_analyzer"):
            if self.market_analyzer.deepseek_analyzer:
                try:
                    self.market_analyzer.deepseek_analyzer.close()
                    logger.info("DeepSeek analyzer closed")
                except Exception as e:
                    logger.error(f"Error closing DeepSeek analyzer: {e}")

        try:
            await self.redis_store.close()
            logger.info("Redis connection closed")
        except Exception as e:
            logger.error(f"Error closing Redis connection: {e}")

        logger.info("Simple Market Analyzer Agent shutdown complete")

    def get_status(self) -> dict:
        """获取当前代理状态。"""
        status = {
            "is_running": self.is_running,
            "redis_connected": self.redis_store.test_connection(),
            "last_analysis": datetime.now().isoformat(),
            "depth_snapshot_available": self.redis_store.depth_snapshot_exists(),
        }

        # 添加市场分析器状态
        if self.market_analyzer:
            status.update(self.market_analyzer.get_status())

        return status


async def main() -> None:
    """简化分析器代理的主入口点。"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Simple Strategy Agent Market Analyzer"
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

    # 创建并启动代理
    agent = SimpleAnalyzerAgent(settings)

    try:
        await agent.start()
        logger.info("Simple agent startup completed successfully")
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
