#!/usr/bin/env python3
"""新版入口文件 - 简化市场分析器代理。

实现用户要求的三步分析流程：
1. 初始化
2. 读取redis中的历史数据并按照要求进行聚合
3. 将orderbook和trades_window聚合后的数据通过prompt发送给deepseek进行分析
并返回标准的json输出结果: {"grid_delta": 2.0, "grid_quantity": 0.001, "active_side": "Buy"}
"""

import asyncio
import logging
import signal
import sys
from pathlib import Path
from datetime import datetime

# 添加src到Python路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

from core.redis_client import RedisDataStore
from core.simplified_market_analyzer import SimplifiedMarketAnalyzer
from utils.config import Settings

logger = logging.getLogger(__name__)

# 配置常量
SHUTDOWN_TASK_TIMEOUT = 5.0
RETRY_DELAY_ON_ERROR = 10


class SimplifiedAnalyzerAgent:
    """简化市场分析器代理，专注三步分析流程。

    这个代理执行简化的三步市场分析流程：
    1. 初始化所有依赖组件
    2. 从Redis读取历史数据并进行聚合
    3. 调用DeepSeek分析并返回标准JSON格式结果
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

        # 初始化DeepSeek配置
        deepseek_config = self._prepare_deepseek_config(settings)

        # 初始化简化市场分析器
        self.market_analyzer = SimplifiedMarketAnalyzer(
            redis_store=self.redis_store,
            deepseek_config=deepseek_config,
            price_aggregation_precision=getattr(
                settings.analyzer, "price_aggregation_precision", 1.0
            ),
            vp_aggregation_precision=getattr(
                settings.analyzer, "volume_profile_aggregation_precision", 10.0
            ),
        )

        logger.info("Simplified analyzer agent initialized with 3-step analysis flow")

        # 控制标志
        self.is_running = False
        self.shutdown_event = asyncio.Event()

    def _prepare_deepseek_config(self, settings: Settings) -> dict:
        """准备DeepSeek配置。

        Args:
            settings: 配置设置

        Returns:
            DeepSeek配置字典
        """
        if (not settings.analyzer.deepseek.enable or
            not settings.analyzer.deepseek.api_key):
            logger.error("DeepSeek LLM analysis is required for simplified analyzer")
            raise ValueError("DeepSeek API key is required")

        return {
            "api_key": settings.analyzer.deepseek.api_key,
            "base_url": settings.analyzer.deepseek.base_url,
            "model": settings.analyzer.deepseek.model,
            "max_tokens": 3000,  # 减少令牌数以获得更简洁的JSON输出
            "temperature": settings.analyzer.deepseek.temperature,
            "timeout": settings.analyzer.deepseek.timeout,
            "max_retries": 3,
        }

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
        """启动简化分析流程。"""
        logger.info("🚀 Starting Simplified Market Analyzer Agent")

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

        logger.info("✅ Redis connection established")

        # 主分析循环
        try:
            self.is_running = True
            await self._analysis_loop()
        except asyncio.CancelledError:
            logger.info("Analysis loop cancelled")
        except Exception as e:
            logger.error(f"Simplified analyzer agent error: {e}")
        finally:
            await self._shutdown()

    async def _analysis_loop(self) -> None:
        """主分析循环。"""
        interval = self.settings.analyzer.analysis.interval_seconds
        symbol = self.settings.binance.symbol

        logger.info(f"🔄 Starting analysis loop for {symbol} (interval: {interval}s)")

        while self.is_running:
            try:
                logger.debug("Starting simplified market analysis cycle")
                await self._perform_simplified_analysis_cycle()

                # 等待下一个周期，支持取消
                try:
                    await asyncio.wait_for(self.shutdown_event.wait(), timeout=interval)
                    logger.info("Shutdown event triggered, exiting analysis loop")
                    break
                except TimeoutError:
                    # 正常超时，继续下一个周期
                    continue

            except Exception as e:
                logger.error(f"Simplified analysis cycle error: {e}")
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

    async def _perform_simplified_analysis_cycle(self) -> None:
        """执行简化分析周期。"""
        try:
            # 执行三步分析流程
            symbol = self.settings.binance.symbol
            analysis_result = await self.market_analyzer.analyze_market(symbol)

            if analysis_result["status"] == "success":
                logger.info(
                    f"✅ Simplified market analysis completed successfully for {symbol}"
                )

                # 打印分析摘要
                self._log_analysis_summary(analysis_result)

                # 可以在这里添加其他处理逻辑
                # 比如发送交易参数到交易系统
                trading_params = analysis_result.get("trading_params")
                if trading_params:
                    logger.info(f"🎯 Trading Parameters Ready: {trading_params}")

            elif analysis_result["status"] == "no_data":
                logger.info("ℹ️  No data available for simplified analysis")
            else:
                logger.error(
                    f"❌ Simplified analysis failed: {analysis_result.get('error', 'Unknown error')}"
                )

        except Exception as e:
            logger.error(f"❌ Simplified analysis cycle failed: {e}")

    def _log_analysis_summary(self, analysis_result: dict) -> None:
        """记录简化分析摘要。

        Args:
            analysis_result: 分析结果字典
        """
        symbol = analysis_result.get("symbol", "UNKNOWN")

        logger.info(f"=== 📊 {symbol} Simplified Analysis Summary ===")
        logger.info("🔍 Analysis Mode: Simplified 3-Step Analysis")

        # 市场数据摘要
        market_summary = analysis_result.get("market_data_summary", {})
        logger.info(
            f"📊 Market Data: {market_summary.get('bid_levels', 0)} bid levels, "
            f"{market_summary.get('ask_levels', 0)} ask levels"
        )
        logger.info(
            f"📈 Volume Profile: {market_summary.get('vp_price_levels', 0)} price levels, "
            f"total_volume={market_summary.get('total_volume', 0):.2f}"
        )

        # 交易参数摘要
        trading_params = analysis_result.get("trading_params")
        if trading_params:
            logger.info("✅ Trading Parameters Generated:")
            logger.info(f"   💰 Grid Delta: {trading_params.get('grid_delta', 'N/A')}")
            logger.info(f"   📊 Grid Quantity: {trading_params.get('grid_quantity', 'N/A')}")
            logger.info(f"   🎯 Active Side: {trading_params.get('active_side', 'N/A')}")
        else:
            logger.info("❌ Failed to generate trading parameters")

        logger.info("=" * 50)

    async def _shutdown(self) -> None:
        """清理和关闭简化代理。"""
        logger.info("🔄 Shutting down Simplified Market Analyzer Agent")

        self.is_running = False
        self._shutdown_requested = True

        # 取消所有待处理的任务
        tasks = [
            task for task in asyncio.all_tasks()
            if task is not asyncio.current_task()
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

        # 关闭简化分析器资源
        if self.market_analyzer:
            try:
                await self.market_analyzer.close()
                logger.info("✅ Simplified market analyzer closed")
            except Exception as e:
                logger.error(f"Error closing simplified market analyzer: {e}")

        # 关闭Redis连接
        try:
            await self.redis_store.close()
            logger.info("✅ Redis connection closed")
        except Exception as e:
            logger.error(f"Error closing Redis connection: {e}")

        logger.info("✅ Simplified Market Analyzer Agent shutdown complete")

    def get_status(self) -> dict:
        """获取当前简化代理状态。"""
        base_status = {
            "agent_type": "simplified_analyzer",
            "is_running": self.is_running,
            "redis_connected": self.redis_store.test_connection(),
            "last_analysis": datetime.now().isoformat(),
            "depth_snapshot_available": self.redis_store.depth_snapshot_exists(),
            "trade_window_count": self.redis_store.get_trade_window_count(),
        }

        # 添加简化分析器状态
        if self.market_analyzer:
            analyzer_status = self.market_analyzer.get_status()
            base_status.update(analyzer_status)

        return base_status


async def main() -> None:
    """简化分析器代理的主入口点。"""
    import argparse
    import os

    parser = argparse.ArgumentParser(
        description="Simplified Strategy Agent Market Analyzer - 3-Step Analysis Flow"
    )
    parser.add_argument(
        "--config", default="config/development.yaml", help="Configuration file path"
    )
    parser.add_argument(
        "--single-run", action="store_true", help="Run analysis once and exit"
    )
    args = parser.parse_args()

    # 加载设置
    try:
        settings = Settings.load_from_file(args.config)
        settings.setup_logging()
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        sys.exit(1)

    # 验证DeepSeek API密钥（必需）
    if not settings.analyzer.deepseek.enable or not settings.analyzer.deepseek.api_key:
        logger.error(
            "❌ DeepSeek API key is required for simplified analyzer. "
            "Please set DEEPSEEK_API_KEY environment variable and enable DeepSeek in configuration."
        )
        logger.info("Example: export DEEPSEEK_API_KEY='your-api-key'")
        sys.exit(1)

    # 创建并启动简化代理
    try:
        agent = SimplifiedAnalyzerAgent(settings)
        logger.info("✅ Simplified analyzer agent created successfully")
    except Exception as e:
        logger.error(f"❌ Failed to create analyzer agent: {e}")
        sys.exit(1)

    try:
        if args.single_run:
            # 单次运行模式
            logger.info("🔧 Running in single-run mode")
            symbol = settings.binance.symbol
            analysis_result = await agent.market_analyzer.analyze_market(symbol)
            agent._log_analysis_summary(analysis_result)
            await agent.close()
        else:
            # 持续运行模式
            await agent.start()
            logger.info("✅ Simplified agent startup completed successfully")
    except KeyboardInterrupt:
        logger.info("👋 Interrupted by user - initiating shutdown")
    except asyncio.CancelledError:
        logger.info("🛑 Tasks cancelled - shutting down")
    except Exception as e:
        logger.error(f"💥 Fatal error: {e}")
        sys.exit(1)
    finally:
        logger.info("🏁 Main function exiting")


if __name__ == "__main__":
    # 显示启动信息
    print("🚀 Starting Simplified Market Analyzer Agent")
    print("📋 3-Step Analysis Flow:")
    print("   1. Initialize components")
    print("   2. Read and aggregate Redis data")
    print("   3. Send to DeepSeek for analysis")
    print("📤 Expected Output: {\"grid_delta\": 2.0, \"grid_quantity\": 0.001, \"active_side\": \"Buy\"}  # delta: 1.0-50.0, quantity: 0.0001-0.02")
    print("=" * 60)

    asyncio.run(main())