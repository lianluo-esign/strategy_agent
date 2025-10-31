#!/usr/bin/env python3
"""优化版Agent Analyzer入口文件 - 基于trades_window数据的AI趋势分析。

该文件是agent_analyzer.py的优化版本，主要改进：
1. 移除对5000层深度快照的依赖，专注于trades_window数据
2. 集成Deepseek AI进行趋势分析，输出标准JSON格式
3. 支持Discord webhook通知功能
4. 优化性能和可靠性

使用方法:
    python agent_analyzer_optimized.py --config config/development.yaml
    python agent_analyzer_optimized.py --single-run
"""

import argparse
import asyncio
import logging
import os
import signal
import sys
from pathlib import Path
from typing import Optional

# 添加src到Python路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

from core.agent_analyzer_optimized import OptimizedAgentAnalyzer
from core.redis_client import RedisDataStore
from utils.config import Settings

# 配置常量
SHUTDOWN_TASK_TIMEOUT = 5.0
RETRY_DELAY_ON_ERROR = 10

logger = logging.getLogger(__name__)


class OptimizedAnalyzerAgent:
    """优化版分析器代理，专注于trades_window数据的AI趋势分析。"""

    def __init__(self, settings: Settings):
        """初始化优化版分析器代理。

        Args:
            settings: 配置设置
        """
        self.settings = settings
        self.symbol = settings.binance.symbol

        # 初始化Redis连接
        self.redis_store = RedisDataStore(
            host=settings.redis.host,
            port=settings.redis.port,
            db=settings.redis.db
        )

        # 准备Deepseek配置
        deepseek_config = self._prepare_deepseek_config(settings)

        # 获取Discord webhook URL
        discord_config = getattr(settings.analyzer, 'discord', None)
        discord_webhook_url = ""
        if discord_config and discord_config.enable:
            discord_webhook_url = discord_config.webhook_url
            if not discord_webhook_url:
                logger.warning("Discord功能已启用但webhook URL未配置，将跳过通知功能")
        else:
            logger.info("Discord通知功能未启用")

        # 创建优化版分析器
        self.analyzer = OptimizedAgentAnalyzer(
            redis_store=self.redis_store,
            deepseek_config=deepseek_config,
            discord_webhook_url=discord_webhook_url,
            analysis_window_minutes=getattr(
                settings.analyzer, "analysis_window_minutes", 4320
            )
        )

        # 控制标志
        self.is_running = False
        self.shutdown_event = asyncio.Event()

        logger.info("优化版分析器代理初始化完成")

    def _prepare_deepseek_config(self, settings: Settings) -> dict:
        """准备Deepseek配置。

        Args:
            settings: 配置设置

        Returns:
            Deepseek配置字典
        """
        deepseek_settings = getattr(settings.analyzer, 'deepseek', None)

        if not deepseek_settings or not deepseek_settings.enable:
            logger.error("Deepseek分析功能必须启用")
            raise ValueError("Deepseek功能未启用")

        if not deepseek_settings.api_key:
            logger.error("Deepseek API密钥未配置")
            raise ValueError("请在配置文件中指定deepseek.api_key")

        return {
            "api_key": deepseek_settings.api_key,
            "base_url": deepseek_settings.base_url,
            "model": deepseek_settings.model,
            "max_tokens": getattr(deepseek_settings, "max_tokens", 4000),
            "temperature": deepseek_settings.temperature,
            "timeout": getattr(deepseek_settings, "timeout", 90),
            "max_retries": getattr(deepseek_settings, "max_retries", 3),
        }

    def setup_signal_handlers(self) -> None:
        """设置信号处理器。"""
        self._shutdown_requested = False

    def _signal_handler(self) -> None:
        """处理关闭信号。"""
        logger.info("接收到关闭信号，正在停止...")
        self._shutdown_requested = True
        self.is_running = False
        self.shutdown_event.set()

    async def start(self) -> None:
        """启动优化版分析流程。"""
        logger.info("🚀 启动优化版市场分析代理")

        # 设置信号处理器
        self.setup_signal_handlers()

        # 添加信号处理器
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, self._signal_handler)
                logger.debug(f"信号处理器已注册: {sig}")
            except Exception as e:
                logger.error(f"信号处理器注册失败: {sig}: {e}")
                raise RuntimeError(f"信号处理器注册失败: {e}") from e

        # 测试Redis连接
        if not self.redis_store.test_connection():
            logger.error("Redis连接失败，退出...")
            return

        logger.info("✅ Redis连接已建立")

        # Discord通知功能说明（不进行连接测试）
        if self.analyzer.discord_manager:
            logger.info("✅ Discord通知功能已启用，分析完成后将自动发送结果")
        else:
            logger.info("ℹ️ Discord通知功能未配置")

        # 执行健康检查
        health = await self.analyzer.health_check()
        if health["overall_status"] != "healthy":
            logger.warning(f"⚠️ 健康检查发现问题: {health.get('failed_checks', [])}")
        else:
            logger.info("✅ 健康检查通过")

        # 主分析循环
        try:
            self.is_running = True
            await self._analysis_loop()
        except asyncio.CancelledError:
            logger.info("分析循环被取消")
        except Exception as e:
            logger.error(f"优化版分析器代理错误: {e}")
        finally:
            await self._shutdown()

    async def _analysis_loop(self) -> None:
        """主分析循环。"""
        interval = getattr(self.settings.analyzer.analysis, "interval_seconds", 300)

        logger.info(f"🔄 开始分析循环: {self.symbol} (间隔: {interval}s)")

        while self.is_running:
            try:
                logger.debug("开始优化版市场分析周期")
                await self._perform_analysis_cycle()

                # 等待下一个周期
                try:
                    await asyncio.wait_for(self.shutdown_event.wait(), timeout=interval)
                    logger.info("关闭事件触发，退出分析循环")
                    break
                except TimeoutError:
                    # 正常超时，继续下一个周期
                    continue

            except Exception as e:
                logger.error(f"优化版分析周期错误: {e}")
                # 重试前等待
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
        """执行分析周期。"""
        try:
            # 执行优化版分析
            result = await self.analyzer.analyze_market(self.symbol)

            if result["status"] == "success":
                logger.info(f"✅ 优化版市场分析完成: {self.symbol}")

                # 输出分析摘要
                self._log_analysis_summary(result)

                # 输出JSON结果
                analysis_result = result["analysis_result"]
                logger.info(f"📤 JSON输出: {analysis_result}")

            else:
                logger.error(f"❌ 优化版分析失败: {result.get('error', '未知错误')}")

        except Exception as e:
            logger.error(f"❌ 优化版分析周期失败: {e}")

    def _log_analysis_summary(self, result: dict) -> None:
        """记录分析摘要。

        Args:
            result: 分析结果字典
        """
        symbol = result.get("symbol", "UNKNOWN")
        processing_time = result.get("processing_time", 0)
        discord_sent = result.get("discord_notification_sent", False)

        logger.info(f"=== 📊 {symbol} 优化版分析摘要 ===")
        logger.info(f"🔍 分析模式: 基于原始trades_window数据的AI趋势分析")

        # 分析结果摘要
        analysis_result = result.get("analysis_result", {})
        if analysis_result:
            logger.info(f"📈 趋势判断: {analysis_result.get('trend', '未知')}")
            logger.info(f"🎯 置信度: {analysis_result.get('confidence', 0):.1%}")

            strength_levels = analysis_result.get("strength_levels", {})
            if strength_levels:
                logger.info("💪 支撑/阻力强度:")
                for level, value in strength_levels.items():
                    logger.info(f"   {level}: {value:.2f}")

        # 数据摘要
        raw_data = result.get("raw_data", {})
        if raw_data:
            logger.info(
                f"📊 数据统计: {raw_data.get('data_points_count', 0)} 个分钟数据点, "
                f"分析窗口 {len(raw_data.get('minute_data_points', []))} 分钟"
            )

        logger.info(f"⏱️ 处理时间: {processing_time:.2f}s")
        logger.info(f"📢 Discord通知: {'已发送' if discord_sent else '未发送'}")
        logger.info("=" * 50)

    async def _shutdown(self) -> None:
        """清理和关闭代理。"""
        logger.info("🔄 正在关闭优化版市场分析代理")

        self.is_running = False
        self._shutdown_requested = True

        # 取消所有待处理的任务
        tasks = [
            task for task in asyncio.all_tasks()
            if task is not asyncio.current_task()
        ]
        if tasks:
            logger.info(f"正在取消 {len(tasks)} 个待处理任务...")
            for task in tasks:
                task.cancel()

            # 等待任务完成
            try:
                await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=SHUTDOWN_TASK_TIMEOUT,
                )
            except TimeoutError:
                logger.warning("部分任务未在超时时间内完成")

        # 关闭优化版分析器
        if self.analyzer:
            try:
                await self.analyzer.close()
                logger.info("✅ 优化版分析器已关闭")
            except Exception as e:
                logger.error(f"关闭优化版分析器时出错: {e}")

        logger.info("✅ 优化版市场分析代理已完全关闭")

    def get_status(self) -> dict:
        """获取代理状态。"""
        base_status = {
            "agent_type": "optimized_analyzer",
            "is_running": self.is_running,
            "symbol": self.symbol,
            "discord_webhook_configured": True,
        }

        # 添加优化版分析器状态
        if self.analyzer:
            analyzer_status = self.analyzer.get_status()
            base_status.update(analyzer_status)

        return base_status


async def main() -> None:
    """优化版分析器代理的主入口点。"""
    parser = argparse.ArgumentParser(
        description="优化版Strategy Agent市场分析器 - 基于trades_window数据的AI趋势分析"
    )
    parser.add_argument(
        "--config", default="config/development.yaml", help="配置文件路径"
    )
    parser.add_argument(
        "--single-run", action="store_true", help="运行分析一次后退出"
    )
    args = parser.parse_args()

    # 加载设置
    try:
        settings = Settings.load_from_file(args.config)
        settings.setup_logging()
    except Exception as e:
        logger.error(f"加载配置失败: {e}")
        sys.exit(1)

    # 验证Deepseek配置
    try:
        deepseek_settings = getattr(settings.analyzer, 'deepseek', None)
        if not deepseek_settings or not deepseek_settings.enable:
            logger.error("❌ Deepseek功能必须在配置中启用")
            logger.info("请在配置文件中设置 analyzer.deepseek.enable = true")
            sys.exit(1)
    except AttributeError:
        logger.error("❌ 配置文件中缺少Deepseek配置")
        sys.exit(1)

    # 创建并启动代理
    try:
        agent = OptimizedAnalyzerAgent(settings)
        logger.info("✅ 优化版分析器代理创建成功")
    except Exception as e:
        logger.error(f"❌ 创建优化版分析器代理失败: {e}")
        sys.exit(1)

    try:
        if args.single_run:
            # 单次运行模式
            logger.info("🔧 单次运行模式")
            json_result = await agent.analyzer.analyze_single_cycle()
            print("=== 📤 JSON输出结果 ===")
            print(json_result)
            print("=" * 50)
            await agent._shutdown()

        else:
            # 持续运行模式
            await agent.start()
            logger.info("✅ 优化版代理启动完成")

    except KeyboardInterrupt:
        logger.info("👋 用户中断 - 正在关闭")
    except asyncio.CancelledError:
        logger.info("🛑 任务被取消 - 正在关闭")
    except Exception as e:
        logger.error(f"💥 致命错误: {e}")
        sys.exit(1)
    finally:
        logger.info("🏁 主函数退出")


if __name__ == "__main__":
    # 显示启动信息
    print("🚀 启动优化版市场分析代理")
    print("📋 优化功能:")
    print("   • 基于原始trades_window数据进行AI趋势分析")
    print("   • 直接使用最近4小时每分钟数据点")
    print("   • 无二次聚合，保持数据原始性")
    print("   • 集成Deepseek AI分析")
    print("   • 标准JSON输出格式")
    print("   • 分析完成后直接发送Discord通知")
    print("📤 输出格式:")
    print("   • timestamp: 最新分钟时间")
    print("   • trend: 趋势判断(震荡/微弱看涨/看涨/强力看涨/微弱看跌/看跌/强力看跌)")
    print("   • strength_levels: 各档位强度值")
    print("   • reason: 分析原因")
    print("   • confidence: 置信度")
    print("   • 分析完成后自动发送Discord通知")
    print("=" * 60)

    asyncio.run(main())