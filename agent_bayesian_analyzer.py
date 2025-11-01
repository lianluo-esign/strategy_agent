#!/usr/bin/env python3
"""贝叶斯趋势分析器入口文件 - 集成静态订单簿和动态数据的贝叶斯分析。

该文件是agent_analyzer_optimized.py的贝叶斯增强版本，主要改进：
1. 集成depth_snapshot_5000静态订单簿数据（10美元精度聚合）
2. 采用贝叶斯思维框架进行概率化趋势分析
3. 贝叶斯化的Deepseek AI分析提示词
4. 概率化的输出结果（置信度、不确定性量化）
5. 支持Discord webhook通知功能
6. 提供贝叶斯证据权重和推理链条分析

使用方法:
    python agent_bayesian_analyzer.py --config config/development.yaml
    python agent_bayesian_analyzer.py --single-run
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

from core.agent_analyzer_optimized.bayesian_optimized_analyzer import BayesianOptimizedAnalyzer
from core.redis_client import RedisDataStore
from utils.config import Settings

# 配置常量
SHUTDOWN_TASK_TIMEOUT = 5.0
RETRY_DELAY_ON_ERROR = 10

logger = logging.getLogger(__name__)


class BayesianAnalyzerAgent:
    """贝叶斯分析器代理，专注于概率化市场趋势分析。

    核心功能：
    1. 静态深度订单簿分析（depth_snapshot_5000）
    2. 动态trades_window数据聚合
    3. 贝叶斯概率化趋势预测
    4. AI增强的贝叶斯分析
    5. 概率化的结果输出

    主要特点：
    - 贝叶斯思维框架的证据权重分析
    - 概率化的趋势预测和置信度评估
    - 静态和动态数据的综合分析
    - 量化的不确定性和风险评估
    """

    def __init__(self, settings: Settings):
        """初始化贝叶斯分析器代理。

        Args:
            settings: 配置设置
        """
        self.settings = settings
        self.symbol = settings.binance.symbol

        # 初始化Redis连接
        self.redis_store = RedisDataStore(
            host=settings.redis.host,
            port=settings.redis.port,
            db=settings.redis.db,
            storage_dir=settings.redis.storage_dir,
            max_storage_files=settings.redis.max_storage_files
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

        # 创建贝叶斯优化分析器
        self.analyzer = BayesianOptimizedAnalyzer(
            redis_store=self.redis_store,
            deepseek_config=deepseek_config,
            discord_webhook_url=discord_webhook_url,
            analysis_window_minutes=getattr(
                settings.analyzer, "analysis_window_minutes", 240
            ),
            orderbook_precision=getattr(
                settings.analyzer, "orderbook_aggregation_precision", 10.0
            )
        )

        # 控制标志
        self.is_running = False
        self.shutdown_event = asyncio.Event()

        # 数据驱动模式的状态跟踪
        self.last_analysis_timestamp = None
        self.last_trades_window_hash = None
        self.last_depth_snapshot_hash = None
        self.min_data_points_for_analysis = 60  # 至少需要60个数据点才进行分析
        self.check_interval = 5  # 每5秒检查一次是否有新数据

        logger.info("贝叶斯分析器代理初始化完成 (贝叶斯思维框架)")

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
        """启动贝叶斯分析流程。"""
        logger.info("🚀 启动贝叶斯市场分析代理")

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
            logger.error(f"贝叶斯分析器代理错误: {e}")
        finally:
            await self._shutdown()

    async def _analysis_loop(self) -> None:
        """数据驱动的贝叶斯分析循环 - 仅在有新数据时触发分析。"""
        logger.info(f"🔄 开始贝叶斯数据驱动分析循环: {self.symbol}")
        logger.info(f"📊 最小数据点要求: {self.min_data_points_for_analysis}")
        logger.info(f"🔍 数据检查间隔: {self.check_interval}秒")
        logger.info(f"🧠 采用贝叶斯思维框架进行概率化分析")

        while self.is_running:
            try:
                # 检查是否有新的数据点
                has_new_data = await self._check_for_new_data()

                if has_new_data:
                    logger.info("🆕 检测到新数据点，触发贝叶斯分析")
                    await self._perform_analysis_cycle()
                else:
                    logger.debug("📋 没有新数据点，等待下次检查")

                # 等待下次检查或关闭信号
                try:
                    await asyncio.wait_for(self.shutdown_event.wait(), timeout=self.check_interval)
                    logger.info("关闭事件触发，退出分析循环")
                    break
                except TimeoutError:
                    # 正常超时，继续检查数据变化
                    continue

            except Exception as e:
                logger.error(f"贝叶斯数据驱动分析检查错误: {e}")
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

    async def _check_for_new_data(self) -> bool:
        """检查是否有新的数据点（静态或动态）。

        Returns:
            bool: 如果有新数据且满足分析条件返回True，否则返回False
        """
        try:
            # 获取当前数据状态
            current_trades_count = self.redis_store.get_trade_window_count()
            latest_trades_timestamp = self.redis_store.get_latest_trade_timestamp()
            current_trades_hash = self.redis_store.get_trades_window_hash()
            current_depth_hash = self.redis_store.get_depth_snapshot_hash()

            logger.debug(
                f"数据检查: 动态数据数量={current_trades_count}, "
                f"最新时间={latest_trades_timestamp}, "
                f"动态哈希={current_trades_hash}, "
                f"静态哈希={current_depth_hash}"
            )

            # 检查是否有足够的数据点
            if current_trades_count < self.min_data_points_for_analysis:
                if current_trades_count > 0:
                    logger.debug(f"动态数据点不足: {current_trades_count}/{self.min_data_points_for_analysis}")
                # 即使动态数据不足，如果有静态数据也可以进行分析
                if self.redis_store.depth_snapshot_exists():
                    logger.info("动态数据不足但有静态数据，将进行分析")
                    return self._check_static_data_change(current_depth_hash)
                return False

            # 首次运行时，只要有足够数据就分析
            if self.last_analysis_timestamp is None:
                logger.info(f"首次运行，检测到 {current_trades_count} 个动态数据点，将进行贝叶斯分析")
                self.last_analysis_timestamp = latest_trades_timestamp
                self.last_trades_window_hash = current_trades_hash
                self.last_depth_snapshot_hash = current_depth_hash
                return True

            # 检查动态数据时间戳变化
            if latest_trades_timestamp and latest_trades_timestamp > self.last_analysis_timestamp:
                time_diff = latest_trades_timestamp - self.last_analysis_timestamp
                logger.info(f"检测到新的动态数据点，时间差: {time_diff}")
                self.last_analysis_timestamp = latest_trades_timestamp
                self.last_trades_window_hash = current_trades_hash
                return True

            # 检查动态数据内容变化
            if current_trades_hash and current_trades_hash != self.last_trades_window_hash:
                logger.info("检测到动态trades_window内容发生变化")
                self.last_trades_window_hash = current_trades_hash
                return True

            # 检查静态数据变化
            if current_depth_hash and current_depth_hash != self.last_depth_snapshot_hash:
                logger.info("检测到静态深度快照内容发生变化")
                self.last_depth_snapshot_hash = current_depth_hash
                return True

            # 没有检测到新数据
            return False

        except Exception as e:
            logger.error(f"检查新数据时出错: {e}")
            return False

    def _check_static_data_change(self, current_depth_hash: Optional[str]) -> bool:
        """检查静态数据变化。

        Args:
            current_depth_hash: 当前深度快照哈希

        Returns:
            bool: 如果有变化返回True
        """
        if self.last_depth_snapshot_hash is None:
            # 首次检测到静态数据
            self.last_depth_snapshot_hash = current_depth_hash
            return True

        if current_depth_hash and current_depth_hash != self.last_depth_snapshot_hash:
            # 静态数据发生变化
            self.last_depth_snapshot_hash = current_depth_hash
            return True

        return False

    async def _perform_analysis_cycle(self) -> None:
        """执行贝叶斯分析周期。"""
        try:
            # 执行贝叶斯分析
            result = await self.analyzer.analyze_market(self.symbol)

            if result["status"] == "success":
                logger.info(f"✅ 贝叶斯市场分析完成: {self.symbol}")

                # 输出分析摘要
                self._log_analysis_summary(result)

                # 输出JSON结果
                analysis_result = result["analysis_result"]
                logger.info(f"📤 贝叶斯JSON输出: {analysis_result}")

                # 输出概率分布摘要
                self._log_probability_summary(analysis_result)

            else:
                logger.error(f"❌ 贝叶斯分析失败: {result.get('error', '未知错误')}")

        except Exception as e:
            logger.error(f"❌ 贝叶斯分析周期失败: {e}")

    def _log_analysis_summary(self, result: dict) -> None:
        """记录贝叶斯分析摘要。

        Args:
            result: 分析结果字典
        """
        symbol = result.get("symbol", "UNKNOWN")
        processing_time = result.get("processing_time", 0)
        discord_sent = result.get("discord_notification_sent", False)

        logger.info(f"=== 📊 {symbol} 贝叶斯分析摘要 ===")
        logger.info(f"🧠 分析模式: 贝叶斯思维框架的概率化趋势分析")

        # 显示触发信息
        current_data_points = self.redis_store.get_trade_window_count()
        static_available = self.redis_store.depth_snapshot_exists()
        logger.info(f"🎯 触发条件: 检测到新数据点")
        logger.info(f"📋 数据源状态: 动态数据={current_data_points}点, 静态数据={'可用' if static_available else '不可用'}")

        # 贝叶斯分析结果摘要
        analysis_result = result.get("analysis_result", {})
        if analysis_result:
            trend_analysis = analysis_result.get("trend_analysis", {})
            logger.info(f"📈 最可能趋势: {trend_analysis.get('most_likely_trend', '未知')}")
            logger.info(f"🎯 置信度: {trend_analysis.get('confidence', 0):.1%}")
            logger.info(f"⚠️ 不确定性: {trend_analysis.get('uncertainty', 0):.1%}")
            logger.info(f"📊 置信等级: {trend_analysis.get('confidence_level', 'unknown')}")
            logger.info(f"🎲 风险等级: {trend_analysis.get('risk_level', 'unknown')}")

            # 概率分布摘要
            prob_dist = analysis_result.get("probability_distribution", {})
            top_three = prob_dist.get("top_three_trends", [])
            if top_three:
                logger.info("🎯 概率分布Top3:")
                for i, trend_info in enumerate(top_three, 1):
                    trend = trend_info.get("trend", "unknown")
                    prob = trend_info.get("probability", 0)
                    logger.info(f"   {i}. {trend}: {prob:.1%}")

        # 数据源统计
        data_source_stats = result.get("data_source_stats", {})
        logger.info(
            f"📊 数据源质量: 静态={'✅' if data_source_stats.get('static_data_available') else '❌'}, "
            f"动态={'✅' if data_source_stats.get('dynamic_data_available') else '❌'}"
        )

        logger.info(f"⏱️ 处理时间: {processing_time:.2f}s")
        logger.info(f"📢 Discord通知: {'已发送' if discord_sent else '未发送'}")
        logger.info(f"🧠 贝叶斯优势: 概率化预测，量化不确定性，证据权重分析")
        logger.info("=" * 60)

    def _log_probability_summary(self, analysis_result: dict) -> None:
        """记录概率分布摘要。

        Args:
            analysis_result: 分析结果字典
        """
        prob_dist = analysis_result.get("probability_distribution", {})
        full_dist = prob_dist.get("full_distribution", {})

        if not full_dist:
            return

        logger.info("🎲 完整概率分布:")
        sorted_probs = sorted(full_dist.items(), key=lambda x: x[1], reverse=True)
        for trend, probability in sorted_probs:
            bar_length = int(probability * 20)  # 20字符表示100%
            bar = "█" * bar_length + "░" * (20 - bar_length)
            logger.info(f"   {trend:8} | {bar} | {probability:.1%}")

        # 显示贝叶斯洞察
        bayesian_analysis = analysis_result.get("bayesian_analysis", {})
        key_insights = bayesian_analysis.get("key_insights", {})
        if key_insights:
            logger.info("🧠 贝叶斯洞察:")
            logger.info(f"   主要驱动因素: {key_insights.get('primary_driver', 'unknown')}")
            logger.info(f"   证据一致性: {key_insights.get('evidence_consistency', 'unknown')}")

            strength_factors = key_insights.get("strength_factors", [])
            if strength_factors:
                logger.info(f"   支撑因素: {', '.join(strength_factors)}")

            weakness_factors = key_insights.get("weakness_factors", [])
            if weakness_factors:
                logger.info(f"   风险因素: {', '.join(weakness_factors)}")

    async def _shutdown(self) -> None:
        """清理和关闭代理。"""
        logger.info("🔄 正在关闭贝叶斯市场分析代理")

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

        # 关闭贝叶斯分析器
        if self.analyzer:
            try:
                await self.analyzer.close()
                logger.info("✅ 贝叶斯分析器已关闭")
            except Exception as e:
                logger.error(f"关闭贝叶斯分析器时出错: {e}")

        logger.info("✅ 贝叶斯市场分析代理已完全关闭")

    def get_status(self) -> dict:
        """获取代理状态。"""
        base_status = {
            "agent_type": "bayesian_analyzer",
            "mode": "bayesian_inference",
            "is_running": self.is_running,
            "symbol": self.symbol,
            "discord_webhook_configured": True,
            # 贝叶斯模式状态
            "last_analysis_timestamp": self.last_analysis_timestamp.isoformat() if self.last_analysis_timestamp else None,
            "min_data_points_required": self.min_data_points_for_analysis,
            "check_interval_seconds": self.check_interval,
            "current_data_points": self.redis_store.get_trade_window_count(),
            "latest_trade_timestamp": self.redis_store.get_latest_trade_timestamp().isoformat() if self.redis_store.get_latest_trade_timestamp() else None,
            "static_data_available": self.redis_store.depth_snapshot_exists(),
        }

        # 添加贝叶斯分析器状态
        if self.analyzer:
            analyzer_status = self.analyzer.get_status()
            base_status.update(analyzer_status)

        return base_status


async def main() -> None:
    """贝叶斯分析器代理的主入口点。"""
    parser = argparse.ArgumentParser(
        description="贝叶斯Strategy Agent市场分析器 - 概率化趋势分析"
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
        agent = BayesianAnalyzerAgent(settings)
        logger.info("✅ 贝叶斯分析器代理创建成功")
    except Exception as e:
        logger.error(f"❌ 创建贝叶斯分析器代理失败: {e}")
        sys.exit(1)

    try:
        if args.single_run:
            # 单次运行模式
            logger.info("🔧 单次运行模式")
            json_result = await agent.analyzer.analyze_single_cycle()
            print("=== 📤 贝叶斯JSON输出结果 ===")
            print(json_result)
            print("=" * 60)
            await agent._shutdown()

        else:
            # 持续运行模式
            await agent.start()
            logger.info("✅ 贝叶斯代理启动完成")

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
    print("🚀 启动贝叶斯市场分析代理")
    print("📋 贝叶斯增强功能:")
    print("   • 静态深度订单簿分析 (depth_snapshot_5000)")
    print("   • 10美元精度聚合的订单簿数据")
    print("   • 贝叶斯思维框架的概率化分析")
    print("   • 动态trades_window数据流分析")
    print("   • 贝叶斯证据权重和似然计算")
    print("   • AI增强的贝叶斯推理分析")
    print("   • 概率化输出（置信度、不确定性）")
    print("   • 贝叶斯推理链条和证据分析")
    print("   • 集成Discord通知功能")
    print("")
    print("🧠 贝叶斯思维优势:")
    print("   • 概率化预测：避免非黑即白的判断")
    print("   • 不确定性量化：明确表达预测的置信程度")
    print("   • 证据权重：根据证据强度调整预测")
    print("   • 动态更新：新证据实时更新概率")
    print("   • 风险评估：量化决策风险和不确定性")
    print("")
    print("📊 数据源整合:")
    print("   • 静态流动性：深度订单簿的供需结构")
    print("   • 动态成交：历史交易的价格动能")
    print("   • 贝叶斯融合：多源证据的概率综合")
    print("")
    print("📤 输出格式:")
    print("   • most_likely_trend: 最可能的趋势")
    print("   • confidence: 置信度 (0.0-1.0)")
    print("   • uncertainty: 不确定性 (0.0-1.0)")
    print("   • posterior_probabilities: 完整概率分布")
    print("   • evidence_summary: 证据分析摘要")
    print("   • 分析完成后自动发送Discord通知")
    print("=" * 70)

    asyncio.run(main())