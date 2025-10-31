#!/usr/bin/env python3
"""量价结合策略启动器。

基于支撑阻力位和成交量确认的稳定交易策略。
"""

import asyncio
import json
import logging
import signal
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))

from src.core.redis_client import RedisDataStore
from src.core.models import MinuteTradeData
from src.core.volume_price_strategy_analyzer import (
    VolumePriceStrategyAnalyzer,
    VolumePriceSignal,
    MomentumDirection
)
from src.core.support_resistance_analyzer import VolumePriceAnalysis

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/volume_price_strategy.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class VolumePriceStrategyLauncher:
    """量价结合策略启动器。"""

    def __init__(
        self,
        analysis_window_minutes: int = 15,
        analysis_interval_seconds: int = 60,
        analysis_delay_seconds: int = 5,
        min_volume_ratio: float = 1.5,
        max_distance_from_level_percent: float = 0.2,
        min_strength_threshold: float = 0.6
    ):
        """初始化策略启动器。

        Args:
            analysis_window_minutes: 分析窗口（分钟）
            analysis_interval_seconds: 分析间隔（秒）
            analysis_delay_seconds: 分析延迟（秒）- 等待数据写入
            min_volume_ratio: 最小成交量放大倍数
            max_distance_from_level_percent: 距离关键位最大百分比
            min_strength_threshold: 最小信号强度阈值
        """
        self.analysis_window_minutes = analysis_window_minutes
        self.analysis_interval_seconds = analysis_interval_seconds
        self.analysis_delay_seconds = analysis_delay_seconds

        # 初始化Redis客户端
        self.redis_store = RedisDataStore()

        # 初始化量价策略分析器
        self.analyzer = VolumePriceStrategyAnalyzer(
            window_minutes=analysis_window_minutes,
            min_volume_ratio=min_volume_ratio,
            max_distance_from_level_percent=max_distance_from_level_percent,
            min_strength_threshold=min_strength_threshold
        )

        self.running = False
        self.last_analysis_time = None

    async def start(self):
        """启动策略。"""
        logger.info("🚀 启动量价结合策略")
        logger.info(f"📊 分析窗口: {self.analysis_window_minutes}分钟")
        logger.info(f"⏰ 分析间隔: {self.analysis_interval_seconds}秒")
        logger.info(f"⏱️ 分析延迟: {self.analysis_delay_seconds}秒")

        # 测试Redis连接
        if not self.redis_store.test_connection():
            logger.error("❌ Redis连接失败")
            return

        logger.info("✅ Redis连接成功")

        # 创建信号目录
        Path("signals").mkdir(exist_ok=True)
        Path("logs").mkdir(exist_ok=True)

        # 设置信号处理
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        self.running = True

        # 等待到下一个分析时间点
        next_analysis_time = self._calculate_next_analysis_time()
        logger.info(f"⏰ 首次分析将在 {next_analysis_time.strftime('%H:%M:%S')} 开始")

        # 主循环
        while self.running:
            try:
                current_time = datetime.now()

                # 检查是否到了分析时间
                if current_time >= next_analysis_time:
                    await self._perform_analysis()
                    next_analysis_time = self._calculate_next_analysis_time()
                    logger.info(f"⏰ 下次分析将在 {next_analysis_time.strftime('%H:%M:%S')} 开始")

                # 短暂休眠
                await asyncio.sleep(1)

            except Exception as e:
                logger.error(f"策略运行错误: {e}")
                await asyncio.sleep(5)

        logger.info("🛑 量价策略已停止")

    async def _perform_analysis(self):
        """执行一次分析。"""
        try:
            logger.info("🔍 开始执行量价分析...")

            # 加载最近15分钟的数据
            minute_data = await self._load_recent_data()
            if not minute_data:
                logger.warning("❌ 没有足够的数据进行分析")
                return

            logger.info(f"✅ 成功加载 {len(minute_data)} 个时间点的数据")

            # 执行量价分析
            volume_price_signal = self.analyzer.analyze_volume_price_strategy(
                minute_data, "BTCFDUSD"
            )

            if volume_price_signal:
                # 生成动量信号
                momentum_signal = self.analyzer.convert_to_momentum_signal(volume_price_signal)

                # 保存信号
                await self._save_signal(momentum_signal, volume_price_signal)

                logger.info(f"🎯 生成交易信号:")
                logger.info(f"  方向: {momentum_signal['direction'].upper()}")
                logger.info(f"  强度: {momentum_signal['strength']:.3f}")
                logger.info(f"  置信度: {momentum_signal['confidence']:.3f}")
                logger.info(f"  信号类型: {volume_price_signal.signal_type}")
                logger.info(f"  入场价格: {float(volume_price_signal.entry_price):.4f}")
                logger.info(f"  风险收益比: {volume_price_signal.risk_reward_ratio:.2f}")
                logger.info(f"  成交量确认: {volume_price_signal.volume_confirmation:.2f}x")

            else:
                logger.info("📊 当前市场条件不符合交易信号要求")

            self.last_analysis_time = datetime.now()

        except Exception as e:
            logger.error(f"❌ 分析执行失败: {e}")
            import traceback
            traceback.print_exc()

    async def _load_recent_data(self) -> list[MinuteTradeData]:
        """加载最近的数据。"""
        try:
            # 从Redis获取最近15分钟的数据
            minute_data = self.redis_store.get_recent_trade_data(minutes=self.analysis_window_minutes)

            if not minute_data:
                logger.warning("Redis中没有找到数据")
                return []

            # 按时间排序
            minute_data.sort(key=lambda x: x.timestamp)

            # 确保数据量充足
            if len(minute_data) < 5:
                logger.warning(f"数据点不足: {len(minute_data)} < 5")
                return []

            return minute_data

        except Exception as e:
            logger.error(f"加载数据失败: {e}")
            return []

    async def _save_signal(self, momentum_signal: dict, volume_price_signal: VolumePriceSignal):
        """保存信号。"""
        try:
            # 保存动量信号（兼容现有格式）
            signal_file = "signals/latest_volume_price_signal.json"
            signal_data = {
                "timestamp": momentum_signal["timestamp"].isoformat() if hasattr(momentum_signal["timestamp"], 'isoformat') else momentum_signal["timestamp"],
                "symbol": momentum_signal["symbol"],
                "strategy_type": "volume_price",
                "analysis_window_minutes": self.analysis_window_minutes,
                "signal": {
                    "timestamp": momentum_signal["timestamp"].isoformat() if hasattr(momentum_signal["timestamp"], 'isoformat') else momentum_signal["timestamp"],
                    "symbol": momentum_signal["symbol"],
                    "direction": momentum_signal["direction"],
                    "strength": momentum_signal["strength"],
                    "confidence": momentum_signal["confidence"],
                    "raw_score": momentum_signal["raw_score"],
                    "indicators": momentum_signal["indicators"],
                    "timeframe": momentum_signal["timeframe"],
                    "analysis_window_minutes": momentum_signal["analysis_window_minutes"],
                    "trade_count": momentum_signal["trade_count"],
                    "signal_quality_score": momentum_signal["signal_quality_score"],
                    "market_condition": momentum_signal["market_condition"]
                },
                "volume_price_analysis": {
                    "signal_type": volume_price_signal.signal_type,
                    "entry_price": float(volume_price_signal.entry_price),
                    "stop_loss": float(volume_price_signal.stop_loss),
                    "take_profit": float(volume_price_signal.take_profit),
                    "risk_reward_ratio": volume_price_signal.risk_reward_ratio,
                    "volume_confirmation": volume_price_signal.volume_confirmation,
                    "price_distance_from_level": volume_price_signal.price_distance_from_level,
                    "support_level": {
                        "price": float(volume_price_signal.support_level.price),
                        "strength": volume_price_signal.support_level.strength,
                        "volume_concentration": volume_price_signal.support_level.volume_concentration
                    } if volume_price_signal.support_level else None,
                    "resistance_level": {
                        "price": float(volume_price_signal.resistance_level.price),
                        "strength": volume_price_signal.resistance_level.strength,
                        "volume_concentration": volume_price_signal.resistance_level.volume_concentration
                    } if volume_price_signal.resistance_level else None
                },
                "analysis_statistics": {
                    "analysis_time": datetime.now().isoformat(),
                    "data_points": self.analysis_window_minutes,
                    "strategy_params": {
                        "min_volume_ratio": self.analyzer.min_volume_ratio,
                        "max_distance_from_level_percent": self.analyzer.max_distance_from_level_percent,
                        "min_strength_threshold": self.analyzer.min_strength_threshold
                    }
                },
                "processing_time_ms": 0.0,
                "memory_usage_mb": 0.0
            }

            with open(signal_file, 'w', encoding='utf-8') as f:
                json.dump(signal_data, f, indent=2, ensure_ascii=False)

            # 记录到日志
            log_file = "logs/volume_price_signals.log"
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"{datetime.now().isoformat()},{momentum_signal['symbol']},{momentum_signal['direction']},"
                       f"{momentum_signal['strength']:.3f},{momentum_signal['confidence']:.3f},"
                       f"{volume_price_signal.signal_type},{volume_price_signal.volume_confirmation:.2f}x\n")

        except Exception as e:
            logger.error(f"保存信号失败: {e}")

    def _calculate_next_analysis_time(self) -> datetime:
        """计算下次分析时间。"""
        now = datetime.now()

        # 计算下一个分钟的第5秒
        next_minute = now.replace(second=self.analysis_delay_seconds, microsecond=0)
        if now.second >= self.analysis_delay_seconds:
            next_minute += timedelta(minutes=1)

        return next_minute

    def _signal_handler(self, signum, frame):
        """信号处理器。"""
        logger.info(f"收到信号 {signum}，正在停止策略...")
        self.running = False

    async def stop(self):
        """停止策略。"""
        logger.info("正在停止量价策略...")
        self.running = False
        await self.redis_store.close()


async def main():
    """主函数。"""
    launcher = VolumePriceStrategyLauncher(
        analysis_window_minutes=15,
        analysis_interval_seconds=60,
        analysis_delay_seconds=5,
        min_volume_ratio=1.5,
        max_distance_from_level_percent=0.2,
        min_strength_threshold=0.6
    )

    try:
        await launcher.start()
    except KeyboardInterrupt:
        logger.info("收到中断信号")
    except Exception as e:
        logger.error(f"策略运行异常: {e}")
    finally:
        await launcher.stop()


if __name__ == "__main__":
    asyncio.run(main())