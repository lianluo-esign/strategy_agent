#!/usr/bin/env python3
"""精确定时动量策略。

每分钟的第5秒开始分析，确保Redis数据已写入完成。
"""

import asyncio
import logging
import signal
import sys
import time
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import yaml

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.core.models import Trade, MinuteTradeData, PriceLevelData
from src.core.order_flow_momentum_analyzer import OrderFlowMomentumAnalyzer
from src.core.redis_client import RedisDataStore
from src.core.constants import REDIS_TRADES_WINDOW_KEY


class TimedMomentumStrategyLauncher:
    """精确定时动量策略启动器。"""

    def __init__(self, config_path: str = "config/momentum_strategy.yaml"):
        """初始化策略启动器。

        Args:
            config_path: 配置文件路径
        """
        self.config = self._load_config(config_path)

        # 使用订单流动量分析器
        self.analyzer = OrderFlowMomentumAnalyzer(
            window_size_minutes=self.config["analyzer"]["window_size_minutes"],
            buy_threshold=self.config["analyzer"].get("buy_threshold", 0.15),
            sell_threshold=self.config["analyzer"].get("sell_threshold", -0.15),
            neutral_range=self.config["analyzer"].get("neutral_range", 0.05),
        )

        # 初始化Redis连接
        redis_config = self.config.get("redis", {})
        self.redis_store = RedisDataStore(
            host=redis_config.get("host", "localhost"),
            port=redis_config.get("port", 6379),
            db=redis_config.get("db", 0),
            storage_dir=redis_config.get("storage_dir", "storage")
        )

        # 测试Redis连接
        if not self.redis_store.test_connection():
            self.logger.error("❌ Redis连接失败，请检查Redis服务是否运行")
            sys.exit(1)

        # 设置日志
        self._setup_logging()

        # 运行状态
        self.is_running = False
        self.signal_count = 0
        self.start_time = None

        # 定时配置
        self.analysis_delay_seconds = self.config.get("timing", {}).get("analysis_delay_seconds", 5)
        self.analysis_window_minutes = self.config["analyzer"]["window_size_minutes"]

        # 注册信号处理器
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _load_config(self, config_path: str) -> dict[str, Any]:
        """加载配置文件。"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config
        except FileNotFoundError:
            logging.warning(f"配置文件 {config_path} 不存在，使用默认配置")
            return self._get_default_config()
        except yaml.YAMLError as e:
            logging.error(f"配置文件解析错误: {e}")
            sys.exit(1)

    def _get_default_config(self) -> dict[str, Any]:
        """获取默认配置。"""
        return {
            "analyzer": {
                "window_size_minutes": 3,          # 3分钟窗口
                "buy_threshold": 0.12,
                "sell_threshold": -0.12,
                "neutral_range": 0.03,
            },
            "data": {
                "symbol": "BTCFDUSD",
                "data_source": "redis",
            },
            "redis": {
                "host": "localhost",
                "port": 6379,
                "db": 0,
                "storage_dir": "storage",
            },
            "output": {
                "console": True,
                "file": True,
                "file_path": "logs/timed_momentum_signals.log",
                "signal_file": "signals/timed_latest_signal.json",
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            },
            "timing": {
                "analysis_delay_seconds": 5,       # 每分钟第5秒开始分析
                "max_wait_time_seconds": 30,       # 最大等待时间
            }
        }

    def _setup_logging(self) -> None:
        """设置日志配置。"""
        log_config = self.config["logging"]

        # 创建日志目录
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        # 配置日志
        logging.basicConfig(
            level=getattr(logging, log_config["level"]),
            format=log_config["format"],
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler("logs/timed_momentum_strategy.log", encoding='utf-8')
            ]
        )

        self.logger = logging.getLogger(__name__)

    def _signal_handler(self, signum: int, frame) -> None:
        """信号处理器。"""
        self.logger.info(f"收到信号 {signum}，正在停止策略...")
        self.is_running = False

    def _calculate_next_analysis_time(self) -> datetime:
        """计算下次分析时间（每分钟的第5秒）。"""
        now = datetime.now()

        # 计算下一分钟的第5秒
        next_minute = now.replace(second=self.analysis_delay_seconds, microsecond=0)
        if now.second >= self.analysis_delay_seconds:
            next_minute += timedelta(minutes=1)

        return next_minute

    def _wait_for_analysis_time(self) -> None:
        """等待到分析时间。"""
        next_time = self._calculate_next_analysis_time()
        now = datetime.now()

        if next_time > now:
            wait_seconds = (next_time - now).total_seconds()
            self.logger.info(f"⏰ 等待到 {next_time.strftime('%H:%M:%S')} 开始分析（等待 {wait_seconds:.1f} 秒）")

            # 分段等待，支持信号中断
            while wait_seconds > 0 and self.is_running:
                sleep_time = min(1.0, wait_seconds)  # 每次最多等待1秒
                time.sleep(sleep_time)
                wait_seconds -= sleep_time

    async def _load_minute_data_from_redis(self) -> list[MinuteTradeData]:
        """从Redis加载指定时间窗口的分钟数据。"""
        try:
            # 获取最近N分钟的订单流数据
            minute_trade_data_list = self.redis_store.get_recent_trade_data(
                minutes=self.analysis_window_minutes
            )

            if not minute_trade_data_list:
                self.logger.warning(f"Redis中没有找到最近{self.analysis_window_minutes}分钟的数据")
                return []

            # 验证数据新鲜度
            if len(minute_trade_data_list) < self.analysis_window_minutes:
                self.logger.warning(
                    f"数据不足：期望{self.analysis_window_minutes}分钟，实际获得{len(minute_trade_data_list)}分钟"
                )

            # 检查最新数据的时间戳
            latest_data = minute_trade_data_list[0] if minute_trade_data_list else None
            if latest_data:
                data_age = (datetime.now() - latest_data.timestamp).total_seconds()
                if data_age > 120:  # 超过2分钟认为数据过期
                    self.logger.warning(f"数据可能过期：最新数据时间是 {latest_data.timestamp}（{data_age:.0f}秒前）")

            self.logger.info(
                f"从Redis加载了 {len(minute_trade_data_list)} 个时间点的订单流数据 "
                f"(最新: {latest_data.timestamp.strftime('%H:%M:%S') if latest_data else 'N/A'})"
            )
            return minute_trade_data_list

        except Exception as e:
            self.logger.error(f"从Redis加载订单流数据失败: {e}")
            return []

    async def _wait_for_fresh_data(self, max_wait_seconds: int = 30) -> bool:
        """等待最新分钟数据写入Redis。"""
        self.logger.info("🔄 等待Data Collector写入最新数据...")

        start_time = datetime.now()

        while (datetime.now() - start_time).total_seconds() < max_wait_seconds:
            # 检查最新数据是否已更新
            minute_data = await self._load_minute_data_from_redis()

            if minute_data:
                latest_data = minute_data[0]
                current_minute = datetime.now().replace(second=0, microsecond=0)

                # 检查最新数据是否是当前分钟的数据
                if latest_data.timestamp >= current_minute - timedelta(minutes=1):
                    self.logger.info(
                        f"✅ 检测到最新数据：{latest_data.timestamp.strftime('%H:%M:%S')}"
                    )
                    return True

                # 检查是否有最近30秒内更新的数据
                data_age = (datetime.now() - latest_data.timestamp).total_seconds()
                if data_age <= 30:
                    self.logger.info(
                        f"✅ 数据较新：{latest_data.timestamp.strftime('%H:%M:%S')}（{data_age:.0f}秒前）"
                    )
                    return True

            # 等待1秒后重试
            await asyncio.sleep(1)

        self.logger.warning("⚠️ 等待超时，将使用现有数据进行分析")
        return False

    def _output_signal(self, result: Any) -> None:
        """输出交易信号。"""
        output_config = self.config["output"]

        # 控制台输出
        if output_config.get("console", True):
            self._print_signal(result)

        # 文件输出
        if output_config.get("file", True):
            self._save_signal_to_file(result)

        # 保存最新信号到文件
        self._save_latest_signal(result)

    def _print_signal(self, result: Any) -> None:
        """打印信号到控制台。"""
        signal = result.signal
        indicators = signal.indicators

        print("\n" + "=" * 80)
        print(f"🎯 定时动量信号 #{self.signal_count + 1}")
        print("=" * 80)
        print(f"📊 交易对: {signal.symbol}")
        print(f"⏰ 分析时间: {signal.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🎯 信号方向: {self._get_direction_emoji(signal.direction)} {signal.direction.value.upper()}")
        print(f"💪 信号强度: {signal.strength:.3f}")
        print(f"🔍 置信度: {signal.confidence:.3f}")
        print(f"📈 原始分数: {signal.raw_score:.4f}")
        print(f"🏪 市场条件: {signal.market_condition}")
        print(f"📋 交易数量: {signal.trade_count}")
        print(f"⚡ 处理时间: {result.processing_time_ms:.2f}ms")
        print(f"🕐 数据窗口: {self.analysis_window_minutes}分钟")

        print("\n📊 关键指标:")
        print(f"  💰 订单流动量: {indicators.order_flow_momentum:+.4f}")
        print(f"  ⚖️ 成交量不平衡: {indicators.volume_imbalance:+.4f}")
        print(f"  📈 价格动量: {indicators.price_momentum:+.4f}")
        print(f"  🔄 趋势强度: {indicators.trend_strength:+.4f}")
        print(f"  📊 流动一致性: {indicators.flow_consistency:+.4f}")
        print(f"  📉 实现波动率: {indicators.realized_volatility:.4f}")

        print("=" * 80)

    def _get_direction_emoji(self, direction: Any) -> str:
        """获取方向对应的emoji。"""
        from src.core.momentum_models import MomentumDirection

        if direction == MomentumDirection.BUY:
            return "🟢"
        elif direction == MomentumDirection.SELL:
            return "🔴"
        else:
            return "🟡"

    def _save_signal_to_file(self, result: Any) -> None:
        """保存信号到日志文件。"""
        try:
            log_file = Path(self.config["output"]["file_path"])
            log_file.parent.mkdir(exist_ok=True)

            with open(log_file, 'a', encoding='utf-8') as f:
                signal = result.signal
                f.write(f"{signal.timestamp.isoformat()},"
                       f"{signal.symbol},"
                       f"{signal.direction.value},"
                       f"{signal.strength:.3f},"
                       f"{signal.confidence:.3f},"
                       f"{signal.raw_score:.4f},"
                       f"{signal.trade_count},"
                       f"{result.processing_time_ms:.2f},"
                       f"timed_analysis\n")
        except Exception as e:
            self.logger.error(f"保存信号日志失败: {e}")

    def _save_latest_signal(self, result: Any) -> None:
        """保存最新信号到文件。"""
        try:
            signal_file = Path(self.config["output"]["signal_file"])
            signal_file.parent.mkdir(exist_ok=True)

            import json
            result_dict = result.to_dict()

            # 转换datetime对象为字符串
            def convert_datetime(obj):
                if hasattr(obj, 'isoformat'):
                    return obj.isoformat()
                elif isinstance(obj, dict):
                    return {k: convert_datetime(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_datetime(item) for item in obj]
                return obj

            result_dict = convert_datetime(result_dict)
            result_dict["analysis_type"] = "timed_momentum"

            with open(signal_file, 'w', encoding='utf-8') as f:
                json.dump(result_dict, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"保存最新信号失败: {e}")

    async def run_analysis_loop(self) -> None:
        """运行精确定时分析循环。"""
        self.logger.info("🚀 启动精确定时动量策略分析...")
        self.logger.info(f"⏱️ 分析配置：每分钟第{self.analysis_delay_seconds}秒开始分析")
        self.logger.info(f"📊 数据窗口：{self.analysis_window_minutes}分钟")

        self.is_running = True
        self.start_time = datetime.now()

        try:
            while self.is_running:
                cycle_start = time.time()

                # 1. 等待到精确的分析时间
                self._wait_for_analysis_time()

                if not self.is_running:
                    break

                # 2. 等待最新数据写入
                await self._wait_for_fresh_data(
                    max_wait_seconds=self.config.get("timing", {}).get("max_wait_time_seconds", 30)
                )

                # 3. 获取订单流数据
                minute_data = await self._load_minute_data_from_redis()

                if not minute_data:
                    self.logger.warning("未获取到订单流数据，跳过本次分析")
                    continue

                # 4. 执行订单流动量分析
                try:
                    result = self.analyzer.analyze_order_flow_momentum(
                        minute_data,
                        self.config["data"]["symbol"]
                    )

                    # 5. 输出信号
                    self._output_signal(result)

                    self.signal_count += 1

                    # 记录分析完成时间
                    analysis_time = time.time() - cycle_start
                    self.logger.info(f"✅ 分析完成，耗时 {analysis_time:.3f}秒")

                except Exception as e:
                    self.logger.error(f"动量分析失败: {e}")

        except KeyboardInterrupt:
            self.logger.info("收到键盘中断，正在停止...")
        except Exception as e:
            self.logger.error(f"分析循环发生错误: {e}")
        finally:
            await self._shutdown()

    async def _shutdown(self) -> None:
        """关闭策略。"""
        self.is_running = False

        if self.start_time:
            runtime = datetime.now() - self.start_time
            self.logger.info(f"策略运行时长: {runtime}")

        self.logger.info(f"总共生成 {self.signal_count} 个信号")

        # 关闭Redis连接
        try:
            await self.redis_store.close()
            self.logger.info("Redis连接已关闭")
        except Exception as e:
            self.logger.warning(f"关闭Redis连接时发生错误: {e}")

        self.logger.info("精确定时动量策略已停止")

    async def run(self) -> None:
        """启动策略。"""
        self.logger.info("🚀 启动精确定时动量策略")
        self.logger.info(f"配置: {self.config}")

        try:
            await self.run_analysis_loop()
        except Exception as e:
            self.logger.error(f"策略运行失败: {e}")
            raise


async def main():
    """主函数。"""
    import argparse

    parser = argparse.ArgumentParser(description="精确定时动量策略启动器")
    parser.add_argument(
        "--config",
        default="config/momentum_strategy.yaml",
        help="配置文件路径"
    )

    args = parser.parse_args()

    # 创建启动器
    launcher = TimedMomentumStrategyLauncher(args.config)

    try:
        await launcher.run()
    except KeyboardInterrupt:
        print("\n👋 策略已手动停止")
    except Exception as e:
        print(f"❌ 策略启动失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())