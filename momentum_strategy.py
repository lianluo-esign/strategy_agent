#!/usr/bin/env python3
"""短期动量策略启动入口文件。

实时分析BTC-FDUSD交易数据，生成短期动量交易信号。
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
sys.path.append(str(Path(__file__).parent))

from src.core.models import Trade, MinuteTradeData, PriceLevelData
from src.core.short_term_momentum_analyzer import ShortTermMomentumAnalyzer
from src.core.order_flow_momentum_analyzer import OrderFlowMomentumAnalyzer
from src.core.redis_client import RedisDataStore
from src.core.constants import REDIS_TRADES_WINDOW_KEY


class MomentumStrategyLauncher:
    """短期动量策略启动器。"""

    def __init__(self, config_path: str = "config/momentum_strategy.yaml"):
        """初始化策略启动器。

        Args:
            config_path: 配置文件路径
        """
        self.config = self._load_config(config_path)
        # 使用优化的订单流动量分析器
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
                "window_size_minutes": 5,
                "min_trades": 10,
                "min_volume": 0.01,
                "buy_threshold": 0.15,
                "sell_threshold": -0.15,
                "neutral_range": 0.05,
            },
            "data": {
                "symbol": "BTCFDUSD",
                "data_source": "redis",  # redis, mock, file, websocket
                "mock_trades_per_second": 20,
                "data_file_path": "storage/trades_data.json",
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
                "file_path": "logs/momentum_signals.log",
                "signal_file": "signals/latest_signal.json",
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
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
                logging.FileHandler("logs/momentum_strategy.log", encoding='utf-8')
            ]
        )

        self.logger = logging.getLogger(__name__)

    def _signal_handler(self, signum: int, frame) -> None:
        """信号处理器。"""
        self.logger.info(f"收到信号 {signum}，正在停止策略...")
        self.is_running = False

    async def _generate_mock_trades(self) -> list[Trade]:
        """生成模拟交易数据。"""
        trades = []
        current_time = datetime.now()
        base_price = Decimal("42000.00")

        # 模拟价格趋势
        trend_direction = 1 if self.signal_count % 3 == 0 else (-1 if self.signal_count % 3 == 1 else 0)

        trades_per_batch = self.config["data"]["mock_trades_per_second"]

        for i in range(trades_per_batch):
            # 确保交易在5分钟窗口内（密集分布）
            timestamp = current_time - timedelta(seconds=i * 12)  # 每12秒一笔交易

            # 模拟价格变化
            price_change = Decimal(str(trend_direction * i * 0.5))
            price = base_price + price_change + Decimal(str((i % 10) * 0.1))

            # 模拟交易量
            quantity = Decimal(str(0.1 + (i % 5) * 0.02))

            # 模拟买卖方向
            is_buyer_maker = (i + trend_direction) % 3 != 0

            trade = Trade(
                symbol=self.config["data"]["symbol"],
                price=price,
                quantity=quantity,
                is_buyer_maker=is_buyer_maker,
                timestamp=timestamp,
                trade_id=f"mock_trade_{int(time.time())}_{i}"
            )
            trades.append(trade)

        return trades

    async def _load_trades_from_file(self) -> list[Trade]:
        """从文件加载交易数据。"""
        try:
            import json
            file_path = Path(self.config["data"]["data_file_path"])

            if not file_path.exists():
                self.logger.warning(f"数据文件 {file_path} 不存在")
                return []

            with open(file_path, 'r', encoding='utf-8') as f:
                trades_data = json.load(f)

            trades = []
            for trade_data in trades_data[-50:]:  # 只取最近50条
                trade = Trade(
                    symbol=trade_data["symbol"],
                    price=Decimal(str(trade_data["price"])),
                    quantity=Decimal(str(trade_data["quantity"])),
                    is_buyer_maker=trade_data["is_buyer_maker"],
                    timestamp=datetime.fromisoformat(trade_data["timestamp"]),
                    trade_id=trade_data["trade_id"]
                )
                trades.append(trade)

            return trades

        except Exception as e:
            self.logger.error(f"加载交易数据失败: {e}")
            return []

    async def _load_minute_data_from_redis(self) -> list[MinuteTradeData]:
        """从Redis加载trades_window数据（直接使用MinuteTradeData）。"""
        try:
            # 获取最近的分钟交易数据
            window_minutes = self.config["analyzer"]["window_size_minutes"]
            minute_trade_data_list = self.redis_store.get_recent_trade_data(minutes=window_minutes)

            if not minute_trade_data_list:
                self.logger.warning("Redis中没有找到trades_window数据")
                return []

            self.logger.info(f"从Redis加载了 {len(minute_trade_data_list)} 个时间点的订单流数据")
            return minute_trade_data_list

        except Exception as e:
            self.logger.error(f"从Redis加载trades_window数据失败: {e}")
            return []

    async def _get_minute_data(self) -> list[MinuteTradeData]:
        """获取分钟级订单流数据。"""
        data_source = self.config["data"]["data_source"]

        if data_source == "redis":
            return await self._load_minute_data_from_redis()
        elif data_source == "mock":
            # 生成模拟的MinuteTradeData
            return await self._generate_mock_minute_data()
        elif data_source == "file":
            # 从文件加载并转换为MinuteTradeData
            return await self._load_minute_data_from_file()
        elif data_source == "websocket":
            # TODO: 实现WebSocket数据获取
            self.logger.warning("WebSocket数据源尚未实现，使用Redis数据")
            return await self._load_minute_data_from_redis()
        else:
            self.logger.error(f"未知的数据源: {data_source}")
            return []

    async def _generate_mock_minute_data(self) -> list[MinuteTradeData]:
        """生成模拟的MinuteTradeData。"""
        from datetime import timedelta
        import random

        minute_data_list = []
        current_time = datetime.now()
        base_price = 114000.0

        # 生成最近N分钟的模拟数据
        window_minutes = self.config["analyzer"]["window_size_minutes"]
        for i in range(window_minutes):
            timestamp = current_time - timedelta(minutes=i)

            # 模拟价格层级数据
            price_levels = {}
            for j in range(10):  # 10个价格层级
                price = base_price + (j - 5) * 10 + random.uniform(-5, 5)
                buy_volume = random.uniform(0.1, 2.0) if random.random() > 0.3 else 0.0
                sell_volume = random.uniform(0.1, 2.0) if random.random() > 0.3 else 0.0
                trade_count = int((buy_volume + sell_volume) * 10)

                price_levels[str(price)] = {
                    "price_level": price,
                    "buy_volume": buy_volume,
                    "sell_volume": sell_volume,
                    "total_volume": buy_volume + sell_volume,
                    "delta": buy_volume - sell_volume,
                    "trade_count": trade_count
                }

            minute_data = MinuteTradeData(
                timestamp=timestamp,
                symbol=self.config["data"]["symbol"],
                price_levels=price_levels
            )
            minute_data_list.append(minute_data)

        return minute_data_list

    async def _load_minute_data_from_file(self) -> list[MinuteTradeData]:
        """从文件加载交易数据并转换为MinuteTradeData（兼容模式）。"""
        trades = await self._load_trades_from_file()
        if not trades:
            return []

        # 将Trade数据转换为MinuteTradeData（简化版本）
        from collections import defaultdict

        minute_groups = defaultdict(list)
        for trade in trades:
            minute_key = trade.timestamp.replace(second=0, microsecond=0)
            minute_groups[minute_key].append(trade)

        minute_data_list = []
        for minute_time, minute_trades in minute_groups.items():
            price_levels = defaultdict(lambda: {
                "price_level": 0.0,
                "buy_volume": 0.0,
                "sell_volume": 0.0,
                "total_volume": 0.0,
                "delta": 0.0,
                "trade_count": 0
            })

            for trade in minute_trades:
                price_str = str(trade.price)
                price_levels[price_str]["price_level"] = float(trade.price)
                price_levels[price_str]["total_volume"] += float(trade.quantity)
                price_levels[price_str]["trade_count"] += 1

                if not trade.is_buyer_maker:  # 主动买入
                    price_levels[price_str]["buy_volume"] += float(trade.quantity)
                else:  # 主动卖出
                    price_levels[price_str]["sell_volume"] += float(trade.quantity)

            for level_data in price_levels.values():
                level_data["delta"] = level_data["buy_volume"] - level_data["sell_volume"]

            minute_data = MinuteTradeData(
                timestamp=minute_time,
                symbol=trade.symbol,
                price_levels=dict(price_levels)
            )
            minute_data_list.append(minute_data)

        return minute_data_list

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
        print(f"🚀 短期动量信号 #{self.signal_count + 1}")
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

        print("\n📊 关键指标:")
        print(f"  💰 价格动量: {indicators.price_momentum:+.4f}")
        print(f"  📊 成交量动量: {indicators.volume_momentum:+.4f}")
        print(f"  🔄 订单流动量: {indicators.order_flow_momentum:+.4f}")
        print(f"  📈 波动率调整: {indicators.volatility_adjusted:+.4f}")
        print(f"  ⚖️ 成交量不平衡: {indicators.volume_imbalance:+.4f}")
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
                       f"{result.processing_time_ms:.2f}\n")
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

            with open(signal_file, 'w', encoding='utf-8') as f:
                json.dump(result_dict, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"保存最新信号失败: {e}")

    async def run_analysis_loop(self) -> None:
        """运行分析循环。"""
        self.logger.info("开始短期动量策略分析...")
        self.is_running = True
        self.start_time = datetime.now()

        analysis_interval = self.config.get("analysis_interval_seconds", 60)  # 默认每分钟分析一次

        try:
            while self.is_running:
                cycle_start = time.time()

                # 获取分钟级订单流数据
                minute_data = await self._get_minute_data()

                if not minute_data:
                    self.logger.warning("未获取到订单流数据，跳过本次分析")
                    await asyncio.sleep(analysis_interval)
                    continue

                # 执行订单流动量分析
                result = self.analyzer.analyze_order_flow_momentum(
                    minute_data,
                    self.config["data"]["symbol"]
                )

                # 输出信号
                self._output_signal(result)

                self.signal_count += 1

                # 计算处理时间
                cycle_time = time.time() - cycle_start
                sleep_time = max(0, analysis_interval - cycle_time)

                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)

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

        self.logger.info("短期动量策略已停止")

    async def run(self) -> None:
        """启动策略。"""
        self.logger.info("🚀 启动短期动量策略")
        self.logger.info(f"配置: {self.config}")

        try:
            await self.run_analysis_loop()
        except Exception as e:
            self.logger.error(f"策略运行失败: {e}")
            raise


async def main():
    """主函数。"""
    import argparse

    parser = argparse.ArgumentParser(description="短期动量策略启动器")
    parser.add_argument(
        "--config",
        default="config/momentum_strategy.yaml",
        help="配置文件路径"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="模拟运行模式"
    )

    args = parser.parse_args()

    # 创建启动器
    launcher = MomentumStrategyLauncher(args.config)

    try:
        await launcher.run()
    except KeyboardInterrupt:
        print("\n👋 策略已手动停止")
    except Exception as e:
        print(f"❌ 策略启动失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())