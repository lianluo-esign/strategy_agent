"""交易数据收集器 - 简化版trades_window数据处理器。

该模块专注于直接收集trades_window原始数据，移除复杂的聚合处理逻辑，
为Deepseek AI分析提供未经修改的原始市场数据。
"""

import logging
from datetime import datetime, timedelta
from typing import Any

logger = logging.getLogger(__name__)

# 数据收集配置常量
DEFAULT_ANALYSIS_MINUTES = 4320  # 默认分析过去72小时的数据（3天）


class RawTradesData:
    """原始交易数据模型。"""

    def __init__(
        self,
        timestamp: datetime,
        symbol: str,
        minute_data_points: list[dict[str, Any]],
        data_points_count: int,
        time_range: tuple[str, str]
    ):
        self.timestamp = timestamp
        self.symbol = symbol
        self.minute_data_points = minute_data_points
        self.data_points_count = data_points_count
        self.time_range = time_range

    def to_dict(self) -> dict[str, Any]:
        """转换为字典格式。"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "minute_data_points": self.minute_data_points,
            "data_points_count": self.data_points_count,
            "time_range": self.time_range,
            "analysis_minutes": len(self.minute_data_points)
        }


class TradesAggregator:
    """交易数据收集器，专注于直接收集trades_window原始数据。

    该类替代了复杂的聚合处理逻辑，专注于：
    1. 直接收集trades_window原始分钟数据
    2. 保持数据完整性，不做二次加工
    3. 为AI分析提供未经修改的市场数据
    """

    def __init__(
        self,
        minutes_to_collect: int = DEFAULT_ANALYSIS_MINUTES
    ):
        """初始化交易数据收集器。

        Args:
            minutes_to_collect: 收集的时间窗口（分钟）
        """
        if minutes_to_collect <= 0:
            raise ValueError("收集时间窗口必须为正数")

        self.minutes_to_collect = minutes_to_collect

        logger.info(
            f"Initialized TradesAggregator with collection_window={minutes_to_collect} minutes"
        )

    def collect_raw_trades_data(
        self,
        trades_window_data: list[Any],
        symbol: str = "BTCFDUSD"
    ) -> RawTradesData:
        """收集trades_window原始数据。

        Args:
            trades_window_data: 从Redis获取的trades_window数据
            symbol: 交易符号

        Returns:
            原始交易数据集合

        Raises:
            ValueError: 当输入数据无效时
        """
        logger.info(f"Starting collection of raw trades window data for {symbol}")

        if not trades_window_data:
            raise ValueError("trades_window_data不能为空")

        # 转换为原始数据格式，不做任何加工
        minute_data_points = self._convert_to_raw_format(trades_window_data)

        if not minute_data_points:
            raise ValueError("没有有效的交易数据可供收集")

        # 计算时间范围
        timestamps = [point["timestamp"] for point in minute_data_points]
        time_range = (min(timestamps), max(timestamps))

        # 创建原始数据结果
        result = RawTradesData(
            timestamp=datetime.now(),
            symbol=symbol,
            minute_data_points=minute_data_points,
            data_points_count=len(minute_data_points),
            time_range=time_range
        )

        logger.info(
            f"Raw data collection completed: {len(minute_data_points)} 分钟数据点, "
            f"时间范围: {time_range[0]} 到 {time_range[1]}"
        )

        return result

    def _convert_to_raw_format(self, trades_window_data: list[Any]) -> list[dict[str, Any]]:
        """将trades_window数据转换为原始格式。

        Args:
            trades_window_data: 从Redis获取的原始交易数据列表

        Returns:
            标准化的原始数据点列表
        """
        import json

        raw_data_points = []
        cutoff_time = datetime.now() - timedelta(minutes=self.minutes_to_collect)

        for minute_data in trades_window_data:
            try:
                # 处理MinuteTradeData对象格式
                if hasattr(minute_data, 'timestamp') and hasattr(minute_data, 'price_levels'):
                    # 直接访问MinuteTradeData对象的属性
                    data_time = minute_data.timestamp
                    price_levels = minute_data.price_levels

                    # 检查时间是否在收集窗口内
                    if data_time < cutoff_time:
                        continue

                    # 检查价格水平数据
                    if not price_levels:
                        continue

                    # 转换price_levels中的Decimal键为float，以便JSON序列化
                    converted_price_levels = {}
                    for price_level, data in price_levels.items():
                        if hasattr(price_level, 'float'):  # Decimal类型
                            price_key = float(price_level)
                        else:
                            price_key = float(price_level)
                        # 保持原始精度，转换为字符串键
                        converted_price_levels[str(price_key)] = data

                    # 创建原始数据格式
                    raw_point = {
                        "timestamp": data_time.isoformat(),
                        "price_levels": converted_price_levels
                    }

                    raw_data_points.append(raw_point)

                elif isinstance(minute_data, str):
                    # 处理JSON字符串格式（向后兼容）
                    try:
                        parsed_data = json.loads(minute_data)
                    except json.JSONDecodeError:
                        logger.debug(f"Failed to parse JSON data: {minute_data[:100]}...")
                        continue

                    # 检查数据是否包含必要字段
                    if not isinstance(parsed_data, dict) or "timestamp" not in parsed_data or "price_levels" not in parsed_data:
                        continue

                    # 解析时间戳
                    timestamp_str = parsed_data["timestamp"]
                    if isinstance(timestamp_str, str):
                        if timestamp_str.endswith('Z'):
                            timestamp_str = timestamp_str.replace('Z', '+00:00')
                        data_time = datetime.fromisoformat(timestamp_str)
                    else:
                        data_time = timestamp_str

                    # 检查时间是否在收集窗口内
                    if data_time < cutoff_time:
                        continue

                    # 检查价格水平数据
                    price_levels = parsed_data["price_levels"]
                    if not price_levels or not isinstance(price_levels, dict):
                        continue

                    raw_point = {
                        "timestamp": data_time.isoformat(),
                        "price_levels": price_levels
                    }

                    raw_data_points.append(raw_point)

                elif isinstance(minute_data, dict):
                    # 处理字典格式（向后兼容）
                    if "timestamp" not in minute_data or "price_levels" not in minute_data:
                        continue

                    # 解析时间戳
                    timestamp_str = minute_data["timestamp"]
                    if isinstance(timestamp_str, str):
                        if timestamp_str.endswith('Z'):
                            timestamp_str = timestamp_str.replace('Z', '+00:00')
                        data_time = datetime.fromisoformat(timestamp_str)
                    else:
                        data_time = timestamp_str

                    # 检查时间是否在收集窗口内
                    if data_time < cutoff_time:
                        continue

                    price_levels = minute_data["price_levels"]
                    if not price_levels:
                        continue

                    raw_point = {
                        "timestamp": data_time.isoformat(),
                        "price_levels": price_levels
                    }

                    raw_data_points.append(raw_point)
                else:
                    logger.debug(f"Unsupported data type: {type(minute_data)}")
                    continue

            except Exception as e:
                logger.info(f"跳过无效数据点: {e}")
                logger.info(f"Problem data sample: {str(minute_data)[:200]}")
                continue

        logger.info(f"原始数据转换完成: {len(raw_data_points)}/{len(trades_window_data)} 数据点有效")
        return raw_data_points

    def get_data_summary(self, raw_data: RawTradesData) -> dict[str, Any]:
        """生成原始数据摘要。

        Args:
            raw_data: 原始交易数据

        Returns:
            数据摘要字典
        """
        return {
            "collection_timestamp": raw_data.timestamp.isoformat(),
            "symbol": raw_data.symbol,
            "data_summary": {
                "minutes_collected": raw_data.data_points_count,
                "time_range": raw_data.time_range,
                "collection_window_minutes": self.minutes_to_collect
            },
            "note": "原始trades_window数据，未经任何聚合处理，直接提供给AI分析"
        }
