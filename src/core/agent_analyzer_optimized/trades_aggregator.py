"""交易数据聚合器 - 优化版trades_window数据处理器。

该模块专注于trades_window数据的聚合处理，移除了对5000层深度快照的依赖，
提供高效的数据聚合和预处理功能。
"""

import logging
from collections import defaultdict
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# 聚合配置常量
DEFAULT_AGGREGATION_PRECISION = 10.0  # 默认价格聚合精度
MINUTES_TO_ANALYZE = 1440  # 分析过去24小时的数据
VOLUME_THRESHOLD = 0.1  # 最小成交量阈值


class AggregatedTradesData:
    """聚合后的交易数据模型。"""

    def __init__(
        self,
        timestamp: datetime,
        symbol: str,
        price_levels: Dict[float, float],
        total_volume: float,
        trade_count: int,
        price_range: tuple[float, float]
    ):
        self.timestamp = timestamp
        self.symbol = symbol
        self.price_levels = price_levels
        self.total_volume = total_volume
        self.trade_count = trade_count
        self.price_range = price_range

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式。"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "price_levels": self.price_levels,
            "total_volume": self.total_volume,
            "trade_count": self.trade_count,
            "price_range": self.price_range,
            "price_levels_count": len(self.price_levels)
        }


class TradesAggregator:
    """交易数据聚合器，专注于trades_window数据的高效处理。

    该类替代了原有的深度快照数据处理逻辑，专注于：
    1. 高效聚合trades_window数据
    2. 生成价格-成交量分布
    3. 提供市场结构分析基础数据
    """

    def __init__(
        self,
        aggregation_precision: float = DEFAULT_AGGREGATION_PRECISION,
        min_volume_threshold: float = VOLUME_THRESHOLD,
        minutes_to_analyze: int = MINUTES_TO_ANALYZE
    ):
        """初始化交易数据聚合器。

        Args:
            aggregation_precision: 价格聚合精度
            min_volume_threshold: 最小成交量阈值
            minutes_to_analyze: 分析的时间窗口（分钟）
        """
        if aggregation_precision <= 0:
            raise ValueError("聚合精度必须为正数")

        if min_volume_threshold < 0:
            raise ValueError("最小成交量阈值不能为负数")

        if minutes_to_analyze <= 0:
            raise ValueError("分析时间窗口必须为正数")

        self.aggregation_precision = Decimal(str(aggregation_precision))
        self.min_volume_threshold = min_volume_threshold
        self.minutes_to_analyze = minutes_to_analyze

        logger.info(
            f"Initialized TradesAggregator with precision=${aggregation_precision}, "
            f"min_volume_threshold={min_volume_threshold}, "
            f"analysis_window={minutes_to_analyze} minutes"
        )

    def aggregate_trades_window(
        self,
        trades_window_data: List[Any],
        symbol: str = "BTCFDUSD"
    ) -> AggregatedTradesData:
        """聚合trades_window数据。

        Args:
            trades_window_data: 从Redis获取的trades_window数据
            symbol: 交易符号

        Returns:
            聚合后的交易数据

        Raises:
            ValueError: 当输入数据无效时
        """
        logger.info(f"Starting aggregation of trades window data for {symbol}")

        if not trades_window_data:
            raise ValueError("trades_window_data不能为空")

        # 验证数据质量
        valid_data = self._validate_and_filter_data(trades_window_data)
        if not valid_data:
            raise ValueError("没有有效的交易数据可供聚合")

        # 执行聚合
        volume_profile = self._build_volume_profile(valid_data)

        # 计算统计信息
        total_volume = sum(volume_profile.values())
        trade_count = len(valid_data)

        if total_volume <= 0:
            raise ValueError("聚合后的总交易量为0")

        # 计算价格范围
        prices = list(volume_profile.keys())
        price_range = (min(prices), max(prices))

        # 创建聚合结果
        result = AggregatedTradesData(
            timestamp=datetime.now(),
            symbol=symbol,
            price_levels=volume_profile,
            total_volume=float(total_volume),
            trade_count=trade_count,
            price_range=price_range
        )

        logger.info(
            f"Aggregation completed: {len(volume_profile)} price levels, "
            f"total_volume={result.total_volume:.2f}, "
            f"price_range=${price_range[0]:.2f}-${price_range[1]:.2f}"
        )

        return result

    def _validate_and_filter_data(self, trades_window_data: List[Any]) -> List[Any]:
        """验证并过滤无效的交易数据。

        Args:
            trades_window_data: 原始交易数据列表

        Returns:
            过滤后的有效数据列表
        """
        valid_data = []
        cutoff_time = datetime.now() - timedelta(minutes=self.minutes_to_analyze)

        for minute_data in trades_window_data:
            try:
                # 检查数据是否包含必要字段
                if not hasattr(minute_data, "timestamp") or not hasattr(minute_data, "price_levels"):
                    continue

                # 检查时间是否在分析窗口内
                if hasattr(minute_data.timestamp, 'timestamp'):
                    # 处理时间戳格式
                    data_time = datetime.fromtimestamp(minute_data.timestamp.timestamp())
                else:
                    data_time = minute_data.timestamp

                if data_time < cutoff_time:
                    continue

                # 检查价格水平数据
                price_levels = minute_data.price_levels
                if not price_levels or not isinstance(price_levels, dict):
                    continue

                valid_data.append(minute_data)

            except Exception as e:
                logger.debug(f"跳过无效数据点: {e}")
                continue

        logger.info(f"数据验证完成: {len(valid_data)}/{len(trades_window_data)} 数据点有效")
        return valid_data

    def _build_volume_profile(self, valid_data: List[Any]) -> Dict[float, float]:
        """构建成交量分布图。

        Args:
            valid_data: 验证后的有效数据

        Returns:
            价格-成交量字典
        """
        volume_profile: defaultdict[float, float] = defaultdict(float)

        total_processed = 0
        for minute_data in valid_data:
            try:
                price_levels = minute_data.price_levels

                for price_key, level_data in price_levels.items():
                    try:
                        # 提取成交量
                        if isinstance(level_data, dict):
                            volume = float(level_data.get("total_volume", 0))
                        else:
                            # 假设是对象格式
                            volume = float(getattr(level_data, "total_volume", 0))

                        if volume > 0:
                            # 对齐价格到聚合精度
                            price = float(price_key)
                            aligned_price = self._align_price_to_precision(price)
                            volume_profile[aligned_price] += volume
                            total_processed += 1

                    except (ValueError, TypeError, AttributeError) as e:
                        logger.debug(f"跳过无效价格水平数据: {e}")
                        continue

            except Exception as e:
                logger.debug(f"处理分钟数据时出错: {e}")
                continue

        # 过滤低于阈值的成交量
        filtered_profile = {
            price: volume
            for price, volume in volume_profile.items()
            if volume >= self.min_volume_threshold
        }

        logger.info(
            f"成交量分布构建完成: 处理了{total_processed}个数据点, "
            f"生成{len(filtered_profile)}个有效价格水平"
        )

        return filtered_profile

    def _align_price_to_precision(self, price: float) -> float:
        """将价格对齐到聚合精度。

        Args:
            price: 原始价格

        Returns:
            对齐后的价格
        """
        # 向下对齐到聚合精度
        aligned = (price // float(self.aggregation_precision)) * float(self.aggregation_precision)
        return aligned

    def get_market_summary(self, aggregated_data: AggregatedTradesData) -> Dict[str, Any]:
        """生成市场数据摘要。

        Args:
            aggregated_data: 聚合后的交易数据

        Returns:
            市场摘要字典
        """
        price_levels = aggregated_data.price_levels

        if not price_levels:
            return {"error": "没有价格水平数据"}

        # 计算统计信息
        volumes = list(price_levels.values())
        total_volume = sum(volumes)
        max_volume = max(volumes)
        min_volume = min(volumes)
        avg_volume = total_volume / len(volumes)

        # 找到成交量最大的价格（POC - Point of Control）
        poc_price = max(price_levels.items(), key=lambda x: x[1])

        # 计算成交量分布
        sorted_volumes = sorted(volumes, reverse=True)
        top_10_percent_volume = sum(sorted_volumes[: max(1, len(sorted_volumes) // 10)])

        # 价格范围分析
        price_range = aggregated_data.price_range
        price_spread = price_range[1] - price_range[0]

        return {
            "analysis_timestamp": aggregated_data.timestamp.isoformat(),
            "symbol": aggregated_data.symbol,
            "data_summary": {
                "total_volume": total_volume,
                "trade_count": aggregated_data.trade_count,
                "price_levels_count": len(price_levels),
                "price_range": price_range,
                "price_spread": price_spread
            },
            "volume_analysis": {
                "max_volume": max_volume,
                "min_volume": min_volume,
                "avg_volume": avg_volume,
                "volume_concentration": max_volume / avg_volume if avg_volume > 0 else 0,
                "top_10_percent_volume_ratio": top_10_percent_volume / total_volume if total_volume > 0 else 0
            },
            "poc_analysis": {
                "poc_price": poc_price[0],
                "poc_volume": poc_price[1],
                "poc_volume_percentage": poc_price[1] / total_volume if total_volume > 0 else 0
            }
        }