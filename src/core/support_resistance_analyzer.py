#!/usr/bin/env python3
"""支撑阻力位分析器。

基于成交量集中度和价格行为识别关键支撑和阻力位。
"""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import List, Dict, Tuple, Optional
import logging

from src.core.models import MinuteTradeData

logger = logging.getLogger(__name__)


@dataclass
class PriceLevel:
    """价格级别数据。"""
    price: Decimal
    total_volume: Decimal
    buy_volume: Decimal
    sell_volume: Decimal
    trade_count: int
    buy_ratio: float
    sell_ratio: float
    volume_imbalance: float
    delta: float  # 买入量 - 卖出量


@dataclass
class SupportResistanceLevel:
    """支撑阻力位。"""
    price: Decimal
    level_type: str  # 'support' or 'resistance'
    strength: float  # 0.0 - 1.0
    volume_concentration: float  # 成交量集中度
    touch_count: int  # 被触及次数
    total_volume: Decimal
    avg_volume_per_touch: Decimal
    formation_time: datetime
    last_updated: datetime
    price_range_start: Decimal  # 支撑/阻力位的价格区间起点
    price_range_end: Decimal    # 支撑/阻力位的价格区间终点


@dataclass
class VolumePriceAnalysis:
    """量价分析结果。"""
    current_price: Decimal
    support_levels: List[SupportResistanceLevel]
    resistance_levels: List[SupportResistanceLevel]
    nearest_support: Optional[SupportResistanceLevel]
    nearest_resistance: Optional[SupportResistanceLevel]
    price_position: str  # 'above_support', 'below_resistance', 'in_range'
    volume_profile: Dict[str, float]  # 价格成交量分布
    trend_direction: str  # 'bullish', 'bearish', 'neutral'
    volume_momentum: float  # 成交量动量


class SupportResistanceAnalyzer:
    """支撑阻力位分析器。"""

    def __init__(
        self,
        min_touches: int = 3,  # 最少触及次数
        price_tolerance_percent: float = 0.1,  # 价格容差百分比
        volume_threshold: float = 0.5,  # 成交量阈值
        window_minutes: int = 15  # 分析窗口
    ):
        """初始化支撑阻力位分析器。

        Args:
            min_touches: 最少触及次数
            price_tolerance_percent: 价格容差百分比
            volume_threshold: 成交量阈值
            window_minutes: 分析窗口分钟数
        """
        self.min_touches = min_touches
        self.price_tolerance_percent = price_tolerance_percent
        self.volume_threshold = volume_threshold
        self.window_minutes = window_minutes

    def aggregate_price_levels(self, minute_data_list: List[MinuteTradeData]) -> List[PriceLevel]:
        """聚合价格级别数据。"""
        price_levels = {}

        for minute_data in minute_data_list:
            for price_str, price_data in minute_data.price_levels.items():
                price = Decimal(price_str)

                # 处理price_data可能是字典或对象的情况
                if isinstance(price_data, dict):
                    buy_volume = Decimal(str(price_data.get("buy_volume", 0)))
                    sell_volume = Decimal(str(price_data.get("sell_volume", 0)))
                    trade_count = price_data.get("trade_count", 0)
                else:
                    buy_volume = price_data.buy_volume
                    sell_volume = price_data.sell_volume
                    trade_count = price_data.trade_count

                total_volume = buy_volume + sell_volume

                if total_volume > 0:
                    if price not in price_levels:
                        price_levels[price] = PriceLevel(
                            price=price,
                            total_volume=Decimal('0'),
                            buy_volume=Decimal('0'),
                            sell_volume=Decimal('0'),
                            trade_count=0,
                            buy_ratio=0.0,
                            sell_ratio=0.0,
                            volume_imbalance=0.0,
                            delta=0.0
                        )

                    level = price_levels[price]
                    level.total_volume += total_volume
                    level.buy_volume += buy_volume
                    level.sell_volume += sell_volume
                    level.trade_count += trade_count

        # 计算衍生指标
        for level in price_levels.values():
            if level.total_volume > 0:
                level.buy_ratio = float(level.buy_volume / level.total_volume)
                level.sell_ratio = float(level.sell_volume / level.total_volume)
                level.volume_imbalance = float((level.buy_volume - level.sell_volume) / level.total_volume)
                level.delta = float(level.buy_volume - level.sell_volume)

        return sorted(price_levels.values(), key=lambda x: x.price)

    def identify_support_resistance_levels(
        self,
        price_levels: List[PriceLevel]
    ) -> Tuple[List[SupportResistanceLevel], List[SupportResistanceLevel]]:
        """识别支撑和阻力位。"""
        support_levels = []
        resistance_levels = []

        if not price_levels:
            return support_levels, resistance_levels

        # 按成交量排序，识别高成交量区域
        volume_sorted_levels = sorted(
            price_levels,
            key=lambda x: float(x.total_volume),
            reverse=True
        )

        # 计算成交量基准
        total_volume = sum(float(level.total_volume) for level in price_levels)
        volume_threshold = total_volume * self.volume_threshold / len(price_levels)

        # 识别潜在的支撑和阻力位
        for i, level in enumerate(volume_sorted_levels):
            if float(level.total_volume) < volume_threshold:
                continue

            # 计算价格范围（考虑相邻价格的成交量）
            price_range_width = self._calculate_price_range(price_levels, level.price)
            price_range_start = level.price - price_range_width / 2
            price_range_end = level.price + price_range_width / 2

            # 判断是支撑位还是阻力位
            level_type, strength = self._classify_level(
                price_levels, level, i, total_volume
            )

            sr_level = SupportResistanceLevel(
                price=level.price,
                level_type=level_type,
                strength=strength,
                volume_concentration=float(level.total_volume) / total_volume,
                touch_count=self._estimate_touch_count(price_levels, level, price_range_width),
                total_volume=level.total_volume,
                avg_volume_per_touch=level.total_volume / max(1, self._estimate_touch_count(price_levels, level, price_range_width)),
                formation_time=datetime.now(),
                last_updated=datetime.now(),
                price_range_start=price_range_start,
                price_range_end=price_range_end
            )

            if level_type == 'support':
                support_levels.append(sr_level)
            else:
                resistance_levels.append(sr_level)

        # 按强度排序
        support_levels.sort(key=lambda x: x.strength, reverse=True)
        resistance_levels.sort(key=lambda x: x.strength, reverse=True)

        return support_levels, resistance_levels

    def _calculate_price_range(self, price_levels: List[PriceLevel], target_price: Decimal) -> Decimal:
        """计算价格级别的影响范围。"""
        price_tolerance = float(target_price) * self.price_tolerance_percent / 100

        # 找到邻近的价格级别
        nearby_levels = []
        for level in price_levels:
            if abs(float(level.price - target_price)) <= price_tolerance:
                nearby_levels.append(level)

        if len(nearby_levels) <= 1:
            return Decimal(str(price_tolerance))

        # 计算价格范围
        min_price = min(level.price for level in nearby_levels)
        max_price = max(level.price for level in nearby_levels)
        return max_price - min_price

    def _classify_level(
        self,
        price_levels: List[PriceLevel],
        target_level: PriceLevel,
        index: int,
        total_volume: float
    ) -> Tuple[str, float]:
        """分类价格级别为支撑位或阻力位。"""

        # 基于买卖压力判断
        if target_level.volume_imbalance > 0.3:  # 买入压力明显
            return 'support', min(1.0, target_level.volume_imbalance + 0.3)
        elif target_level.volume_imbalance < -0.3:  # 卖出压力明显
            return 'resistance', min(1.0, abs(target_level.volume_imbalance) + 0.3)

        # 基于位置判断（简单的高低点识别）
        if index < len(price_levels) / 3:  # 价格较低位置，可能是支撑
            return 'support', 0.5
        elif index > 2 * len(price_levels) / 3:  # 价格较高位置，可能是阻力
            return 'resistance', 0.5
        else:
            return 'support', 0.4  # 中性区域，偏向支撑

    def _estimate_touch_count(
        self,
        price_levels: List[PriceLevel],
        target_level: PriceLevel,
        price_range: Decimal
    ) -> int:
        """估算触及次数。"""
        touch_count = 0
        price_tolerance = float(price_range) / 2

        for level in price_levels:
            if abs(float(level.price - target_level.price)) <= price_tolerance:
                touch_count += level.trade_count

        return max(self.min_touches, touch_count // 10)  # 估算触及次数

    def analyze_volume_price_relationship(
        self,
        minute_data_list: List[MinuteTradeData],
        current_price: Decimal
    ) -> VolumePriceAnalysis:
        """分析量价关系。"""

        # 聚合价格级别
        price_levels = self.aggregate_price_levels(minute_data_list)

        if not price_levels:
            return VolumePriceAnalysis(
                current_price=current_price,
                support_levels=[],
                resistance_levels=[],
                nearest_support=None,
                nearest_resistance=None,
                price_position='unknown',
                volume_profile={},
                trend_direction='neutral',
                volume_momentum=0.0
            )

        # 识别支撑阻力位
        support_levels, resistance_levels = self.identify_support_resistance_levels(price_levels)

        # 找到最近的支撑和阻力位
        nearest_support = self._find_nearest_level(support_levels, current_price, 'support')
        nearest_resistance = self._find_nearest_level(resistance_levels, current_price, 'resistance')

        # 判断价格位置
        price_position = self._determine_price_position(
            current_price, nearest_support, nearest_resistance
        )

        # 构建成交量分布
        volume_profile = self._build_volume_profile(price_levels)

        # 判断趋势方向
        trend_direction = self._determine_trend_direction(minute_data_list)

        # 计算成交量动量
        volume_momentum = self._calculate_volume_momentum(minute_data_list)

        return VolumePriceAnalysis(
            current_price=current_price,
            support_levels=support_levels,
            resistance_levels=resistance_levels,
            nearest_support=nearest_support,
            nearest_resistance=nearest_resistance,
            price_position=price_position,
            volume_profile=volume_profile,
            trend_direction=trend_direction,
            volume_momentum=volume_momentum
        )

    def _find_nearest_level(
        self,
        levels: List[SupportResistanceLevel],
        current_price: Decimal,
        level_type: str
    ) -> Optional[SupportResistanceLevel]:
        """找到最近的支撑或阻力位。"""
        if not levels:
            return None

        # 找到最近且有效的级别
        valid_levels = [level for level in levels if level.strength > 0.3]
        if not valid_levels:
            return None

        nearest = min(
            valid_levels,
            key=lambda x: abs(float(x.price - current_price))
        )
        return nearest

    def _determine_price_position(
        self,
        current_price: Decimal,
        nearest_support: Optional[SupportResistanceLevel],
        nearest_resistance: Optional[SupportResistanceLevel]
    ) -> str:
        """判断价格相对于支撑阻力位的位置。"""

        if nearest_support and nearest_resistance:
            support_distance = float(current_price - nearest_support.price)
            resistance_distance = float(nearest_resistance.price - current_price)

            if support_distance > 0 and resistance_distance > 0:
                # 在支撑和阻力之间
                if support_distance < resistance_distance:
                    return 'near_support'
                else:
                    return 'near_resistance'
            elif support_distance > 0:
                return 'above_support'
            elif resistance_distance > 0:
                return 'below_resistance'

        return 'unknown'

    def _build_volume_profile(self, price_levels: List[PriceLevel]) -> Dict[str, float]:
        """构建成交量分布。"""
        profile = {}
        for level in price_levels:
            profile[str(level.price)] = float(level.total_volume)
        return profile

    def _determine_trend_direction(self, minute_data_list: List[MinuteTradeData]) -> str:
        """判断趋势方向。"""
        if len(minute_data_list) < 2:
            return 'neutral'

        # 比较首尾的VWAP
        first_vwap = self._calculate_vwap(minute_data_list[0])
        last_vwap = self._calculate_vwap(minute_data_list[-1])

        if last_vwap > first_vwap * Decimal('1.001'):  # 0.1%以上涨幅
            return 'bullish'
        elif last_vwap < first_vwap * Decimal('0.999'):  # 0.1%以上跌幅
            return 'bearish'
        else:
            return 'neutral'

    def _calculate_vwap(self, minute_data: MinuteTradeData) -> Decimal:
        """计算VWAP。"""
        total_value = Decimal('0')
        total_volume = Decimal('0')

        for price_str, price_data in minute_data.price_levels.items():
            price = Decimal(price_str)

            if isinstance(price_data, dict):
                buy_volume = Decimal(str(price_data.get("buy_volume", 0)))
                sell_volume = Decimal(str(price_data.get("sell_volume", 0)))
            else:
                buy_volume = price_data.buy_volume
                sell_volume = price_data.sell_volume

            volume = buy_volume + sell_volume
            total_value += price * volume
            total_volume += volume

        return total_value / total_volume if total_volume > 0 else Decimal('0')

    def _calculate_volume_momentum(self, minute_data_list: List[MinuteTradeData]) -> float:
        """计算成交量动量。"""
        if len(minute_data_list) < 2:
            return 0.0

        # 比较最近几个分钟的成交量变化
        recent_volumes = []
        for minute_data in minute_data_list[-3:]:  # 最近3分钟
            minute_volume = 0.0
            for price_data in minute_data.price_levels.values():
                if isinstance(price_data, dict):
                    total_volume = float(price_data.get("total_volume", 0))
                else:
                    total_volume = float(price_data.total_volume)
                minute_volume += total_volume
            recent_volumes.append(minute_volume)

        if len(recent_volumes) < 2:
            return 0.0

        # 计算成交量变化率
        current_volume = recent_volumes[-1]
        avg_volume = sum(recent_volumes[:-1]) / len(recent_volumes[:-1]) if recent_volumes[:-1] else 0.0

        if avg_volume > 0:
            return (current_volume - avg_volume) / avg_volume
        else:
            return 0.0