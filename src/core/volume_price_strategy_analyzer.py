#!/usr/bin/env python3
"""量价结合策略分析器。

结合支撑阻力位识别和成交量确认来生成稳定的交易信号。
"""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import List, Dict, Tuple, Optional
import logging

from src.core.models import MinuteTradeData
from src.core.support_resistance_analyzer import (
    SupportResistanceAnalyzer,
    VolumePriceAnalysis,
    SupportResistanceLevel
)

# 定义动量相关枚举
class MomentumDirection:
    BUY = "buy"
    SELL = "sell"
    NEUTRAL = "neutral"

logger = logging.getLogger(__name__)


@dataclass
class VolumePriceSignal:
    """量价结合信号。"""
    timestamp: datetime
    symbol: str
    direction: MomentumDirection
    strength: float  # 0.0 - 1.0
    confidence: float  # 0.0 - 1.0
    signal_type: str  # 'support_bounce', 'resistance_break', 'volume_breakout'
    entry_price: Decimal
    stop_loss: Decimal
    take_profit: Decimal
    risk_reward_ratio: float
    support_level: Optional[SupportResistanceLevel]
    resistance_level: Optional[SupportResistanceLevel]
    volume_confirmation: float  # 成交量确认强度
    price_distance_from_level: float  # 价格距离关键位的百分比


class VolumePriceStrategyAnalyzer:
    """量价结合策略分析器。

    基于支撑阻力位和成交量确认来生成更稳定的交易信号。
    """

    def __init__(
        self,
        window_minutes: int = 15,  # 分析窗口
        min_volume_ratio: float = 1.5,  # 最小成交量放大倍数
        max_distance_from_level_percent: float = 0.2,  # 距离关键位最大百分比
        min_strength_threshold: float = 0.6,  # 最小信号强度阈值
        risk_reward_ratio_min: float = 1.5  # 最小风险收益比
    ):
        """初始化量价策略分析器。

        Args:
            window_minutes: 分析窗口分钟数
            min_volume_ratio: 最小成交量放大倍数
            max_distance_from_level_percent: 距离关键位最大百分比
            min_strength_threshold: 最小信号强度阈值
            risk_reward_ratio_min: 最小风险收益比
        """
        self.window_minutes = window_minutes
        self.min_volume_ratio = min_volume_ratio
        self.max_distance_from_level_percent = max_distance_from_level_percent
        self.min_strength_threshold = min_strength_threshold
        self.risk_reward_ratio_min = risk_reward_ratio_min

        # 初始化支撑阻力分析器
        self.support_resistance_analyzer = SupportResistanceAnalyzer(
            min_touches=2,
            price_tolerance_percent=0.1,
            volume_threshold=0.3,
            window_minutes=window_minutes
        )

    def analyze_volume_price_strategy(
        self,
        minute_data_list: List[MinuteTradeData],
        symbol: str
    ) -> Optional[VolumePriceSignal]:
        """分析量价结合策略。

        Args:
            minute_data_list: 分钟级交易数据列表
            symbol: 交易对符号

        Returns:
            量价结合信号，如果没有符合条件的信号则返回None
        """
        if len(minute_data_list) < 5:
            logger.warning(f"数据点不足，无法进行量价分析: {len(minute_data_list)} < 5")
            return None

        try:
            # 计算当前价格
            current_price = self._calculate_current_price(minute_data_list[-1])
            if current_price == 0:
                return None

            # 执行支撑阻力分析
            volume_analysis = self.support_resistance_analyzer.analyze_volume_price_relationship(
                minute_data_list, current_price
            )

            # 分析成交量特征
            volume_analysis_result = self._analyze_volume_characteristics(minute_data_list)

            # 生成交易信号
            signal = self._generate_volume_price_signal(
                minute_data_list,
                volume_analysis,
                volume_analysis_result,
                symbol,
                current_price
            )

            return signal

        except Exception as e:
            logger.error(f"量价分析失败: {e}")
            return None

    def _calculate_current_price(self, latest_data: MinuteTradeData) -> Decimal:
        """计算当前价格（使用VWAP）。"""
        total_value = Decimal('0')
        total_volume = Decimal('0')

        for price_str, price_data in latest_data.price_levels.items():
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

    def _analyze_volume_characteristics(self, minute_data_list: List[MinuteTradeData]) -> Dict[str, float]:
        """分析成交量特征。"""
        if len(minute_data_list) < 3:
            return {
                'volume_ratio': 1.0,
                'volume_trend': 0.0,
                'volume_spike': False
            }

        # 计算每个分钟的总成交量
        minute_volumes = []
        for data in minute_data_list:
            minute_volume = Decimal('0')
            for price_data in data.price_levels.values():
                if isinstance(price_data, dict):
                    buy_volume = Decimal(str(price_data.get("buy_volume", 0)))
                    sell_volume = Decimal(str(price_data.get("sell_volume", 0)))
                else:
                    buy_volume = price_data.buy_volume
                    sell_volume = price_data.sell_volume
                minute_volume += buy_volume + sell_volume
            minute_volumes.append(float(minute_volume))

        # 计算成交量比率（当前vs平均）
        if len(minute_volumes) > 0:
            current_volume = minute_volumes[-1]
            avg_volume = sum(minute_volumes[:-1]) / len(minute_volumes[:-1]) if len(minute_volumes) > 1 else current_volume
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
        else:
            volume_ratio = 1.0

        # 计算成交量趋势
        volume_trend = 0.0
        if len(minute_volumes) >= 3:
            recent_avg = sum(minute_volumes[-3:]) / 3
            earlier_avg = sum(minute_volumes[:-3]) / len(minute_volumes[:-3]) if len(minute_volumes) > 3 else recent_avg
            volume_trend = (recent_avg - earlier_avg) / earlier_avg if earlier_avg > 0 else 0.0

        # 检测成交量激增
        volume_spike = volume_ratio > self.min_volume_ratio

        return {
            'volume_ratio': volume_ratio,
            'volume_trend': volume_trend,
            'volume_spike': volume_spike
        }

    def _generate_volume_price_signal(
        self,
        minute_data_list: List[MinuteTradeData],
        volume_analysis: VolumePriceAnalysis,
        volume_characteristics: Dict[str, float],
        symbol: str,
        current_price: Decimal
    ) -> Optional[VolumePriceSignal]:
        """生成量价结合信号。"""

        # 检查是否在支撑位附近且有支撑反弹迹象
        support_signal = self._check_support_bounce_signal(
            volume_analysis, volume_characteristics, current_price
        )

        if support_signal:
            return support_signal

        # 检查是否在阻力位附近且有突破迹象
        resistance_signal = self._check_resistance_breakout_signal(
            volume_analysis, volume_characteristics, current_price
        )

        if resistance_signal:
            return resistance_signal

        # 检查成交量突破信号
        breakout_signal = self._check_volume_breakout_signal(
            volume_analysis, volume_characteristics, current_price
        )

        return breakout_signal

    def _check_support_bounce_signal(
        self,
        volume_analysis: VolumePriceAnalysis,
        volume_characteristics: Dict[str, float],
        current_price: Decimal
    ) -> Optional[VolumePriceSignal]:
        """检查支撑位反弹信号。"""
        if not volume_analysis.nearest_support:
            return None

        support = volume_analysis.nearest_support
        distance_from_support = float(current_price - support.price) / float(support.price)

        # 价格在支撑位附近（0.2%以内）
        if distance_from_support < 0 and abs(distance_from_support) <= self.max_distance_from_level_percent / 100:
            # 检查是否有买入压力和成交量确认
            if (volume_characteristics['volume_spike'] and
                volume_characteristics['volume_trend'] > 0 and
                support.strength >= self.min_strength_threshold):

                strength = min(1.0, support.strength * volume_characteristics['volume_ratio'] / self.min_volume_ratio)
                confidence = min(1.0, strength * (support.volume_concentration * 2))

                # 计算止损和止盈位
                stop_loss = support.price * Decimal('0.998')  # 支撑位下方0.2%
                take_profit = current_price * Decimal('1.002')  # 当前价格上方0.2%
                risk_reward_ratio = float(take_profit - current_price) / float(current_price - stop_loss)

                if risk_reward_ratio >= self.risk_reward_ratio_min:
                    return VolumePriceSignal(
                        timestamp=datetime.now(),
                        symbol="BTCFDUSD",
                        direction=MomentumDirection.BUY,
                        strength=strength,
                        confidence=confidence,
                        signal_type="support_bounce",
                        entry_price=current_price,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        risk_reward_ratio=risk_reward_ratio,
                        support_level=support,
                        resistance_level=None,
                        volume_confirmation=volume_characteristics['volume_ratio'],
                        price_distance_from_level=abs(distance_from_support)
                    )

        return None

    def _check_resistance_breakout_signal(
        self,
        volume_analysis: VolumePriceAnalysis,
        volume_characteristics: Dict[str, float],
        current_price: Decimal
    ) -> Optional[VolumePriceSignal]:
        """检查阻力位突破信号。"""
        if not volume_analysis.nearest_resistance:
            return None

        resistance = volume_analysis.nearest_resistance
        distance_to_resistance = float(resistance.price - current_price) / float(resistance.price)

        # 价格接近阻力位（0.2%以内）或刚刚突破
        if (0 <= distance_to_resistance <= self.max_distance_from_level_percent / 100 or
            distance_to_resistance < 0 and abs(distance_to_resistance) <= self.max_distance_from_level_percent / 100):

            # 检查是否有强劲的成交量支持突破
            if (volume_characteristics['volume_spike'] and
                volume_characteristics['volume_ratio'] >= self.min_volume_ratio * 1.5 and
                volume_analysis.trend_direction == 'bullish'):

                strength = min(1.0, resistance.strength * volume_characteristics['volume_ratio'] / (self.min_volume_ratio * 1.5))
                confidence = min(1.0, strength * resistance.volume_concentration * 1.5)

                # 计算止损和止盈位
                stop_loss = resistance.price * Decimal('0.998')  # 阻力位下方0.2%
                take_profit = current_price * Decimal('1.003')  # 当前价格上方0.3%
                risk_reward_ratio = float(take_profit - current_price) / float(current_price - stop_loss)

                if risk_reward_ratio >= self.risk_reward_ratio_min:
                    return VolumePriceSignal(
                        timestamp=datetime.now(),
                        symbol="BTCFDUSD",
                        direction=MomentumDirection.BUY,
                        strength=strength,
                        confidence=confidence,
                        signal_type="resistance_breakout",
                        entry_price=current_price,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        risk_reward_ratio=risk_reward_ratio,
                        support_level=None,
                        resistance_level=resistance,
                        volume_confirmation=volume_characteristics['volume_ratio'],
                        price_distance_from_level=abs(distance_to_resistance)
                    )

        return None

    def _check_volume_breakout_signal(
        self,
        volume_analysis: VolumePriceAnalysis,
        volume_characteristics: Dict[str, float],
        current_price: Decimal
    ) -> Optional[VolumePriceSignal]:
        """检查成交量突破信号。"""
        # 需要非常强劲的成交量激增
        if (volume_characteristics['volume_ratio'] >= self.min_volume_ratio * 2 and
            volume_characteristics['volume_trend'] > 0.5):

            # 根据趋势方向决定信号
            if volume_analysis.trend_direction == 'bullish':
                strength = min(1.0, volume_characteristics['volume_ratio'] / (self.min_volume_ratio * 2))
                confidence = strength * 0.7  # 成交量突破的置信度稍低

                # 计算止损和止盈位
                stop_loss = current_price * Decimal('0.997')  # 下方0.3%
                take_profit = current_price * Decimal('1.004')  # 上方0.4%
                risk_reward_ratio = float(take_profit - current_price) / float(current_price - stop_loss)

                if risk_reward_ratio >= self.risk_reward_ratio_min:
                    return VolumePriceSignal(
                        timestamp=datetime.now(),
                        symbol="BTCFDUSD",
                        direction=MomentumDirection.BUY,
                        strength=strength,
                        confidence=confidence,
                        signal_type="volume_breakout",
                        entry_price=current_price,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        risk_reward_ratio=risk_reward_ratio,
                        support_level=None,
                        resistance_level=None,
                        volume_confirmation=volume_characteristics['volume_ratio'],
                        price_distance_from_level=0.0
                    )

        return None

    def convert_to_momentum_signal(self, volume_price_signal: VolumePriceSignal) -> dict:
        """将量价信号转换为动量信号格式。"""
        # 构建指标数据
        indicators = {
            "price_momentum": 0.0,
            "volume_momentum": volume_price_signal.volume_confirmation,
            "order_flow_momentum": volume_price_signal.strength,
            "volatility_adjusted": 0.0,
            "volume_imbalance": 0.0,
            "trend_strength": volume_price_signal.strength,
            "buy_pressure": volume_price_signal.strength if volume_price_signal.direction == MomentumDirection.BUY else 0.0,
            "sell_pressure": volume_price_signal.strength if volume_price_signal.direction == MomentumDirection.SELL else 0.0,
            "flow_consistency": volume_price_signal.confidence,
            "large_trade_ratio": 0.0,
            "volume_trend": volume_price_signal.volume_confirmation,
            "realized_volatility": 0.0,
            "risk_adjusted_return": volume_price_signal.risk_reward_ratio,
            "price_change_rate": 0.0,
            "weighted_price_momentum": 0.0
        }

        return {
            "timestamp": volume_price_signal.timestamp,
            "symbol": volume_price_signal.symbol,
            "direction": volume_price_signal.direction,
            "strength": volume_price_signal.strength,
            "confidence": volume_price_signal.confidence,
            "raw_score": volume_price_signal.strength * volume_price_signal.confidence,
            "indicators": indicators,
            "timeframe": f"{self.window_minutes}m",
            "analysis_window_minutes": self.window_minutes,
            "trade_count": 0,  # 量价策略不基于具体交易数量
            "signal_quality_score": volume_price_signal.confidence,
            "market_condition": "trending"
        }