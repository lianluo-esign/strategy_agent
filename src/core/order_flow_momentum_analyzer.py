#!/usr/bin/env python3
"""基于订单流数据的动量分析器。

直接使用MinuteTradeData进行动量分析，避免数据转换损失。
"""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any
import logging

from src.core.models import MinuteTradeData, PriceLevelData
from src.core.momentum_models import (
    MomentumDirection,
    MomentumIndicators,
    MomentumSignal,
    MomentumAnalysisResult,
    TradeWindow
)


@dataclass
class OrderFlowMetrics:
    """订单流指标。"""
    total_buy_volume: Decimal
    total_sell_volume: Decimal
    volume_delta: Decimal
    volume_imbalance: Decimal
    weighted_buy_price: Decimal
    weighted_sell_price: Decimal
    vwap: Decimal
    price_levels_count: int
    dominant_side: str
    concentration_ratio: Decimal  # 成交量集中度比率


class OrderFlowMomentumAnalyzer:
    """基于订单流的动量分析器。"""

    def __init__(
        self,
        window_size_minutes: int = 5,
        buy_threshold: float = 0.15,
        sell_threshold: float = -0.15,
        neutral_range: float = 0.05,
    ):
        """初始化订单流动量分析器。

        Args:
            window_size_minutes: 分析时间窗口（分钟）
            buy_threshold: 买入信号阈值
            sell_threshold: 卖出信号阈值
            neutral_range: 中性区间范围
        """
        self.window_size_minutes = window_size_minutes
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.neutral_range = neutral_range
        self.logger = logging.getLogger(__name__)

    def calculate_order_flow_metrics(self, minute_data: MinuteTradeData) -> OrderFlowMetrics:
        """计算单个分钟数据的订单流指标。

        Args:
            minute_data: 分钟交易数据

        Returns:
            订单流指标
        """
        total_buy_volume = Decimal('0')
        total_sell_volume = Decimal('0')
        total_volume = Decimal('0')
        weighted_buy_value = Decimal('0')
        weighted_sell_value = Decimal('0')
        total_value = Decimal('0')

        # 统计每个价位的交易数据
        price_volumes = []
        max_volume = Decimal('0')

        for price_level_str, price_data in minute_data.price_levels.items():
            # 处理price_data可能是字典或对象的情况
            if isinstance(price_data, dict):
                buy_volume = Decimal(str(price_data.get("buy_volume", 0)))
                sell_volume = Decimal(str(price_data.get("sell_volume", 0)))
                trade_count = price_data.get("trade_count", 0)
            else:
                # 如果是PriceLevelData对象
                buy_volume = price_data.buy_volume
                sell_volume = price_data.sell_volume
                trade_count = price_data.trade_count

            price = Decimal(price_level_str)
            level_volume = buy_volume + sell_volume

            if level_volume > 0:
                total_buy_volume += buy_volume
                total_sell_volume += sell_volume
                total_volume += level_volume

                weighted_buy_value += buy_volume * price
                weighted_sell_value += sell_volume * price
                total_value += level_volume * price

                price_volumes.append((price, level_volume))
                max_volume = max(max_volume, level_volume)

        # 计算衍生指标
        volume_delta = total_buy_volume - total_sell_volume
        volume_imbalance = volume_delta / total_volume if total_volume > 0 else Decimal('0')

        # 计算VWAP
        vwap = total_value / total_volume if total_volume > 0 else Decimal('0')
        weighted_buy_price = weighted_buy_value / total_buy_volume if total_buy_volume > 0 else Decimal('0')
        weighted_sell_price = weighted_sell_value / total_sell_volume if total_sell_volume > 0 else Decimal('0')

        # 确定主导方
        dominant_side = "buy" if total_buy_volume > total_sell_volume else "sell" if total_sell_volume > total_buy_volume else "neutral"

        # 计算成交量集中度（前3个价位占总成交量的比例）
        price_volumes.sort(key=lambda x: x[1], reverse=True)
        top3_volume = sum(vol for _, vol in price_volumes[:3])
        concentration_ratio = top3_volume / total_volume if total_volume > 0 else Decimal('0')

        return OrderFlowMetrics(
            total_buy_volume=total_buy_volume,
            total_sell_volume=total_sell_volume,
            volume_delta=volume_delta,
            volume_imbalance=volume_imbalance,
            weighted_buy_price=weighted_buy_price,
            weighted_sell_price=weighted_sell_price,
            vwap=vwap,
            price_levels_count=len(price_volumes),
            dominant_side=dominant_side,
            concentration_ratio=concentration_ratio
        )

    def calculate_momentum_indicators(self, metrics_list: list[OrderFlowMetrics]) -> MomentumIndicators:
        """基于订单流指标计算动量指标。

        Args:
            metrics_list: 多个时间点的订单流指标列表

        Returns:
            动量指标
        """
        if len(metrics_list) < 2:
            # 数据不足时返回中性指标
            return MomentumIndicators(
                price_momentum=0.0,
                price_change_rate=0.0,
                trend_strength=0.0,
                weighted_price_momentum=0.0,
                volume_momentum=0.0,
                volume_imbalance=0.0,
                large_trade_ratio=0.0,
                volume_trend=0.0,
                order_flow_momentum=0.0,
                buy_pressure=0.0,
                sell_pressure=0.0,
                flow_consistency=0.0,
                volatility_adjusted=0.0,
                realized_volatility=0.0,
                risk_adjusted_return=0.0
            )

        # 计算VWAP动量
        vwap_values = [float(m.vwap) for m in metrics_list if m.vwap > 0]
        if len(vwap_values) >= 2:
            price_momentum = (vwap_values[-1] - vwap_values[-2]) / vwap_values[-2]
            price_change_rate = price_momentum
        else:
            price_momentum = 0.0
            price_change_rate = 0.0

        # 计算成交量动量（基于不平衡度）
        volume_imbalances = [float(m.volume_imbalance) for m in metrics_list]
        current_volume_imbalance = volume_imbalances[-1] if volume_imbalances else 0.0

        # 计算成交量动量的变化
        if len(volume_imbalances) >= 2:
            volume_momentum = current_volume_imbalance - volume_imbalances[-2]
        else:
            volume_momentum = 0.0

        # 计算订单流动量（基于买卖压力变化）
        buy_pressures = [float(m.total_buy_volume) for m in metrics_list]
        sell_pressures = [float(m.total_sell_volume) for m in metrics_list]

        total_volume = sum(buy_pressures[-1] + sell_pressures[-1] for _ in [0] if buy_pressures and sell_pressures)
        if total_volume > 0:
            buy_pressure = buy_pressures[-1] / total_volume
            sell_pressure = sell_pressures[-1] / total_volume
        else:
            buy_pressure = 0.5
            sell_pressure = 0.5

        # 订单流动量 = 买卖压力差的变化率
        if len(buy_pressures) >= 2 and len(sell_pressures) >= 2:
            current_pressure_diff = buy_pressures[-1] - sell_pressures[-1]
            prev_pressure_diff = buy_pressures[-2] - sell_pressures[-2]
            order_flow_momentum = (current_pressure_diff - prev_pressure_diff) / (abs(prev_pressure_diff) + 1)
        else:
            order_flow_momentum = 0.0

        # 计算成交量集中度动量
        concentrations = [float(m.concentration_ratio) for m in metrics_list]
        large_trade_ratio = concentrations[-1] if concentrations else 0.0

        # 计算趋势强度（基于主导方的连续性）
        dominant_sides = [m.dominant_side for m in metrics_list]
        buy_count = sum(1 for side in dominant_sides if side == "buy")
        sell_count = sum(1 for side in dominant_sides if side == "sell")

        if len(dominant_sides) > 0:
            trend_strength = max(buy_count, sell_count) / len(dominant_sides)
        else:
            trend_strength = 0.0

        # 计算成交量趋势
        total_volumes = [float(m.total_buy_volume + m.total_sell_volume) for m in metrics_list]
        if len(total_volumes) >= 2:
            volume_trend = (total_volumes[-1] - total_volumes[-2]) / total_volumes[-2] if total_volumes[-2] > 0 else 0.0
        else:
            volume_trend = 0.0

        # 计算一致性（买卖压力的一致性）
        if len(volume_imbalances) >= 3:
            consistent_directions = sum(1 for i in range(1, len(volume_imbalances))
                                      if (volume_imbalances[i] > 0 and volume_imbalances[i-1] > 0) or
                                         (volume_imbalances[i] < 0 and volume_imbalances[i-1] < 0))
            flow_consistency = consistent_directions / (len(volume_imbalances) - 1)
        else:
            flow_consistency = 0.5

        # 计算波动率
        if len(vwap_values) >= 3:
            returns = [(vwap_values[i] - vwap_values[i-1]) / vwap_values[i-1] for i in range(1, len(vwap_values))]
            avg_return = sum(returns) / len(returns)
            variance = sum((r - avg_return) ** 2 for r in returns) / len(returns)
            realized_volatility = variance ** 0.5
        else:
            realized_volatility = 0.0

        # 波动率调整后的动量
        volatility_adjusted = price_momentum / (realized_volatility + 0.0001)

        # 风险调整收益
        risk_adjusted_return = price_momentum / (realized_volatility + 0.0001) if realized_volatility > 0 else 0.0

        return MomentumIndicators(
            price_momentum=price_momentum,
            price_change_rate=price_change_rate,
            trend_strength=trend_strength,
            weighted_price_momentum=price_momentum,  # VWAP已经是加权价格
            volume_momentum=volume_momentum,
            volume_imbalance=current_volume_imbalance,
            large_trade_ratio=large_trade_ratio,
            volume_trend=volume_trend,
            order_flow_momentum=order_flow_momentum,
            buy_pressure=buy_pressure,
            sell_pressure=sell_pressure,
            flow_consistency=flow_consistency,
            volatility_adjusted=volatility_adjusted,
            realized_volatility=realized_volatility,
            risk_adjusted_return=risk_adjusted_return
        )

    def generate_momentum_signal(
        self,
        indicators: MomentumIndicators,
        symbol: str,
        timestamp: datetime,
        trade_count: int
    ) -> MomentumSignal:
        """基于订单流指标生成动量信号。

        Args:
            indicators: 动量指标
            symbol: 交易对
            timestamp: 时间戳
            trade_count: 交易数量

        Returns:
            动量信号
        """
        # 综合评分计算（优化权重配置）
        weights = {
            'order_flow_momentum': 0.3,      # 订单流动量权重最高
            'volume_imbalance': 0.2,         # 成交量不平衡
            'price_momentum': 0.15,          # 价格动量
            'trend_strength': 0.1,           # 趋势强度
            'flow_consistency': 0.1,         # 一致性
            'volume_trend': 0.05,            # 成交量趋势
            'volatility_adjusted': 0.1       # 波动率调整
        }

        # 计算加权分数
        raw_score = (
            indicators.order_flow_momentum * weights['order_flow_momentum'] +
            indicators.volume_imbalance * weights['volume_imbalance'] +
            indicators.price_momentum * weights['price_momentum'] +
            (indicators.trend_strength - 0.5) * weights['trend_strength'] * 2 +  # 转换到[-1,1]
            (indicators.flow_consistency - 0.5) * weights['flow_consistency'] * 2 +  # 转换到[-1,1]
            indicators.volume_trend * weights['volume_trend'] +
            indicators.volatility_adjusted * weights['volatility_adjusted']
        )

        # 标准化到[-1,1]范围
        raw_score = max(-1.0, min(1.0, raw_score))

        # 确定信号方向
        if raw_score > self.neutral_range:
            direction = MomentumDirection.BUY
        elif raw_score < -self.neutral_range:
            direction = MomentumDirection.SELL
        else:
            direction = MomentumDirection.NEUTRAL

        # 计算信号强度（基于分数偏离阈值的程度）
        if direction == MomentumDirection.BUY:
            strength = min(1.0, (raw_score - self.neutral_range) / (self.buy_threshold - self.neutral_range))
        elif direction == MomentumDirection.SELL:
            strength = min(1.0, (abs(raw_score) - self.neutral_range) / (abs(self.sell_threshold) - self.neutral_range))
        else:
            strength = 1.0 - abs(raw_score) / self.neutral_range

        # 计算置信度（基于指标一致性）
        confidence_factors = [
            1.0 - abs(indicators.volume_imbalance),  # 不平衡程度越高置信度越高
            indicators.flow_consistency,              # 一致性
            indicators.trend_strength,                # 趋势强度
            min(1.0, trade_count / 50.0),           # 交易数量充足性
        ]

        confidence = sum(confidence_factors) / len(confidence_factors)
        confidence = max(0.1, min(1.0, confidence))

        return MomentumSignal(
            timestamp=timestamp,
            symbol=symbol,
            direction=direction,
            strength=strength,
            confidence=confidence,
            raw_score=raw_score,
            indicators=indicators,
            timeframe=f"{self.window_size_minutes}m",
            analysis_window_minutes=self.window_size_minutes,
            trade_count=trade_count,
            signal_quality_score=confidence,
            market_condition=self._determine_market_condition(indicators)
        )

    def _determine_market_condition(self, indicators: MomentumIndicators) -> str:
        """判断市场条件。"""
        if indicators.realized_volatility > 0.002:
            return "volatile"
        elif abs(indicators.volume_imbalance) > 0.3:
            return "trending"
        elif indicators.flow_consistency > 0.7:
            return "directional"
        else:
            return "ranging"

    def analyze_order_flow_momentum(
        self,
        minute_data_list: list[MinuteTradeData],
        symbol: str = "BTCFDUSD",
        end_time: datetime | None = None,
    ) -> MomentumAnalysisResult:
        """分析订单流动量。

        Args:
            minute_data_list: 分钟交易数据列表
            symbol: 交易对
            end_time: 分析结束时间

        Returns:
            动量分析结果
        """
        start_time = datetime.now()

        if not minute_data_list:
            raise ValueError("分钟交易数据不能为空")

        # 设置分析时间窗口
        if end_time is None:
            end_time = minute_data_list[-1].timestamp

        # 计算每个时间点的订单流指标
        metrics_list = []
        total_trade_count = 0

        for minute_data in minute_data_list:
            metrics = self.calculate_order_flow_metrics(minute_data)
            metrics_list.append(metrics)

            # 计算总交易数量
            for price_data in minute_data.price_levels.values():
                if isinstance(price_data, dict):
                    total_trade_count += price_data.get("trade_count", 0)
                else:
                    total_trade_count += price_data.trade_count

        # 计算动量指标
        indicators = self.calculate_momentum_indicators(metrics_list)

        # 生成信号
        signal = self.generate_momentum_signal(
            indicators=indicators,
            symbol=symbol,
            timestamp=end_time,
            trade_count=total_trade_count
        )

        # 创建交易窗口摘要（基于订单流数据）
        first_metrics = metrics_list[0] if metrics_list else None
        last_metrics = metrics_list[-1] if metrics_list else None

        if first_metrics and last_metrics and first_metrics.vwap > 0 and last_metrics.vwap > 0:
            price_change = float(last_metrics.vwap - first_metrics.vwap)
            price_change_rate = price_change / float(first_metrics.vwap)
        else:
            price_change = 0.0
            price_change_rate = 0.0

        trade_window_summary = {
            "symbol": symbol,
            "start_time": minute_data_list[0].timestamp,
            "end_time": end_time,
            "total_volume": float(last_metrics.total_buy_volume + last_metrics.total_sell_volume) if last_metrics else 0.0,
            "total_trades": total_trade_count,
            "buy_volume": float(last_metrics.total_buy_volume) if last_metrics else 0.0,
            "sell_volume": float(last_metrics.total_sell_volume) if last_metrics else 0.0,
            "open_price": float(first_metrics.vwap) if first_metrics and first_metrics.vwap > 0 else 0.0,
            "close_price": float(last_metrics.vwap) if last_metrics and last_metrics.vwap > 0 else 0.0,
            "high_price": 0.0,  # 订单流数据中无高低价
            "low_price": 0.0,   # 订单流数据中无高低价
            "vwap": float(last_metrics.vwap) if last_metrics and last_metrics.vwap > 0 else 0.0,
            "price_change": price_change,
            "price_change_rate": price_change_rate,
            "volume_imbalance": float(last_metrics.volume_imbalance) if last_metrics else 0.0
        }

        # 分析统计
        analysis_statistics = {
            "trade_statistics": {
                "total_trades": total_trade_count,
                "total_volume": trade_window_summary["total_volume"],
                "avg_trade_size": trade_window_summary["total_volume"] / total_trade_count if total_trade_count > 0 else 0.0,
                "buy_ratio": float(last_metrics.total_buy_volume / (last_metrics.total_buy_volume + last_metrics.total_sell_volume)) if last_metrics and (last_metrics.total_buy_volume + last_metrics.total_sell_volume) > 0 else 0.5,
                "sell_ratio": float(last_metrics.total_sell_volume / (last_metrics.total_buy_volume + last_metrics.total_sell_volume)) if last_metrics and (last_metrics.total_buy_volume + last_metrics.total_sell_volume) > 0 else 0.5
            },
            "price_statistics": {
                "open_price": trade_window_summary["open_price"],
                "close_price": trade_window_summary["close_price"],
                "high_price": trade_window_summary["high_price"],
                "low_price": trade_window_summary["low_price"],
                "vwap": trade_window_summary["vwap"],
                "price_change": trade_window_summary["price_change"],
                "price_change_rate": trade_window_summary["price_change_rate"]
            },
            "indicator_summary": {
                "strongest_indicator": self._find_strongest_indicator(indicators),
                "overall_momentum": signal.raw_score,
                "volatility_level": "high" if indicators.realized_volatility > 0.002 else "low"
            }
        }

        processing_time = (datetime.now() - start_time).total_seconds() * 1000

        return MomentumAnalysisResult(
            timestamp=end_time,
            symbol=symbol,
            analysis_window_minutes=self.window_size_minutes,
            signal=signal,
            trade_window_summary=trade_window_summary,
            analysis_statistics=analysis_statistics,
            processing_time_ms=processing_time,
            memory_usage_mb=0.0
        )

    def _find_strongest_indicator(self, indicators: MomentumIndicators) -> str:
        """找出最强的指标。"""
        indicator_values = {
            "order_flow": abs(indicators.order_flow_momentum),
            "volume": abs(indicators.volume_imbalance),
            "price": abs(indicators.price_momentum),
            "trend": abs(indicators.trend_strength - 0.5) * 2
        }

        return max(indicator_values, key=indicator_values.get)