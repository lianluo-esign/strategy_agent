"""短期动量分析器。

基于trades_window数据计算短期价格动量方向和强度的核心算法实现。
"""

import logging
import time
from datetime import datetime, timedelta
from decimal import Decimal, DecimalException
from typing import Any

import numpy as np
from scipy import stats

from .momentum_models import (
    MomentumAnalysisResult,
    MomentumDirection,
    MomentumIndicators,
    MomentumSignal,
    TradeWindow,
)

logger = logging.getLogger(__name__)


class ShortTermMomentumAnalyzer:
    """短期动量分析器。

    专注于基于trades_window数据的短期动量分析，提供：
    - 价格动量计算
    - 成交量动量分析
    - 订单流动量评估
    - 波动率调整后的综合信号
    """

    # 默认参数配置
    DEFAULT_WINDOW_SIZE_MINUTES = 5
    DEFAULT_MIN_TRADES = 10
    DEFAULT_MIN_VOLUME = 0.1

    # 信号阈值
    BUY_THRESHOLD = 0.15
    SELL_THRESHOLD = -0.15
    NEUTRAL_RANGE = 0.15

    # 权重配置
    PRICE_MOMENTUM_WEIGHT = 0.30
    VOLUME_MOMENTUM_WEIGHT = 0.30
    ORDER_FLOW_WEIGHT = 0.25
    VOLATILITY_ADJUSTED_WEIGHT = 0.15

    def __init__(
        self,
        window_size_minutes: int = DEFAULT_WINDOW_SIZE_MINUTES,
        min_trades: int = DEFAULT_MIN_TRADES,
        min_volume: float = DEFAULT_MIN_VOLUME,
        buy_threshold: float = BUY_THRESHOLD,
        sell_threshold: float = SELL_THRESHOLD,
        neutral_range: float = NEUTRAL_RANGE,
    ):
        """初始化短期动量分析器。

        Args:
            window_size_minutes: 分析窗口大小（分钟）
            min_trades: 最小交易数量
            min_volume: 最小成交量
            buy_threshold: 买入信号阈值
            sell_threshold: 卖出信号阈值
            neutral_range: 中性区间范围
        """
        logger.info(f"Initializing ShortTermMomentumAnalyzer with {window_size_minutes}min window")

        self.window_size_minutes = window_size_minutes
        self.min_trades = min_trades
        self.min_volume = Decimal(str(min_volume))
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.neutral_range = neutral_range

        # 验证权重总和
        total_weight = (
            self.PRICE_MOMENTUM_WEIGHT +
            self.VOLUME_MOMENTUM_WEIGHT +
            self.ORDER_FLOW_WEIGHT +
            self.VOLATILITY_ADJUSTED_WEIGHT
        )
        if abs(total_weight - 1.0) > 0.01:
            logger.warning(f"Weight sum ({total_weight}) is not close to 1.0")

    def analyze_momentum(
        self,
        trades_data: list[Any],
        symbol: str = "BTCFDUSD",
        end_time: datetime | None = None,
    ) -> MomentumAnalysisResult:
        """分析短期动量。

        Args:
            trades_data: 交易数据列表
            symbol: 交易符号
            end_time: 分析结束时间，默认为当前时间

        Returns:
            动量分析结果
        """
        start_time = time.time()

        if end_time is None:
            end_time = datetime.now()

        window_start = end_time - timedelta(minutes=self.window_size_minutes)

        logger.info(f"Analyzing momentum for {symbol} from {window_start} to {end_time}")

        try:
            # 1. 创建交易窗口
            trade_window = self._create_trade_window(trades_data, symbol, window_start, end_time)

            # 2. 数据质量检查
            quality_check = self._validate_trade_window(trade_window)
            if not quality_check["is_valid"]:
                return self._create_error_result(
                    symbol, end_time, quality_check["error"], start_time
                )

            # 3. 计算各类动量指标（增加性能监控）
            indicators_start = time.time()
            indicators = self._calculate_momentum_indicators(trade_window)
            indicators_time = (time.time() - indicators_start) * 1000

            if indicators_time > 100:  # 如果计算超过100ms，记录警告
                logger.warning(f"Momentum indicators calculation took {indicators_time:.2f}ms")

            # 4. 生成综合信号
            signal = self._generate_momentum_signal(indicators, trade_window, end_time)

            # 5. 创建分析结果
            result = MomentumAnalysisResult(
                timestamp=end_time,
                symbol=symbol,
                analysis_window_minutes=self.window_size_minutes,
                signal=signal,
                trade_window_summary=trade_window.to_dict(),
                analysis_statistics=self._calculate_analysis_statistics(trade_window, indicators),
                processing_time_ms=(time.time() - start_time) * 1000,
            )

            logger.info(
                f"Momentum analysis completed: direction={signal.direction.value}, "
                f"strength={signal.strength:.3f}, confidence={signal.confidence:.3f}"
            )

            return result

        except (ValueError, DecimalException) as e:
            logger.error(f"Calculation error in momentum analysis: {e}")
            return self._create_error_result(symbol, end_time, f"Calculation error: {str(e)}", start_time)
        except Exception as e:
            logger.error(f"Unexpected error in momentum analysis: {e}")
            return self._create_error_result(symbol, end_time, f"Unexpected error: {str(e)}", start_time)

    def _create_trade_window(
        self,
        trades_data: list[Any],
        symbol: str,
        start_time: datetime,
        end_time: datetime,
    ) -> TradeWindow:
        """创建交易窗口。"""
        trade_window = TradeWindow(symbol=symbol, start_time=start_time, end_time=end_time)

        for trade in trades_data:
            trade_window.add_trade(trade)

        logger.debug(f"Trade window created with {len(trade_window.trades)} trades")
        return trade_window

    def _validate_trade_window(self, trade_window: TradeWindow) -> dict[str, Any]:
        """验证交易窗口数据质量。"""
        if len(trade_window.trades) < self.min_trades:
            return {
                "is_valid": False,
                "error": f"Insufficient trades: {len(trade_window.trades)} < {self.min_trades}",
            }

        if trade_window.total_volume < self.min_volume:
            return {
                "is_valid": False,
                "error": f"Insufficient volume: {trade_window.total_volume} < {self.min_volume}",
            }

        if trade_window.open_price is None or trade_window.close_price is None:
            return {
                "is_valid": False,
                "error": "Missing price data (open/close prices)",
            }

        return {"is_valid": True, "error": None}

    def _calculate_momentum_indicators(self, trade_window: TradeWindow) -> MomentumIndicators:
        """计算动量指标。"""
        indicators = MomentumIndicators()

        # 1. 计算价格动量指标
        price_indicators = self._calculate_price_momentum(trade_window)
        indicators.price_momentum = price_indicators["momentum"]
        indicators.price_change_rate = price_indicators["change_rate"]
        indicators.trend_strength = price_indicators["trend_strength"]
        indicators.weighted_price_momentum = price_indicators["weighted_momentum"]

        # 2. 计算成交量动量指标
        volume_indicators = self._calculate_volume_momentum(trade_window)
        indicators.volume_momentum = volume_indicators["momentum"]
        indicators.volume_imbalance = volume_indicators["imbalance"]
        indicators.large_trade_ratio = volume_indicators["large_trade_ratio"]
        indicators.volume_trend = volume_indicators["trend"]

        # 3. 计算订单流动量指标
        flow_indicators = self._calculate_order_flow_momentum(trade_window)
        indicators.order_flow_momentum = flow_indicators["momentum"]
        indicators.buy_pressure = flow_indicators["buy_pressure"]
        indicators.sell_pressure = flow_indicators["sell_pressure"]
        indicators.flow_consistency = flow_indicators["consistency"]

        # 4. 计算波动率调整指标
        volatility_indicators = self._calculate_volatility_adjusted(trade_window, indicators)
        indicators.volatility_adjusted = volatility_indicators["adjusted"]
        indicators.realized_volatility = volatility_indicators["realized_vol"]
        indicators.risk_adjusted_return = volatility_indicators["risk_adjusted_return"]

        return indicators

    def _calculate_price_momentum(self, trade_window: TradeWindow) -> dict[str, float]:
        """计算价格动量指标。"""
        price_series = trade_window.get_price_series()
        if len(price_series) < 2:
            return {
                "momentum": 0.0,
                "change_rate": 0.0,
                "trend_strength": 0.0,
                "weighted_momentum": 0.0,
            }

        # 1. 简单价格变化率
        change_rate = trade_window.calculate_price_change_rate()

        # 2. 线性回归趋势强度
        x = np.arange(len(price_series))
        y = np.array([float(p) for p in price_series])

        if len(x) > 1:
            slope, _, r_value, _, _ = stats.linregress(x, y)
            trend_strength = abs(slope * r_value) / float(trade_window.open_price or 1)
        else:
            trend_strength = 0.0

        # 3. 加权价格动量（基于成交量）
        volume_series = trade_window.get_volume_series()
        if len(volume_series) == len(price_series) and len(volume_series) > 0:
            weighted_prices = []
            for i, (price, volume) in enumerate(zip(price_series, volume_series, strict=True)):
                if i > 0:
                    weighted_change = (float(price) - float(price_series[i-1])) * float(volume)
                    weighted_prices.append(weighted_change)

            weighted_momentum = np.mean(weighted_prices) if weighted_prices else 0.0
        else:
            weighted_momentum = 0.0

        # 4. 综合价格动量
        momentum = (change_rate * 0.4 + trend_strength * 0.3 + weighted_momentum * 0.3)

        return {
            "momentum": momentum,
            "change_rate": change_rate,
            "trend_strength": trend_strength,
            "weighted_momentum": weighted_momentum,
        }

    def _calculate_volume_momentum(self, trade_window: TradeWindow) -> dict[str, float]:
        """计算成交量动量指标。"""
        # 1. 成交量不平衡
        volume_imbalance = trade_window.calculate_volume_imbalance()

        # 2. 大单分析
        if trade_window.trades:
            avg_trade_size = trade_window.total_volume / len(trade_window.trades)
            large_trades = [
                t for t in trade_window.trades
                if t.quantity > avg_trade_size * 2
            ]
            large_trade_ratio = len(large_trades) / len(trade_window.trades)
        else:
            large_trade_ratio = 0.0

        # 3. 成交量趋势（时间序列分析）
        volume_trend = self._calculate_volume_trend(trade_window)

        # 4. 综合成交量动量
        momentum = (volume_imbalance * 0.4 + large_trade_ratio * 0.3 + volume_trend * 0.3)

        return {
            "momentum": momentum,
            "imbalance": volume_imbalance,
            "large_trade_ratio": large_trade_ratio,
            "trend": volume_trend,
        }

    def _calculate_volume_trend(self, trade_window: TradeWindow) -> float:
        """计算成交量趋势。"""
        # 将交易按时间分组（每分钟）
        minute_volumes = {}
        for trade in trade_window.trades:
            minute_key = trade.timestamp.replace(second=0, microsecond=0)
            if minute_key not in minute_volumes:
                minute_volumes[minute_key] = Decimal("0")
            minute_volumes[minute_key] += trade.quantity

        if len(minute_volumes) < 2:
            return 0.0

        # 计算成交量线性趋势
        volumes = list(minute_volumes.values())
        x = np.arange(len(volumes))
        y = np.array([float(v) for v in volumes])

        if len(x) > 1:
            slope, _, r_value, _, _ = stats.linregress(x, y)
            # 归一化趋势
            avg_volume = np.mean(y)
            trend = (slope * r_value) / avg_volume if avg_volume > 0 else 0.0
            return float(np.clip(trend, -1, 1))
        else:
            return 0.0

    def _calculate_order_flow_momentum(self, trade_window: TradeWindow) -> dict[str, float]:
        """计算订单流动量指标。"""
        if not trade_window.trades:
            return {
                "momentum": 0.0,
                "buy_pressure": 0.0,
                "sell_pressure": 0.0,
                "consistency": 0.0,
            }

        # 1. 计算买卖压力
        total_volume = trade_window.total_volume
        if total_volume > 0:
            buy_pressure = float(trade_window.buy_volume / total_volume)
            sell_pressure = float(trade_window.sell_volume / total_volume)
        else:
            buy_pressure = sell_pressure = 0.0

        # 2. 计算流向一致性
        flow_consistency = self._calculate_flow_consistency(trade_window)

        # 3. 综合订单流动量
        momentum = (buy_pressure * 0.4 - sell_pressure * 0.4 + flow_consistency * 0.2)

        return {
            "momentum": momentum,
            "buy_pressure": buy_pressure,
            "sell_pressure": sell_pressure,
            "consistency": flow_consistency,
        }

    def _calculate_flow_consistency(self, trade_window: TradeWindow) -> float:
        """计算订单流一致性。"""
        if len(trade_window.trades) < 3:
            return 0.0

        # 计算连续同方向交易的强度
        consecutive_sequences = []
        current_sequence = 1
        current_direction = not trade_window.trades[0].is_buyer_maker  # True for aggressive buy

        for trade in trade_window.trades[1:]:
            trade_direction = not trade.is_buyer_maker  # True for aggressive buy

            if trade_direction == current_direction:
                current_sequence += 1
            else:
                consecutive_sequences.append(current_sequence)
                current_sequence = 1
                current_direction = trade_direction

        consecutive_sequences.append(current_sequence)

        # 计算平均连续长度
        if consecutive_sequences:
            avg_consecutive = np.mean(consecutive_sequences)
            # 归一化到0-1范围
            consistency = min(avg_consecutive / len(trade_window.trades), 1.0)
            return float(consistency) if isinstance(consistency, (int, float)) else float(consistency)
        else:
            return 0.0

    def _calculate_volatility_adjusted(
        self,
        trade_window: TradeWindow,
        indicators: MomentumIndicators,
    ) -> dict[str, float]:
        """计算波动率调整指标。"""
        price_series = trade_window.get_price_series()
        if len(price_series) < 2:
            return {
                "adjusted": 0.0,
                "realized_vol": 0.0,
                "risk_adjusted_return": 0.0,
            }

        # 1. 计算实现波动率
        price_returns = []
        for i in range(1, len(price_series)):
            if float(price_series[i-1]) > 0:
                ret = (float(price_series[i]) - float(price_series[i-1])) / float(price_series[i-1])
                price_returns.append(ret)

        if price_returns:
            realized_vol = np.std(price_returns) * np.sqrt(len(price_returns))  # 年化因子
        else:
            realized_vol = 0.0

        # 2. 风险调整收益
        if realized_vol > 0:
            risk_adjusted_return = indicators.price_momentum / realized_vol
        else:
            risk_adjusted_return = 0.0

        # 3. 波动率调整后的动量
        # 高波动率时降低信号强度
        volatility_factor = 1.0 / (1.0 + realized_vol * 10)  # 调整因子
        adjusted_momentum = indicators.price_momentum * volatility_factor

        return {
            "adjusted": adjusted_momentum,
            "realized_vol": realized_vol,
            "risk_adjusted_return": risk_adjusted_return,
        }

    def _generate_momentum_signal(
        self,
        indicators: MomentumIndicators,
        trade_window: TradeWindow,
        timestamp: datetime,
    ) -> MomentumSignal:
        """生成动量交易信号。"""
        # 1. 计算综合动量分数
        raw_score = (
            indicators.price_momentum * self.PRICE_MOMENTUM_WEIGHT +
            indicators.volume_momentum * self.VOLUME_MOMENTUM_WEIGHT +
            indicators.order_flow_momentum * self.ORDER_FLOW_WEIGHT +
            indicators.volatility_adjusted * self.VOLATILITY_ADJUSTED_WEIGHT
        )

        # 2. 确定方向
        if raw_score > self.buy_threshold:
            direction = MomentumDirection.BUY
        elif raw_score < self.sell_threshold:
            direction = MomentumDirection.SELL
        else:
            direction = MomentumDirection.NEUTRAL

        # 3. 计算强度（归一化到0-1）
        strength = min(abs(raw_score) * 2, 1.0)

        # 4. 计算置信度
        confidence = self._calculate_signal_confidence(indicators, trade_window)

        # 5. 评估信号质量
        quality_score = self._assess_signal_quality(indicators, trade_window)

        # 6. 确定市场条件
        market_condition = self._determine_market_condition(indicators, trade_window)

        return MomentumSignal(
            timestamp=timestamp,
            symbol=trade_window.symbol,
            direction=direction,
            strength=strength,
            confidence=confidence,
            raw_score=raw_score,
            indicators=indicators,
            timeframe=f"{self.window_size_minutes}m",
            analysis_window_minutes=self.window_size_minutes,
            trade_count=len(trade_window.trades),
            signal_quality_score=quality_score,
            market_condition=market_condition,
        )

    def _calculate_signal_confidence(
        self,
        indicators: MomentumIndicators,
        trade_window: TradeWindow,
    ) -> float:
        """计算信号置信度。"""
        # 基于多个因素计算置信度
        factors = []

        # 1. 指标一致性
        indicator_signs = [
            np.sign(indicators.price_momentum),
            np.sign(indicators.volume_momentum),
            np.sign(indicators.order_flow_momentum),
        ]
        consistency = len(set(indicator_signs)) / len(indicator_signs)
        factors.append(consistency)

        # 2. 数据质量
        data_quality = min(len(trade_window.trades) / 50, 1.0)  # 50笔交易为满分
        factors.append(data_quality)

        # 3. 趋势强度
        trend_factor = min(indicators.trend_strength * 10, 1.0)
        factors.append(trend_factor)

        # 4. 流动性充足度
        liquidity_factor = min(float(trade_window.total_volume) / 1.0, 1.0)  # 1 BTC为满分
        factors.append(liquidity_factor)

        # 综合置信度
        confidence = np.mean(factors)
        return float(np.clip(confidence, 0.0, 1.0))

    def _assess_signal_quality(
        self,
        indicators: MomentumIndicators,
        trade_window: TradeWindow,
    ) -> float:
        """评估信号质量分数。"""
        quality_factors = []

        # 1. 多维度确认
        price_confirmation = abs(indicators.price_momentum) > 0.05
        volume_confirmation = abs(indicators.volume_imbalance) > 0.1
        flow_confirmation = abs(indicators.order_flow_momentum) > 0.05

        confirmation_score = sum([price_confirmation, volume_confirmation, flow_confirmation]) / 3
        quality_factors.append(confirmation_score)

        # 2. 波动率合理性
        volatility_reasonable = 0.001 < indicators.realized_volatility < 0.1
        quality_factors.append(float(volatility_reasonable))

        # 3. 数据完整性
        completeness = 1.0 if (trade_window.open_price and trade_window.close_price) else 0.0
        quality_factors.append(completeness)

        # 4. 交易活跃度
        activity_score = min(len(trade_window.trades) / 20, 1.0)  # 20笔交易为满分
        quality_factors.append(activity_score)

        return float(np.mean(quality_factors))

    def _determine_market_condition(
        self,
        indicators: MomentumIndicators,
        trade_window: TradeWindow,
    ) -> str:
        """确定市场条件。"""
        volatility = indicators.realized_volatility
        trend_strength = indicators.trend_strength
        price_change = abs(trade_window.calculate_price_change_rate())

        if volatility > 0.05:
            return "volatile"
        elif trend_strength > 0.01 and price_change > 0.002:
            return "trending"
        elif volatility < 0.01 and price_change < 0.001:
            return "ranging"
        else:
            return "normal"

    def _calculate_analysis_statistics(
        self,
        trade_window: TradeWindow,
        indicators: MomentumIndicators,
    ) -> dict[str, Any]:
        """计算分析统计信息。"""
        return {
            "trade_statistics": {
                "total_trades": len(trade_window.trades),
                "total_volume": float(trade_window.total_volume),
                "avg_trade_size": float(trade_window.total_volume / len(trade_window.trades)) if trade_window.trades else 0,
                "buy_ratio": float(trade_window.buy_volume / trade_window.total_volume) if trade_window.total_volume > 0 else 0,
                "sell_ratio": float(trade_window.sell_volume / trade_window.total_volume) if trade_window.total_volume > 0 else 0,
            },
            "price_statistics": {
                "open_price": float(trade_window.open_price) if trade_window.open_price else None,
                "close_price": float(trade_window.close_price) if trade_window.close_price else None,
                "high_price": float(trade_window.high_price) if trade_window.high_price else None,
                "low_price": float(trade_window.low_price) if trade_window.low_price else None,
                "vwap": float(trade_window.vwap) if trade_window.vwap else None,
                "price_change": float(trade_window.calculate_price_change()),
                "price_change_rate": trade_window.calculate_price_change_rate(),
            },
            "indicator_summary": {
                "strongest_indicator": max([
                    ("price", abs(indicators.price_momentum)),
                    ("volume", abs(indicators.volume_momentum)),
                    ("flow", abs(indicators.order_flow_momentum)),
                ], key=lambda x: x[1])[0],
                "overall_momentum": (
                    indicators.price_momentum * self.PRICE_MOMENTUM_WEIGHT +
                    indicators.volume_momentum * self.VOLUME_MOMENTUM_WEIGHT +
                    indicators.order_flow_momentum * self.ORDER_FLOW_WEIGHT
                ),
                "volatility_level": "high" if indicators.realized_volatility > 0.05 else "normal" if indicators.realized_volatility > 0.01 else "low",
            },
        }

    def _create_error_result(
        self,
        symbol: str,
        timestamp: datetime,
        error_message: str,
        start_time: float,
    ) -> MomentumAnalysisResult:
        """创建错误分析结果。"""
        processing_time = (time.time() - start_time) * 1000

        # 创建中性信号
        neutral_indicators = MomentumIndicators()
        neutral_signal = MomentumSignal(
            timestamp=timestamp,
            symbol=symbol,
            direction=MomentumDirection.NEUTRAL,
            strength=0.0,
            confidence=0.0,
            raw_score=0.0,
            indicators=neutral_indicators,
            timeframe=f"{self.window_size_minutes}m",
            analysis_window_minutes=self.window_size_minutes,
            trade_count=0,
            signal_quality_score=0.0,
            market_condition="error",
        )

        return MomentumAnalysisResult(
            timestamp=timestamp,
            symbol=symbol,
            analysis_window_minutes=self.window_size_minutes,
            signal=neutral_signal,
            trade_window_summary={"error": error_message},
            analysis_statistics={"error": error_message},
            processing_time_ms=processing_time,
        )

    def get_analyzer_status(self) -> dict[str, Any]:
        """获取分析器状态信息。"""
        return {
            "analyzer_type": "short_term_momentum_analyzer",
            "window_size_minutes": self.window_size_minutes,
            "min_trades": self.min_trades,
            "min_volume": float(self.min_volume),
            "thresholds": {
                "buy_threshold": self.buy_threshold,
                "sell_threshold": self.sell_threshold,
                "neutral_range": self.neutral_range,
            },
            "weights": {
                "price_momentum": self.PRICE_MOMENTUM_WEIGHT,
                "volume_momentum": self.VOLUME_MOMENTUM_WEIGHT,
                "order_flow": self.ORDER_FLOW_WEIGHT,
                "volatility_adjusted": self.VOLATILITY_ADJUSTED_WEIGHT,
            },
        }
