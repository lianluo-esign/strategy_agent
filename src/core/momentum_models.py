"""动量分析数据模型。

定义短期动量策略分析所需的数据结构和枚举类型。
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class MomentumDirection(Enum):
    """动量方向枚举。"""
    BUY = "buy"
    SELL = "sell"
    NEUTRAL = "neutral"


@dataclass
class MomentumIndicators:
    """动量指标数据结构。"""

    # 价格动量指标
    price_momentum: float = 0.0
    price_change_rate: float = 0.0
    trend_strength: float = 0.0
    weighted_price_momentum: float = 0.0

    # 成交量动量指标
    volume_momentum: float = 0.0
    volume_imbalance: float = 0.0
    large_trade_ratio: float = 0.0
    volume_trend: float = 0.0

    # 订单流动量指标
    order_flow_momentum: float = 0.0
    buy_pressure: float = 0.0
    sell_pressure: float = 0.0
    flow_consistency: float = 0.0

    # 波动率调整指标
    volatility_adjusted: float = 0.0
    realized_volatility: float = 0.0
    risk_adjusted_return: float = 0.0

    def to_dict(self) -> dict[str, float]:
        """转换为字典格式。"""
        return {
            "price_momentum": self.price_momentum,
            "price_change_rate": self.price_change_rate,
            "trend_strength": self.trend_strength,
            "weighted_price_momentum": self.weighted_price_momentum,
            "volume_momentum": self.volume_momentum,
            "volume_imbalance": self.volume_imbalance,
            "large_trade_ratio": self.large_trade_ratio,
            "volume_trend": self.volume_trend,
            "order_flow_momentum": self.order_flow_momentum,
            "buy_pressure": self.buy_pressure,
            "sell_pressure": self.sell_pressure,
            "flow_consistency": self.flow_consistency,
            "volatility_adjusted": self.volatility_adjusted,
            "realized_volatility": self.realized_volatility,
            "risk_adjusted_return": self.risk_adjusted_return,
        }


@dataclass
class MomentumSignal:
    """动量交易信号。"""

    # 基本信息
    timestamp: datetime
    symbol: str
    direction: MomentumDirection
    strength: float  # 0.0 - 1.0
    confidence: float  # 0.0 - 1.0

    # 原始分数和指标
    raw_score: float
    indicators: MomentumIndicators

    # 元数据
    timeframe: str = "5m"
    analysis_window_minutes: int = 5
    trade_count: int = 0

    # 信号质量评估
    signal_quality_score: float = 0.0
    market_condition: str = "normal"  # normal, volatile, trending, ranging

    def to_dict(self) -> dict[str, Any]:
        """转换为字典格式。"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "direction": self.direction.value,
            "strength": self.strength,
            "confidence": self.confidence,
            "raw_score": self.raw_score,
            "indicators": self.indicators.to_dict(),
            "timeframe": self.timeframe,
            "analysis_window_minutes": self.analysis_window_minutes,
            "trade_count": self.trade_count,
            "signal_quality_score": self.signal_quality_score,
            "market_condition": self.market_condition,
        }


@dataclass
class TradeWindow:
    """交易窗口数据结构。"""

    # 基本信息
    symbol: str
    start_time: datetime
    end_time: datetime

    # 交易数据
    trades: list[Any] = field(default_factory=list)  # List[Trade] objects

    # 聚合数据
    total_volume: Decimal = Decimal("0")
    total_trades: int = 0
    buy_volume: Decimal = Decimal("0")
    sell_volume: Decimal = Decimal("0")

    # 价格统计
    open_price: Decimal | None = None
    close_price: Decimal | None = None
    high_price: Decimal | None = None
    low_price: Decimal | None = None
    vwap: Decimal | None = None  # 成交量加权平均价

    def add_trade(self, trade: Any) -> None:
        """添加交易数据到窗口。"""

        if not self._validate_trade(trade):
            return

        self.trades.append(trade)
        self._update_aggregates(trade)
        self._update_price_statistics(trade.price)
        self._update_vwap()

    def _validate_trade(self, trade: Any) -> bool:
        """验证交易数据。"""
        from .models import Trade

        if not isinstance(trade, Trade):
            logger.warning(f"Invalid trade type: {type(trade)}")
            return False

        # 检查时间范围
        if not (self.start_time <= trade.timestamp <= self.end_time):
            return False

        return True

    def _update_aggregates(self, trade: Any) -> None:
        """更新聚合统计数据。"""
        self.total_trades += 1
        self.total_volume += trade.quantity

        # 更新买卖量
        if trade.is_buyer_maker:
            self.sell_volume += trade.quantity
        else:
            self.buy_volume += trade.quantity

    def _update_price_statistics(self, trade_price: Decimal) -> None:
        """更新价格统计。"""
        if self.open_price is None:
            self.open_price = trade_price

        self.close_price = trade_price

        if self.high_price is None or trade_price > self.high_price:
            self.high_price = trade_price

        if self.low_price is None or trade_price < self.low_price:
            self.low_price = trade_price

    def _update_vwap(self) -> None:
        """更新VWAP（增加数值溢出保护）。"""
        if self.total_volume > 0:
            try:
                total_value = Decimal("0")
                for trade in self.trades:
                    # 防止数值溢出
                    trade_value = trade.price * trade.quantity
                    if abs(trade_value) > Decimal("1e10"):
                        logger.warning(f"Large trade value detected: {trade_value}")
                    total_value += trade_value
                self.vwap = total_value / self.total_volume
            except (ValueError, ArithmeticError) as e:
                logger.error(f"VWAP calculation error: {e}")
                self.vwap = None

    def get_price_series(self) -> list[Decimal]:
        """获取价格序列。"""
        return [trade.price for trade in self.trades]

    def get_volume_series(self) -> list[Decimal]:
        """获取成交量序列。"""
        return [trade.quantity for trade in self.trades]

    def get_timestamp_series(self) -> list[datetime]:
        """获取时间戳序列。"""
        return [trade.timestamp for trade in self.trades]

    def calculate_price_change(self) -> Decimal:
        """计算价格变化。"""
        if self.open_price and self.close_price:
            return self.close_price - self.open_price
        return Decimal("0")

    def calculate_price_change_rate(self) -> float:
        """计算价格变化率。"""
        if self.open_price and self.close_price and self.open_price > 0:
            change = self.close_price - self.open_price
            return float(change / self.open_price)
        return 0.0

    def calculate_volume_imbalance(self) -> float:
        """计算成交量不平衡率。"""
        if self.total_volume > 0:
            return float((self.buy_volume - self.sell_volume) / self.total_volume)
        return 0.0

    def to_dict(self) -> dict[str, Any]:
        """转换为字典格式。"""
        return {
            "symbol": self.symbol,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "total_volume": float(self.total_volume),
            "total_trades": self.total_trades,
            "buy_volume": float(self.buy_volume),
            "sell_volume": float(self.sell_volume),
            "open_price": float(self.open_price) if self.open_price else None,
            "close_price": float(self.close_price) if self.close_price else None,
            "high_price": float(self.high_price) if self.high_price else None,
            "low_price": float(self.low_price) if self.low_price else None,
            "vwap": float(self.vwap) if self.vwap else None,
            "price_change": float(self.calculate_price_change()),
            "price_change_rate": self.calculate_price_change_rate(),
            "volume_imbalance": self.calculate_volume_imbalance(),
        }

    def cleanup_old_trades(self, cutoff_time: datetime) -> None:
        """清理旧交易数据以防止内存泄漏。

        Args:
            cutoff_time: 截断时间，早于此时间的交易将被删除
        """
        original_count = len(self.trades)
        self.trades = [trade for trade in self.trades if trade.timestamp >= cutoff_time]

        # 重新计算聚合数据
        if len(self.trades) != original_count:
            self._recalculate_aggregates()
            logger.debug(f"Cleaned up {original_count - len(self.trades)} old trades")

    def _recalculate_aggregates(self) -> None:
        """重新计算聚合数据。"""
        self.total_volume = Decimal("0")
        self.total_trades = 0
        self.buy_volume = Decimal("0")
        self.sell_volume = Decimal("0")
        self.open_price = None
        self.close_price = None
        self.high_price = None
        self.low_price = None
        self.vwap = None

        # 重新计算所有聚合指标
        for trade in self.trades:
            self.total_volume += trade.quantity
            self.total_trades += 1

            if trade.is_buyer_maker:
                self.sell_volume += trade.quantity
            else:
                self.buy_volume += trade.quantity

            # 更新价格统计
            if self.open_price is None:
                self.open_price = trade.price
            self.close_price = trade.price

            if self.high_price is None or trade.price > self.high_price:
                self.high_price = trade.price
            if self.low_price is None or trade.price < self.low_price:
                self.low_price = trade.price

        # 重新计算VWAP
        if self.total_volume > 0:
            total_value = sum(trade.price * trade.quantity for trade in self.trades)
            self.vwap = total_value / self.total_volume

    def get_memory_usage_estimate(self) -> int:
        """估算内存使用量（字节）。

        Returns:
            估算的内存使用量
        """
        import sys

        # 估算单个Trade对象的大小
        trade_size = sys.getsizeof(self.trades[0]) if self.trades else 200
        return len(self.trades) * trade_size + 1000  # 加上基础结构开销


@dataclass
class MomentumAnalysisResult:
    """动量分析结果。"""

    # 基本信息
    timestamp: datetime
    symbol: str
    analysis_window_minutes: int

    # 主要信号
    signal: MomentumSignal

    # 原始数据摘要
    trade_window_summary: dict[str, Any]

    # 分析统计
    analysis_statistics: dict[str, Any] = field(default_factory=dict)

    # 性能指标
    processing_time_ms: float = 0.0
    memory_usage_mb: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """转换为字典格式。"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "analysis_window_minutes": self.analysis_window_minutes,
            "signal": self.signal.to_dict(),
            "trade_window_summary": self.trade_window_summary,
            "analysis_statistics": self.analysis_statistics,
            "processing_time_ms": self.processing_time_ms,
            "memory_usage_mb": self.memory_usage_mb,
        }
