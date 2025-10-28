"""动量数据模型单元测试。"""

import pytest
from datetime import datetime, timedelta
from decimal import Decimal

from src.core.models import Trade
from src.core.momentum_models import (
    MomentumDirection,
    MomentumIndicators,
    MomentumSignal,
    TradeWindow,
    MomentumAnalysisResult,
)


class TestMomentumDirection:
    """动量方向枚举测试。"""

    def test_direction_values(self):
        """测试方向值。"""
        assert MomentumDirection.BUY.value == "buy"
        assert MomentumDirection.SELL.value == "sell"
        assert MomentumDirection.NEUTRAL.value == "neutral"


class TestMomentumIndicators:
    """动量指标测试。"""

    def test_default_initialization(self):
        """测试默认初始化。"""
        indicators = MomentumIndicators()

        assert indicators.price_momentum == 0.0
        assert indicators.volume_momentum == 0.0
        assert indicators.order_flow_momentum == 0.0
        assert indicators.volatility_adjusted == 0.0

    def test_custom_initialization(self):
        """测试自定义初始化。"""
        indicators = MomentumIndicators(
            price_momentum=0.5,
            volume_momentum=0.3,
            order_flow_momentum=0.2,
            volatility_adjusted=0.1
        )

        assert indicators.price_momentum == 0.5
        assert indicators.volume_momentum == 0.3
        assert indicators.order_flow_momentum == 0.2
        assert indicators.volatility_adjusted == 0.1

    def test_to_dict(self):
        """测试转换为字典。"""
        indicators = MomentumIndicators(
            price_momentum=0.1,
            volume_momentum=0.2,
            order_flow_momentum=0.3,
            volatility_adjusted=0.4
        )

        result = indicators.to_dict()

        assert isinstance(result, dict)
        assert result["price_momentum"] == 0.1
        assert result["volume_momentum"] == 0.2
        assert result["order_flow_momentum"] == 0.3
        assert result["volatility_adjusted"] == 0.4

        # 验证所有指标都包含在内
        expected_keys = [
            "price_momentum", "price_change_rate", "trend_strength", "weighted_price_momentum",
            "volume_momentum", "volume_imbalance", "large_trade_ratio", "volume_trend",
            "order_flow_momentum", "buy_pressure", "sell_pressure", "flow_consistency",
            "volatility_adjusted", "realized_volatility", "risk_adjusted_return"
        ]

        for key in expected_keys:
            assert key in result


class TestTradeWindow:
    """交易窗口测试。"""

    def setup_method(self):
        """测试方法设置。"""
        self.symbol = "BTCFDUSD"
        self.start_time = datetime.now() - timedelta(minutes=5)
        self.end_time = datetime.now()

    def test_default_initialization(self):
        """测试默认初始化。"""
        window = TradeWindow(self.symbol, self.start_time, self.end_time)

        assert window.symbol == self.symbol
        assert window.start_time == self.start_time
        assert window.end_time == self.end_time
        assert len(window.trades) == 0
        assert window.total_volume == Decimal("0")
        assert window.total_trades == 0

    def test_add_valid_trade(self):
        """测试添加有效交易。"""
        window = TradeWindow(self.symbol, self.start_time, self.end_time)

        trade_time = self.start_time + timedelta(minutes=2)
        trade = Trade(
            symbol=self.symbol,
            price=Decimal("42000.00"),
            quantity=Decimal("0.1"),
            is_buyer_maker=False,
            timestamp=trade_time,
            trade_id="test_trade_1"
        )

        window.add_trade(trade)

        assert len(window.trades) == 1
        assert window.total_trades == 1
        assert window.total_volume == Decimal("0.1")
        assert window.open_price == Decimal("42000.00")
        assert window.close_price == Decimal("42000.00")
        assert window.high_price == Decimal("42000.00")
        assert window.low_price == Decimal("42000.00")

    def test_add_trade_outside_time_window(self):
        """测试添加时间窗口外的交易。"""
        window = TradeWindow(self.symbol, self.start_time, self.end_time)

        # 时间窗口之前的交易
        early_trade = Trade(
            symbol=self.symbol,
            price=Decimal("42000.00"),
            quantity=Decimal("0.1"),
            is_buyer_maker=False,
            timestamp=self.start_time - timedelta(minutes=1),
            trade_id="early_trade"
        )

        # 时间窗口之后的交易
        late_trade = Trade(
            symbol=self.symbol,
            price=Decimal("42001.00"),
            quantity=Decimal("0.1"),
            is_buyer_maker=False,
            timestamp=self.end_time + timedelta(minutes=1),
            trade_id="late_trade"
        )

        window.add_trade(early_trade)
        window.add_trade(late_trade)

        # 都不应该被添加
        assert len(window.trades) == 0
        assert window.total_trades == 0

    def test_add_invalid_trade(self):
        """测试添加无效交易。"""
        window = TradeWindow(self.symbol, self.start_time, self.end_time)

        # 添加无效类型
        window.add_trade("invalid_trade")
        window.add_trade(None)

        assert len(window.trades) == 0

    def test_buy_sell_volume_calculation(self):
        """测试买卖成交量计算。"""
        window = TradeWindow(self.symbol, self.start_time, self.end_time)

        # 主动买入交易（is_buyer_maker=False）
        buy_trade = Trade(
            symbol=self.symbol,
            price=Decimal("42000.00"),
            quantity=Decimal("0.2"),
            is_buyer_maker=False,  # 主动买入
            timestamp=self.start_time + timedelta(minutes=1),
            trade_id="buy_trade"
        )

        # 主动卖出交易（is_buyer_maker=True）
        sell_trade = Trade(
            symbol=self.symbol,
            price=Decimal("42000.00"),
            quantity=Decimal("0.15"),
            is_buyer_maker=True,  # 主动卖出
            timestamp=self.start_time + timedelta(minutes=2),
            trade_id="sell_trade"
        )

        window.add_trade(buy_trade)
        window.add_trade(sell_trade)

        assert window.buy_volume == Decimal("0.2")
        assert window.sell_volume == Decimal("0.15")
        assert window.total_volume == Decimal("0.35")

    def test_price_statistics(self):
        """测试价格统计。"""
        window = TradeWindow(self.symbol, self.start_time, self.end_time)

        prices = [
            Decimal("42000.00"),
            Decimal("42001.00"),
            Decimal("41999.00"),
            Decimal("42002.00"),
            Decimal("41998.00")
        ]

        for i, price in enumerate(prices):
            trade = Trade(
                symbol=self.symbol,
                price=price,
                quantity=Decimal("0.1"),
                is_buyer_maker=False,
                timestamp=self.start_time + timedelta(seconds=i * 60),
                trade_id=f"price_test_{i}"
            )
            window.add_trade(trade)

        assert window.open_price == Decimal("42000.00")
        assert window.close_price == Decimal("41998.00")
        assert window.high_price == Decimal("42002.00")
        assert window.low_price == Decimal("41998.00")

    def test_vwap_calculation(self):
        """测试VWAP计算。"""
        window = TradeWindow(self.symbol, self.start_time, self.end_time)

        # 添加不同价格和数量的交易
        trades_data = [
            (Decimal("42000.00"), Decimal("0.1")),
            (Decimal("42001.00"), Decimal("0.2")),
            (Decimal("42002.00"), Decimal("0.1")),
        ]

        for i, (price, quantity) in enumerate(trades_data):
            trade = Trade(
                symbol=self.symbol,
                price=price,
                quantity=quantity,
                is_buyer_maker=False,
                timestamp=self.start_time + timedelta(seconds=i * 60),
                trade_id=f"vwap_test_{i}"
            )
            window.add_trade(trade)

        # 手动计算期望的VWAP
        total_value = sum(price * quantity for price, quantity in trades_data)
        total_volume = sum(quantity for _, quantity in trades_data)
        expected_vwap = total_value / total_volume

        assert window.vwap == expected_vwap

    def test_price_change_calculation(self):
        """测试价格变化计算。"""
        window = TradeWindow(self.symbol, self.start_time, self.end_time)

        # 添加两笔交易
        open_trade = Trade(
            symbol=self.symbol,
            price=Decimal("42000.00"),
            quantity=Decimal("0.1"),
            is_buyer_maker=False,
            timestamp=self.start_time + timedelta(minutes=1),
            trade_id="open_trade"
        )

        close_trade = Trade(
            symbol=self.symbol,
            price=Decimal("42050.00"),
            quantity=Decimal("0.1"),
            is_buyer_maker=False,
            timestamp=self.start_time + timedelta(minutes=2),
            trade_id="close_trade"
        )

        window.add_trade(open_trade)
        window.add_trade(close_trade)

        price_change = window.calculate_price_change()
        assert price_change == Decimal("50.00")

        change_rate = window.calculate_price_change_rate()
        expected_rate = 50.00 / 42000.00
        assert abs(change_rate - expected_rate) < 0.0001

    def test_volume_imbalance_calculation(self):
        """测试成交量不平衡计算。"""
        window = TradeWindow(self.symbol, self.start_time, self.end_time)

        # 添加更多买入交易
        buy_volume = Decimal("0.8")
        sell_volume = Decimal("0.2")

        # 买入交易
        for i in range(4):
            trade = Trade(
                symbol=self.symbol,
                price=Decimal("42000.00"),
                quantity=Decimal("0.2"),
                is_buyer_maker=False,  # 主动买入
                timestamp=self.start_time + timedelta(minutes=i),
                trade_id=f"buy_trade_{i}"
            )
            window.add_trade(trade)

        # 卖出交易
        for i in range(1):
            trade = Trade(
                symbol=self.symbol,
                price=Decimal("42000.00"),
                quantity=Decimal("0.2"),
                is_buyer_maker=True,  # 主动卖出
                timestamp=self.start_time + timedelta(minutes=4),
                trade_id="sell_trade"
            )
            window.add_trade(trade)

        imbalance = window.calculate_volume_imbalance()
        expected_imbalance = float((buy_volume - sell_volume) / (buy_volume + sell_volume))
        assert abs(imbalance - expected_imbalance) < 0.0001

    def test_series_methods(self):
        """测试序列方法。"""
        window = TradeWindow(self.symbol, self.start_time, self.end_time)

        prices = [Decimal("42000.00"), Decimal("42001.00"), Decimal("42002.00")]
        volumes = [Decimal("0.1"), Decimal("0.2"), Decimal("0.15")]

        for i, (price, volume) in enumerate(zip(prices, volumes)):
            trade = Trade(
                symbol=self.symbol,
                price=price,
                quantity=volume,
                is_buyer_maker=False,
                timestamp=self.start_time + timedelta(seconds=i * 60),
                trade_id=f"series_test_{i}"
            )
            window.add_trade(trade)

        # 测试价格序列
        price_series = window.get_price_series()
        assert price_series == prices

        # 测试成交量序列
        volume_series = window.get_volume_series()
        assert volume_series == volumes

        # 测试时间戳序列
        timestamp_series = window.get_timestamp_series()
        assert len(timestamp_series) == 3

    def test_to_dict(self):
        """测试转换为字典。"""
        window = TradeWindow(self.symbol, self.start_time, self.end_time)

        trade = Trade(
            symbol=self.symbol,
            price=Decimal("42000.00"),
            quantity=Decimal("0.1"),
            is_buyer_maker=False,
            timestamp=self.start_time + timedelta(minutes=1),
            trade_id="dict_test"
        )
        window.add_trade(trade)

        result = window.to_dict()

        assert isinstance(result, dict)
        assert result["symbol"] == self.symbol
        assert result["total_volume"] == 0.1
        assert result["total_trades"] == 1
        assert result["open_price"] == 42000.0
        assert result["price_change_rate"] == 0.0  # 只有一笔交易，变化率为0


class TestMomentumSignal:
    """动量信号测试。"""

    def setup_method(self):
        """测试方法设置。"""
        self.timestamp = datetime.now()
        self.symbol = "BTCFDUSD"
        self.direction = MomentumDirection.BUY
        self.indicators = MomentumIndicators(price_momentum=0.2)

    def test_initialization(self):
        """测试初始化。"""
        signal = MomentumSignal(
            timestamp=self.timestamp,
            symbol=self.symbol,
            direction=self.direction,
            strength=0.8,
            confidence=0.9,
            raw_score=0.15,
            indicators=self.indicators
        )

        assert signal.timestamp == self.timestamp
        assert signal.symbol == self.symbol
        assert signal.direction == self.direction
        assert signal.strength == 0.8
        assert signal.confidence == 0.9
        assert signal.raw_score == 0.15
        assert signal.indicators == self.indicators

    def test_to_dict(self):
        """测试转换为字典。"""
        signal = MomentumSignal(
            timestamp=self.timestamp,
            symbol=self.symbol,
            direction=self.direction,
            strength=0.7,
            confidence=0.85,
            raw_score=0.12,
            indicators=self.indicators,
            timeframe="5m",
            trade_count=25
        )

        result = signal.to_dict()

        assert isinstance(result, dict)
        assert result["timestamp"] == self.timestamp.isoformat()
        assert result["symbol"] == self.symbol
        assert result["direction"] == "buy"
        assert result["strength"] == 0.7
        assert result["confidence"] == 0.85
        assert result["raw_score"] == 0.12
        assert result["timeframe"] == "5m"
        assert result["trade_count"] == 25

        # 验证指标也被正确转换
        assert "indicators" in result
        assert isinstance(result["indicators"], dict)


class TestMomentumAnalysisResult:
    """动量分析结果测试。"""

    def setup_method(self):
        """测试方法设置。"""
        self.timestamp = datetime.now()
        self.symbol = "BTCFDUSD"
        self.signal = MomentumSignal(
            timestamp=self.timestamp,
            symbol=self.symbol,
            direction=MomentumDirection.BUY,
            strength=0.8,
            confidence=0.9,
            raw_score=0.15,
            indicators=MomentumIndicators()
        )
        self.trade_window_summary = {"total_trades": 50, "total_volume": 5.0}

    def test_initialization(self):
        """测试初始化。"""
        result = MomentumAnalysisResult(
            timestamp=self.timestamp,
            symbol=self.symbol,
            analysis_window_minutes=5,
            signal=self.signal,
            trade_window_summary=self.trade_window_summary,
            processing_time_ms=150.0
        )

        assert result.timestamp == self.timestamp
        assert result.symbol == self.symbol
        assert result.analysis_window_minutes == 5
        assert result.signal == self.signal
        assert result.trade_window_summary == self.trade_window_summary
        assert result.processing_time_ms == 150.0

    def test_to_dict(self):
        """测试转换为字典。"""
        result = MomentumAnalysisResult(
            timestamp=self.timestamp,
            symbol=self.symbol,
            analysis_window_minutes=5,
            signal=self.signal,
            trade_window_summary=self.trade_window_summary,
            analysis_statistics={"test_stat": 123},
            processing_time_ms=150.0,
            memory_usage_mb=25.5
        )

        dict_result = result.to_dict()

        assert isinstance(dict_result, dict)
        assert dict_result["timestamp"] == self.timestamp.isoformat()
        assert dict_result["symbol"] == self.symbol
        assert dict_result["analysis_window_minutes"] == 5
        assert dict_result["processing_time_ms"] == 150.0
        assert dict_result["memory_usage_mb"] == 25.5

        # 验证嵌套结构
        assert "signal" in dict_result
        assert "trade_window_summary" in dict_result
        assert "analysis_statistics" in dict_result

        # 验证信号也被正确转换
        assert isinstance(dict_result["signal"], dict)
        assert dict_result["signal"]["direction"] == "buy"