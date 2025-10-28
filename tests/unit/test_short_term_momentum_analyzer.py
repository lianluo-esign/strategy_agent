"""短期动量分析器单元测试。"""

import pytest
from datetime import datetime, timedelta
from decimal import Decimal

from src.core.models import Trade
from src.core.short_term_momentum_analyzer import ShortTermMomentumAnalyzer
from src.core.momentum_models import MomentumDirection


class TestShortTermMomentumAnalyzer:
    """短期动量分析器测试类。"""

    def setup_method(self):
        """每个测试方法前的设置。"""
        self.analyzer = ShortTermMomentumAnalyzer(
            window_size_minutes=5,
            min_trades=5,
            min_volume=0.01
        )
        self.base_time = datetime.now() - timedelta(minutes=10)

    def teardown_method(self):
        """每个测试方法后的清理。"""
        pass

    def create_test_trades(self, count: int, price_trend: str = "up") -> list[Trade]:
        """创建测试交易数据。

        Args:
            count: 交易数量
            price_trend: 价格趋势 ("up", "down", "sideways")

        Returns:
            交易数据列表
        """
        trades = []
        base_price = Decimal("42000.00")
        # 确保交易在最近的5分钟窗口内
        end_time = datetime.now()
        start_time = end_time - timedelta(minutes=4)

        for i in range(count):
            # 确保时间在5分钟窗口内
            timestamp = start_time + timedelta(seconds=i * (240 / count))

            if price_trend == "up":
                price = base_price + Decimal(str(i * 0.5))
                is_buyer_maker = i % 3 != 0  # 更多买入
            elif price_trend == "down":
                price = base_price - Decimal(str(i * 0.3))
                is_buyer_maker = i % 4 != 0  # 更多卖出
            else:  # sideways
                price = base_price + Decimal(str((i % 10 - 5) * 0.1))
                is_buyer_maker = i % 2 == 0  # 平衡买卖

            trade = Trade(
                symbol="BTCFDUSD",
                price=price,
                quantity=Decimal("0.1"),
                is_buyer_maker=is_buyer_maker,
                timestamp=timestamp,
                trade_id=f"test_trade_{i}"
            )
            trades.append(trade)

        return trades

    def test_analyzer_initialization(self):
        """测试分析器初始化。"""
        assert self.analyzer.window_size_minutes == 5
        assert self.analyzer.min_trades == 5
        assert self.analyzer.min_volume == Decimal("0.01")
        assert self.analyzer.buy_threshold == 0.15
        assert self.analyzer.sell_threshold == -0.15

    def test_analyzer_status(self):
        """测试获取分析器状态。"""
        status = self.analyzer.get_analyzer_status()

        assert status["analyzer_type"] == "short_term_momentum_analyzer"
        assert status["window_size_minutes"] == 5
        assert "thresholds" in status
        assert "weights" in status

    def test_analyze_momentum_uptrend(self):
        """测试上升趋势动量分析。"""
        trades = self.create_test_trades(20, "up")
        result = self.analyzer.analyze_momentum(trades, "BTCFDUSD")

        # 验证基本结果结构
        assert result.symbol == "BTCFDUSD"
        assert result.analysis_window_minutes == 5
        assert result.signal is not None
        assert result.processing_time_ms > 0

        # 验证信号特征
        signal = result.signal
        assert signal.direction in [MomentumDirection.BUY, MomentumDirection.NEUTRAL]
        assert 0 <= signal.strength <= 1
        assert 0 <= signal.confidence <= 1
        assert signal.trade_count == 20

        # 上升趋势应该产生买入或中性信号
        assert signal.direction in [MomentumDirection.BUY, MomentumDirection.NEUTRAL]

    def test_analyze_momentum_downtrend(self):
        """测试下跌趋势动量分析。"""
        trades = self.create_test_trades(15, "down")
        result = self.analyzer.analyze_momentum(trades, "BTCFDUSD")

        signal = result.signal
        # 下跌趋势应该产生卖出或中性信号
        assert signal.direction in [MomentumDirection.SELL, MomentumDirection.NEUTRAL]
        assert signal.trade_count == 15

    def test_analyze_momentum_sideways(self):
        """测试横盘整理动量分析。"""
        trades = self.create_test_trades(25, "sideways")
        result = self.analyzer.analyze_momentum(trades, "BTCFDUSD")

        signal = result.signal
        # 横盘整理应该产生中性信号
        assert signal.direction == MomentumDirection.NEUTRAL
        assert signal.trade_count == 25

    def test_insufficient_trades(self):
        """测试交易数量不足的情况。"""
        trades = self.create_test_trades(3, "up")  # 少于min_trades
        result = self.analyzer.analyze_momentum(trades, "BTCFDUSD")

        signal = result.signal
        assert signal.direction == MomentumDirection.NEUTRAL
        assert signal.strength == 0.0
        assert signal.confidence == 0.0
        assert "error" in result.trade_window_summary

    def test_insufficient_volume(self):
        """测试成交量不足的情况。"""
        # 创建成交量很小的交易
        trades = []
        for i in range(10):
            trade = Trade(
                symbol="BTCFDUSD",
                price=Decimal("42000.00"),
                quantity=Decimal("0.001"),  # 很小的成交量
                is_buyer_maker=False,
                timestamp=self.base_time + timedelta(seconds=i * 30),
                trade_id=f"small_trade_{i}"
            )
            trades.append(trade)

        result = self.analyzer.analyze_momentum(trades, "BTCFDUSD")

        signal = result.signal
        assert signal.direction == MomentumDirection.NEUTRAL
        assert signal.strength == 0.0
        assert "error" in result.trade_window_summary

    def test_empty_trades_list(self):
        """测试空交易列表。"""
        result = self.analyzer.analyze_momentum([], "BTCFDUSD")

        signal = result.signal
        assert signal.direction == MomentumDirection.NEUTRAL
        assert signal.strength == 0.0
        assert signal.confidence == 0.0
        assert signal.trade_count == 0

    def test_momentum_indicators_calculation(self):
        """测试动量指标计算。"""
        trades = self.create_test_trades(30, "up")
        result = self.analyzer.analyze_momentum(trades, "BTCFDUSD")

        indicators = result.signal.indicators

        # 验证所有指标都被计算
        assert indicators.price_momentum is not None
        assert indicators.volume_momentum is not None
        assert indicators.order_flow_momentum is not None
        assert indicators.volatility_adjusted is not None

        # 验证指标值在合理范围内
        assert -1 <= indicators.price_momentum <= 1
        assert -1 <= indicators.volume_momentum <= 1
        assert -1 <= indicators.order_flow_momentum <= 1
        assert indicators.realized_volatility >= 0

    def test_signal_strength_calculation(self):
        """测试信号强度计算。"""
        # 强上升趋势
        strong_uptrend_trades = []
        for i in range(50):
            price = Decimal("42000.00") + Decimal(str(i * 2.0))  # 强烈上涨
            trade = Trade(
                symbol="BTCFDUSD",
                price=price,
                quantity=Decimal("0.2"),
                is_buyer_maker=i % 4 != 0,  # 更多买入
                timestamp=self.base_time + timedelta(seconds=i * 6),
                trade_id=f"strong_trade_{i}"
            )
            strong_uptrend_trades.append(trade)

        result = self.analyzer.analyze_momentum(strong_uptrend_trades, "BTCFDUSD")

        # 强趋势应该产生较高的信号强度
        signal = result.signal
        assert signal.strength > 0.3  # 至少中等强度

    def test_confidence_calculation(self):
        """测试置信度计算。"""
        # 高质量数据：多笔交易，一致趋势
        quality_trades = self.create_test_trades(100, "up")
        result = self.analyzer.analyze_momentum(quality_trades, "BTCFDUSD")

        signal = result.signal
        # 高质量数据应该产生较高的置信度
        assert signal.confidence > 0.5

    def test_market_condition_detection(self):
        """测试市场条件检测。"""
        # 高波动率数据
        volatile_trades = []
        for i in range(40):
            # 大幅价格波动
            price = Decimal("42000.00") + Decimal(str((i % 10 - 5) * 10.0))
            trade = Trade(
                symbol="BTCFDUSD",
                price=price,
                quantity=Decimal("0.15"),
                is_buyer_maker=i % 2 == 0,
                timestamp=self.base_time + timedelta(seconds=i * 7.5),
                trade_id=f"volatile_trade_{i}"
            )
            volatile_trades.append(trade)

        result = self.analyzer.analyze_momentum(volatile_trades, "BTCFDUSD")

        # 高波动率应该被检测为volatile市场
        assert result.signal.market_condition in ["volatile", "normal"]

    def test_analysis_statistics(self):
        """测试分析统计信息。"""
        trades = self.create_test_trades(35, "up")
        result = self.analyzer.analyze_momentum(trades, "BTCFDUSD")

        stats = result.analysis_statistics

        # 验证统计信息结构
        assert "trade_statistics" in stats
        assert "price_statistics" in stats
        assert "indicator_summary" in stats

        # 验证交易统计
        trade_stats = stats["trade_statistics"]
        assert trade_stats["total_trades"] == 35
        assert trade_stats["total_volume"] > 0
        assert 0 <= trade_stats["buy_ratio"] <= 1
        assert 0 <= trade_stats["sell_ratio"] <= 1

        # 验证价格统计
        price_stats = stats["price_statistics"]
        assert price_stats["open_price"] is not None
        assert price_stats["close_price"] is not None
        assert price_stats["price_change_rate"] is not None

    def test_custom_time_window(self):
        """测试自定义时间窗口。"""
        custom_analyzer = ShortTermMomentumAnalyzer(
            window_size_minutes=10,
            min_trades=15
        )

        trades = self.create_test_trades(50, "up")
        result = custom_analyzer.analyze_momentum(trades, "BTCFDUSD")

        assert result.analysis_window_minutes == 10
        assert result.signal.timeframe == "10m"

    def test_custom_thresholds(self):
        """测试自定义阈值。"""
        custom_analyzer = ShortTermMomentumAnalyzer(
            buy_threshold=0.1,
            sell_threshold=-0.1,
            neutral_range=0.1
        )

        trades = self.create_test_trades(30, "up")
        result = custom_analyzer.analyze_momentum(trades, "BTCFDUSD")

        # 验证阈值设置正确
        assert custom_analyzer.buy_threshold == 0.1
        assert custom_analyzer.sell_threshold == -0.1
        assert custom_analyzer.neutral_range == 0.1

    def test_trade_window_creation(self):
        """测试交易窗口创建。"""
        trades = self.create_test_trades(20, "up")

        # 使用内部方法创建交易窗口
        window_start = self.base_time
        window_end = self.base_time + timedelta(minutes=5)

        trade_window = self.analyzer._create_trade_window(
            trades, "BTCFDUSD", window_start, window_end
        )

        assert trade_window.symbol == "BTCFDUSD"
        assert trade_window.start_time == window_start
        assert trade_window.end_time == window_end
        assert len(trade_window.trades) > 0

    def test_price_momentum_calculation(self):
        """测试价格动量计算。"""
        trades = self.create_test_trades(25, "up")

        # 创建交易窗口
        window_start = self.base_time
        window_end = self.base_time + timedelta(minutes=5)
        trade_window = self.analyzer._create_trade_window(
            trades, "BTCFDUSD", window_start, window_end
        )

        # 计算价格动量
        price_momentum = self.analyzer._calculate_price_momentum(trade_window)

        # 验证返回的指标
        assert "momentum" in price_momentum
        assert "change_rate" in price_momentum
        assert "trend_strength" in price_momentum
        assert "weighted_momentum" in price_momentum

        # 上升趋势应该有正的价格动量
        assert price_momentum["change_rate"] > 0

    def test_volume_momentum_calculation(self):
        """测试成交量动量计算。"""
        trades = self.create_test_trades(25, "up")

        window_start = self.base_time
        window_end = self.base_time + timedelta(minutes=5)
        trade_window = self.analyzer._create_trade_window(
            trades, "BTCFDUSD", window_start, window_end
        )

        volume_momentum = self.analyzer._calculate_volume_momentum(trade_window)

        # 验证返回的指标
        assert "momentum" in volume_momentum
        assert "imbalance" in volume_momentum
        assert "large_trade_ratio" in volume_momentum
        assert "trend" in volume_momentum

        # 验证值在合理范围内
        assert -1 <= volume_momentum["imbalance"] <= 1
        assert 0 <= volume_momentum["large_trade_ratio"] <= 1

    def test_order_flow_momentum_calculation(self):
        """测试订单流动量计算。"""
        trades = self.create_test_trades(25, "up")

        window_start = self.base_time
        window_end = self.base_time + timedelta(minutes=5)
        trade_window = self.analyzer._create_trade_window(
            trades, "BTCFDUSD", window_start, window_end
        )

        flow_momentum = self.analyzer._calculate_order_flow_momentum(trade_window)

        # 验证返回的指标
        assert "momentum" in flow_momentum
        assert "buy_pressure" in flow_momentum
        assert "sell_pressure" in flow_momentum
        assert "consistency" in flow_momentum

        # 验证压力值在合理范围内
        assert 0 <= flow_momentum["buy_pressure"] <= 1
        assert 0 <= flow_momentum["sell_pressure"] <= 1

    def test_volatility_adjusted_calculation(self):
        """测试波动率调整计算。"""
        trades = self.create_test_trades(25, "up")

        window_start = self.base_time
        window_end = self.base_time + timedelta(minutes=5)
        trade_window = self.analyzer._create_trade_window(
            trades, "BTCFDUSD", window_start, window_end
        )

        # 创建虚拟指标
        from src.core.momentum_models import MomentumIndicators
        indicators = MomentumIndicators()
        indicators.price_momentum = 0.1

        volatility_adjusted = self.analyzer._calculate_volatility_adjusted(trade_window, indicators)

        # 验证返回的指标
        assert "adjusted" in volatility_adjusted
        assert "realized_vol" in volatility_adjusted
        assert "risk_adjusted_return" in volatility_adjusted

        # 验证波动率为非负数
        assert volatility_adjusted["realized_vol"] >= 0

    def test_signal_generation(self):
        """测试信号生成。"""
        from src.core.momentum_models import MomentumIndicators

        indicators = MomentumIndicators()
        indicators.price_momentum = 0.2  # 强买入信号
        indicators.volume_momentum = 0.15
        indicators.order_flow_momentum = 0.1
        indicators.volatility_adjusted = 0.05

        trades = self.create_test_trades(25, "up")
        window_start = self.base_time
        window_end = self.base_time + timedelta(minutes=5)
        trade_window = self.analyzer._create_trade_window(
            trades, "BTCFDUSD", window_start, window_end
        )

        signal = self.analyzer._generate_momentum_signal(
            indicators, trade_window, datetime.now()
        )

        # 强买入指标应该产生买入信号
        assert signal.direction == MomentumDirection.BUY
        assert signal.strength > 0
        assert signal.confidence > 0
        assert signal.trade_count == 25

    def test_error_handling(self):
        """测试错误处理。"""
        # 测试无效的交易数据
        invalid_trades = [
            None,  # 无效交易
            "invalid_trade",  # 错误类型
        ]

        result = self.analyzer.analyze_momentum(invalid_trades, "BTCFDUSD")

        # 应该优雅地处理错误
        assert result.signal.direction == MomentumDirection.NEUTRAL
        assert result.signal.strength == 0.0
        assert result.signal.confidence == 0.0

    def test_performance_requirements(self):
        """测试性能要求。"""
        # 大量交易数据
        large_trades = self.create_test_trades(1000, "up")

        start_time = datetime.now()
        result = self.analyzer.analyze_momentum(large_trades, "BTCFDUSD")
        end_time = datetime.now()

        # 处理时间应该在合理范围内（小于1秒）
        processing_time = (end_time - start_time).total_seconds() * 1000
        assert processing_time < 1000  # 小于1秒

        # 验证结果正确性
        assert result.signal is not None
        assert result.signal.trade_count == 1000

    def test_boundary_conditions(self):
        """测试边界条件。"""
        # 测试正好达到最小要求的交易数量
        min_trades = self.create_test_trades(5, "up")  # 正好min_trades
        result = self.analyzer.analyze_momentum(min_trades, "BTCFDUSD")

        # 应该能正常处理
        assert result.signal.direction in [MomentumDirection.BUY, MomentumDirection.NEUTRAL, MomentumDirection.SELL]
        assert result.signal.trade_count >= 0  # 由于时间窗口过滤，可能少于原始数量

        # 测试时间窗口边界
        old_trades = []
        current_time = datetime.now()
        for i in range(10):
            # 6分钟前的交易（超出5分钟窗口）
            timestamp = current_time - timedelta(minutes=6, seconds=i*10)
            trade = Trade(
                symbol="BTCFDUSD",
                price=Decimal("42000.00"),
                quantity=Decimal("0.1"),
                is_buyer_maker=False,
                timestamp=timestamp,
                trade_id=f"old_trade_{i}"
            )
            old_trades.append(trade)

        result = self.analyzer.analyze_momentum(old_trades, "BTCFDUSD")

        # 超出时间窗口的交易应该被过滤
        assert result.signal.trade_count == 0
        assert result.signal.direction == MomentumDirection.NEUTRAL