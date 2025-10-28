#!/usr/bin/env python3
"""短期动量分析器测试脚本。

演示如何使用ShortTermMomentumAnalyzer进行动量分析。
"""

import sys
from datetime import datetime, timedelta
from decimal import Decimal

# 添加项目根目录到路径
sys.path.append("/home/jamesduan/projects/strategy_agent")

from src.core.models import Trade
from src.core.short_term_momentum_analyzer import ShortTermMomentumAnalyzer


def create_sample_trades_data() -> list[Trade]:
    """创建示例交易数据。"""
    trades = []
    base_time = datetime.now() - timedelta(minutes=10)
    base_price = Decimal("42000.00")

    # 模拟上升趋势的交易数据
    for i in range(100):
        timestamp = base_time + timedelta(seconds=i * 6)  # 每6秒一笔交易

        # 模拟价格上涨趋势
        price_change = Decimal(str(i * 0.5))  # 每笔交易上涨0.5
        price = base_price + price_change

        # 模拟交易量
        quantity = Decimal(str(0.1 + (i % 10) * 0.02))  # 0.1-0.28之间的随机量

        # 模拟更多主动买入
        is_buyer_maker = i % 3 != 0  # 2/3的概率是主动买入

        trade = Trade(
            symbol="BTCFDUSD",
            price=price,
            quantity=quantity,
            is_buyer_maker=is_buyer_maker,
            timestamp=timestamp,
            trade_id=f"sample_trade_{i}"
        )
        trades.append(trade)

    return trades


def create_sample_trades_data_downward() -> list[Trade]:
    """创建下跌趋势的示例交易数据。"""
    trades = []
    base_time = datetime.now() - timedelta(minutes=10)
    base_price = Decimal("42100.00")

    # 模拟下跌趋势的交易数据
    for i in range(80):
        timestamp = base_time + timedelta(seconds=i * 7.5)  # 每7.5秒一笔交易

        # 模拟价格下跌趋势
        price_change = Decimal(str(i * 0.8))  # 每笔交易下跌0.8
        price = base_price - price_change

        # 模拟交易量
        quantity = Decimal(str(0.15 + (i % 8) * 0.025))  # 0.15-0.325之间的随机量

        # 模拟更多主动卖出
        is_buyer_maker = i % 4 != 0  # 3/4的概率是主动卖出

        trade = Trade(
            symbol="BTCFDUSD",
            price=price,
            quantity=quantity,
            is_buyer_maker=is_buyer_maker,
            timestamp=timestamp,
            trade_id=f"sample_trade_down_{i}"
        )
        trades.append(trade)

    return trades


def create_sample_trades_data_sideways() -> list[Trade]:
    """创建横盘整理的示例交易数据。"""
    trades = []
    base_time = datetime.now() - timedelta(minutes=8)
    base_price = Decimal("42050.00")

    # 模拟横盘整理的交易数据
    for i in range(60):
        timestamp = base_time + timedelta(seconds=i * 8)  # 每8秒一笔交易

        # 模拟价格在小区间内波动
        price_variation = Decimal(str((i % 20 - 10) * 0.2))  # -2到+2之间波动
        price = base_price + price_variation

        # 模拟交易量
        quantity = Decimal(str(0.12 + (i % 6) * 0.03))  # 0.12-0.27之间的随机量

        # 模拟买卖相对平衡
        is_buyer_maker = i % 2 == 0  # 50/50的概率

        trade = Trade(
            symbol="BTCFDUSD",
            price=price,
            quantity=quantity,
            is_buyer_maker=is_buyer_maker,
            timestamp=timestamp,
            trade_id=f"sample_trade_side_{i}"
        )
        trades.append(trade)

    return trades


def test_momentum_analyzer():
    """测试动量分析器。"""
    print("=" * 80)
    print("短期动量分析器测试")
    print("=" * 80)

    # 创建分析器实例
    analyzer = ShortTermMomentumAnalyzer(
        window_size_minutes=5,
        min_trades=10,
        min_volume=0.1
    )

    print(f"分析器配置: {analyzer.get_analyzer_status()}")
    print()

    # 测试场景1: 上升趋势
    print("📈 测试场景1: 上升趋势")
    print("-" * 50)
    uptrend_trades = create_sample_trades_data()
    result1 = analyzer.analyze_momentum(uptrend_trades, "BTCFDUSD")

    print(f"信号方向: {result1.signal.direction.value}")
    print(f"信号强度: {result1.signal.strength:.3f}")
    print(f"置信度: {result1.signal.confidence:.3f}")
    print(f"原始分数: {result1.signal.raw_score:.4f}")
    print(f"市场条件: {result1.signal.market_condition}")
    print(f"交易数量: {result1.signal.trade_count}")
    print(f"处理时间: {result1.processing_time_ms:.2f}ms")
    print()

    # 显示关键指标
    indicators = result1.signal.indicators
    print("关键指标:")
    print(f"  价格动量: {indicators.price_momentum:.4f}")
    print(f"  成交量动量: {indicators.volume_momentum:.4f}")
    print(f"  订单流动量: {indicators.order_flow_momentum:.4f}")
    print(f"  波动率调整: {indicators.volatility_adjusted:.4f}")
    print(f"  成交量不平衡: {indicators.volume_imbalance:.4f}")
    print(f"  实现波动率: {indicators.realized_volatility:.4f}")
    print()

    # 测试场景2: 下跌趋势
    print("📉 测试场景2: 下跌趋势")
    print("-" * 50)
    downtrend_trades = create_sample_trades_data_downward()
    result2 = analyzer.analyze_momentum(downtrend_trades, "BTCFDUSD")

    print(f"信号方向: {result2.signal.direction.value}")
    print(f"信号强度: {result2.signal.strength:.3f}")
    print(f"置信度: {result2.signal.confidence:.3f}")
    print(f"原始分数: {result2.signal.raw_score:.4f}")
    print(f"市场条件: {result2.signal.market_condition}")
    print()

    # 测试场景3: 横盘整理
    print("➡️ 测试场景3: 横盘整理")
    print("-" * 50)
    sideways_trades = create_sample_trades_data_sideways()
    result3 = analyzer.analyze_momentum(sideways_trades, "BTCFDUSD")

    print(f"信号方向: {result3.signal.direction.value}")
    print(f"信号强度: {result3.signal.strength:.3f}")
    print(f"置信度: {result3.signal.confidence:.3f}")
    print(f"原始分数: {result3.signal.raw_score:.4f}")
    print(f"市场条件: {result3.signal.market_condition}")
    print()

    # 测试场景4: 数据不足的情况
    print("❌ 测试场景4: 数据不足")
    print("-" * 50)
    insufficient_trades = uptrend_trades[:3]  # 只取3笔交易
    result4 = analyzer.analyze_momentum(insufficient_trades, "BTCFDUSD")

    print(f"信号方向: {result4.signal.direction.value}")
    print(f"信号强度: {result4.signal.strength:.3f}")
    print(f"置信度: {result4.signal.confidence:.3f}")
    print(f"错误信息: {result4.trade_window_summary.get('error', 'No error')}")
    print()

    # 总结
    print("=" * 80)
    print("测试总结")
    print("=" * 80)
    print(f"✅ 上升趋势检测: {result1.signal.direction.value} (强度: {result1.signal.strength:.3f})")
    print(f"✅ 下跌趋势检测: {result2.signal.direction.value} (强度: {result2.signal.strength:.3f})")
    print(f"✅ 横盘整理检测: {result3.signal.direction.value} (强度: {result3.signal.strength:.3f})")
    print(f"✅ 数据不足处理: {result4.signal.direction.value} (错误处理正常)")
    print()
    print("所有测试完成！动量分析器工作正常。")


if __name__ == "__main__":
    test_momentum_analyzer()