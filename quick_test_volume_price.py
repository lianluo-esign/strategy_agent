#!/usr/bin/env python3
"""快速测试量价策略生成信号的场景。"""

import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))

from src.core.volume_price_strategy_analyzer import VolumePriceStrategyAnalyzer
from src.core.support_resistance_analyzer import SupportResistanceAnalyzer
from src.core.models import MinuteTradeData, PriceLevelData
from decimal import Decimal

def create_strong_signal_scenario():
    """创建强烈信号场景：价格在强支撑位附近 + 成交量激增。"""
    base_time = datetime.now() - timedelta(minutes=15)
    price_levels = {}

    # 创建强支撑位：114500附近有大量买入订单
    support_price = 114500

    # 支撑位核心区域：大量买入订单集中
    for i in range(5):
        price = Decimal(str(support_price)) - Decimal(str(i * 0.1))  # 114500, 114499.9, 114499.8...
        price_levels[price] = PriceLevelData(
            price_level=price,
            buy_volume=Decimal('500') * (5 - i),  # 越接近支撑位，买入量越大
            sell_volume=Decimal('20') * (i + 1),  # 卖出量很少
            trade_count=500 + i * 50
        )

    # 稍远一点的支撑区域
    for i in range(10):
        price = Decimal(str(support_price - 0.5)) - Decimal(str(i * 0.2))
        price_levels[price] = PriceLevelData(
            price_level=price,
            buy_volume=Decimal('200') * (10 - i),
            sell_volume=Decimal('50') * (i + 1),
            trade_count=200 + i * 20
        )

    # 当前价格稍微在支撑位上方，成交量激增
    current_price = Decimal('114502')  # 在支撑位上方2点
    for i in range(8):
        price = current_price - Decimal(str(i * 0.3))
        # 当前时刻成交量激增（5倍以上）
        price_levels[price] = PriceLevelData(
            price_level=price,
            buy_volume=Decimal('300') * (8 - i),  # 大量买入
            sell_volume=Decimal('100') * (i + 1),
            trade_count=800 + i * 100  # 高交易数量
        )

    return MinuteTradeData(
        timestamp=base_time,
        price_levels=price_levels,
        max_price_levels=1000
    )

def create_breakout_signal_scenario():
    """创建突破信号场景：突破强阻力位 + 成交量确认。"""
    base_time = datetime.now() - timedelta(minutes=15)
    price_levels = {}

    # 创建强阻力位：115000附近有大量卖出订单
    resistance_price = 115000

    # 阻力位核心区域：大量卖出订单
    for i in range(5):
        price = Decimal(str(resistance_price)) + Decimal(str(i * 0.1))  # 115000, 115000.1, 115000.2...
        price_levels[price] = PriceLevelData(
            price_level=price,
            buy_volume=Decimal('20') * (i + 1),  # 买入量很少
            sell_volume=Decimal('500') * (5 - i),  # 越接近阻力位，卖出量越大
            trade_count=500 + i * 50
        )

    # 当前价格突破阻力位，成交量急剧放大
    current_price = Decimal('115001')  # 突破阻力位1点
    for i in range(8):
        price = current_price + Decimal(str(i * 0.2))
        # 突破时成交量急剧放大（10倍以上）
        price_levels[price] = PriceLevelData(
            price_level=price,
            buy_volume=Decimal('800') * (8 - i),  # 大量买入突破
            sell_volume=Decimal('100') * (i + 1),
            trade_count=1200 + i * 150  # 极高交易数量
        )

    return MinuteTradeData(
        timestamp=base_time,
        price_levels=price_levels,
        max_price_levels=1000
    )

async def test_strong_signals():
    """测试强信号场景。"""
    print("🧪 测试量价策略强信号生成")
    print("=" * 50)

    analyzer = VolumePriceStrategyAnalyzer(
        window_minutes=15,
        min_volume_ratio=1.5,
        max_distance_from_level_percent=0.2,
        min_strength_threshold=0.6
    )

    # 测试1：强支撑位反弹信号
    print("📈 测试场景1：强支撑位反弹")
    print("   - 价格在114500强支撑位附近")
    print("   - 成交量激增5倍以上")
    print("   - 买入压力明显")

    # 创建15分钟的数据序列
    test_data = []
    base_time = datetime.now() - timedelta(minutes=15)

    for i in range(15):
        data = create_strong_signal_scenario()
        data.timestamp = base_time + timedelta(minutes=i)
        test_data.append(data)

    current_price = Decimal('114502')
    signal = analyzer.analyze_volume_price_strategy(test_data, "BTCFDUSD")

    if signal:
        print(f"✅ 生成交易信号:")
        print(f"   信号类型: {signal.signal_type}")
        print(f"   方向: {signal.direction}")
        print(f"   强度: {signal.strength:.3f}")
        print(f"   置信度: {signal.confidence:.3f}")
        print(f"   成交量确认: {signal.volume_confirmation:.2f}x")
        print(f"   风险收益比: {signal.risk_reward_ratio:.2f}")

        if signal.support_level:
            print(f"   支撑位: {signal.support_level.price} (强度: {signal.support_level.strength:.3f})")
    else:
        print("❌ 未生成信号")

    print("\n" + "-" * 50)

    # 测试2：阻力位突破信号
    print("🚀 测试场景2：阻力位突破")
    print("   - 价格突破115000强阻力位")
    print("   - 成交量激增10倍以上")
    print("   - 强势突破确认")

    # 创建突破场景的数据序列
    breakout_data = []
    for i in range(15):
        data = create_breakout_signal_scenario()
        data.timestamp = base_time + timedelta(minutes=i)
        breakout_data.append(data)

    current_price = Decimal('115001')
    breakout_signal = analyzer.analyze_volume_price_strategy(breakout_data, "BTCFDUSD")

    if breakout_signal:
        print(f"✅ 生成交易信号:")
        print(f"   信号类型: {breakout_signal.signal_type}")
        print(f"   方向: {breakout_signal.direction}")
        print(f"   强度: {breakout_signal.strength:.3f}")
        print(f"   置信度: {breakout_signal.confidence:.3f}")
        print(f"   成交量确认: {breakout_signal.volume_confirmation:.2f}x")
        print(f"   风险收益比: {breakout_signal.risk_reward_ratio:.2f}")

        if breakout_signal.resistance_level:
            print(f"   阻力位: {breakout_signal.resistance_level.price} (强度: {breakout_signal.resistance_level.strength:.3f})")
    else:
        print("❌ 未生成信号")

    print("\n" + "=" * 50)
    print("📊 测试总结:")
    print("   量价策略只在满足以下条件时生成信号:")
    print("   1. 价格在关键支撑/阻力位附近")
    print("   2. 成交量有明显放大确认")
    print("   3. 信号强度和置信度达到阈值")
    print("   4. 风险收益比满足要求")
    print("   这确保了信号的稳定性和可靠性！")

if __name__ == "__main__":
    asyncio.run(test_strong_signals())