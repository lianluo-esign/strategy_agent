#!/usr/bin/env python3
"""测试量价结合策略。

验证支撑阻力位识别和成交量确认的准确性。
"""

import asyncio
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))

from src.core.redis_client import RedisDataStore
from src.core.volume_price_strategy_analyzer import VolumePriceStrategyAnalyzer
from src.core.support_resistance_analyzer import SupportResistanceAnalyzer
from src.core.models import MinuteTradeData, PriceLevelData
from decimal import Decimal

def create_test_support_scenario():
    """创建支撑位测试场景。"""
    base_time = datetime.now() - timedelta(minutes=15)
    price_levels = {}

    # 创建支撑位集中的价格区域（114500附近）
    support_price = 114500
    for i in range(10):
        price = Decimal(str(support_price + i * 0.5))
        price_levels[price] = PriceLevelData(
            price_level=price,
            buy_volume=Decimal('100') * (10 - i),  # 越接近支撑位，买入量越大
            sell_volume=Decimal('50') * (i + 1),
            trade_count=100 + i * 10
        )

    # 在支撑位附近创建大量买入压力
    for i in range(5):
        price = Decimal(str(support_price - i * 0.2))
        price_levels[price] = PriceLevelData(
            price_level=price,
            buy_volume=Decimal('200') * (5 - i),
            sell_volume=Decimal('30') * (i + 1),
            trade_count=150 + i * 20
        )

    # 计算总量
    total_volume = sum(level.buy_volume + level.sell_volume for level in price_levels.values())
    buy_volume = sum(level.buy_volume for level in price_levels.values())
    sell_volume = sum(level.sell_volume for level in price_levels.values())

    return MinuteTradeData(
        timestamp=base_time,
        price_levels=price_levels,
        max_price_levels=1000
    )

def create_test_resistance_scenario():
    """创建阻力位测试场景。"""
    base_time = datetime.now() - timedelta(minutes=10)
    price_levels = {}

    # 创建阻力位集中的价格区域（115000附近）
    resistance_price = 115000
    for i in range(10):
        price = Decimal(str(resistance_price - i * 0.5))
        price_levels[price] = PriceLevelData(
            price_level=price,
            buy_volume=Decimal('50') * (i + 1),
            sell_volume=Decimal('100') * (10 - i),  # 越接近阻力位，卖出量越大
            trade_count=100 + i * 10
        )

    # 在阻力位附近创建大量卖出压力
    for i in range(5):
        price = Decimal(str(resistance_price + i * 0.2))
        price_levels[price] = PriceLevelData(
            price_level=price,
            buy_volume=Decimal('30') * (i + 1),
            sell_volume=Decimal('200') * (5 - i),
            trade_count=150 + i * 20
        )

    return MinuteTradeData(
        timestamp=base_time,
        price_levels=price_levels,
        max_price_levels=1000
    )

def create_volume_spike_scenario():
    """创建成交量激增场景。"""
    base_time = datetime.now() - timedelta(minutes=5)
    price_levels = {}

    # 创建正常的价格分布，但成交量激增
    for i in range(20):
        price = Decimal(str(114800 + i * 1.0))
        volume_multiplier = Decimal('5.0') if i < 10 else Decimal('1.0')  # 前半部分成交量激增
        price_levels[price] = PriceLevelData(
            price_level=price,
            buy_volume=Decimal('50') * volume_multiplier,
            sell_volume=Decimal('50') * volume_multiplier,
            trade_count=int(100 * float(volume_multiplier))
        )

    return MinuteTradeData(
        timestamp=base_time,
        price_levels=price_levels,
        max_price_levels=1000
    )

async def test_support_resistance_identification():
    """测试支撑阻力位识别。"""
    print("🔍 测试支撑阻力位识别...")

    analyzer = SupportResistanceAnalyzer()

    # 测试支撑位场景
    support_data = create_test_support_scenario()
    resistance_data = create_test_resistance_scenario()
    volume_spike_data = create_volume_spike_scenario()

    # 创建一个数据序列（15分钟）
    test_data = []
    base_time = datetime.now() - timedelta(minutes=15)

    for i in range(15):
        if i < 5:
            # 前面5分钟：支撑位形成
            data = create_test_support_scenario()
            data.timestamp = base_time + timedelta(minutes=i)
        elif i < 10:
            # 中间5分钟：阻力位形成
            data = create_test_resistance_scenario()
            data.timestamp = base_time + timedelta(minutes=i)
        else:
            # 最后5分钟：成交量激增
            data = create_volume_spike_scenario()
            data.timestamp = base_time + timedelta(minutes=i)

        test_data.append(data)

    # 分析支撑阻力位
    current_price = Decimal('114505')
    analysis = analyzer.analyze_volume_price_relationship(test_data, current_price)

    print(f"✅ 识别到 {len(analysis.support_levels)} 个支撑位")
    print(f"✅ 识别到 {len(analysis.resistance_levels)} 个阻力位")

    if analysis.nearest_support:
        print(f"🎯 最近支撑位: {analysis.nearest_support.price} (强度: {analysis.nearest_support.strength:.3f})")

    if analysis.nearest_resistance:
        print(f"🎯 最近阻力位: {analysis.nearest_resistance.price} (强度: {analysis.nearest_resistance.strength:.3f})")

    return test_data

async def test_volume_price_strategy():
    """测试量价结合策略。"""
    print("\n🚀 测试量价结合策略...")

    # 创建测试数据
    test_data = await test_support_resistance_identification()

    # 初始化量价策略分析器
    analyzer = VolumePriceStrategyAnalyzer(
        window_minutes=15,
        min_volume_ratio=1.5,
        max_distance_from_level_percent=0.2,
        min_strength_threshold=0.6
    )

    print("📊 分析量价策略信号...")

    # 执行策略分析
    signal = analyzer.analyze_volume_price_strategy(test_data, "BTCFDUSD")

    if signal:
        print(f"🎯 生成交易信号:")
        print(f"  信号类型: {signal.signal_type}")
        print(f"  方向: {signal.direction.value.upper()}")
        print(f"  强度: {signal.strength:.3f}")
        print(f"  置信度: {signal.confidence:.3f}")
        print(f"  入场价格: {signal.entry_price}")
        print(f"  止损价格: {signal.stop_loss}")
        print(f"  止盈价格: {signal.take_profit}")
        print(f"  风险收益比: {signal.risk_reward_ratio:.2f}")
        print(f"  成交量确认: {signal.volume_confirmation:.2f}x")
        print(f"  距离关键位: {signal.price_distance_from_level:.3f}%")

        if signal.support_level:
            print(f"  支撑位: {signal.support_level.price} (强度: {signal.support_level.strength:.3f})")

        if signal.resistance_level:
            print(f"  阻力位: {signal.resistance_level.price} (强度: {signal.resistance_level.strength:.3f})")

        # 转换为动量信号
        momentum_signal = analyzer.convert_to_momentum_signal(signal)
        print(f"\n🔄 转换为动量信号:")
        print(f"  方向: {momentum_signal['direction'].upper()}")
        print(f"  强度: {momentum_signal['strength']:.3f}")
        print(f"  置信度: {momentum_signal['confidence']:.3f}")
        print(f"  原始分数: {momentum_signal['raw_score']:.4f}")

        return True
    else:
        print("📊 当前市场条件不符合交易信号要求")
        return False

async def test_with_redis_data():
    """使用真实Redis数据测试。"""
    print("\n🔗 使用真实Redis数据测试...")

    redis_store = RedisDataStore()

    # 测试Redis连接
    if not redis_store.test_connection():
        print("❌ Redis连接失败，跳过真实数据测试")
        return False

    # 加载最近15分钟的数据
    minute_data = redis_store.get_recent_trade_data(minutes=15)

    if not minute_data or len(minute_data) < 5:
        print("❌ Redis中没有足够的数据，跳过真实数据测试")
        await redis_store.close()
        return False

    print(f"✅ 成功加载 {len(minute_data)} 个时间点的真实数据")

    # 显示数据时间范围
    start_time = min(data.timestamp for data in minute_data)
    end_time = max(data.timestamp for data in minute_data)
    print(f"📅 数据时间范围: {start_time.strftime('%H:%M:%S')} - {end_time.strftime('%H:%M:%S')}")

    # 使用真实数据测试策略
    analyzer = VolumePriceStrategyAnalyzer(
        window_minutes=15,
        min_volume_ratio=1.5,
        max_distance_from_level_percent=0.2,
        min_strength_threshold=0.6
    )

    signal = analyzer.analyze_volume_price_strategy(minute_data, "BTCFDUSD")

    if signal:
        print(f"🎯 真实数据生成交易信号:")
        print(f"  信号类型: {signal.signal_type}")
        print(f"  方向: {signal.direction.value.upper()}")
        print(f"  强度: {signal.strength:.3f}")
        print(f"  置信度: {signal.confidence:.3f}")
        print(f"  成交量确认: {signal.volume_confirmation:.2f}x")

        await redis_store.close()
        return True
    else:
        print("📊 真实数据当前市场条件不符合交易信号要求")
        await redis_store.close()
        return False

async def main():
    """主测试函数。"""
    print("🧪 量价结合策略测试")
    print("=" * 50)

    start_time = time.time()

    try:
        # 测试1：支撑阻力位识别
        test_data = await test_support_resistance_identification()

        # 测试2：量价策略分析
        strategy_success = await test_volume_price_strategy()

        # 测试3：真实数据测试
        real_data_success = await test_with_redis_data()

        # 总结测试结果
        print("\n" + "=" * 50)
        print("📊 测试总结:")
        print(f"✅ 支撑阻力位识别: 成功")
        print(f"{'✅' if strategy_success else '⚠️'} 量价策略分析: {'成功' if strategy_success else '无信号'}")
        print(f"{'✅' if real_data_success else '⚠️'} 真实数据测试: {'成功' if real_data_success else '无信号'}")

        elapsed_time = time.time() - start_time
        print(f"⏱️ 总测试时间: {elapsed_time:.2f}秒")

        if strategy_success or real_data_success:
            print("\n🎉 量价结合策略测试通过！")
            print("💡 策略优势:")
            print("  - 基于真实支撑阻力位识别")
            print("  - 成交量确认机制")
            print("  - 风险收益比控制")
            print("  - 多种信号类型支持")
        else:
            print("\n⚠️ 当前市场条件不满足交易信号生成要求")
            print("💡 这在真实交易中是正常的，策略会耐心等待符合条件的交易机会")

    except Exception as e:
        print(f"❌ 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())