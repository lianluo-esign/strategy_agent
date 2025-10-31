#!/usr/bin/env python3
"""快速测试精确定时动量策略。"""

import asyncio
import sys
import time
from datetime import datetime
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))

from src.core.redis_client import RedisDataStore
from src.core.order_flow_momentum_analyzer import OrderFlowMomentumAnalyzer


async def quick_test():
    """快速测试定时策略逻辑。"""
    print("🚀 快速测试精确定时动量策略")

    # 初始化Redis和分析器
    redis_store = RedisDataStore()
    analyzer = OrderFlowMomentumAnalyzer(window_size_minutes=3)

    # 测试Redis连接
    if not redis_store.test_connection():
        print("❌ Redis连接失败")
        return

    print("✅ Redis连接成功")

    # 加载最近3分钟数据
    print("📊 加载最近3分钟数据...")
    minute_data = redis_store.get_recent_trade_data(minutes=3)

    if not minute_data:
        print("❌ 没有找到数据")
        return

    print(f"✅ 成功加载 {len(minute_data)} 个时间点的数据")

    # 显示数据时间戳
    for i, data in enumerate(minute_data):
        print(f"  数据点 {i+1}: {data.timestamp.strftime('%H:%M:%S')}")

    # 执行分析
    print("🔍 执行动量分析...")
    try:
        result = analyzer.analyze_order_flow_momentum(minute_data, "BTCFDUSD")

        print(f"\n🎯 分析结果:")
        print(f"  信号方向: {result.signal.direction.value.upper()}")
        print(f"  信号强度: {result.signal.strength:.3f}")
        print(f"  置信度: {result.signal.confidence:.3f}")
        print(f"  原始分数: {result.signal.raw_score:.4f}")
        print(f"  交易数量: {result.signal.trade_count}")
        print(f"  处理时间: {result.processing_time_ms:.2f}ms")

        print(f"\n📊 关键指标:")
        indicators = result.signal.indicators
        print(f"  订单流动量: {indicators.order_flow_momentum:+.4f}")
        print(f"  成交量不平衡: {indicators.volume_imbalance:+.4f}")
        print(f"  价格动量: {indicators.price_momentum:+.4f}")
        print(f"  趋势强度: {indicators.trend_strength:+.4f}")

        print(f"\n✅ 分析成功完成！")

    except Exception as e:
        print(f"❌ 分析失败: {e}")
        import traceback
        traceback.print_exc()

    # 关闭连接
    await redis_store.close()


if __name__ == "__main__":
    asyncio.run(quick_test())