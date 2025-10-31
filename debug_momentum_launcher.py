#!/usr/bin/env python3
"""调试短期动量策略启动器。"""

import asyncio
import sys
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))

from src.core.models import Trade
from src.core.short_term_momentum_analyzer import ShortTermMomentumAnalyzer


async def debug_momentum_analysis():
    """调试动量分析过程。"""
    print("🔍 调试短期动量分析器")
    print("=" * 60)

    # 创建分析器
    analyzer = ShortTermMomentumAnalyzer(
        window_size_minutes=5,
        min_trades=10,
        min_volume=0.01
    )

    # 创建测试交易数据
    trades = []
    current_time = datetime.now()
    base_price = Decimal("42000.00")

    print(f"📅 当前时间: {current_time}")
    print(f"📊 分析窗口: 5分钟")

    # 生成在5分钟窗口内的交易数据
    for i in range(20):
        # 确保交易在5分钟窗口内
        timestamp = current_time - timedelta(minutes=4, seconds=i * 15)

        price = base_price + Decimal(str(i * 0.5))
        quantity = Decimal("0.1")
        is_buyer_maker = i % 3 != 0

        trade = Trade(
            symbol="BTCFDUSD",
            price=price,
            quantity=quantity,
            is_buyer_maker=is_buyer_maker,
            timestamp=timestamp,
            trade_id=f"debug_trade_{i}"
        )
        trades.append(trade)

        print(f"  交易 {i}: {timestamp} - 价格: {price} - 买入: {not is_buyer_maker}")

    print(f"\n📈 生成了 {len(trades)} 笔交易")

    # 计算时间窗口
    window_start = current_time - timedelta(minutes=5)
    window_end = current_time

    print(f"⏰ 分析窗口: {window_start} 到 {window_end}")

    # 检查哪些交易在窗口内
    valid_trades = []
    for i, trade in enumerate(trades):
        if window_start <= trade.timestamp <= window_end:
            valid_trades.append(trade)
            print(f"  ✅ 交易 {i}: 在窗口内")
        else:
            print(f"  ❌ 交易 {i}: 超出窗口")

    print(f"\n✅ 有效交易数量: {len(valid_trades)}")

    if len(valid_trades) >= 10:
        print("🚀 开始动量分析...")

        # 执行分析
        result = analyzer.analyze_momentum(trades, "BTCFDUSD")

        # 显示结果
        signal = result.signal
        print(f"\n📊 分析结果:")
        print(f"  方向: {signal.direction.value}")
        print(f"  强度: {signal.strength:.3f}")
        print(f"  置信度: {signal.confidence:.3f}")
        print(f"  交易数量: {signal.trade_count}")
        print(f"  市场条件: {signal.market_condition}")

        indicators = signal.indicators
        print(f"\n📈 关键指标:")
        print(f"  价格动量: {indicators.price_momentum:+.4f}")
        print(f"  成交量动量: {indicators.volume_momentum:+.4f}")
        print(f"  订单流动量: {indicators.order_flow_momentum:+.4f}")
        print(f"  波动率调整: {indicators.volatility_adjusted:+.4f}")
    else:
        print("❌ 交易数量不足，无法进行分析")


if __name__ == "__main__":
    asyncio.run(debug_momentum_analysis())