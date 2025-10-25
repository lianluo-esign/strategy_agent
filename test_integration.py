import asyncio
from datetime import datetime
from decimal import Decimal

from src.core.analyzers_enhanced import EnhancedMarketAnalyzer
from src.core.models import DepthLevel, DepthSnapshot


async def test_basic_functionality():
    print("🧪 开始集成测试...")

    # 测试数据
    bids = [
        DepthLevel(Decimal("99.50"), Decimal("10.0")),
        DepthLevel(Decimal("99.20"), Decimal("5.0")),
        DepthLevel(Decimal("98.90"), Decimal("8.0")),
        DepthLevel(Decimal("99.10"), Decimal("12.0")),
    ]

    asks = [
        DepthLevel(Decimal("100.10"), Decimal("8.0")),
        DepthLevel(Decimal("100.50"), Decimal("6.0")),
        DepthLevel(Decimal("101.30"), Decimal("4.0")),
    ]

    snapshot = DepthSnapshot("BTCFDUSD", datetime.now(), bids, asks)

    print("✅ 测试数据创建完成")

    # 设置日志级别以隐藏预期的警告信息
    import logging

    logging.getLogger("src.core.wave_peak_analyzer").setLevel(logging.ERROR)
    logging.getLogger("src.core.analyzers").setLevel(logging.ERROR)
    logging.getLogger("src.core.analyzers_enhanced").setLevel(logging.INFO)

    # 测试增强分析器
    enhanced_analyzer = EnhancedMarketAnalyzer(
        min_volume_threshold=Decimal("1.0"), analysis_window_minutes=180
    )

    enhanced_result = enhanced_analyzer.analyze_market(
        snapshot=snapshot, trade_data_list=[], symbol="BTCFDUSD", enhanced_mode=True
    )

    print("✅ 增强分析器工作:")
    print(f"   - 波峰检测: {len(enhanced_result.wave_peaks)} 个")
    print(f"   - 支撑区域: {len(enhanced_result.support_zones)} 个")
    print(f"   - 阻力区域: {len(enhanced_result.resistance_zones)} 个")
    print(
        f"   - 聚合数据: {len(enhanced_result.aggregated_bids)} bids, {len(enhanced_result.aggregated_asks)} asks"
    )

    # 性能测试
    import time

    start_time = time.time()

    # 运行10次分析以测试性能
    for i in range(10):
        result = enhanced_analyzer.analyze_market(
            snapshot=snapshot, trade_data_list=[], symbol="BTCFDUSD", enhanced_mode=True
        )

    end_time = time.time()
    avg_time = (end_time - start_time) / 10

    print(f"✅ 性能测试完成: 平均分析时间 {avg_time:.4f}s")

    if avg_time < 0.1:  # 100ms以内
        print("🚀 性能优秀 - 满足生产要求")
    elif avg_time < 0.5:  # 500ms以内
        print("✅ 性能良好 - 满足生产要求")
    else:
        print("⚠️ 性能需要优化")

    print("🎯 集成测试完成！")


if __name__ == "__main__":
    asyncio.run(test_basic_functionality())
