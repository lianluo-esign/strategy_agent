#!/usr/bin/env python3
"""测试Volume Profile分析功能。"""

import asyncio

from src.core.redis_client import RedisDataStore
from src.core.volume_profile_analyzer import VolumeProfileAnalyzer
from src.utils.config import Settings


async def test_vp_analysis_simple():
    """测试简化的Volume Profile分析功能。"""
    # 加载设置
    settings = Settings.load_from_file("config/development.yaml")

    # 初始化Redis连接
    redis_store = RedisDataStore(
        host=settings.redis.host, port=settings.redis.port, db=settings.redis.db
    )

    # 创建VP分析器
    vp_analyzer = VolumeProfileAnalyzer(aggregation_precision=10.0)

    # 获取交易数据
    print("Getting 24h trade data...")
    trade_data = redis_store.get_recent_trade_data(minutes=1440)

    print(f"Trade data count: {len(trade_data)}")
    if trade_data:
        # 测试VP分析
        vp_result = vp_analyzer.analyze_volume_profile(trade_data, "BTCFDUSD")

        print("VP Analysis Result:")
        print("  Status:", vp_result.get("status"))
        if vp_result.get("status") == "success":
            vp_data = vp_result.get("vp_data", {})
            print("  VP data keys:", list(vp_result.keys()))
            print("  Price levels:", len(vp_data))
            print("  Total volume:", vp_result.get("total_volume", 0))

            poc_analysis = vp_result.get("poc_analysis", {})
            print("  POC price:", poc_analysis.get("poc_price", "N/A"))
            print("  POC volume:", poc_analysis.get("poc_volume", "N/A"))

            # 显示前5个最高成交量的价格水平
            sorted_levels = sorted(vp_data.items(), key=lambda x: x[1], reverse=True)[
                :5
            ]
            print("  Top 5 price levels:")
            for price, volume in sorted_levels:
                print(f"    ${float(price):,.0f}: {float(volume):.2f}")
        else:
            print("  Error:", vp_result.get("error", "Unknown error"))
    else:
        print("No trade data available")

    # 关闭连接
    await redis_store.close()


if __name__ == "__main__":
    asyncio.run(test_vp_analysis_simple())
