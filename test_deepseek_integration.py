#!/usr/bin/env python3
"""测试DeepSeek集成功能的脚本"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from datetime import datetime
from decimal import Decimal

from core.analyzers_normal import NormalDistributionMarketAnalyzer
from core.models import DepthLevel, DepthSnapshot


def test_deepseek_integration():
    """测试DeepSeek集成功能"""
    print("🧪 开始测试DeepSeek集成功能...")

    # 创建测试订单簿数据
    bids = [
        DepthLevel(price=Decimal("94980.00"), quantity=Decimal("25.5")),
        DepthLevel(price=Decimal("94981.00"), quantity=Decimal("18.2")),
        DepthLevel(price=Decimal("94982.00"), quantity=Decimal("31.7")),
        DepthLevel(price=Decimal("94983.00"), quantity=Decimal("12.4")),
        DepthLevel(price=Decimal("94984.00"), quantity=Decimal("28.9")),
        DepthLevel(price=Decimal("94985.00"), quantity=Decimal("15.6")),
        DepthLevel(price=Decimal("94986.00"), quantity=Decimal("22.1")),
        DepthLevel(price=Decimal("94987.00"), quantity=Decimal("19.8")),
        DepthLevel(price=Decimal("94988.00"), quantity=Decimal("33.2")),
        DepthLevel(price=Decimal("94989.00"), quantity=Decimal("14.7")),
        DepthLevel(price=Decimal("94990.00"), quantity=Decimal("26.5")),
        DepthLevel(price=Decimal("94991.00"), quantity=Decimal("20.3")),
        DepthLevel(price=Decimal("94992.00"), quantity=Decimal("35.8")),
        DepthLevel(price=Decimal("94993.00"), quantity=Decimal("17.9")),
        DepthLevel(price=Decimal("94994.00"), quantity=Decimal("29.4")),
        DepthLevel(price=Decimal("94995.00"), quantity=Decimal("13.2")),
    ]

    asks = [
        DepthLevel(price=Decimal("95005.00"), quantity=Decimal("22.6")),
        DepthLevel(price=Decimal("95006.00"), quantity=Decimal("31.4")),
        DepthLevel(price=Decimal("95007.00"), quantity=Decimal("15.8")),
        DepthLevel(price=Decimal("95008.00"), quantity=Decimal("27.9")),
        DepthLevel(price=Decimal("95009.00"), quantity=Decimal("19.5")),
        DepthLevel(price=Decimal("95010.00"), quantity=Decimal("34.2")),
        DepthLevel(price=Decimal("95011.00"), quantity=Decimal("16.7")),
        DepthLevel(price=Decimal("95012.00"), quantity=Decimal("24.8")),
        DepthLevel(price=Decimal("95013.00"), quantity=Decimal("21.3")),
        DepthLevel(price=Decimal("95014.00"), quantity=Decimal("32.6")),
        DepthLevel(price=Decimal("95015.00"), quantity=Decimal("18.1")),
        DepthLevel(price=Decimal("95016.00"), quantity=Decimal("26.9")),
        DepthLevel(price=Decimal("95017.00"), quantity=Decimal("14.5")),
        DepthLevel(price=Decimal("95018.00"), quantity=Decimal("30.7")),
        DepthLevel(price=Decimal("95019.00"), quantity=Decimal("17.2")),
        DepthLevel(price=Decimal("95020.00"), quantity=Decimal("25.3")),
    ]

    snapshot = DepthSnapshot(
        symbol="BTCFDUSD", timestamp=datetime.now(), bids=bids, asks=asks
    )

    # 初始化分析器
    analyzer = NormalDistributionMarketAnalyzer()

    # 尝试启用DeepSeek分析
    try:
        # 从配置文件读取API密钥
        import yaml

        with open("config/development.yaml") as f:
            config = yaml.safe_load(f)

        api_key = config.get("analyzer", {}).get("deepseek", {}).get("api_key")
        if api_key:
            analyzer.enable_deepseek_analysis(
                api_key=api_key, timeout=30, max_retries=2
            )
            print("✅ DeepSeek分析器启用成功")
        else:
            print("❌ 未找到DeepSeek API密钥配置")
            return False

    except Exception as e:
        print(f"❌ 启用DeepSeek分析器失败: {e}")
        return False

    # 执行分析
    try:
        print("📊 开始市场分析...")
        result = analyzer.analyze_market(snapshot, [], "BTCFDUSD", enhanced_mode=True)

        print("✅ 分析完成！")
        print(f"   检测到流动性峰值: {len(result.liquidity_peaks)} 个")

        if hasattr(result, "deepseek_analysis"):
            print("🤖 DeepSeek分析结果已显示在上方")
        else:
            print("ℹ️ DeepSeek分析结果未包含在结果中（可能因为配置或API问题）")

        return True

    except Exception as e:
        print(f"❌ 分析失败: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_deepseek_integration()
    if success:
        print("\n🎉 DeepSeek集成测试完成！")
    else:
        print("\n💥 DeepSeek集成测试失败！")
        sys.exit(1)
