#!/usr/bin/env python3
"""测试DeepSeek分析结果打印功能"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from datetime import datetime
from decimal import Decimal

from src.core.analyzers_normal import NormalDistributionMarketAnalyzer
from src.core.models import DepthLevel, DepthSnapshot
from src.utils.config import Settings


def test_deepseek_analysis_printing():
    """测试DeepSeek分析结果打印功能"""
    print("🧪 测试DeepSeek分析结果打印...")

    # 加载配置
    try:
        settings = Settings.load_from_file("config/development.yaml")
        print("✅ 配置加载成功")
        print(f"   DeepSeek启用: {settings.analyzer.deepseek.enable}")
        print(
            f"   API密钥: {'已设置' if settings.analyzer.deepseek.api_key else '未设置'}"
        )
    except Exception as e:
        print(f"❌ 配置加载失败: {e}")
        return False

    # 创建测试数据
    print("\n📊 创建测试订单簿数据...")
    bids = [
        DepthLevel(price=Decimal("111960.00"), quantity=Decimal("0.25")),
        DepthLevel(price=Decimal("111959.00"), quantity=Decimal("0.15")),
        DepthLevel(price=Decimal("111958.00"), quantity=Decimal("0.35")),
        DepthLevel(price=Decimal("111957.00"), quantity=Decimal("0.22")),
        DepthLevel(price=Decimal("111956.00"), quantity=Decimal("0.18")),
        DepthLevel(price=Decimal("111955.00"), quantity=Decimal("0.42")),
        DepthLevel(price=Decimal("111954.00"), quantity=Decimal("0.28")),
        DepthLevel(price=Decimal("111953.00"), quantity=Decimal("0.31")),
        DepthLevel(price=Decimal("111952.00"), quantity=Decimal("0.19")),
        DepthLevel(price=Decimal("111951.00"), quantity=Decimal("0.26")),
    ]

    asks = [
        DepthLevel(price=Decimal("112015.00"), quantity=Decimal("0.77")),
        DepthLevel(price=Decimal("112016.00"), quantity=Decimal("0.45")),
        DepthLevel(price=Decimal("112017.00"), quantity=Decimal("0.33")),
        DepthLevel(price=Decimal("112018.00"), quantity=Decimal("0.58")),
        DepthLevel(price=Decimal("112019.00"), quantity=Decimal("0.41")),
        DepthLevel(price=Decimal("112020.00"), quantity=Decimal("0.29")),
        DepthLevel(price=Decimal("112021.00"), quantity=Decimal("0.36")),
        DepthLevel(price=Decimal("112022.00"), quantity=Decimal("0.24")),
        DepthLevel(price=Decimal("112023.00"), quantity=Decimal("0.52")),
        DepthLevel(price=Decimal("112024.00"), quantity=Decimal("0.38")),
    ]

    snapshot = DepthSnapshot(
        symbol="BTCFDUSD", timestamp=datetime.now(), bids=bids, asks=asks
    )

    print(f"   买盘数据: {len(bids)} 级")
    print(f"   卖盘数据: {len(asks)} 级")

    # 初始化分析器
    print("\n🔧 初始化市场分析器...")
    analyzer = NormalDistributionMarketAnalyzer(
        min_volume_threshold=Decimal("1.0"),
        analysis_window_minutes=180,
        confidence_level=0.95,
    )

    # 启用DeepSeek分析
    print("\n🤖 启用DeepSeek分析...")
    try:
        analyzer.enable_deepseek_analysis(
            api_key=settings.analyzer.deepseek.api_key, timeout=30, max_retries=2
        )
        print("✅ DeepSeek分析器启用成功")
    except Exception as e:
        print(f"❌ DeepSeek分析器启用失败: {e}")
        return False

    # 执行分析
    print("\n📈 执行市场分析...")
    try:
        print("🔄 开始分析...")
        result = analyzer.analyze_market(snapshot, [], "BTCFDUSD", enhanced_mode=True)

        print("\n✅ 分析完成！")
        print(f"   流动性峰值数量: {len(result.liquidity_peaks)}")

        # 检查DeepSeek结果
        if hasattr(result, "deepseek_analysis_result"):
            print(f"   DeepSeek分析结果: {result.deepseek_analysis_result}")
        else:
            print("   DeepSeek分析结果: 未包含在结果对象中")

        return True

    except Exception as e:
        print(f"❌ 分析执行失败: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_deepseek_analysis_printing()
    if success:
        print("\n🎉 DeepSeek分析打印测试完成！")
        print("现在启动analyzer agent应该能看到DeepSeek分析结果了。")
    else:
        print("\n💥 DeepSeek分析打印测试失败！")
        sys.exit(1)
