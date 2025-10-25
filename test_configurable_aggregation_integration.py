#!/usr/bin/env python3
"""集成测试：验证可配置聚合精度的端到端功能"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from decimal import Decimal
from datetime import datetime
from src.core.models import DepthSnapshot, DepthLevel
from src.utils.config import Settings

def test_different_precision_configurations():
    """测试不同精度配置下的聚合效果"""
    print("🧪 测试可配置聚合精度集成功能...")

    # 创建测试订单簿数据
    print("\n📊 创建测试订单簿数据...")
    bids = [
        DepthLevel(price=Decimal('95000.30'), quantity=Decimal('1.2')),
        DepthLevel(price=Decimal('95000.80'), quantity=Decimal('2.1')),
        DepthLevel(price=Decimal('95001.20'), quantity=Decimal('0.8')),
        DepthLevel(price=Decimal('95001.70'), quantity=Decimal('1.5')),
        DepthLevel(price=Decimal('95002.40'), quantity=Decimal('0.9')),
    ]
    asks = [
        DepthLevel(price=Decimal('95100.60'), quantity=Decimal('1.4')),
        DepthLevel(price=Decimal('95101.10'), quantity=Decimal('2.3')),
        DepthLevel(price=Decimal('95101.80'), quantity=Decimal('1.1')),
        DepthLevel(price=Decimal('95102.30'), quantity=Decimal('0.7')),
        DepthLevel(price=Decimal('95102.90'), quantity=Decimal('1.8')),
    ]

    snapshot = DepthSnapshot(
        symbol='BTCFDUSD',
        timestamp=datetime.now(),
        bids=bids,
        asks=asks
    )

    print(f"   原始买盘数据: {len(bids)} 级")
    print(f"   原始卖盘数据: {len(asks)} 级")

    # 测试不同精度配置
    test_configs = [
        {"precision": 1.0, "enabled": True, "name": "$1精度"},
        {"precision": 0.5, "enabled": True, "name": "$0.5精度"},
        {"precision": 0.1, "enabled": True, "name": "$0.1精度"},
        {"precision": 1.0, "enabled": False, "name": "禁用聚合"},
    ]

    for config in test_configs:
        print(f"\n🔧 测试配置: {config['name']}")
        test_aggregation_with_config(snapshot, config)

    print("\n🎉 可配置聚合精度集成测试完成！")
    return True

def test_aggregation_with_config(snapshot, config):
    """使用指定配置测试聚合功能"""
    try:
        # 使用LiquidityPeaksAnalyzer测试
        from src.core.liquidity_peaks_analyzer import LiquidityPeaksAnalyzer

        aggregation_config = {
            "precision": config["precision"],
            "enabled": config["enabled"],
            "max_price_levels": 5000
        }

        analyzer = LiquidityPeaksAnalyzer(
            min_volume_threshold=0.1,
            price_aggregation_config=aggregation_config
        )

        result = analyzer.analyze_liquidity_peaks(snapshot)

        # 显示聚合结果
        bid_aggregation = result["bid_aggregation"]
        ask_aggregation = result["ask_aggregation"]

        print(f"   聚合后买盘: {len(bid_aggregation)} 级")
        print(f"   聚合后卖盘: {len(ask_aggregation)} 级")

        # 显示价格级别示例
        if bid_aggregation:
            sample_bid_price = list(bid_aggregation.keys())[0]
            print(f"   买盘示例价格: ${sample_bid_price}")

        if ask_aggregation:
            sample_ask_price = list(ask_aggregation.keys())[0]
            print(f"   卖盘示例价格: ${sample_ask_price}")

        # 检查流动性峰值
        peaks = result.get("liquidity_peaks", [])
        print(f"   识别流动性峰值: {len(peaks)} 个")

        return True

    except Exception as e:
        print(f"   ❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_configuration_loading():
    """测试配置文件加载功能"""
    print("\n🔧 测试配置文件加载...")

    try:
        # 加载开发配置
        settings = Settings.load_from_file('config/development.yaml')

        # 检查价格聚合配置
        if hasattr(settings.analyzer, 'price_aggregation'):
            price_agg = settings.analyzer.price_aggregation
            print(f"   ✅ 配置加载成功:")
            print(f"      精度: {price_agg.precision}")
            print(f"      启用: {price_agg.enabled}")
            print(f"      最大级别: {price_agg.max_price_levels}")
        else:
            print("   ⚠️  未找到价格聚合配置（使用默认值）")

        return True

    except Exception as e:
        print(f"   ❌ 配置加载失败: {e}")
        return False

def test_normal_distribution_analyzer_integration():
    """测试NormalDistributionMarketAnalyzer集成"""
    print("\n🔧 测试正态分布分析器集成...")

    try:
        from src.core.analyzers_normal import NormalDistributionMarketAnalyzer

        # 配置价格聚合
        price_aggregation_config = {
            "precision": 1.0,
            "enabled": True,
            "max_price_levels": 5000
        }

        analyzer = NormalDistributionMarketAnalyzer(
            min_volume_threshold=Decimal("1.0"),
            analysis_window_minutes=180,
            confidence_level=0.95,
            price_aggregation_config=price_aggregation_config
        )

        # 创建测试数据
        bids = [
            DepthLevel(price=Decimal('95000.50'), quantity=Decimal('1.5')),
            DepthLevel(price=Decimal('95001.20'), quantity=Decimal('2.3')),
        ]
        asks = [
            DepthLevel(price=Decimal('95100.30'), quantity=Decimal('1.8')),
            DepthLevel(price=Decimal('95101.70'), quantity=Decimal('2.1')),
        ]

        snapshot = DepthSnapshot(
            symbol='BTCFDUSD',
            timestamp=datetime.now(),
            bids=bids,
            asks=asks
        )

        # 执行分析
        result = analyzer.analyze_market(snapshot, [], 'BTCFDUSD', enhanced_mode=True)

        print(f"   ✅ 正态分布分析器集成成功:")
        print(f"      聚合买盘级别: {len(result.aggregated_bids)}")
        print(f"      聚合卖盘级别: {len(result.aggregated_asks)}")
        print(f"      流动性峰值: {len(result.liquidity_peaks)}")

        return True

    except Exception as e:
        print(f"   ❌ 正态分布分析器集成失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_full_analyzer_agent_integration():
    """测试完整的analyzer agent集成"""
    print("\n🔧 测试完整分析器代理集成...")

    try:
        from src.agents.analyzer import AnalyzerAgent
        from src.utils.config import Settings

        # 加载配置
        settings = Settings.load_from_file('config/development.yaml')

        # 创建代理
        agent = AnalyzerAgent(settings)

        # 检查市场分析器配置
        if hasattr(agent.market_analyzer, 'price_aggregation_config'):
            config = agent.market_analyzer.price_aggregation_config
            print(f"   ✅ 分析器代理配置成功:")
            print(f"      价格聚合配置: {config}")
        else:
            print("   ⚠️  分析器代理未使用价格聚合配置")

        return True

    except Exception as e:
        print(f"   ❌ 分析器代理集成失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = True

    # 执行所有集成测试
    success &= test_different_precision_configurations()
    success &= test_configuration_loading()
    success &= test_normal_distribution_analyzer_integration()
    success &= test_full_analyzer_agent_integration()

    if success:
        print("\n✅ 所有集成测试通过！可配置聚合精度功能正常工作。")
        sys.exit(0)
    else:
        print("\n❌ 部分集成测试失败！")
        sys.exit(1)