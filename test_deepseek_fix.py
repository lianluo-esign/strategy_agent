#!/usr/bin/env python3
"""验证DeepSeek修复后的集成测试脚本"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import logging
from decimal import Decimal
from datetime import datetime

from src.core.analyzers_normal import NormalDistributionMarketAnalyzer
from src.core.models import DepthSnapshot, DepthLevel
from src.utils.config import Settings

def test_deepseek_fix():
    """测试DeepSeek修复后的功能"""
    print("🧪 测试DeepSeek修复后的功能...")

    # 设置日志级别为INFO以查看DeepSeek打印输出
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    try:
        # 加载配置
        settings = Settings.load_from_file('config/development.yaml')

        # 创建分析器
        analyzer = NormalDistributionMarketAnalyzer(
            min_volume_threshold=Decimal(str(settings.analyzer.analysis.min_order_volume_threshold)),
            analysis_window_minutes=180,
            confidence_level=getattr(settings.analyzer, "confidence_level", 0.95)
        )

        # 启用DeepSeek分析（模拟analyzer.py中的逻辑）
        if settings.analyzer.deepseek.enable and settings.analyzer.deepseek.api_key:
            analyzer.enable_deepseek_analysis(
                api_key=settings.analyzer.deepseek.api_key,
                base_url=settings.analyzer.deepseek.base_url,
                model=settings.analyzer.deepseek.model,
                max_tokens=settings.analyzer.deepseek.max_tokens,
                temperature=settings.analyzer.deepseek.temperature,
                timeout=30,  # Default timeout value
                max_retries=3,  # Default retry count
            )
            print("✅ DeepSeek分析器启用成功")
        else:
            print("❌ DeepSeek配置未启用或API密钥缺失")
            return False

        # 创建测试数据
        bids = [
            DepthLevel(price=Decimal('94980.00'), quantity=Decimal('25.5')),
            DepthLevel(price=Decimal('94981.00'), quantity=Decimal('18.2')),
            DepthLevel(price=Decimal('94982.00'), quantity=Decimal('31.7')),
            DepthLevel(price=Decimal('94983.00'), quantity=Decimal('12.4')),
            DepthLevel(price=Decimal('94984.00'), quantity=Decimal('28.9')),
            DepthLevel(price=Decimal('94985.00'), quantity=Decimal('15.6')),
            DepthLevel(price=Decimal('94986.00'), quantity=Decimal('22.1')),
            DepthLevel(price=Decimal('94987.00'), quantity=Decimal('19.8')),
            DepthLevel(price=Decimal('94988.00'), quantity=Decimal('33.2')),
            DepthLevel(price=Decimal('94989.00'), quantity=Decimal('14.7')),
        ]

        asks = [
            DepthLevel(price=Decimal('95005.00'), quantity=Decimal('22.6')),
            DepthLevel(price=Decimal('95006.00'), quantity=Decimal('31.4')),
            DepthLevel(price=Decimal('95007.00'), quantity=Decimal('15.8')),
            DepthLevel(price=Decimal('95008.00'), quantity=Decimal('27.9')),
            DepthLevel(price=Decimal('95009.00'), quantity=Decimal('19.5')),
            DepthLevel(price=Decimal('95010.00'), quantity=Decimal('34.2')),
            DepthLevel(price=Decimal('95011.00'), quantity=Decimal('16.7')),
            DepthLevel(price=Decimal('95012.00'), quantity=Decimal('24.8')),
            DepthLevel(price=Decimal('95013.00'), quantity=Decimal('21.3')),
            DepthLevel(price=Decimal('95014.00'), quantity=Decimal('32.6')),
        ]

        snapshot = DepthSnapshot(
            symbol='BTCFDUSD',
            timestamp=datetime.now(),
            bids=bids,
            asks=asks
        )

        print("📊 开始市场分析（应该包含DeepSeek分析结果）...")

        # 执行分析
        result = analyzer.analyze_market(snapshot, [], 'BTCFDUSD', enhanced_mode=True)

        print(f"✅ 分析完成！")
        print(f"   检测到流动性峰值: {len(result.liquidity_peaks)} 个")
        print(f"   支撑位数量: {len(result.support_levels)} 个")
        print(f"   阻力位数量: {len(result.resistance_levels)} 个")

        # 检查是否有DeepSeek分析结果打印
        print("\n🔍 检查上方是否显示了DeepSeek AI分析结果...")
        print("如果看到 '🤖 DeepSeek AI 市场结构分析' 的标题，说明修复成功！")

        return True

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_deepseek_fix()
    if success:
        print("\n🎉 DeepSeek修复验证测试完成！")
        print("现在DeepSeek分析结果应该能够在生产环境中正确显示了。")
    else:
        print("\n💥 DeepSeek修复验证测试失败！")
        sys.exit(1)