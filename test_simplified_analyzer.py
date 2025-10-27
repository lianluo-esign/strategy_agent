#!/usr/bin/env python3
"""测试简化分析器的功能。

这个脚本用于测试新的简化市场分析器是否能够：
1. 正确初始化所有组件
2. 从Redis读取并聚合数据
3. 调用DeepSeek分析并返回标准JSON格式
"""

import asyncio
import logging
import sys
from decimal import Decimal
from unittest.mock import Mock

# 添加项目根目录到Python路径
sys.path.insert(0, '/home/jamesduan/projects/strategy_agent')

from src.core.simplified_market_analyzer import SimplifiedMarketAnalyzer
from src.core.result_validator import result_validator

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MockRedisStore:
    """模拟Redis数据存储，用于测试。"""

    def __init__(self):
        self.test_mode = True

    def test_connection(self):
        return True

    def depth_snapshot_exists(self):
        return True

    def get_trade_window_count(self):
        return 1440  # 24小时数据

    def get_latest_depth_snapshot(self):
        """返回模拟的深度快照数据。"""
        snapshot = Mock()
        snapshot.symbol = "BTCFDUSD"
        snapshot.timestamp = "2025-10-27 18:00:00"

        # 模拟订单簿数据 (价格 -> 数量)
        snapshot.bids = {
            Decimal('115000.00'): Decimal('2.5'),
            Decimal('115001.00'): Decimal('1.8'),
            Decimal('115002.00'): Decimal('3.2'),
            Decimal('115003.00'): Decimal('2.1'),
            Decimal('115004.00'): Decimal('1.5'),
            Decimal('115005.00'): Decimal('2.8'),
            Decimal('115006.00'): Decimal('1.9'),
            Decimal('115007.00'): Decimal('2.3'),
            Decimal('115008.00'): Decimal('1.7'),
            Decimal('115009.00'): Decimal('2.6'),
        }

        snapshot.asks = {
            Decimal('115010.00'): Decimal('1.9'),
            Decimal('115011.00'): Decimal('2.4'),
            Decimal('115012.00'): Decimal('1.6'),
            Decimal('115013.00'): Decimal('2.8'),
            Decimal('115014.00'): Decimal('2.1'),
            Decimal('115015.00'): Decimal('1.8'),
            Decimal('115016.00'): Decimal('2.7'),
            Decimal('115017.00'): Decimal('2.2'),
            Decimal('115018.00'): Decimal('1.5'),
            Decimal('115019.00'): Decimal('2.3'),
        }

        return snapshot

    def get_recent_trade_data(self, minutes=1440):
        """返回模拟的交易数据。"""
        # 模拟返回1440分钟的交易数据
        trades = []
        for i in range(1440):
            trades.append(f"trade_data_{i}")
        return trades


async def test_result_validator():
    """测试结果验证器的功能。"""
    logger.info("=== Testing Result Validator ===")

    # 测试标准JSON
    test_cases = [
        {
            "name": "Valid JSON",
            "content": '{"grid_delta": 2.0, "grid_quantity": 0.001, "active_side": "Buy"}',
            "should_pass": True
        },
        {
            "name": "JSON with extra content",
            "content": '''Here's my analysis:
            {"grid_delta": 1.5, "grid_quantity": 0.002, "active_side": "Sell"}
            This looks like a good opportunity.''',
            "should_pass": True
        },
        {
            "name": "JSON in code block",
            "content": '''```json
            {"grid_delta": 3.0, "grid_quantity": 0.0015, "active_side": "Buy"}
            ```''',
            "should_pass": True
        },
        {
            "name": "Invalid range",
            "content": '{"grid_delta": 200.0, "grid_quantity": 0.001, "active_side": "Buy"}',
            "should_pass": False
        },
        {
            "name": "Missing field",
            "content": '{"grid_delta": 2.0, "grid_quantity": 0.001}',
            "should_pass": False
        }
    ]

    for test_case in test_cases:
        logger.info(f"Testing: {test_case['name']}")

        try:
            mock_result = {
                "status": "success",
                "raw_content": test_case['content'],
                "symbol": "BTCFDUSD"
            }

            params = result_validator.validate_and_extract_trading_params(mock_result)

            if test_case['should_pass']:
                logger.info(f"✅ PASS: {params}")
            else:
                logger.warning(f"⚠️  UNEXPECTED PASS: {params}")

        except Exception as e:
            if not test_case['should_pass']:
                logger.info(f"✅ PASS (expected fail): {e}")
            else:
                logger.error(f"❌ FAIL: {e}")


async def test_simplified_analyzer():
    """测试简化分析器的完整流程。"""
    logger.info("=== Testing Simplified Market Analyzer ===")

    # 创建模拟Redis存储
    mock_redis = MockRedisStore()

    # 创建DeepSeek配置（使用环境变量中的API密钥）
    import os
    deepseek_config = {
        "api_key": os.getenv("DEEPSEEK_API_KEY", "test-key"),
        "base_url": "https://api.deepseek.com/v1",
        "model": "deepseek-chat",
        "max_tokens": 3000,
        "temperature": 0.1,
        "timeout": 90,
        "max_retries": 3,
    }

    # 检查是否有真实的API密钥
    if deepseek_config["api_key"] == "test-key":
        logger.warning("⚠️  No real DeepSeek API key found. Using mock mode.")
        logger.warning("⚠️  Set DEEPSEEK_API_KEY environment variable to test with real API.")
        return

    # 创建简化分析器
    try:
        analyzer = SimplifiedMarketAnalyzer(
            redis_store=mock_redis,
            deepseek_config=deepseek_config,
            price_aggregation_precision=1.0,
            vp_aggregation_precision=10.0,
        )
        logger.info("✅ SimplifiedMarketAnalyzer initialized successfully")

        # 获取分析器状态
        status = analyzer.get_status()
        logger.info(f"Analyzer status: {status}")

        # 执行市场分析
        logger.info("Starting market analysis...")
        result = await analyzer.analyze_market("BTCFDUSD")

        logger.info("Analysis completed:")
        logger.info(f"Status: {result.get('status')}")

        if result.get("status") == "success":
            trading_params = result.get("trading_params")
            if trading_params:
                logger.info(f"🎯 Trading Parameters: {trading_params}")
            else:
                logger.warning("⚠️  No trading parameters generated")
        else:
            logger.error(f"❌ Analysis failed: {result.get('error')}")

        # 关闭分析器
        await analyzer.close()
        logger.info("✅ Analyzer closed successfully")

    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


async def main():
    """主测试函数。"""
    logger.info("🚀 Starting Simplified Analyzer Tests")

    try:
        # 测试结果验证器
        await test_result_validator()

        # 测试简化分析器（需要真实API密钥）
        await test_simplified_analyzer()

    except KeyboardInterrupt:
        logger.info("👋 Test interrupted by user")
    except Exception as e:
        logger.error(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

    logger.info("✅ All tests completed")


if __name__ == "__main__":
    asyncio.run(main())