#!/usr/bin/env python3
"""
测试简化版Discord通知功能
"""

import asyncio
import json
import logging
from datetime import datetime
from src.core.agent_analyzer_optimized.discord_notifier import DiscordNotificationManager

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_simplified_notification():
    """测试简化版Discord通知"""

    # 这里需要你的Discord webhook URL
    webhook_url = "https://discord.com/api/webhooks/YOUR_WEBHOOK_URL"

    if "YOUR_WEBHOOK_URL" in webhook_url:
        logger.error("请先设置正确的Discord webhook URL")
        return

    try:
        # 创建通知管理器
        discord_manager = DiscordNotificationManager(webhook_url)
        logger.info("Discord通知管理器初始化成功")

        # 测试1: 简化的分析结果
        simple_result = {
            "timestamp": datetime.now().isoformat(),
            "trend": "震荡",
            "confidence": 0.65,
            "reason": "市场成交量平稳，价格在窄幅区间内波动，多空力量相对均衡",
            "strength_levels": {
                "strong_support": 0.3,
                "weak_support": 0.6,
                "strong_resistance": 0.2,
                "weak_resistance": 0.5
            }
        }

        logger.info("发送简化版分析结果...")
        success1 = await discord_manager.send_trend_alert(simple_result, "BTCFDUSD")
        logger.info(f"简化版通知发送结果: {'成功' if success1 else '失败'}")

        # 测试2: 贝叶斯分析结果（如果有的话）
        bayesian_result = {
            "timestamp": datetime.now().isoformat(),
            "analysis_type": "bayesian_trend_analysis",
            "trend_analysis": {
                "most_likely_trend": "微弱看涨",
                "confidence": 0.58,
                "uncertainty": 0.42
            },
            "probability_distribution": {
                "full_distribution": {
                    "微弱看涨": 0.35,
                    "震荡": 0.30,
                    "看涨": 0.20,
                    "微弱看跌": 0.15
                }
            },
            "bayesian_analysis": {
                "analysis_reason": "基于近期成交量放大和价格微弱上涨，市场显示轻微看涨倾向，但不确定性较高"
            }
        }

        logger.info("发送贝叶斯分析结果...")
        success2 = await discord_manager.send_trend_alert(bayesian_result, "BTCFDUSD")
        logger.info(f"贝叶斯通知发送结果: {'成功' if success2 else '失败'}")

        # 测试3: 连接测试
        logger.info("测试Discord连接...")
        connection_ok = await discord_manager.get_notifier().test_connection()
        logger.info(f"Discord连接测试: {'正常' if connection_ok else '异常'}")

        # 显示统计信息
        stats = discord_manager.get_notifier().get_stats()
        logger.info(f"Discord统计信息: {json.dumps(stats, indent=2)}")

        await discord_manager.close()

    except Exception as e:
        logger.error(f"测试过程中发生错误: {e}")

if __name__ == "__main__":
    asyncio.run(test_simplified_notification())