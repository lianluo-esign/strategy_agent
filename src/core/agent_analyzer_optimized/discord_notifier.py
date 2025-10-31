"""Discord Webhook通知模块 - 优化版消息推送服务。

该模块专门用于将AI分析结果发送到Discord webhook，
提供格式化的消息和可靠的通知服务。
"""

import json
import logging
import time
from datetime import datetime
from typing import Any, Dict, Optional

import aiohttp
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

# Discord配置常量
DEFAULT_TIMEOUT = 30
MAX_RETRIES = 3
RETRY_INITIAL_DELAY = 1
RETRY_MAX_DELAY = 16
MAX_MESSAGE_LENGTH = 2000

# 趋势表情符号映射
TREND_EMOJIS = {
    "震荡": "⚖️",
    "微弱看涨": "📈",
    "看涨": "🚀",
    "强力看涨": "🔥",
    "微弱看跌": "📉",
    "看跌": "⬇️",
    "强力看跌": "💥"
}

# 强度等级表情符号
STRENGTH_EMOJIS = {
    "strong_support": "🟢",
    "weak_support": "🟡",
    "strong_resistance": "🔴",
    "weak_resistance": "🟠"
}


class DiscordNotifier:
    """Discord Webhook通知器，用于发送AI分析结果。

    该类提供：
    1. 格式化的Discord消息
    2. 重试机制和错误处理
    3. 消息发送状态跟踪
    4. 性能监控和统计
    """

    def __init__(
        self,
        webhook_url: str,
        timeout: int = DEFAULT_TIMEOUT,
        max_retries: int = MAX_RETRIES,
        enable_embeds: bool = True
    ):
        """初始化Discord通知器。

        Args:
            webhook_url: Discord webhook URL
            timeout: 请求超时时间
            max_retries: 最大重试次数
            enable_embeds: 是否启用嵌入格式
        """
        if not webhook_url or not webhook_url.startswith("https://discord.com/api/webhooks/"):
            raise ValueError("无效的Discord webhook URL")

        self.webhook_url = webhook_url
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.max_retries = max_retries
        self.enable_embeds = enable_embeds

        # 统计信息
        self.stats = {
            "total_notifications": 0,
            "successful_notifications": 0,
            "failed_notifications": 0,
            "average_response_time": 0
        }

        logger.info(f"Initialized DiscordNotifier with webhook URL: {webhook_url[:50]}...")

    @retry(
        stop=stop_after_attempt(MAX_RETRIES),
        wait=wait_exponential(multiplier=RETRY_INITIAL_DELAY, max=RETRY_MAX_DELAY),
        reraise=True
    )
    async def send_analysis_result(
        self,
        analysis_result: Dict[str, Any],
        symbol: str = "BTCFDUSD"
    ) -> bool:
        """发送分析结果到Discord。

        Args:
            analysis_result: AI分析结果
            symbol: 交易符号

        Returns:
            发送是否成功
        """
        start_time = time.time()
        self.stats["total_notifications"] += 1

        try:
            # 验证输入数据
            if not analysis_result or not isinstance(analysis_result, dict):
                raise ValueError("分析结果不能为空且必须是字典格式")

            # 格式化消息
            message_payload = self._format_discord_message(analysis_result, symbol)

            # 发送到Discord
            success = await self._send_to_discord(message_payload)

            # 更新统计信息
            response_time = time.time() - start_time
            self._update_stats(response_time, success)

            if success:
                logger.info(
                    f"Discord通知发送成功: trend={analysis_result.get('trend', 'unknown')}, "
                    f"response_time={response_time:.2f}s"
                )
            else:
                logger.error("Discord通知发送失败")

            return success

        except Exception as e:
            response_time = time.time() - start_time
            self._update_stats(response_time, False)
            logger.error(f"Discord通知发送异常: {e}")
            return False

    def _format_discord_message(
        self,
        analysis_result: Dict[str, Any],
        symbol: str
    ) -> Dict[str, Any]:
        """格式化Discord消息。

        Args:
            analysis_result: 分析结果
            symbol: 交易符号

        Returns:
            Discord消息载荷
        """
        # 提取关键信息
        timestamp = analysis_result.get("timestamp", datetime.now().isoformat())
        trend = analysis_result.get("trend", "未知")
        strength_levels = analysis_result.get("strength_levels", {})
        reason = analysis_result.get("reason", "暂无分析原因")
        confidence = analysis_result.get("confidence", 0.5)

        # 获取表情符号
        trend_emoji = TREND_EMOJIS.get(trend, "❓")
        confidence_bar = self._create_confidence_bar(confidence)

        if self.enable_embeds:
            # 使用嵌入格式
            embed = {
                "title": f"{trend_emoji} {symbol} 市场趋势分析",
                "description": f"**趋势判断**: {trend}",
                "color": self._get_trend_color(trend),
                "timestamp": timestamp,
                "fields": [
                    {
                        "name": "📊 趋势详情",
                        "value": f"**当前趋势**: {trend} {trend_emoji}\n**置信度**: {confidence:.1%} {confidence_bar}",
                        "inline": False
                    },
                    {
                        "name": "💪 支撑/阻力强度",
                        "value": self._format_strength_levels(strength_levels),
                        "inline": True
                    },
                    {
                        "name": "📝 分析原因",
                        "value": self._truncate_text(reason, 1024),
                        "inline": False
                    }
                ],
                "footer": {
                    "text": "AI分析 | BTC-FDUSD流动性分析Agent"
                }
            }

            return {"embeds": [embed]}

        else:
            # 使用简单文本格式
            message = f"""{trend_emoji} **{symbol} 市场趋势分析报告**

**📊 趋势判断**: {trend}
**🎯 置信度**: {confidence:.1%} {confidence_bar}

**💪 支撑/阻力强度**:
{self._format_strength_levels_text(strength_levels)}

**📝 分析原因**:
{self._truncate_text(reason, 1000)}

---
*分析时间: {timestamp}*
*由AI分析生成 | BTC-FDUSD流动性分析Agent*"""

            return {"content": message}

    def _format_strength_levels(self, strength_levels: Dict[str, float]) -> str:
        """格式化强度等级（嵌入格式）。

        Args:
            strength_levels: 强度等级字典

        Returns:
            格式化的强度字符串
        """
        lines = []
        for level, value in strength_levels.items():
            emoji = STRENGTH_EMOJIS.get(level, "📍")
            level_name = self._translate_strength_level(level)
            bar = self._create_strength_bar(value)
            lines.append(f"{emoji} {level_name}: `{value:.2f}` {bar}")

        return "\n".join(lines) if lines else "暂无强度数据"

    def _format_strength_levels_text(self, strength_levels: Dict[str, float]) -> str:
        """格式化强度等级（文本格式）。

        Args:
            strength_levels: 强度等级字典

        Returns:
            格式化的强度字符串
        """
        lines = []
        for level, value in strength_levels.items():
            emoji = STRENGTH_EMOJIS.get(level, "📍")
            level_name = self._translate_strength_level(level)
            bar = self._create_strength_bar(value)
            lines.append(f"  {emoji} {level_name}: {value:.2f} {bar}")

        return "\n".join(lines) if lines else "  暂无强度数据"

    def _translate_strength_level(self, level: str) -> str:
        """翻译强度等级名称。

        Args:
            level: 强度等级英文

        Returns:
            中文名称
        """
        translations = {
            "strong_support": "强支撑",
            "weak_support": "弱支撑",
            "strong_resistance": "强阻力",
            "weak_resistance": "弱阻力"
        }
        return translations.get(level, level)

    def _create_confidence_bar(self, confidence: float) -> str:
        """创建置信度条。

        Args:
            confidence: 置信度值 (0-1)

        Returns:
            置信度条字符串
        """
        filled_bars = int(confidence * 10)
        empty_bars = 10 - filled_bars
        return f"[{'█' * filled_bars}{'░' * empty_bars}]"

    def _create_strength_bar(self, strength: float) -> str:
        """创建强度条。

        Args:
            strength: 强度值 (0-1)

        Returns:
            强度条字符串
        """
        filled_bars = int(strength * 5)
        empty_bars = 5 - filled_bars
        return f"[{'■' * filled_bars}{'□' * empty_bars}]"

    def _get_trend_color(self, trend: str) -> int:
        """获取趋势对应的颜色。

        Args:
            trend: 趋势类型

        Returns:
            Discord颜色代码
        """
        colors = {
            "震荡": 0x808080,      # 灰色
            "微弱看涨": 0x90EE90,   # 浅绿色
            "看涨": 0x00FF00,       # 绿色
            "强力看涨": 0x006400,    # 深绿色
            "微弱看跌": 0xFFB6C1,   # 浅红色
            "看跌": 0xFF0000,       # 红色
            "强力看跌": 0x8B0000     # 深红色
        }
        return colors.get(trend, 0x808080)

    def _truncate_text(self, text: str, max_length: int) -> str:
        """截断文本到指定长度。

        Args:
            text: 原始文本
            max_length: 最大长度

        Returns:
            截断后的文本
        """
        if len(text) <= max_length:
            return text
        return text[:max_length - 3] + "..."

    async def _send_to_discord(self, payload: Dict[str, Any]) -> bool:
        """发送消息到Discord。

        Args:
            payload: 消息载荷

        Returns:
            发送是否成功
        """
        headers = {"Content-Type": "application/json"}

        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            try:
                async with session.post(
                    self.webhook_url,
                    headers=headers,
                    json=payload
                ) as response:
                    if response.status in [200, 204]:
                        return True
                    else:
                        error_text = await response.text()
                        logger.error(
                            f"Discord API返回错误: status={response.status}, "
                            f"error={error_text}"
                        )
                        return False

            except aiohttp.ClientError as e:
                logger.error(f"Discord网络请求失败: {str(e)}")
                return False
            except Exception as e:
                logger.error(f"Discord发送异常: {str(e)}")
                return False

    def _update_stats(self, response_time: float, success: bool) -> None:
        """更新统计信息。

        Args:
            response_time: 响应时间
            success: 是否成功
        """
        if success:
            self.stats["successful_notifications"] += 1
        else:
            self.stats["failed_notifications"] += 1

        # 更新平均响应时间
        total_requests = self.stats["total_notifications"]
        current_avg = self.stats["average_response_time"]
        self.stats["average_response_time"] = (
            (current_avg * (total_requests - 1) + response_time) / total_requests
        )

    async def test_connection(self) -> bool:
        """测试Discord webhook连接。

        Returns:
            连接是否正常
        """
        test_message = {
            "content": "🧪 **测试消息** - BTC-FDUSD流动性分析Agent已连接",
            "embeds": [
                {
                    "title": "连接测试",
                    "description": "这是一条测试消息，用于验证Discord webhook连接是否正常。",
                    "color": 0x00FF00,
                    "timestamp": datetime.now().isoformat()
                }
            ]
        }

        try:
            return await self._send_to_discord(test_message)
        except Exception as e:
            logger.error(f"Discord连接测试失败: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息。

        Returns:
            统计信息字典
        """
        success_rate = (
            self.stats["successful_notifications"] / self.stats["total_notifications"]
            if self.stats["total_notifications"] > 0 else 0
        )

        return {
            **self.stats,
            "success_rate": success_rate,
            "webhook_configured": True,
            "embeds_enabled": self.enable_embeds
        }

    async def close(self) -> None:
        """关闭资源。"""
        logger.info("DiscordNotifier closed")


class DiscordNotificationManager:
    """Discord通知管理器，提供高级通知功能。"""

    def __init__(self, webhook_url: str):
        """初始化通知管理器。

        Args:
            webhook_url: Discord webhook URL
        """
        self.notifier = DiscordNotifier(webhook_url)

    async def send_trend_alert(
        self,
        analysis_result: Dict[str, Any],
        symbol: str = "BTCFDUSD",
        include_raw_data: bool = False
    ) -> bool:
        """发送趋势警报。

        Args:
            analysis_result: 分析结果
            symbol: 交易符号
            include_raw_data: 是否包含原始数据

        Returns:
            发送是否成功
        """
        # 检查是否为显著趋势
        trend = analysis_result.get("trend", "")
        confidence = analysis_result.get("confidence", 0.0)

        # 只对高置信度的强趋势发送警报
        if confidence < 0.7 and trend not in ["强力看涨", "强力看跌"]:
            logger.info("趋势不够显著，跳过警报发送")
            return True

        return await self.notifier.send_analysis_result(analysis_result, symbol)

    async def send_error_notification(self, error_message: str, context: str = "") -> bool:
        """发送错误通知。

        Args:
            error_message: 错误消息
            context: 错误上下文

        Returns:
            发送是否成功
        """
        error_payload = {
            "content": f"🚨 **系统错误通知**",
            "embeds": [
                {
                    "title": "分析系统错误",
                    "description": f"**错误信息**: {error_message}",
                    "color": 0xFF0000,
                    "fields": [
                        {
                            "name": "上下文",
                            "value": context or "无上下文信息",
                            "inline": False
                        },
                        {
                            "name": "时间",
                            "value": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "inline": False
                        }
                    ],
                    "footer": {
                        "text": "BTC-FDUSD流动性分析Agent"
                    }
                }
            ]
        }

        try:
            return await self.notifier._send_to_discord(error_payload)
        except Exception as e:
            logger.error(f"错误通知发送失败: {e}")
            return False

    def get_notifier(self) -> DiscordNotifier:
        """获取通知器实例。

        Returns:
            Discord通知器
        """
        return self.notifier

    async def close(self) -> None:
        """关闭资源。"""
        await self.notifier.close()