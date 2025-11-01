"""Discord Webhook通知模块 - 优化版消息推送服务。

该模块专门用于将AI分析结果发送到Discord webhook，
提供格式化的消息和可靠的通知服务。
"""

import logging
import time
from datetime import datetime
from typing import Any

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
            "average_response_time": 0.0
        }

        logger.info(f"Initialized DiscordNotifier with webhook URL: {webhook_url[:50]}...")

    @retry(
        stop=stop_after_attempt(MAX_RETRIES),
        wait=wait_exponential(multiplier=RETRY_INITIAL_DELAY, max=RETRY_MAX_DELAY),
        reraise=True
    )
    async def send_analysis_result(
        self,
        analysis_result: dict[str, Any],
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
        analysis_result: dict[str, Any],
        symbol: str
    ) -> dict[str, Any]:
        """格式化Discord消息。

        Args:
            analysis_result: 分析结果
            symbol: 交易符号

        Returns:
            Discord消息载荷
        """
        # 提取关键信息 - 支持贝叶斯分析格式
        timestamp = analysis_result.get("timestamp", datetime.now().isoformat())

        # 尝试从不同位置提取趋势信息
        trend = "未知"
        confidence = 0.5
        reason = "暂无分析原因"

        # 1. 首先尝试从trend_analysis提取（贝叶斯格式）
        trend_analysis = analysis_result.get("trend_analysis", {})
        if trend_analysis:
            trend = trend_analysis.get("most_likely_trend", "未知")
            confidence = trend_analysis.get("confidence", 0.5)

        # 2. 如果没有找到，尝试从顶级字段提取（传统格式）
        if trend == "未知":
            trend = analysis_result.get("trend", "未知")
            confidence = analysis_result.get("confidence", 0.5)

        # 3. 提取分析原因
        bayesian_analysis = analysis_result.get("bayesian_analysis", {})
        if bayesian_analysis.get("analysis_reason"):
            reason = bayesian_analysis["analysis_reason"]
        else:
            reason = analysis_result.get("reason", "暂无分析原因")

        # 提取原始数据用于价格位置分析
        raw_data = analysis_result.get("raw_data", {})
        depth_snapshot = raw_data.get("depth_snapshot", {})
        minute_data_points = raw_data.get("minute_data_points", [])

        # 计算关键价格位置
        price_analysis = self._analyze_price_positions(minute_data_points, depth_snapshot)

        # 获取表情符号
        trend_emoji = TREND_EMOJIS.get(trend, "❓")
        confidence_bar = self._create_confidence_bar(confidence)

        # 尝试提取概率分布信息（用于贝叶斯分析）
        probability_summary = self._extract_probability_summary(analysis_result)

        if self.enable_embeds:
            # 构建字段列表
            fields = [
                {
                    "name": "📊 趋势详情",
                    "value": f"**当前趋势**: {trend} {trend_emoji}\n**置信度**: {confidence:.1%} {confidence_bar}",
                    "inline": False
                }
            ]

            # 添加价格分析
            price_analysis_text = self._format_data_statistics_for_discord(analysis_result)
            fields.append({
                "name": "📊 价格分析",
                "value": price_analysis_text,
                "inline": True
            })

            # 如果有概率分布信息，添加概率字段
            if probability_summary:
                fields.append({
                    "name": "📈 概率分布",
                    "value": probability_summary,
                    "inline": True
                })

            # 添加详细分析原因
            detailed_reason = self._enhance_analysis_reason(reason, price_analysis, depth_snapshot)
            fields.append({
                "name": "📝 详细分析",
                "value": self._truncate_text(detailed_reason, 1024),
                "inline": False
            })

            # 使用嵌入格式
            embed = {
                "title": f"{trend_emoji} {symbol} 市场趋势分析",
                "description": f"**趋势判断**: {trend}",
                "color": self._get_trend_color(trend),
                "timestamp": timestamp,
                "fields": fields,
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

**📊 价格分析**:
{self._format_data_statistics_text(analysis_result)}

**📝 详细分析**:
{self._truncate_text(self._enhance_analysis_reason(reason, price_analysis, depth_snapshot), 1000)}

---
*分析时间: {timestamp}*
*由AI分析生成 | BTC-FDUSD流动性分析Agent*"""

            return {"content": message}

    def _extract_probability_summary(self, analysis_result: dict[str, Any]) -> str:
        """提取概率分布摘要（用于传统格式化器）。

        Args:
            analysis_result: 分析结果

        Returns:
            格式化的概率分布字符串
        """
        try:
            # 尝试从metadata的response_raw中解析概率分布
            metadata = analysis_result.get("metadata", {})
            response_raw = metadata.get("response_raw", "")

            if response_raw:
                import json
                try:
                    raw_data = json.loads(response_raw)
                    posterior_probs = raw_data.get("posterior_probabilities", {})
                    if posterior_probs:
                        return self._format_probability_distribution_simple(posterior_probs)
                except json.JSONDecodeError:
                    pass

            # 尝试从probability_distribution中提取
            prob_distribution = analysis_result.get("probability_distribution", {})
            full_dist = prob_distribution.get("full_distribution", {})
            if full_dist:
                return self._format_probability_distribution_simple(full_dist)

        except Exception as e:
            logger.debug(f"提取概率分布失败: {e}")

        return ""

    def _format_probability_distribution_simple(self, probabilities: dict[str, float]) -> str:
        """格式化概率分布（简化版）。

        Args:
            probabilities: 概率分布字典

        Returns:
            格式化的概率字符串
        """
        if not probabilities:
            return ""

        lines = []
        # 按概率排序，只显示前3个
        sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)

        for trend, prob in sorted_probs[:3]:
            emoji = TREND_EMOJIS.get(trend, "❓")
            bar_length = int(prob * 5)
            bar = "●" * bar_length + "○" * (5 - bar_length)
            lines.append(f"{emoji} {trend}: `{prob:.0%}` {bar}")

        return "\n".join(lines)

    def _format_strength_levels(self, strength_levels: dict[str, float]) -> str:
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

    def _format_strength_levels_text(self, strength_levels: dict[str, float]) -> str:
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

    async def send_raw_message(self, payload: dict[str, Any]) -> bool:
        """发送原始消息到Discord（公共方法）。

        Args:
            payload: Discord消息载荷

        Returns:
            发送是否成功
        """
        return await self._send_to_discord(payload)

    async def _send_to_discord(self, payload: dict[str, Any]) -> bool:
        """发送消息到Discord（私有方法）。

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

    def get_stats(self) -> dict[str, Any]:
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

    def _analyze_price_positions(self, minute_data_points: list[dict], depth_snapshot: dict) -> dict[str, Any]:
        """分析关键价格位置。

        Args:
            minute_data_points: 分钟数据点
            depth_snapshot: 深度快照数据

        Returns:
            价格位置分析结果
        """
        price_analysis = {
            "support_levels": [],
            "resistance_levels": [],
            "current_price": 0.0,
            "volume_hotspots": [],
            "order_book_levels": {}
        }

        try:
            # 从深度快照获取当前价格信息
            if depth_snapshot:
                price_analysis["current_price"] = depth_snapshot.get("mid_price", 0.0)
                spread = depth_snapshot.get("spread", 0.0)
                bid_volume = depth_snapshot.get("bid_volume", 0)
                ask_volume = depth_snapshot.get("ask_volume", 0)

                # 计算订单簿关键位置
                if depth_snapshot.get("bids") and depth_snapshot.get("asks"):
                    best_bid = depth_snapshot["bids"][0]["price"] if depth_snapshot["bids"] else 0.0
                    best_ask = depth_snapshot["asks"][0]["price"] if depth_snapshot["asks"] else 0.0

                    price_analysis["order_book_levels"] = {
                        "best_bid": best_bid,
                        "best_ask": best_ask,
                        "spread": spread,
                        "bid_ask_ratio": bid_volume / ask_volume if ask_volume > 0 else 1.0
                    }

            # 从分钟数据分析成交量热点和支撑阻力
            all_volumes = {}
            for point in minute_data_points:
                price_levels = point.get("price_levels", {})
                for price_str, level_data in price_levels.items():
                    try:
                        price = float(price_str)
                        volume = float(level_data.get("total_volume", 0))
                        if volume > 0:
                            all_volumes[price] = all_volumes.get(price, 0) + volume
                    except (ValueError, TypeError):
                        continue

            if all_volumes and price_analysis["current_price"] > 0:
                # 按成交量排序，找出热点价格
                sorted_volumes = sorted(all_volumes.items(), key=lambda x: x[1], reverse=True)
                current_price = price_analysis["current_price"]

                # 找出支撑位（当前价格下方的高成交量位置）
                support_candidates = [(price, vol) for price, vol in sorted_volumes if price < current_price]
                resistance_candidates = [(price, vol) for price, vol in sorted_volumes if price > current_price]

                # 取前3个支撑位和阻力位
                price_analysis["support_levels"] = [
                    {"price": price, "volume": volume, "distance": (current_price - price) / current_price * 100}
                    for price, volume in support_candidates[:3]
                ]

                price_analysis["resistance_levels"] = [
                    {"price": price, "volume": volume, "distance": (price - current_price) / current_price * 100}
                    for price, volume in resistance_candidates[:3]
                ]

                # 成交量热点（前5个）
                price_analysis["volume_hotspots"] = [
                    {"price": price, "volume": volume}
                    for price, volume in sorted_volumes[:5]
                ]

        except Exception as e:
            logger.debug(f"价格位置分析失败: {e}")

        return price_analysis

    def _format_price_positions(self, price_analysis: dict[str, Any]) -> str:
        """格式化价格位置信息（嵌入格式）。

        Args:
            price_analysis: 价格位置分析结果

        Returns:
            格式化的价格位置字符串
        """
        if not price_analysis:
            return "暂无价格位置数据"

        lines = []
        current_price = price_analysis.get("current_price", 0.0)

        if current_price > 0:
            lines.append(f"**当前价格**: ${current_price:.2f}")

        # 支撑位
        support_levels = price_analysis.get("support_levels", [])
        if support_levels:
            lines.append("**🟢 支撑位**:")
            for i, support in enumerate(support_levels[:3], 1):
                price = support["price"]
                distance = support["distance"]
                volume = support["volume"]
                lines.append(f"  {i}. ${price:.2f} (距离 {distance:.2f}%) • 量: {volume:.0f}")

        # 阻力位
        resistance_levels = price_analysis.get("resistance_levels", [])
        if resistance_levels:
            lines.append("**🔴 阻力位**:")
            for i, resistance in enumerate(resistance_levels[:3], 1):
                price = resistance["price"]
                distance = resistance["distance"]
                volume = resistance["volume"]
                lines.append(f"  {i}. ${price:.2f} (距离 {distance:.2f}%) • 量: {volume:.0f}")

        # 订单簿信息
        order_book = price_analysis.get("order_book_levels", {})
        if order_book:
            best_bid = order_book.get("best_bid", 0.0)
            best_ask = order_book.get("best_ask", 0.0)
            spread = order_book.get("spread", 0.0)
            ratio = order_book.get("bid_ask_ratio", 1.0)

            if best_bid > 0 and best_ask > 0:
                lines.append("**📊 订单簿**:")
                lines.append(f"  买一: ${best_bid:.2f} | 卖一: ${best_ask:.2f}")
                lines.append(f"  价差: ${spread:.4f} | 买卖比: {ratio:.2f}")

        return "\n".join(lines) if lines else "暂无价格位置数据"

    def _format_price_positions_text(self, price_analysis: dict[str, Any]) -> str:
        """格式化价格位置信息（文本格式）。

        Args:
            price_analysis: 价格位置分析结果

        Returns:
            格式化的价格位置字符串
        """
        if not price_analysis:
            return "  暂无价格位置数据"

        lines = []
        current_price = price_analysis.get("current_price", 0.0)

        if current_price > 0:
            lines.append(f"  **当前价格**: ${current_price:.2f}")

        # 支撑位
        support_levels = price_analysis.get("support_levels", [])
        if support_levels:
            lines.append("  **🟢 支撑位**:")
            for i, support in enumerate(support_levels[:3], 1):
                price = support["price"]
                distance = support["distance"]
                lines.append(f"    {i}. ${price:.2f} (距离 {distance:.2f}%)")

        # 阻力位
        resistance_levels = price_analysis.get("resistance_levels", [])
        if resistance_levels:
            lines.append("  **🔴 阻力位**:")
            for i, resistance in enumerate(resistance_levels[:3], 1):
                price = resistance["price"]
                distance = resistance["distance"]
                lines.append(f"    {i}. ${price:.2f} (距离 {distance:.2f}%)")

        return "\n".join(lines) if lines else "  暂无价格位置数据"

    def _enhance_analysis_reason(self, original_reason: str, price_analysis: dict[str, Any], depth_snapshot: dict) -> str:
        """增强分析原因，添加更多细节。

        Args:
            original_reason: 原始分析原因
            price_analysis: 价格位置分析
            depth_snapshot: 深度快照数据

        Returns:
            增强后的分析原因
        """
        enhanced_parts = [original_reason]

        try:
            # 添加深度快照洞察
            if depth_snapshot:
                bid_volume = depth_snapshot.get("bid_volume", 0)
                ask_volume = depth_snapshot.get("ask_volume", 0)
                spread = depth_snapshot.get("spread", 0)
                mid_price = depth_snapshot.get("mid_price", 0)

                if bid_volume > 0 and ask_volume > 0:
                    ratio = bid_volume / ask_volume
                    if ratio > 1.2:
                        enhanced_parts.append(f"买盘力量较强(买卖比{ratio:.1f})")
                    elif ratio < 0.8:
                        enhanced_parts.append(f"卖盘压力较大(买卖比{ratio:.1f})")

                if mid_price > 0 and spread > 0:
                    spread_pct = (spread / mid_price * 100)
                    if spread_pct > 0.01:
                        enhanced_parts.append(f"价差较宽({spread_pct:.3f}%)，流动性需关注")
                    else:
                        enhanced_parts.append(f"价差正常({spread_pct:.3f}%)")

            # 添加价格位置洞察
            if price_analysis:
                current_price = price_analysis.get("current_price", 0.0)
                support_levels = price_analysis.get("support_levels", [])
                resistance_levels = price_analysis.get("resistance_levels", [])

                if current_price > 0:
                    # 最近支撑位分析
                    if support_levels:
                        nearest_support = support_levels[0]
                        support_distance = nearest_support["distance"]
                        if support_distance < 1.0:
                            enhanced_parts.append(f"接近近期支撑位(${nearest_support['price']:.2f})")

                    # 最近阻力位分析
                    if resistance_levels:
                        nearest_resistance = resistance_levels[0]
                        resistance_distance = nearest_resistance["distance"]
                        if resistance_distance < 1.0:
                            enhanced_parts.append(f"接近近期阻力位(${nearest_resistance['price']:.2f})")

                    # 价格位置分析
                    if support_levels and resistance_levels:
                        total_levels = len(support_levels) + len(resistance_levels)
                        if total_levels > 0:
                            enhanced_parts.append(f"处于密集成交区，共{total_levels}个关键价位")

        except Exception as e:
            logger.debug(f"增强分析原因失败: {e}")

        # 合并并限制长度
        enhanced_text = " | ".join(enhanced_parts)
        return self._truncate_text(enhanced_text, 500)

    def _format_data_statistics_for_discord(self, analysis_result: dict[str, Any]) -> str:
        """格式化数据统计信息为Discord消息（嵌入格式）。

        Args:
            analysis_result: 分析结果字典

        Returns:
            格式化的价格分析字符串
        """
        try:
            # 尝试从metadata中提取data_statistics
            metadata = analysis_result.get("metadata", {})
            data_stats = metadata.get("data_statistics", {})

            if not data_stats:
                return "暂无统计数据"

            # 提取各项统计数据
            total_volume = data_stats.get("total_volume", 0)
            trade_count = data_stats.get("trade_count", 0)
            price_levels_count = data_stats.get("price_levels_count", 0)
            price_range = data_stats.get("price_range", [0, 0])

            # 格式化显示
            lines = []

            # 总成交量（处理BTC单位显示）
            if total_volume > 0:
                if total_volume >= 1000:
                    volume_display = f"{total_volume/1000:.2f}K BTC"
                else:
                    volume_display = f"{total_volume:.4f} BTC"
                lines.append(f"• 总成交量: {volume_display}")
            else:
                lines.append("• 总成交量: 0 BTC")

            # 交易数量
            lines.append(f"• 交易数量: {trade_count:,} 笔")

            # 价格档位数量
            lines.append(f"• 价格档位: {price_levels_count:,} 个")

            # 价格区间
            if isinstance(price_range, list) and len(price_range) == 2:
                min_price, max_price = price_range
                if min_price > 0 and max_price > 0:
                    lines.append(f"• 价格区间: ${min_price:,.0f} - ${max_price:,.0f}")
                else:
                    lines.append("• 价格区间: 数据异常")
            else:
                lines.append("• 价格区间: 不可用")

            return "\n".join(lines)

        except Exception as e:
            logger.debug(f"格式化数据统计信息失败: {e}")
            return "统计数据格式异常"

    def _format_data_statistics_text(self, analysis_result: dict[str, Any]) -> str:
        """格式化数据统计信息为文本格式。

        Args:
            analysis_result: 分析结果字典

        Returns:
            格式化的价格分析字符串
        """
        try:
            # 尝试从metadata中提取data_statistics
            metadata = analysis_result.get("metadata", {})
            data_stats = metadata.get("data_statistics", {})

            if not data_stats:
                return "  暂无统计数据"

            # 提取各项统计数据
            total_volume = data_stats.get("total_volume", 0)
            trade_count = data_stats.get("trade_count", 0)
            price_levels_count = data_stats.get("price_levels_count", 0)
            price_range = data_stats.get("price_range", [0, 0])

            # 格式化显示
            lines = []

            # 总成交量
            if total_volume > 0:
                if total_volume >= 1000:
                    volume_display = f"{total_volume/1000:.2f}K BTC"
                else:
                    volume_display = f"{total_volume:.4f} BTC"
                lines.append(f"  • 总成交量: {volume_display}")
            else:
                lines.append("  • 总成交量: 0 BTC")

            # 交易数量
            lines.append(f"  • 交易数量: {trade_count:,} 笔")

            # 价格档位数量
            lines.append(f"  • 价格档位: {price_levels_count:,} 个")

            # 价格区间
            if isinstance(price_range, list) and len(price_range) == 2:
                min_price, max_price = price_range
                if min_price > 0 and max_price > 0:
                    lines.append(f"  • 价格区间: ${min_price:,.0f} - ${max_price:,.0f}")
                else:
                    lines.append("  • 价格区间: 数据异常")
            else:
                lines.append("  • 价格区间: 不可用")

            return "\n".join(lines)

        except Exception as e:
            logger.debug(f"格式化数据统计信息失败: {e}")
            return "  统计数据格式异常"


class DiscordNotificationManager:
    """Discord通知管理器，提供高级通知功能。"""

    def __init__(self, webhook_url: str):
        """初始化通知管理器。

        Args:
            webhook_url: Discord webhook URL
        """
        self.notifier = DiscordNotifier(webhook_url)
        self.bayesian_formatter = BayesianDiscordFormatter()

    async def send_trend_alert(
        self,
        analysis_result: dict[str, Any],
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
        # 检查是否为贝叶斯分析结果
        if self._is_bayesian_analysis(analysis_result):
            # 使用贝叶斯格式化器，总是发送（不跳过低置信度）
            formatted_message = self.bayesian_formatter.format_bayesian_analysis(analysis_result, symbol)

            # 发送到Discord
            try:
                success = await self.notifier.send_raw_message(formatted_message)
                if success:
                    logger.info(f"贝叶斯趋势警报发送成功: {symbol}")
                return success
            except Exception as e:
                logger.error(f"贝叶斯趋势警报发送失败: {e}")
                return False
        else:
            # 使用传统格式化器（向后兼容）
            confidence = analysis_result.get("confidence", 0.0)

            # 降低阈值，只对极低置信度的结果跳过
            if confidence < 0.3:
                logger.info(f"置信度过低({confidence:.1%})，跳过警报发送")
                return True

            return await self.notifier.send_analysis_result(analysis_result, symbol)

    def _is_bayesian_analysis(self, analysis_result: dict[str, Any]) -> bool:
        """检查是否为贝叶斯分析结果。

        Args:
            analysis_result: 分析结果

        Returns:
            是否为贝叶斯分析结果
        """
        # 增强贝叶斯分析检查 - 检查更多标志性字段
        bayesian_indicators = [
            "trend_analysis",
            "probability_distribution",
            "bayesian_analysis",
            "metadata",  # DeepSeek响应通常包含metadata
            "analysis_type"  # 包含bayesian字段的analysis_type
        ]

        has_indicator = any(key in analysis_result for key in bayesian_indicators)

        # 进一步检查metadata中是否有response_raw（DeepSeek响应特征）
        if has_indicator:
            metadata = analysis_result.get("metadata", {})
            if metadata.get("response_raw") or metadata.get("model_type") == "bayesian_deepseek":
                return True

            # 检查analysis_type是否包含bayesian
            analysis_type = analysis_result.get("analysis_type", "")
            if "bayesian" in str(analysis_type).lower():
                return True

        return has_indicator

    async def send_error_notification(self, error_message: str, context: str = "") -> bool:
        """发送错误通知。

        Args:
            error_message: 错误消息
            context: 错误上下文

        Returns:
            发送是否成功
        """
        error_payload = {
            "content": "🚨 **系统错误通知**",
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
            return await self.notifier.send_raw_message(error_payload)
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


class BayesianDiscordFormatter:
    """贝叶斯分析结果的Discord格式化器。

    专门用于格式化贝叶斯分析结果，包括：
    1. 概率化趋势分析结果
    2. 波动率和成交量分析
    3. 量价关系和市场活跃度
    4. 贝叶斯证据权重分析
    """

    def __init__(self) -> None:
        """初始化贝叶斯Discord格式化器。"""
        # 贝叶斯趋势表情符号映射
        self.bayesian_trend_emojis = {
            "震荡": "⚖️",
            "微弱看涨": "📈",
            "看涨": "🚀",
            "强力看涨": "🔥",
            "微弱看跌": "📉",
            "看跌": "⬇️",
            "强力看跌": "💥"
        }

        # 置信度等级颜色
        self.confidence_colors = {
            "very_low": 0xFF0000,      # 红色
            "low": 0xFF6B35,           # 橙色
            "medium": 0xFFD700,        # 金色
            "high": 0x90EE90,          # 浅绿色
            "very_high": 0x00FF00      # 绿色
        }

        # 风险等级颜色
        self.risk_colors = {
            "very_high_risk": 0xFF0000,  # 红色
            "high_risk": 0xFF6B35,       # 橙色
            "medium_risk": 0xFFD700,     # 金色
            "low_risk": 0x90EE90,        # 浅绿色
            "very_low_risk": 0x00FF00    # 绿色
        }

    def format_bayesian_analysis(self, analysis_data: dict[str, Any], symbol: str = "BTCFDUSD") -> dict[str, Any]:
        """格式化贝叶斯分析结果为Discord消息（增强版）。

        Args:
            analysis_data: 贝叶斯分析结果数据
            symbol: 交易符号

        Returns:
            Discord消息载荷
        """
        # 从复杂的数据结构中提取核心信息
        core_info = self._extract_core_bayesian_info(analysis_data)

        # 提取关键信息
        most_likely_trend = core_info.get("trend", "未知")
        confidence = core_info.get("confidence", 0.0)
        analysis_reason = core_info.get("reason", "暂无分析原因")
        probabilities = core_info.get("probabilities", {})

        # 获取表情符号和颜色
        trend_emoji = self.bayesian_trend_emojis.get(most_likely_trend, "❓")
        embed_color = self._get_bayesian_color(most_likely_trend, confidence, "unknown")

        # 构建简化的概率信息（只显示前3个）
        prob_summary = self._format_simple_probability_summary(probabilities)

        # 提取价格位置信息
        price_analysis = self._extract_price_positions(analysis_data)
        price_positions = self._format_price_positions_for_bayesian(price_analysis)

        # 构建字段列表
        fields = [
            {
                "name": "📊 概率分布",
                "value": prob_summary,
                "inline": True
            }
        ]

        # 添加价格分析
        price_analysis_text = self._format_data_statistics_for_bayesian(analysis_data)
        fields.append({
            "name": "📊 价格分析",
            "value": price_analysis_text,
            "inline": True
        })

        # 添加详细分析原因
        fields.append({
            "name": "📝 分析原因",
            "value": self._truncate_text(analysis_reason, 400),
            "inline": False
        })

        # 增强版嵌入消息 - 包含价格位置信息
        embed = {
            "title": f"{trend_emoji} {symbol} 市场分析",
            "description": f"**趋势**: {most_likely_trend} {trend_emoji}\n**置信度**: {confidence:.1%}",
            "color": embed_color,
            "fields": fields,
            "footer": {
                "text": "AI分析 | BTC-FDUSD流动性分析Agent"
            }
        }

        return {"embeds": [embed]}

    def _extract_core_bayesian_info(self, analysis_data: dict[str, Any]) -> dict[str, Any]:
        """从复杂的贝叶斯分析数据中提取核心信息。

        Args:
            analysis_data: 完整的贝叶斯分析数据

        Returns:
            包含核心信息的字典
        """
        import json

        core_info = {
            "trend": "未知",
            "confidence": 0.0,
            "timestamp": datetime.now().isoformat(),
            "reason": "暂无分析原因",
            "probabilities": {},
            "evidence": []
        }

        try:
            # 1. 提取趋势分析信息
            trend_analysis = analysis_data.get("trend_analysis", {})
            core_info["trend"] = trend_analysis.get("most_likely_trend") or "未知"
            core_info["confidence"] = trend_analysis.get("confidence") or 0.0
            core_info["timestamp"] = analysis_data.get("timestamp") or core_info["timestamp"]

            # 2. 提取分析原因
            bayesian_analysis = analysis_data.get("bayesian_analysis", {})
            core_info["reason"] = bayesian_analysis.get("analysis_reason") or "暂无分析原因"

            # 3. 提取概率分布
            probabilities = {}

            # 首先尝试从response_raw中解析
            metadata = analysis_data.get("metadata", {})
            response_raw = metadata.get("response_raw", "")

            if response_raw:
                try:
                    raw_data = json.loads(response_raw)
                    posterior_probs = raw_data.get("posterior_probabilities", {})
                    if posterior_probs:
                        # 验证和清洗概率值
                        probabilities = self._validate_and_clean_probabilities(posterior_probs)
                except json.JSONDecodeError:
                    pass

            # 如果response_raw解析失败，尝试其他方式
            if not probabilities:
                prob_distribution = analysis_data.get("probability_distribution", {})
                full_dist = prob_distribution.get("full_distribution", {})
                if full_dist:
                    # 同样需要验证其他来源的概率数据
                    probabilities = self._validate_and_clean_probabilities(full_dist)

            core_info["probabilities"] = probabilities

            # 4. 提取关键证据
            evidence_summary = bayesian_analysis.get("evidence_summary", {})
            if evidence_summary:
                evidence_list = []
                for key, value in evidence_summary.items():
                    if value and len(str(value)) > 10:
                        evidence_list.append(f"**{key}**: {str(value)[:100]}...")
                core_info["evidence"] = evidence_list[:3]  # 只取前3个最重要的证据

        except Exception as e:
            logger.error(f"提取贝叶斯核心信息失败: {e}")

        return core_info

    def _format_simple_probability_summary(self, probabilities: dict[str, float]) -> str:
        """格式化简化版概率分布摘要（只显示前3个）。

        Args:
            probabilities: 概率分布字典

        Returns:
            格式化的概率字符串
        """
        if not probabilities:
            return "暂无概率数据"

        lines = []
        # 按概率排序，只显示前3个
        sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)

        for trend, prob in sorted_probs[:3]:
            emoji = self.bayesian_trend_emojis.get(trend, "❓")
            # 简化的进度条，使用更简洁的符号
            bar_length = int(prob * 5)
            bar = "●" * bar_length + "○" * (5 - bar_length)
            lines.append(f"{emoji} {trend}: `{prob:.0%}` {bar}")

        return "\n".join(lines)

    def _format_probability_summary(self, probabilities: dict[str, float]) -> str:
        """格式化概率分布摘要。

        Args:
            probabilities: 概率分布字典

        Returns:
            格式化的概率字符串
        """
        if not probabilities:
            return "暂无概率数据"

        lines = []
        # 按概率排序，显示前4个
        sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)

        for trend, prob in sorted_probs[:4]:
            emoji = self.bayesian_trend_emojis.get(trend, "❓")
            bar_length = int(prob * 10)
            bar = "█" * bar_length + "░" * (10 - bar_length)
            lines.append(f"{emoji} {trend}: `{prob:.1%}` {bar}")

        return "\n".join(lines)

    def _extract_probability_distribution(self, analysis_data: dict[str, Any], prob_dist: dict[str, Any]) -> dict[str, Any]:
        """从analysis_data中提取概率分布数据。

        Args:
            analysis_data: 完整的分析数据
            prob_dist: 现有的概率分布数据

        Returns:
            包含full_distribution的概率分布数据
        """
        # 如果已经有full_distribution，直接返回
        if prob_dist.get("full_distribution"):
            return prob_dist

        # 尝试从metadata的response_raw中解析概率分布
        import json
        try:
            metadata = analysis_data.get("metadata", {})
            response_raw = metadata.get("response_raw", "")

            if response_raw:
                # 解析JSON字符串
                raw_data = json.loads(response_raw)
                posterior_probs = raw_data.get("posterior_probabilities", {})

                if posterior_probs:
                    # 构造标准格式的概率分布
                    return {
                        "full_distribution": posterior_probs,
                        "entropy": self._calculate_entropy(posterior_probs),
                        "distribution_type": "bayesian_posterior"
                    }

        except (json.JSONDecodeError, AttributeError, KeyError) as e:
            logger.debug(f"解析response_raw失败: {e}")

        # 如果解析失败，返回空的概率分布
        return {"full_distribution": {}}

    def _format_trend_analysis(self, trend_analysis: dict[str, Any]) -> str:
        """格式化趋势分析。

        Args:
            trend_analysis: 趋势分析数据

        Returns:
            格式化的趋势分析字符串
        """
        most_likely_trend = trend_analysis.get("most_likely_trend", "未知")
        confidence = trend_analysis.get("confidence", 0.0)
        uncertainty = trend_analysis.get("uncertainty", 1.0)
        confidence_level = trend_analysis.get("confidence_level", "unknown")
        risk_level = trend_analysis.get("risk_level", "unknown")

        # 置信度条
        confidence_bar = self._create_confidence_bar(confidence)

        # 不确定性条（反向显示）
        uncertainty_bar = self._create_uncertainty_bar(uncertainty)

        lines = [
            f"**最可能趋势**: {most_likely_trend}",
            f"**置信度**: {confidence:.1%} {confidence_bar}",
            f"**不确定性**: {uncertainty:.1%} {uncertainty_bar}",
            f"**置信等级**: {self._translate_confidence_level(confidence_level)}",
            f"**风险等级**: {self._translate_risk_level(risk_level)}"
        ]

        return "\n".join(lines)

    def _format_probability_distribution(self, prob_dist: dict[str, Any]) -> str:
        """格式化概率分布。

        Args:
            prob_dist: 概率分布数据

        Returns:
            格式化的概率分布字符串
        """
        full_dist = prob_dist.get("full_distribution", {})
        if not full_dist:
            return "暂无概率分布数据"

        lines = []
        # 按概率排序
        sorted_probs = sorted(full_dist.items(), key=lambda x: x[1], reverse=True)

        for trend, probability in sorted_probs[:5]:  # 显示前5个
            emoji = self.bayesian_trend_emojis.get(trend, "❓")
            bar_length = int(probability * 10)
            bar = "█" * bar_length + "░" * (10 - bar_length)
            lines.append(f"{emoji} {trend}: `{probability:.1%}` {bar}")

        return "\n".join(lines)

    def _format_bayesian_insights(self, bayesian_analysis: dict[str, Any]) -> str:
        """格式化贝叶斯洞察。

        Args:
            bayesian_analysis: 贝叶斯分析数据

        Returns:
            格式化的贝叶斯洞察字符串
        """
        if not bayesian_analysis:
            return "暂无贝叶斯洞察数据"

        lines = []

        # 主要驱动因素
        primary_driver = bayesian_analysis.get("primary_driver")
        if primary_driver:
            lines.append(f"**主要驱动**: {primary_driver}")

        # 证据一致性
        evidence_consistency = bayesian_analysis.get("evidence_consistency")
        if evidence_consistency:
            consistency_emoji = "✅" if "一致" in evidence_consistency else "⚠️"
            lines.append(f"**证据一致性**: {consistency_emoji} {evidence_consistency}")

        # 支撑因素
        strength_factors = bayesian_analysis.get("strength_factors", [])
        if strength_factors:
            lines.append("**支撑因素**:")
            for factor in strength_factors[:3]:  # 最多显示3个
                lines.append(f"• {factor}")

        # 风险因素
        weakness_factors = bayesian_analysis.get("weakness_factors", [])
        if weakness_factors:
            lines.append("**风险因素**:")
            for factor in weakness_factors[:3]:  # 最多显示3个
                lines.append(f"⚠️ {factor}")

        return "\n".join(lines) if lines else "暂无贝叶斯洞察"

    def _format_volatility_volume_insights(self, volatility_volume_insights: dict[str, Any]) -> str:
        """格式化波动率和成交量洞察。

        Args:
            volatility_volume_insights: 波动率成交量洞察数据

        Returns:
            格式化的波动率成交量洞察字符串
        """
        lines = []

        # 波动率评估
        volatility_assessment = volatility_volume_insights.get("volatility_assessment", "")
        if volatility_assessment:
            lines.append(f"**📊 波动率**: {volatility_assessment}")

        # 成交量趋势分析
        volume_trend = volatility_volume_insights.get("volume_trend_analysis", "")
        if volume_trend:
            lines.append(f"**📈 成交量**: {volume_trend}")

        # 量价配合模式
        coordination_pattern = volatility_volume_insights.get("coordination_pattern", "")
        if coordination_pattern:
            pattern_emoji = "🔄" if "健康" in coordination_pattern else "⚠️"
            lines.append(f"{pattern_emoji} **量价配合**: {coordination_pattern}")

        # 活跃度影响
        activity_impact = volatility_volume_insights.get("activity_level_impact", "")
        if activity_impact:
            lines.append(f"**⚡ 活跃度**: {activity_impact}")

        return "\n".join(lines) if lines else "暂无波动率成交量洞察"

    def _get_bayesian_color(self, trend: str, confidence: float, risk_level: str) -> int:
        """获取贝叶斯分析对应的颜色。

        Args:
            trend: 趋势类型
            confidence: 置信度
            risk_level: 风险等级

        Returns:
            Discord颜色代码
        """
        # 基于置信度调整颜色强度
        if confidence >= 0.8:
            # 高置信度 - 使用趋势的标准颜色
            base_colors = {
                "震荡": 0x808080,      # 灰色
                "微弱看涨": 0x90EE90,   # 浅绿色
                "看涨": 0x00FF00,       # 绿色
                "强力看涨": 0x006400,    # 深绿色
                "微弱看跌": 0xFFB6C1,   # 浅红色
                "看跌": 0xFF0000,       # 红色
                "强力看跌": 0x8B0000     # 深红色
            }
        elif confidence >= 0.6:
            # 中等置信度 - 使用较淡的颜色
            base_colors = {
                "震荡": 0xA0A0A0,      # 浅灰色
                "微弱看涨": 0xB8FFB8,   # 很浅绿色
                "看涨": 0x90FF90,       # 浅绿色
                "强力看涨": 0x32CD32,    # 酸橙绿
                "微弱看跌": 0xFFD0D0,   # 很浅红色
                "看跌": 0xFF6B6B,       # 浅红色
                "强力看跌": 0xCD5C5C     # 印度红
            }
        else:
            # 低置信度 - 使用很淡的颜色，表示不确定性
            base_colors = {
                "震荡": 0xC0C0C0,      # 很浅灰色
                "微弱看涨": 0xE0FFE0,   # 极浅绿色
                "看涨": 0xC8FFC8,       # 极浅绿色
                "强力看涨": 0x98FB98,    # 淡绿色
                "微弱看跌": 0xFFE0E0,   # 极浅红色
                "看跌": 0xFFC0C0,       # 极浅红色
                "强力看跌": 0xF08080     # 浅珊瑚红
            }

        return base_colors.get(trend, 0x808080)

    def _create_confidence_bar(self, confidence: float) -> str:
        """创建置信度条。

        Args:
            confidence: 置信度值 (0-1)

        Returns:
            置信度条字符串
        """
        filled_bars = int(confidence * 10)
        empty_bars = 10 - filled_bars
        color = "🟩" if confidence >= 0.8 else "🟨" if confidence >= 0.6 else "🟥" if confidence >= 0.4 else "🟥"
        return f"[{color * filled_bars}{'⬜' * empty_bars}]"

    def _create_uncertainty_bar(self, uncertainty: float) -> str:
        """创建不确定性条。

        Args:
            uncertainty: 不确定性值 (0-1)

        Returns:
            不确定性条字符串
        """
        filled_bars = int(uncertainty * 10)
        empty_bars = 10 - filled_bars
        return f"[{'🟥' * filled_bars}{'⬜' * empty_bars}]"

    def _translate_confidence_level(self, level: str) -> str:
        """翻译置信度等级。

        Args:
            level: 置信度等级

        Returns:
            中文置信度等级
        """
        translations = {
            "very_low": "极低",
            "low": "低",
            "medium": "中等",
            "high": "高",
            "very_high": "极高"
        }
        return translations.get(level, level)

    def _translate_risk_level(self, level: str) -> str:
        """翻译风险等级。

        Args:
            level: 风险等级

        Returns:
            中文风险等级
        """
        translations = {
            "very_high_risk": "极高风险",
            "high_risk": "高风险",
            "medium_risk": "中等风险",
            "low_risk": "低风险",
            "very_low_risk": "极低风险"
        }
        return translations.get(level, level)

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

    def _validate_and_clean_probabilities(self, probabilities: dict[str, float]) -> dict[str, float]:
        """验证和清洗概率分布数据。

        Args:
            probabilities: 原始概率分布字典

        Returns:
            验证和清洗后的概率分布字典
        """
        cleaned_probs = {}

        for trend, prob in probabilities.items():
            # 验证概率值是否为数字
            try:
                prob_float = float(prob)
            except (ValueError, TypeError):
                logger.warning(f"无效的概率值类型: {trend}={prob}")
                continue

            # 验证概率值范围
            if prob_float < 0:
                logger.warning(f"概率值为负数，已修正为0: {trend}={prob_float}")
                prob_float = 0.0
            elif prob_float > 1:
                logger.warning(f"概率值超过1，已修正为1: {trend}={prob_float}")
                prob_float = 1.0

            # 只保留有效的概率值
            if prob_float > 0:
                cleaned_probs[trend] = prob_float

        # 如果所有概率都无效，返回空字典
        if not cleaned_probs:
            logger.warning("所有概率值都无效，返回空概率分布")
            return {}

        # 归一化概率值，确保总和为1（允许小的浮点误差）
        total_prob = sum(cleaned_probs.values())
        if abs(total_prob - 1.0) > 0.01:  # 如果总和偏差超过1%
            logger.info(f"概率分布总和不为1({total_prob:.3f})，进行归一化处理")
            if total_prob > 0:
                cleaned_probs = {
                    trend: prob / total_prob
                    for trend, prob in cleaned_probs.items()
                }

        return cleaned_probs

    def _calculate_entropy(self, probabilities: dict[str, float]) -> float:
        """计算概率分布的熵（用于衡量不确定性）。

        Args:
            probabilities: 概率分布字典

        Returns:
            熵值（0到log(n)之间）
        """
        import math

        if not probabilities:
            return 0.0

        entropy = 0.0
        for prob in probabilities.values():
            if prob > 0:
                entropy -= prob * math.log2(prob)

        return entropy

    def _extract_price_positions(self, analysis_data: dict[str, Any]) -> dict[str, Any]:
        """从分析数据中提取价格位置信息。

        Args:
            analysis_data: 分析数据

        Returns:
            价格位置分析结果
        """
        price_analysis = {
            "support_levels": [],
            "resistance_levels": [],
            "current_price": 0.0,
            "volume_hotspots": [],
            "order_book_levels": {}
        }

        try:
            # 提取原始数据用于价格位置分析
            raw_data = analysis_data.get("raw_data", {})
            if not raw_data:
                # 如果没有raw_data，尝试从metadata中提取
                metadata = analysis_data.get("metadata", {})
                data_statistics = metadata.get("data_statistics", {})
                if data_statistics:
                    # 从statistics中提取基本信息
                    price_range = data_statistics.get("price_range", [0, 0])
                    if price_range and len(price_range) == 2:
                        price_analysis["current_price"] = (price_range[0] + price_range[1]) / 2
                        price_analysis["volume_hotspots"] = [
                            {"price": price_range[0], "volume": 100.0},
                            {"price": price_range[1], "volume": 100.0}
                        ]
                return price_analysis

            depth_snapshot = raw_data.get("depth_snapshot", {})
            minute_data_points = raw_data.get("minute_data_points", [])

            # 从深度快照获取当前价格信息
            if depth_snapshot:
                price_analysis["current_price"] = depth_snapshot.get("mid_price", 0.0)
                spread = depth_snapshot.get("spread", 0.0)
                bid_volume = depth_snapshot.get("bid_volume", 0)
                ask_volume = depth_snapshot.get("ask_volume", 0)

                # 计算订单簿关键位置
                if depth_snapshot.get("bids") and depth_snapshot.get("asks"):
                    best_bid = depth_snapshot["bids"][0]["price"] if depth_snapshot["bids"] else 0.0
                    best_ask = depth_snapshot["asks"][0]["price"] if depth_snapshot["asks"] else 0.0

                    price_analysis["order_book_levels"] = {
                        "best_bid": best_bid,
                        "best_ask": best_ask,
                        "spread": spread,
                        "bid_ask_ratio": bid_volume / ask_volume if ask_volume > 0 else 1.0
                    }

            # 从分钟数据分析成交量热点和支撑阻力
            all_volumes = {}
            for point in minute_data_points:
                price_levels = point.get("price_levels", {})
                for price_str, level_data in price_levels.items():
                    try:
                        price = float(price_str)
                        volume = float(level_data.get("total_volume", 0))
                        if volume > 0:
                            all_volumes[price] = all_volumes.get(price, 0) + volume
                    except (ValueError, TypeError):
                        continue

            if all_volumes and price_analysis["current_price"] > 0:
                # 按成交量排序，找出热点价格
                sorted_volumes = sorted(all_volumes.items(), key=lambda x: x[1], reverse=True)
                current_price = price_analysis["current_price"]

                # 找出支撑位（当前价格下方的高成交量位置）
                support_candidates = [(price, vol) for price, vol in sorted_volumes if price < current_price]
                resistance_candidates = [(price, vol) for price, vol in sorted_volumes if price > current_price]

                # 取前3个支撑位和阻力位
                price_analysis["support_levels"] = [
                    {"price": price, "volume": volume, "distance": (current_price - price) / current_price * 100}
                    for price, volume in support_candidates[:3]
                ]

                price_analysis["resistance_levels"] = [
                    {"price": price, "volume": volume, "distance": (price - current_price) / current_price * 100}
                    for price, volume in resistance_candidates[:3]
                ]

                # 成交量热点（前5个）
                price_analysis["volume_hotspots"] = [
                    {"price": price, "volume": volume}
                    for price, volume in sorted_volumes[:5]
                ]

        except Exception as e:
            logger.debug(f"贝叶斯格式化器价格位置分析失败: {e}")

        return price_analysis

    def _format_price_positions_for_bayesian(self, price_analysis: dict[str, Any]) -> str:
        """为贝叶斯分析格式化价格位置信息。

        Args:
            price_analysis: 价格位置分析结果

        Returns:
            格式化的价格位置字符串
        """
        if not price_analysis:
            return "暂无价格位置数据"

        lines = []
        current_price = price_analysis.get("current_price", 0.0)

        if current_price > 0:
            lines.append(f"**当前价格**: ${current_price:.2f}")

        # 支撑位
        support_levels = price_analysis.get("support_levels", [])
        if support_levels:
            lines.append("**🟢 支撑位**:")
            for i, support in enumerate(support_levels[:2], 1):  # 只显示前2个
                price = support["price"]
                distance = support["distance"]
                lines.append(f"  {i}. ${price:.2f} (距离 {distance:.2f}%)")

        # 阻力位
        resistance_levels = price_analysis.get("resistance_levels", [])
        if resistance_levels:
            lines.append("**🔴 阻力位**:")
            for i, resistance in enumerate(resistance_levels[:2], 1):  # 只显示前2个
                price = resistance["price"]
                distance = resistance["distance"]
                lines.append(f"  {i}. ${price:.2f} (距离 {distance:.2f}%)")

        # 订单簿信息
        order_book = price_analysis.get("order_book_levels", {})
        if order_book:
            best_bid = order_book.get("best_bid", 0.0)
            best_ask = order_book.get("best_ask", 0.0)
            ratio = order_book.get("bid_ask_ratio", 1.0)

            if best_bid > 0 and best_ask > 0:
                lines.append("**📊 订单簿**:")
                lines.append(f"  买一: ${best_bid:.2f} | 卖一: ${best_ask:.2f}")
                lines.append(f"  买卖比: {ratio:.2f}")

        return "\n".join(lines) if lines else "暂无价格位置数据"

    def _format_data_statistics_for_bayesian(self, analysis_data: dict[str, Any]) -> str:
        """为贝叶斯分析格式化数据统计信息。

        Args:
            analysis_data: 分析数据

        Returns:
            格式化的数据统计字符串
        """
        try:
            # 尝试从metadata中提取data_statistics
            metadata = analysis_data.get("metadata", {})
            data_stats = metadata.get("data_statistics", {})

            if not data_stats:
                return "暂无统计数据"

            # 提取各项统计数据
            total_volume = data_stats.get("total_volume", 0)
            trade_count = data_stats.get("trade_count", 0)
            price_levels_count = data_stats.get("price_levels_count", 0)
            price_range = data_stats.get("price_range", [0, 0])

            # 格式化显示
            lines = []

            # 总成交量（简化显示，适合贝叶斯格式）
            if total_volume > 0:
                if total_volume >= 1000:
                    volume_display = f"{total_volume/1000:.1f}K"
                else:
                    volume_display = f"{total_volume:.2f}"
                lines.append(f"**成交量**: {volume_display} BTC")
            else:
                lines.append("**成交量**: 0 BTC")

            # 交易数量
            lines.append(f"**交易数**: {trade_count:,}")

            # 价格档位数量
            lines.append(f"**档位数**: {price_levels_count:,}")

            # 价格区间
            if isinstance(price_range, list) and len(price_range) == 2:
                min_price, max_price = price_range
                if min_price > 0 and max_price > 0:
                    lines.append(f"**区间**: ${min_price:,.0f}-${max_price:,.0f}")
                else:
                    lines.append("**区间**: 数据异常")
            else:
                lines.append("**区间**: 不可用")

            return "\n".join(lines)

        except Exception as e:
            logger.debug(f"贝叶斯格式化器数据统计失败: {e}")
            return "统计数据异常"
