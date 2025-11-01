"""Deepseek AI分析客户端 - 优化版AI趋势分析。

该模块专门用于基于聚合后的trades_window数据进行AI趋势分析，
提供标准化的JSON输出格式，支持7种趋势分类和强度评估。
"""

import json
import logging
import time
from datetime import datetime
from typing import Any

import aiohttp
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

# API配置常量
DEFAULT_TIMEOUT = 90
MAX_RETRIES = 3
RETRY_INITIAL_DELAY = 1
RETRY_MAX_DELAY = 30

# 趋势分类常量
TREND_TYPES = [
    "震荡",
    "微弱看涨", "看涨", "强力看涨",
    "微弱看跌", "看跌", "强力看跌"
]

# 强度等级常量
STRENGTH_LEVELS = [
    "strong_support", "weak_support",
    "strong_resistance", "weak_resistance"
]


class TrendAnalysisResult:
    """趋势分析结果模型。"""

    def __init__(
        self,
        timestamp: datetime,
        trend: str,
        strength_levels: dict[str, float],
        reason: str,
        confidence: float,
        analysis_metadata: dict[str, Any] | None = None
    ):
        self.timestamp = timestamp
        self.trend = trend
        self.strength_levels = strength_levels
        self.reason = reason
        self.confidence = confidence
        self.analysis_metadata = analysis_metadata or {}

    def validate(self) -> bool:
        """验证结果数据的有效性。"""
        if self.trend not in TREND_TYPES:
            logger.error(f"无效的趋势类型: {self.trend}")
            return False

        if not 0 <= self.confidence <= 1:
            logger.error(f"置信度必须在0-1之间: {self.confidence}")
            return False

        for level, value in self.strength_levels.items():
            if level not in STRENGTH_LEVELS:
                logger.error(f"无效的强度等级: {level}")
                return False
            if not 0 <= value <= 1:
                logger.error(f"强度值必须在0-1之间: {level}={value}")
                return False

        return True

    def to_dict(self) -> dict[str, Any]:
        """转换为字典格式。"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "trend": self.trend,
            "strength_levels": self.strength_levels,
            "reason": self.reason,
            "confidence": self.confidence,
            "analysis_metadata": self.analysis_metadata
        }


class DeepSeekAnalyzer:
    """Deepseek AI分析器，专注于基于交易数据的趋势分析。

    该类提供：
    1. 标准化的AI分析接口
    2. 重试机制和错误处理
    3. 结果验证和格式化
    4. 性能监控和日志记录
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.deepseek.com/v1",
        model: str = "deepseek-chat",
        max_tokens: int = 4000,
        temperature: float = 0.1,
        timeout: int = DEFAULT_TIMEOUT,
        max_retries: int = MAX_RETRIES
    ):
        """初始化Deepseek分析器。

        Args:
            api_key: Deepseek API密钥
            base_url: API基础URL
            model: 使用的模型名称
            max_tokens: 最大令牌数
            temperature: 温度参数
            timeout: 请求超时时间
            max_retries: 最大重试次数
        """
        if not api_key:
            raise ValueError("API密钥不能为空")

        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.max_retries = max_retries

        # 统计信息
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "average_response_time": 0
        }

        logger.info(
            f"Initialized DeepSeekAnalyzer: model={model}, "
            f"max_tokens={max_tokens}, temperature={temperature}"
        )

    @retry(
        stop=stop_after_attempt(MAX_RETRIES),
        wait=wait_exponential(multiplier=RETRY_INITIAL_DELAY, max=RETRY_MAX_DELAY),
        reraise=True
    )
    async def analyze_trend(
        self,
        aggregated_data: dict[str, Any],
        symbol: str = "BTCFDUSD"
    ) -> TrendAnalysisResult:
        """分析市场趋势。

        Args:
            aggregated_data: 聚合后的交易数据
            symbol: 交易符号

        Returns:
            趋势分析结果

        Raises:
            ValueError: 当输入数据无效时
            RuntimeError: 当API调用失败时
        """
        start_time = time.time()
        self.stats["total_requests"] += 1

        try:
            # 验证输入数据
            if not aggregated_data or not isinstance(aggregated_data, dict):
                raise ValueError("聚合数据不能为空且必须是字典格式")

            # 构建分析提示词
            prompt = self._build_analysis_prompt(aggregated_data, symbol)

            # 调用Deepseek API
            response = await self._call_deepseek_api(prompt)

            # 解析响应
            result = self._parse_response(response, symbol)

            # 验证结果
            if not result.validate():
                raise ValueError("AI分析结果验证失败")

            # 更新统计信息
            response_time = time.time() - start_time
            self._update_stats(response_time, success=True)

            logger.info(
                f"趋势分析完成: trend={result.trend}, "
                f"confidence={result.confidence:.2f}, "
                f"response_time={response_time:.2f}s"
            )

            return result

        except Exception as e:
            response_time = time.time() - start_time
            self._update_stats(response_time, success=False)
            logger.error(f"趋势分析失败: {e}")
            raise RuntimeError(f"趋势分析失败: {str(e)}") from e

    def _build_analysis_prompt(self, raw_data: dict[str, Any], symbol: str) -> str:
        """构建AI分析提示词。

        Args:
            raw_data: 原始交易数据
            symbol: 交易符号

        Returns:
            格式化的提示词
        """
        # 提取原始数据
        minute_data_points = raw_data.get("minute_data_points", [])
        data_points_count = raw_data.get("data_points_count", 0)
        depth_snapshot = raw_data.get("depth_snapshot", {})

        # 计算基础统计信息
        total_volume = 0
        all_price_levels: dict[float, float] = {}
        first_timestamp: str | None = None
        price_changes: list[float] = []

        for point in minute_data_points:
            timestamp = point.get("timestamp", "")
            price_levels = point.get("price_levels", {})

            # 记录时间范围
            if first_timestamp is None:
                first_timestamp = timestamp

            # 累加成交量统计和价格变化
            minute_volume: float = 0
            minute_prices: list[float] = []
            for price_str, level_data in price_levels.items():
                try:
                    volume = float(level_data.get("total_volume", 0))
                    if volume > 0:
                        total_volume += volume
                        minute_volume += volume
                        price_float = float(price_str)
                        all_price_levels[price_float] = all_price_levels.get(price_float, 0.0) + volume
                        minute_prices.append(price_float)
                except (ValueError, TypeError, AttributeError):
                    continue

            # 计算每分钟的价格变化（用于波动率分析）
            if minute_prices:
                minute_high = max(minute_prices)
                minute_low = min(minute_prices)
                if minute_low > 0:
                    price_changes.append((minute_high - minute_low) / minute_low)

        # 获取成交量最大的前10个价格水平
        top_price_levels = sorted(
            all_price_levels.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]

        # 格式化价格水平数据
        price_levels_str = "\n".join([
            f"  价格: ${price:.2f}, 成交量: {volume:.2f}"
            for price, volume in top_price_levels
        ])

        # 计算价格范围和波动率指标
        if all_price_levels:
            price_min = min(all_price_levels.keys())
            price_max = max(all_price_levels.keys())
            price_range = price_max - price_min
            avg_price = sum(all_price_levels.keys()) / len(all_price_levels.keys())
            price_volatility = (price_range / avg_price * 100) if avg_price > 0 else 0
        else:
            price_min = price_max = 0
            price_volatility = 0

        # 计算时间序列波动率
        if price_changes:
            volatility_rate = sum(price_changes) / len(price_changes) * 100
            volatility_trend = "上升" if len(price_changes) > 1 and price_changes[-1] > price_changes[0] else "下降"
        else:
            volatility_rate = 0
            volatility_trend = "稳定"

        # 计算成交量分布特征
        if all_price_levels:
            max_volume = max(all_price_levels.values())
            concentration_ratio = max_volume / total_volume if total_volume > 0 else 0
            volume_distribution = "高度集中" if concentration_ratio > 0.5 else "相对分散" if concentration_ratio > 0.2 else "非常分散"
        else:
            concentration_ratio = 0
            volume_distribution = "无数据"

        # 处理深度快照数据
        depth_info = ""
        if depth_snapshot:
            bid_volume = depth_snapshot.get("bid_volume", 0)
            ask_volume = depth_snapshot.get("ask_volume", 0)
            spread = depth_snapshot.get("spread", 0)
            mid_price = depth_snapshot.get("mid_price", 0)

            order_book_ratio = bid_volume / ask_volume if ask_volume > 0 else 1
            spread_ratio = (spread / mid_price * 100) if mid_price > 0 else 0

            depth_info = f"""
**深度快照数据**:
- 中间价: ${mid_price:.2f}
- 买卖价差: ${spread:.2f} ({spread_ratio:.3f}%)
- 买盘总量: {bid_volume:.0f}
- 卖盘总量: {ask_volume:.0f}
- 买卖比例: {order_book_ratio:.2f}
- 流动性: {'充裕' if min(bid_volume, ask_volume) > 1000 else '一般' if min(bid_volume, ask_volume) > 100 else '稀薄'}"""

        prompt = f"""基于{symbol}交易数据做趋势分析。

**数据概览**:
- 数据点: {data_points_count} 个
- 总成交量: {total_volume:.0f}
- 价格区间: ${price_min:.2f} - ${price_max:.2f}
- 价格波动率: {price_volatility:.2f}%
- 时间序列波动: {volatility_rate:.2f}% ({volatility_trend})

**成交量分析**:
- 成交量分布: {volume_distribution}
- 最大单价格占比: {concentration_ratio:.1%}

**主要成交价格**:
{price_levels_str}
{depth_info}

**分析要求**:
基于以上数据进行综合趋势分析，重点考虑：
1. **波动率特征**: 价格波动幅度、波动趋势变化
2. **成交量分析**: 成交量分布、量价关系、买卖力量对比
3. **深度流动性**: 订单簿买卖盘力量对比、价差分析
4. **市场情绪**: 结合价格、成交量、深度的综合判断

**输出格式**:
{{
  "trend": "震荡/微弱看涨/看涨/强力看涨/微弱看跌/看跌/强力看跌",
  "confidence": 0.0-1.0,
  "reason": "100字以内的详细分析原因，必须包含波动率、成交量、深度流动性等关键信息"
}}

只返回JSON，不要其他内容。"""

        return prompt

    async def _call_deepseek_api(self, prompt: str) -> str:
        """调用Deepseek API。

        Args:
            prompt: 分析提示词

        Returns:
            API响应文本

        Raises:
            RuntimeError: 当API调用失败时
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "你是一个专业的加密货币市场分析师，擅长基于交易数据进行趋势分析。请始终返回有效的JSON格式结果。"
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stream": False
        }

        url = f"{self.base_url}/chat/completions"

        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            try:
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise RuntimeError(
                            f"API请求失败: status={response.status}, error={error_text}"
                        )

                    response_data = await response.json()

                    if "choices" not in response_data or not response_data["choices"]:
                        raise RuntimeError("API响应格式错误：缺少choices字段")

                    message = response_data["choices"][0]["message"]
                    if "content" not in message:
                        raise RuntimeError("API响应格式错误：缺少content字段")

                    return message["content"]

            except aiohttp.ClientError as e:
                raise RuntimeError(f"网络请求失败: {str(e)}") from e
            except Exception as e:
                raise RuntimeError(f"API调用异常: {str(e)}") from e

    def _parse_response(self, response: str, symbol: str) -> TrendAnalysisResult:
        """解析API响应。

        Args:
            response: API响应文本
            symbol: 交易符号

        Returns:
            趋势分析结果

        Raises:
            ValueError: 当响应格式无效时
        """
        try:
            # 尝试提取JSON部分
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.endswith("```"):
                response = response[:-3]
            response = response.strip()

            # 解析JSON
            data = json.loads(response)

            # 提取核心字段
            trend = data.get("trend", "震荡")
            reason = data.get("reason", "暂无分析原因")
            confidence = float(data.get("confidence", 0.5))

            # 智能生成强度等级（基于趋势类型）
            strength_levels = self._calculate_strength_levels(trend, confidence)

            # 创建分析元数据
            analysis_metadata = {
                "symbol": symbol,
                "analysis_method": "intelligent_trades_analysis",
                "model": self.model,
                "response_truncated": False
            }

            logger.info(f"解析响应成功: {trend}, 置信度: {confidence:.2f}")

            return TrendAnalysisResult(
                timestamp=datetime.now(),
                trend=trend,
                strength_levels=strength_levels,
                reason=reason,
                confidence=confidence,
                analysis_metadata=analysis_metadata
            )

        except json.JSONDecodeError as e:
            logger.error(f"JSON解析失败: {response[:100]}...")
            raise ValueError(f"JSON解析失败: {str(e)}") from e
        except Exception as e:
            logger.error(f"响应解析失败: {str(e)}")
            raise ValueError(f"响应解析失败: {str(e)}") from e

    def _calculate_strength_levels(self, trend: str, confidence: float) -> dict[str, float]:
        """根据趋势和置信度智能计算强度等级。

        Args:
            trend: 趋势类型
            confidence: 置信度

        Returns:
            强度等级字典
        """
        # 基础强度值基于置信度
        base_strength = confidence * 0.8  # 保留一些余地

        # 根据趋势类型分配强度
        if "强力看涨" in trend:
            return {
                "strong_support": base_strength * 0.9,
                "weak_support": base_strength * 0.7,
                "strong_resistance": base_strength * 0.2,
                "weak_resistance": base_strength * 0.4
            }
        elif "看涨" in trend:
            return {
                "strong_support": base_strength * 0.7,
                "weak_support": base_strength * 0.8,
                "strong_resistance": base_strength * 0.3,
                "weak_resistance": base_strength * 0.5
            }
        elif "微弱看涨" in trend:
            return {
                "strong_support": base_strength * 0.5,
                "weak_support": base_strength * 0.6,
                "strong_resistance": base_strength * 0.4,
                "weak_resistance": base_strength * 0.5
            }
        elif "强力看跌" in trend:
            return {
                "strong_support": base_strength * 0.2,
                "weak_support": base_strength * 0.4,
                "strong_resistance": base_strength * 0.9,
                "weak_resistance": base_strength * 0.7
            }
        elif "看跌" in trend:
            return {
                "strong_support": base_strength * 0.3,
                "weak_support": base_strength * 0.5,
                "strong_resistance": base_strength * 0.7,
                "weak_resistance": base_strength * 0.8
            }
        elif "微弱看跌" in trend:
            return {
                "strong_support": base_strength * 0.4,
                "weak_support": base_strength * 0.5,
                "strong_resistance": base_strength * 0.5,
                "weak_resistance": base_strength * 0.6
            }
        else:  # 震荡
            return {
                "strong_support": base_strength * 0.5,
                "weak_support": base_strength * 0.5,
                "strong_resistance": base_strength * 0.5,
                "weak_resistance": base_strength * 0.5
            }

    def _update_stats(self, response_time: float, success: bool) -> None:
        """更新统计信息。

        Args:
            response_time: 响应时间
            success: 是否成功
        """
        if success:
            self.stats["successful_requests"] += 1
        else:
            self.stats["failed_requests"] += 1

        # 更新平均响应时间
        total_requests = self.stats["total_requests"]
        current_avg = self.stats["average_response_time"]
        self.stats["average_response_time"] = (
            (current_avg * (total_requests - 1) + response_time) / total_requests
        )

    def get_stats(self) -> dict[str, Any]:
        """获取统计信息。

        Returns:
            统计信息字典
        """
        success_rate = (
            self.stats["successful_requests"] / self.stats["total_requests"]
            if self.stats["total_requests"] > 0 else 0
        )

        return {
            **self.stats,
            "success_rate": success_rate,
            "model": self.model,
            "max_tokens": self.max_tokens
        }

    def close(self) -> None:
        """关闭资源。"""
        logger.info("DeepSeekAnalyzer closed")
