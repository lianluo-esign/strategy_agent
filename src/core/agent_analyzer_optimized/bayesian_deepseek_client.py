"""贝叶斯化的Deepseek AI分析客户端。

该模块专门用于基于贝叶斯思维框架的AI趋势分析，提供：
1. 贝叶斯化的提示词生成
2. 概率化的分析结果输出
3. 与静态订单簿数据的结合分析
4. 贝叶斯证据权重优化
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

# 趋势类型常量
TREND_TYPES = [
    "震荡", "微弱看涨", "看涨", "强力看涨",
    "微弱看跌", "看跌", "强力看跌"
]


class BayesianTrendResult:
    """贝叶斯趋势分析结果模型。"""

    def __init__(
        self,
        timestamp: datetime,
        posterior_probabilities: dict[str, float],
        most_likely_trend: str,
        confidence: float,
        uncertainty: float,
        analysis_reason: str,
        evidence_summary: dict[str, str],
        bayesian_metadata: dict[str, Any] | None = None
    ):
        self.timestamp = timestamp
        self.posterior_probabilities = posterior_probabilities
        self.most_likely_trend = most_likely_trend
        self.confidence = confidence
        self.uncertainty = uncertainty
        self.analysis_reason = analysis_reason
        self.evidence_summary = evidence_summary
        self.bayesian_metadata = bayesian_metadata or {}

    def validate(self) -> bool:
        """验证结果数据的有效性。"""
        if self.most_likely_trend not in TREND_TYPES:
            logger.error(f"无效的趋势类型: {self.most_likely_trend}")
            return False

        if not 0 <= self.confidence <= 1:
            logger.error(f"置信度必须在0-1之间: {self.confidence}")
            return False

        if not 0 <= self.uncertainty <= 1:
            logger.error(f"不确定性必须在0-1之间: {self.uncertainty}")
            return False

        # 验证概率分布
        total_prob = sum(self.posterior_probabilities.values())
        if abs(total_prob - 1.0) > 0.01:
            logger.error(f"概率分布总和不等于1: {total_prob}")
            return False

        return True

    def to_dict(self) -> dict[str, Any]:
        """转换为字典格式。"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "posterior_probabilities": self.posterior_probabilities,
            "most_likely_trend": self.most_likely_trend,
            "confidence": self.confidence,
            "uncertainty": self.uncertainty,
            "analysis_reason": self.analysis_reason,
            "evidence_summary": self.evidence_summary,
            "bayesian_metadata": self.bayesian_metadata
        }


class BayesianDeepSeekAnalyzer:
    """贝叶斯化的Deepseek AI分析器。

    该分析器基于贝叶斯思维框架，结合静态订单簿和动态交易数据，
    提供概率化的趋势分析结果。
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
        """初始化贝叶斯化Deepseek分析器。

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
            f"Initialized BayesianDeepSeekAnalyzer: model={model}, "
            f"max_tokens={max_tokens}, temperature={temperature}"
        )

    @retry(
        stop=stop_after_attempt(MAX_RETRIES),
        wait=wait_exponential(multiplier=RETRY_INITIAL_DELAY, max=RETRY_MAX_DELAY),
        reraise=True
    )
    async def analyze_bayesian_trend(
        self,
        static_data: dict[str, Any],
        dynamic_data: dict[str, Any],
        symbol: str = "BTCFDUSD"
    ) -> BayesianTrendResult:
        """分析贝叶斯趋势。

        Args:
            static_data: 静态订单簿分析数据
            dynamic_data: 动态交易数据分析数据
            symbol: 交易符号

        Returns:
            贝叶斯趋势分析结果

        Raises:
            ValueError: 当输入数据无效时
            RuntimeError: 当API调用失败时
        """
        start_time = time.time()
        self.stats["total_requests"] += 1

        try:
            # 验证输入数据
            if not static_data and not dynamic_data:
                raise ValueError("静态数据和动态数据不能都为空")

            # 构建贝叶斯分析提示词
            prompt = self._build_bayesian_analysis_prompt(static_data, dynamic_data, symbol)

            # 调用Deepseek API
            response = await self._call_deepseek_api(prompt)

            # 解析响应
            result = self._parse_bayesian_response(response, symbol)

            # 验证结果
            if not result.validate():
                raise ValueError("贝叶斯AI分析结果验证失败")

            # 更新统计信息
            response_time = time.time() - start_time
            self._update_stats(response_time, success=True)

            logger.info(
                f"贝叶斯趋势分析完成: trend={result.most_likely_trend}, "
                f"confidence={result.confidence:.3f}, "
                "uncertainty={result.uncertainty:.3f}, "
                f"response_time={response_time:.2f}s"
            )

            return result

        except Exception as e:
            response_time = time.time() - start_time
            self._update_stats(response_time, success=False)
            logger.error(f"贝叶斯趋势分析失败: {e}")
            raise RuntimeError(f"贝叶斯趋势分析失败: {str(e)}") from e

    def _build_bayesian_analysis_prompt(
        self,
        static_data: dict[str, Any],
        dynamic_data: dict[str, Any],
        symbol: str
    ) -> str:
        """构建贝叶斯分析提示词。

        Args:
            static_data: 静态订单簿数据
            dynamic_data: 动态交易数据
            symbol: 交易符号

        Returns:
            格式化的贝叶斯分析提示词
        """
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M")

        # 构建静态数据分析部分
        static_analysis = self._format_static_data_for_prompt(static_data)

        # 构建动态数据分析部分
        dynamic_analysis = self._format_dynamic_data_for_prompt(dynamic_data)

        prompt = f"""作为一名专业的量化分析师，请基于贝叶斯思维框架对{symbol}市场进行概率化趋势分析。

**分析时间**: {current_time}

**第一部分：静态订单簿深度分析 (10美元精度聚合)**
{static_analysis}

**第二部分：动态成交量历史数据分析**
{dynamic_analysis}

**第三部分：贝叶斯分析要求**

请严格按照贝叶斯思维框架进行分析：

1. **先验概率设定**：
   基于历史市场统计，各趋势类型的先验概率：
   - 震荡: 30%
   - 微弱看涨/微弱看跌: 各12%
   - 看涨/看跌: 各15%
   - 强力看涨/强力看跌: 各8%

2. **证据收集与似然计算**：

   **静态流动性证据**：
   - 订单簿不平衡度对趋势的指示作用
   - 流动性墙（大订单集中区域）的支撑阻力作用
   - 买卖盘深度对比反映的市场情绪

   **动态成交量证据**：
   - 成交量趋势变化反映的市场参与度
   - 成交量活跃度与趋势强度的关系
   - 成交量分布特征对趋势可持续性的影响

   **价格动能证据**：
   - 近期价格变化的方向和强度
   - 价格动能与成交量的共振关系
   - 动能持续性对趋势概率的影响

3. **贝叶斯更新计算**：
   对于每个趋势类型T，计算后验概率：
   P(T|证据) ∝ P(证据|T) × P(T)

   其中P(证据|T)是各证据的加权似然函数。

4. **概率化输出要求**：

   请严格按照以下JSON格式返回贝叶斯分析结果：

{{
  "posterior_probabilities": {{
    "震荡": 0.0-1.0,
    "微弱看涨": 0.0-1.0,
    "看涨": 0.0-1.0,
    "强力看涨": 0.0-1.0,
    "微弱看跌": 0.0-1.0,
    "看跌": 0.0-1.0,
    "强力看跌": 0.0-1.0
  }},
  "most_likely_trend": "概率最高的趋势类型",
  "confidence": 0.0-1.0,
  "uncertainty": 0.0-1.0,
  "analysis_reason": "详细的贝叶斯分析过程说明，包含关键证据对概率的影响",
  "evidence_summary": {{
    "static_liquidity": "静态流动性证据分析",
    "dynamic_volume": "动态成交量证据分析",
    "price_momentum": "价格动能证据分析"
  }},
  "bayesian_insights": {{
    "key_drivers": "驱动概率变化的关键因素",
    "evidence_consistency": "各证据之间的一致性分析",
    "risk_factors": "可能影响预测准确性的风险因素"
  }}
}}

**数值约束**：
- 所有概率值必须在0.0-1.0之间
- posterior_probabilities中所有概率之和必须约等于1.0 (误差±0.01)
- confidence表示最高概率的置信度
- uncertainty表示预测的不确定性程度

**分析重点**：
- 重视证据之间的相互验证
- 识别关键的概率驱动因素
- 提供清晰的逻辑推理链条
- 量化分析的不确定性

请确保返回的是有效的JSON格式，并严格遵循贝叶斯概率框架。"""

        return prompt

    def _format_static_data_for_prompt(self, static_data: dict[str, Any]) -> str:
        """格式化静态数据用于提示词。"""
        if not static_data or static_data.get("status") != "success":
            return "静态订单簿数据不可用"

        liquidity = static_data.get("liquidity_analysis", {})
        key_levels = static_data.get("key_levels", {})
        imbalance = static_data.get("imbalance_metrics", {})

        return f"""
**流动性概况**：
- 总流动性: {liquidity.get('total_liquidity', 0):.2f}
- 买方流动性: {liquidity.get('bid_side', {}).get('total_volume', 0):.2f} ({liquidity.get('bid_side', {}).get('levels_count', 0)}个价位)
- 卖方流动性: {liquidity.get('ask_side', {}).get('total_volume', 0):.2f} ({liquidity.get('ask_side', {}).get('levels_count', 0)}个价位)
- 买卖比率: {liquidity.get('bid_ask_ratio', 0):.2f}

**关键价格水平**：
- 最强支撑位: ${key_levels.get('strongest_support', {}).get('price', 0):.2f} (成交量: {key_levels.get('strongest_support', {}).get('volume', 0):.2f})
- 最强阻力位: ${key_levels.get('strongest_resistance', {}).get('price', 0):.2f} (成交量: {key_levels.get('strongest_resistance', {}).get('volume', 0):.2f})
- 当前最优买价: ${key_levels.get('best_bid', {}).get('price', 0):.2f}
- 当前最优卖价: ${key_levels.get('best_ask', {}).get('price', 0):.2f}

**订单不平衡分析**：
- 买方占比: {imbalance.get('bid_percentage', 0):.1%}
- 卖方占比: {imbalance.get('ask_percentage', 0):.1%}
- 不平衡强度: {imbalance.get('imbalance_strength', 0):.3f}
- 不平衡方向: {imbalance.get('direction', 'neutral')}"""

    def _format_dynamic_data_for_prompt(self, dynamic_data: dict[str, Any]) -> str:
        """格式化动态数据用于提示词。"""
        if not dynamic_data:
            return "动态交易数据不可用"

        minute_data_points = dynamic_data.get("minute_data_points", [])
        data_points_count = dynamic_data.get("data_points_count", 0)

        if not minute_data_points:
            return "动态交易数据为空"

        # 计算成交量统计
        total_volume = 0
        volume_trend = []
        price_levels_count = set()

        for point in minute_data_points[-20:]:  # 最近20个数据点
            price_levels = point.get("price_levels", {})
            point_volume = 0

            for price_str, level_data in price_levels.items():
                volume = float(level_data.get("total_volume", 0))
                point_volume += volume
                price_levels_count.add(price_str)

            total_volume += point_volume
            volume_trend.append(point_volume)

        # 成交量趋势分析
        if len(volume_trend) >= 10:
            recent_avg = sum(volume_trend[-5:]) / 5
            earlier_avg = sum(volume_trend[-10:-5]) / 5
            trend_direction = "上升" if recent_avg > earlier_avg * 1.2 else "下降" if recent_avg < earlier_avg * 0.8 else "稳定"
        else:
            trend_direction = "无法判断"

        # 价格分析
        prices = []
        for point in minute_data_points[-10:]:
            price_levels = point.get("price_levels", {})
            if price_levels:
                # 使用成交量加权平均价格
                total_weight = 0
                weighted_sum = 0
                for price_str, level_data in price_levels.items():
                    volume = float(level_data.get("total_volume", 0))
                    if volume > 0:
                        price = float(price_str)
                        weighted_sum += price * volume
                        total_weight += volume

                if total_weight > 0:
                    prices.append(weighted_sum / total_weight)

        price_momentum = "中性"
        if len(prices) >= 3:
            price_change = (prices[-1] - prices[0]) / prices[0] if prices[0] > 0 else 0
            if price_change > 0.002:
                price_momentum = f"上涨({price_change:.2%})"
            elif price_change < -0.002:
                price_momentum = f"下跌({price_change:.2%})"

        return f"""
**动态数据概况**：
- 数据时间跨度: {data_points_count} 分钟
- 总成交量: {total_volume:.2f}
- 活跃价格水平: {len(price_levels_count)} 个
- 最近20分钟成交量: {sum(volume_trend):.2f}

**成交量特征**：
- 成交量趋势: {trend_direction}
- 平均每分钟成交量: {sum(volume_trend) / len(volume_trend):.2f}
- 成交量活跃度: {'高' if total_volume > 500 else '中' if total_volume > 100 else '低'}

**价格动能**：
- 价格动量方向: {price_momentum}
- 分析价格点数: {len(prices)}
- 市场活跃度评估: {'活跃' if total_volume > 200 and len(price_levels_count) > 50 else '一般' if total_volume > 50 else '低迷'}"""

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
                    "content": "你是一个专业的量化分析师，精通贝叶斯统计学和概率论。请基于贝叶斯思维框架进行市场分析，返回准确的概率化结果。"
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

    def _parse_bayesian_response(self, response: str, symbol: str) -> BayesianTrendResult:
        """解析贝叶斯API响应。

        Args:
            response: API响应文本
            symbol: 交易符号

        Returns:
            贝叶斯趋势分析结果

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

            # 提取贝叶斯分析字段
            posterior_probabilities = data.get("posterior_probabilities", {})
            most_likely_trend = data.get("most_likely_trend", "震荡")
            confidence = float(data.get("confidence", 0.5))
            uncertainty = float(data.get("uncertainty", 0.5))
            analysis_reason = data.get("analysis_reason", "暂无贝叶斯分析原因")
            evidence_summary = data.get("evidence_summary", {})
            bayesian_insights = data.get("bayesian_insights", {})

            # 验证和修正概率分布
            posterior_probabilities = self._validate_and_normalize_probabilities(posterior_probabilities)

            # 确保所有趋势类型都有概率值
            for trend in TREND_TYPES:
                if trend not in posterior_probabilities:
                    posterior_probabilities[trend] = 0.0

            # 创建贝叶斯元数据
            bayesian_metadata = {
                "symbol": symbol,
                "analysis_method": "bayesian_deepseek_analysis",
                "response_raw": response[:500] + "..." if len(response) > 500 else response,
                "bayesian_insights": bayesian_insights
            }

            return BayesianTrendResult(
                timestamp=datetime.now(),
                posterior_probabilities=posterior_probabilities,
                most_likely_trend=most_likely_trend,
                confidence=confidence,
                uncertainty=uncertainty,
                analysis_reason=analysis_reason,
                evidence_summary=evidence_summary,
                bayesian_metadata=bayesian_metadata
            )

        except json.JSONDecodeError as e:
            raise ValueError(f"JSON解析失败: {str(e)}") from e
        except Exception as e:
            raise ValueError(f"贝叶斯响应解析失败: {str(e)}") from e

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
            "max_tokens": self.max_tokens,
            "analyzer_type": "bayesian_deepseek"
        }

    def _validate_and_normalize_probabilities(self, probabilities: dict[str, float]) -> dict[str, float]:
        """验证并归一化概率分布。

        Args:
            probabilities: 原始概率分布

        Returns:
            验证并归一化后的概率分布
        """
        if not probabilities:
            logger.error("概率分布为空，使用默认值")
            return self._get_default_probabilities()

        total = sum(probabilities.values())
        if total <= 0:
            logger.error("无效的概率分布，总和必须大于0")
            return self._get_default_probabilities()

        # 归一化概率分布
        if abs(total - 1.0) > 0.05:
            logger.warning(f"概率分布归一化: {total:.3f} -> 1.0")
            for trend in probabilities:
                probabilities[trend] = probabilities[trend] / total

        # 验证所有概率值在有效范围内
        for trend, prob in probabilities.items():
            if not 0 <= prob <= 1:
                logger.warning(f"概率值超出范围 [{trend}: {prob}]，设置为0")
                probabilities[trend] = 0.0

        return probabilities

    def _get_default_probabilities(self) -> dict[str, float]:
        """获取默认概率分布。"""
        return {
            "震荡": 0.30,
            "微弱看涨": 0.12,
            "看涨": 0.15,
            "强力看涨": 0.08,
            "微弱看跌": 0.12,
            "看跌": 0.15,
            "强力看跌": 0.08
        }

    def close(self) -> None:
        """关闭资源。"""
        logger.info("BayesianDeepSeekAnalyzer closed")
