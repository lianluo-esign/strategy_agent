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

from .volatility_volume_analyzer import VolatilityVolumeAnalyzer

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

        # 初始化波动率成交量分析器
        self.volatility_volume_analyzer = VolatilityVolumeAnalyzer()

        # 统计信息
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "average_response_time": 0.0
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
        """构建简化的贝叶斯分析提示词。

        Args:
            static_data: 静态订单簿数据
            dynamic_data: 动态交易数据
            symbol: 交易符号

        Returns:
            格式化的贝叶斯分析提示词
        """
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M")

        # 简化的静态数据摘要
        static_summary = self._get_static_summary(static_data)

        # 简化的动态数据摘要
        dynamic_summary = self._get_dynamic_summary(dynamic_data)

        prompt = f"""基于{symbol}市场数据进行简化贝叶斯趋势分析。

**分析时间**: {current_time}

**市场数据概览**:
{static_summary}
{dynamic_summary}

**分析要求**:
基于贝叶斯思维，结合静态订单簿和动态交易数据，给出概率化的趋势判断。

**输出格式**:
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
  "most_likely_trend": "最可能的趋势",
  "confidence": 0.0-1.0,
  "uncertainty": 0.0-1.0,
  "analysis_reason": "50字以内的分析原因，包含关键证据"
}}

**数值要求**:
- 所有概率值0.0-1.0，总和≈1.0
- confidence: 最高概率的置信度
- uncertainty: 预测不确定性
- analysis_reason: 简洁的概率化原因说明

只返回JSON，不要其他内容。"""

        return prompt

    def _get_static_summary(self, static_data: dict[str, Any]) -> str:
        """获取静态数据简化摘要。

        Args:
            static_data: 静态订单簿数据

        Returns:
            简化的静态数据摘要
        """
        if not static_data or static_data.get("status") != "success":
            return "- 静态数据: 不可用\n"

        liquidity = static_data.get("liquidity_analysis", {})
        imbalance = static_data.get("imbalance_metrics", {})

        return f"""- 静态订单簿:
  • 买卖比率: {liquidity.get('bid_ask_ratio', 0):.2f}
  • 不平衡度: {imbalance.get('imbalance_strength', 0):.3f}
  • 不平衡方向: {imbalance.get('direction', 'neutral')}
  • 总流动性: {liquidity.get('total_liquidity', 0):.0f}
"""

    def _get_dynamic_summary(self, dynamic_data: dict[str, Any]) -> str:
        """获取动态数据简化摘要。

        Args:
            dynamic_data: 动态交易数据

        Returns:
            简化的动态数据摘要
        """
        if not dynamic_data or dynamic_data.get("status") != "success":
            return "- 动态数据: 不可用\n"

        minute_data_points = dynamic_data.get("minute_data_points", [])
        if not minute_data_points:
            return "- 动态数据: 无数据点\n"

        # 计算基础统计
        total_volume = 0
        price_levels = set()

        for point in minute_data_points[-10:]:  # 最近10个点
            for price_str, level_data in point.get("price_levels", {}).items():
                volume = float(level_data.get("total_volume", 0))
                total_volume += volume
                price_levels.add(float(price_str))

        # 价格趋势分析
        prices = []
        for point in minute_data_points[-5:]:  # 最近5个点
            vwap = self._calculate_vwap(point.get("price_levels", {}))
            if vwap > 0:
                prices.append(vwap)

        price_momentum = "中性"
        if len(prices) >= 2:
            change = (prices[-1] - prices[0]) / prices[0]
            if change > 0.005:
                price_momentum = "上涨"
            elif change < -0.005:
                price_momentum = "下跌"

        return f"""- 动态交易数据:
  • 数据点: {len(minute_data_points)} 分钟
  • 成交量: {total_volume:.0f}
  • 价格水平: {len(price_levels)} 个
  • 价格动能: {price_momentum}
"""

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
        if not minute_data_points:
            return "动态交易数据为空"

        data_points_count = dynamic_data.get("data_points_count", 0)

        # 计算基础统计数据
        volume_stats = self._calculate_volume_stats(minute_data_points)
        price_analysis = self._analyze_price_momentum(minute_data_points)

        return f"""
**动态数据概况**：
- 数据时间跨度: {data_points_count} 分钟
- 总成交量: {volume_stats['total_volume']:.2f}
- 活跃价格水平: {volume_stats['price_levels_count']} 个
- 最近20分钟成交量: {volume_stats['recent_volume']:.2f}

**成交量特征**：
- 成交量趋势: {volume_stats['trend_direction']}
- 平均每分钟成交量: {volume_stats['avg_volume']:.2f}
- 成交量活跃度: {volume_stats['activity_level']}

**价格动能**：
- 价格动量方向: {price_analysis['momentum']}
- 分析价格点数: {len(price_analysis['prices'])}
- 市场活跃度评估: {price_analysis['market_activity']}"""

    def _calculate_volume_stats(self, minute_data_points: list[dict[str, Any]]) -> dict[str, Any]:
        """计算成交量统计数据。

        Args:
            minute_data_points: 分钟数据点

        Returns:
            成交量统计数据
        """
        total_volume: float = 0
        volume_trend: list[float] = []
        price_levels_count = set()

        for point in minute_data_points[-20:]:  # 最近20个数据点
            price_levels = point.get("price_levels", {})
            point_volume: float = 0

            for price_str, level_data in price_levels.items():
                volume = float(level_data.get("total_volume", 0))
                point_volume += volume
                price_levels_count.add(price_str)

            total_volume += point_volume
            volume_trend.append(point_volume)

        # 成交量趋势分析
        trend_direction = self._determine_volume_trend(volume_trend)
        avg_volume = sum(volume_trend) / len(volume_trend)
        activity_level = self._assess_volume_activity(total_volume)

        return {
            "total_volume": total_volume,
            "recent_volume": sum(volume_trend),
            "avg_volume": avg_volume,
            "trend_direction": trend_direction,
            "activity_level": activity_level,
            "price_levels_count": len(price_levels_count)
        }

    def _determine_volume_trend(self, volume_trend: list[float]) -> str:
        """确定成交量趋势方向。

        Args:
            volume_trend: 成交量趋势数据

        Returns:
            趋势方向字符串
        """
        if len(volume_trend) >= 10:
            recent_avg = sum(volume_trend[-5:]) / 5
            earlier_avg = sum(volume_trend[-10:-5]) / 5
            if recent_avg > earlier_avg * 1.2:
                return "上升"
            elif recent_avg < earlier_avg * 0.8:
                return "下降"
            else:
                return "稳定"
        return "无法判断"

    def _assess_volume_activity(self, total_volume: float) -> str:
        """评估成交量活跃度。

        Args:
            total_volume: 总成交量

        Returns:
            活跃度评估字符串
        """
        if total_volume > 500:
            return "高"
        elif total_volume > 100:
            return "中"
        else:
            return "低"

    def _analyze_price_momentum(self, minute_data_points: list[dict[str, Any]]) -> dict[str, Any]:
        """分析价格动能。

        Args:
            minute_data_points: 分钟数据点

        Returns:
            价格动能分析结果
        """
        prices = []
        for point in minute_data_points[-10:]:
            price_levels = point.get("price_levels", {})
            if price_levels:
                vwap = self._calculate_vwap(price_levels)
                if vwap > 0:
                    prices.append(vwap)

        momentum = self._calculate_momentum(prices)
        market_activity = self._assess_market_activity_level(prices, minute_data_points)

        return {
            "prices": prices,
            "momentum": momentum,
            "market_activity": market_activity
        }

    def _calculate_vwap(self, price_levels: dict[str, Any]) -> float:
        """计算成交量加权平均价格。

        Args:
            price_levels: 价格水平数据

        Returns:
            VWAP价格
        """
        total_weight: float = 0
        weighted_sum: float = 0

        for price_str, level_data in price_levels.items():
            volume = float(level_data.get("total_volume", 0))
            if volume > 0:
                price = float(price_str)
                weighted_sum += price * volume
                total_weight += volume

        return weighted_sum / total_weight if total_weight > 0 else 0

    def _calculate_momentum(self, prices: list[float]) -> str:
        """计算价格动能。

        Args:
            prices: 价格序列

        Returns:
            价格动能字符串
        """
        if len(prices) >= 3 and prices[0] > 0:
            price_change = (prices[-1] - prices[0]) / prices[0]
            if price_change > 0.002:
                return f"上涨({price_change:.2%})"
            elif price_change < -0.002:
                return f"下跌({price_change:.2%})"
        return "中性"

    def _assess_market_activity_level(self, prices: list[float], minute_data_points: list[dict[str, Any]]) -> str:
        """评估市场活跃度水平。

        Args:
            prices: 价格序列
            minute_data_points: 分钟数据点

        Returns:
            市场活跃度评估字符串
        """
        if len(minute_data_points) == 0:
            return "低迷"

        # 计算总成交量和价格水平数量
        total_volume: float = 0
        price_levels_count = set()

        for point in minute_data_points[-20:]:
            price_levels = point.get("price_levels", {})
            for price_str, level_data in price_levels.items():
                volume = float(level_data.get("total_volume", 0))
                total_volume += volume
                price_levels_count.add(price_str)

        if total_volume > 200 and len(price_levels_count) > 50:
            return "活跃"
        elif total_volume > 50:
            return "一般"
        else:
            return "低迷"

    def _analyze_volatility_volume_for_prompt(self, dynamic_data: dict[str, Any]) -> str:
        """分析波动率和成交量数据并格式化为提示词。

        Args:
            dynamic_data: 动态交易数据

        Returns:
            格式化的波动率和成交量分析字符串
        """
        if not dynamic_data or dynamic_data.get("status") != "success":
            return "波动率和成交量分析数据不可用"

        # 执行波动率和成交量分析
        analysis_result = self.volatility_volume_analyzer.analyze_volatility_volume(dynamic_data)

        if analysis_result.get("status") != "success":
            return f"波动率和成交量分析失败: {analysis_result.get('error', '未知错误')}"

        # 提取分析结果
        volatility_analysis = analysis_result.get("volatility_analysis", {})
        volume_analysis = analysis_result.get("volume_analysis", {})
        price_volume_relation = analysis_result.get("price_volume_relationship", {})
        market_activity = analysis_result.get("market_activity_assessment", {})
        summary = analysis_result.get("analysis_summary", {})

        # 格式化波动率分析
        vol_metrics = volatility_analysis.get("volatility_metrics", {})
        price_range = volatility_analysis.get("price_range_analysis", {})
        trend_analysis = volatility_analysis.get("trend_analysis", {})

        volatility_section = f"""
**价格波动率分析**：
- 当前波动率水平: {vol_metrics.get('volatility_level', '未知')}
- 实际波动率值: {vol_metrics.get('current_volatility', 0):.4f}
- 5分钟波动率: {vol_metrics.get('volatility_5min', 0):.4f}
- 15分钟波动率: {vol_metrics.get('volatility_15min', 0):.4f}
- 价格范围: ${price_range.get('lowest_price', 0):.2f} - ${price_range.get('highest_price', 0):.2f}
- 价格范围百分比: {price_range.get('price_range_percentage', 0):.2f}%
- 价格趋势方向: {trend_analysis.get('trend_direction', '未知')}
- 趋势强度: {trend_analysis.get('trend_strength_level', '未知')} ({trend_analysis.get('trend_strength_percent', 0):.3f}%)"""

        # 格式化成交量分析
        volume_basic = volume_analysis.get("basic_statistics", {})
        volume_trend = volume_analysis.get("trend_analysis", {})
        volume_activity = volume_analysis.get("activity_assessment", {})

        volume_section = f"""
**成交量特征分析**：
- 总成交量: {volume_basic.get('total_volume', 0):.2f}
- 平均每分钟成交量: {volume_basic.get('average_volume', 0):.2f}
- 成交量中位数: {volume_basic.get('median_volume', 0):.2f}
- 成交量标准差: {volume_basic.get('std_volume', 0):.2f}
- 成交量趋势方向: {volume_trend.get('trend_direction', '未知')} ({volume_trend.get('volume_trend_percent', 0):.1f}%)
- 成交量活跃度: {volume_activity.get('activity_level', '未知')}
- 成交量突增次数: {volume_activity.get('spike_count', 0)} 次"""

        # 格式化量价关系分析
        correlation = price_volume_relation.get("correlation_analysis", {})
        coordination = price_volume_relation.get("coordination_patterns", {})
        relationship = price_volume_relation.get("relationship_assessment", {})

        price_volume_section = f"""
**量价关系分析**：
- 价格-成交量相关系数: {correlation.get('correlation_coefficient', 0):.3f}
- 相关性强度: {correlation.get('correlation_strength', '未知')}
- 相关性方向: {correlation.get('correlation_direction', '未知')}
- 主导量价模式: {coordination.get('dominant_pattern', '无明显模式')}
- 健康上涨模式占比: {relationship.get('healthy_uptrend', 0):.1%}
- 抛售压力模式占比: {relationship.get('selling_pressure', 0):.1%}
- 上涨乏力模式占比: {relationship.get('weak_uptrend', 0):.1%}
- 止跌迹象模式占比: {relationship.get('stabilization', 0):.1%}"""

        # 格式化市场活跃度分析
        activity_scoring = market_activity.get("activity_scoring", {})
        activity_assessment = market_activity.get("activity_assessment", {})

        market_activity_section = f"""
**市场活跃度评估**：
- 综合活跃度评分: {activity_scoring.get('total_score', 0):.1f} / 100
- 活跃度等级: {activity_assessment.get('activity_level', '未知')}
- 市场状态: {activity_assessment.get('market_state', '未知')}
- 成交量评分: {activity_scoring.get('volume_score', 0):.1f}
- 波动率评分: {activity_scoring.get('volatility_score', 0):.1f}
- 价格多样性评分: {activity_scoring.get('diversity_score', 0):.1f}
- 交易频率评分: {activity_scoring.get('trades_score', 0):.1f}"""

        # 格式化关键洞察
        key_insights = summary.get("key_insights", [])

        insights_section = ""
        if key_insights:
            insights_section = "\n**关键洞察摘要**：\n"
            for insight in key_insights:
                insights_section += f"- {insight}\n"

        return volatility_section + volume_section + price_volume_section + market_activity_section + insights_section

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

                    return str(message["content"])

            except aiohttp.ClientError as e:
                raise RuntimeError(f"网络请求失败: {str(e)}") from e
            except Exception as e:
                raise RuntimeError(f"API调用异常: {str(e)}") from e

    def _parse_bayesian_response(self, response: str, symbol: str) -> BayesianTrendResult:
        """解析简化的贝叶斯API响应。

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

            # 提取核心字段
            posterior_probabilities = data.get("posterior_probabilities", {})
            most_likely_trend = data.get("most_likely_trend", "震荡")
            confidence = float(data.get("confidence", 0.5))
            uncertainty = float(data.get("uncertainty", 0.5))
            analysis_reason = data.get("analysis_reason", "暂无分析原因")

            # 验证和修正概率分布
            posterior_probabilities = self._validate_and_normalize_probabilities(posterior_probabilities)

            # 确保所有趋势类型都有概率值
            for trend in TREND_TYPES:
                if trend not in posterior_probabilities:
                    posterior_probabilities[trend] = 0.0

            # 简化的贝叶斯元数据，包含原始响应用于概率分布提取
            bayesian_metadata = {
                "symbol": symbol,
                "analysis_method": "simplified_bayesian_analysis",
                "simplified": True,
                "response_raw": response[:1000] + "..." if len(response) > 1000 else response
            }

            logger.info(f"解析简化贝叶斯响应成功: {most_likely_trend}, 置信度: {confidence:.2f}, 概率分布数量: {len(posterior_probabilities)}")

            return BayesianTrendResult(
                timestamp=datetime.now(),
                posterior_probabilities=posterior_probabilities,
                most_likely_trend=most_likely_trend,
                confidence=confidence,
                uncertainty=uncertainty,
                analysis_reason=analysis_reason,
                evidence_summary={},  # 简化为空字典
                bayesian_metadata=bayesian_metadata
            )

        except json.JSONDecodeError as e:
            logger.error(f"贝叶斯JSON解析失败: {response[:100]}...")
            raise ValueError(f"JSON解析失败: {str(e)}") from e
        except Exception as e:
            logger.error(f"贝叶斯响应解析失败: {str(e)}")
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
        current_avg: float = self.stats["average_response_time"]
        self.stats["average_response_time"] = float(
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
