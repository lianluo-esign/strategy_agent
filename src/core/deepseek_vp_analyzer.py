"""DeepSeek Volume Profile分析器 - 专门用于VP数据的AI分析。

这个模块提供DeepSeek LLM集成，专门分析Volume Profile数据：
1. 分析动态成交量和POC点
2. 识别主动成交密集区域
3. 提供流动性做市建议
"""

import json
import logging
from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


class DeepSeekVPAnalyzer:
    """DeepSeek LLM分析器，专门用于Volume Profile数据分析。

    这个分析器使用DeepSeek LLM分析Volume Profile数据，
    识别POC点、成交密集区域和流动性做市机会。
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.deepseek.com/v1",
        model: str = "deepseek-chat",
        max_tokens: int = 4000,
        temperature: float = 0.1,
        timeout: int = 60,
        max_retries: int = 3,
    ):
        """初始化DeepSeek VP分析器。

        Args:
            api_key: DeepSeek API密钥
            base_url: API基础URL
            model: 模型名称
            max_tokens: 最大令牌数
            temperature: 温度设置
            timeout: 请求超时时间（秒）
            max_retries: 最大重试次数
        """
        if not api_key:
            raise ValueError("DeepSeek API密钥是必需的")

        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries

        # 初始化HTTP客户端
        self.client = httpx.Client(
            base_url=base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
        )

        logger.info(
            f"Initialized DeepSeekVPAnalyzer with model={model}, "
            f"max_tokens={max_tokens}, temperature={temperature}"
        )

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def analyze_volume_profile_with_llm(
        self,
        vp_result: dict[str, Any],
    ) -> dict[str, Any]:
        """使用DeepSeek LLM分析Volume Profile数据。

        Args:
            vp_result: Volume Profile分析结果

        Returns:
            包含LLM分析结果的字典
        """
        try:
            symbol = vp_result.get("symbol", "UNKNOWN")
            logger.info(f"Starting DeepSeek LLM Volume Profile analysis for {symbol}")

            # 准备系统提示词
            system_prompt = self._get_volume_profile_system_prompt()

            # 准备用户提示词和VP数据
            user_prompt = self._create_volume_profile_prompt(vp_result)

            # 发起API请求
            response_data = self._make_api_request(system_prompt, user_prompt)

            # 解析和结构化分析结果
            analysis_result = self._parse_vp_analysis_response(response_data, symbol)

            logger.info(f"DeepSeek Volume Profile analysis completed for {symbol}")
            return analysis_result

        except Exception as e:
            logger.error(f"DeepSeek Volume Profile analysis failed: {e}")
            return self._create_error_analysis(symbol, str(e))

    def _get_volume_profile_system_prompt(self) -> str:
        """获取Volume Profile分析的系统提示词。"""
        return """你是一个专业的加密货币Volume Profile分析师，专门分析BTC-FDUSD现货市场的动态成交量数据。

你的任务是分析Volume Profile（成交量分布）数据，识别动态市场特征和流动性做市机会。

**Volume Profile分析重点**：
1. **POC点分析**：Point of Control（最大成交量价格）的市场意义
2. **成交密集区域**：高成交量集中的价格区间和其市场影响
3. **市场结构洞察**：通过成交量分布反映的市场参与度
4. **流动性做市策略**：基于VP数据的流动性部署建议

**分析原则**：
- Volume Profile反映了真实的成交价格分布，比静态订单簿更能体现市场共识
- POC点是市场最重要的价格水平，通常有最强的支撑/阻力作用
- 成交密集区域是流动性集中区，适合大额交易和做市活动
- 结合价格行为和成交量分布可以预测价格突破方向

**输出要求**：
请提供结构化的中文分析，避免任何直接交易建议，专注于市场结构分析。

**分析框架**：
- 使用结构化JSON格式输出分析结果
- 重点关注动态成交数据反映的市场特征
- 提供具体的流动性做市建议"""

    def _create_volume_profile_prompt(self, vp_result: dict[str, Any]) -> str:
        """创建Volume Profile分析的用户提示词。

        Args:
            vp_result: Volume Profile分析结果

        Returns:
            格式化的提示词字符串
        """
        symbol = vp_result.get("symbol", "UNKNOWN")
        vp_data = vp_result.get("vp_data", {})
        poc_analysis = vp_result.get("poc_analysis", {})

        # 获取顶级价格水平
        sorted_levels = sorted(vp_data.items(), key=lambda x: x[1], reverse=True)
        top_levels = sorted_levels[:15]  # 前15个最高成交量价格水平

        # 格式化价格水平数据
        price_levels_data = []
        for price, volume in top_levels:
            price_levels_data.append(
                {
                    "价格": f"${float(price):,.2f}",
                    "成交量": f"{float(volume):.2f}",
                    "价格区间": self._get_price_range_description(float(price)),
                }
            )

        # 计算统计信息
        total_volume = vp_result.get("total_volume", 0)
        poc_price = poc_analysis.get("poc_price", 0)
        poc_volume = poc_analysis.get("poc_volume", 0)
        value_area_high = poc_analysis.get("value_area_high", 0)
        value_area_low = poc_analysis.get("value_area_low", 0)
        value_area_range = poc_analysis.get("value_area_range", 0)

        prompt = f"""请分析以下{symbol}的24小时Volume Profile数据：

**Volume Profile基础信息**：
- 分析周期：24小时动态成交数据
- 总成交量：{total_volume:.2f}
- 价格水平数量：{len(vp_data)}个
- POC点价格：${float(poc_price):,.2f}
- POC点成交量：{poc_volume:.2f}
- 价值区间：${float(value_area_low):,.2f} - ${float(value_area_high):,.2f}
- 价值区间宽度：${float(value_area_range):,.2f}

**顶级成交量价格水平（按成交量排序）**：
{self._format_price_levels_data(price_levels_data)}

**市场背景**：
- Volume Profile反映了24小时内所有实际成交的价格分布
- POC点是成交最密集的价格，代表市场最重要的共识价位
- 价值区间包含了70%的总成交量，是主要的价格活动区域
- 成交量分布的形态可以揭示市场的参与度和情绪

**分析任务**：
请基于以上Volume Profile数据，进行专业的动态市场分析：

1. **POC点深度分析**：
   - POC点作为核心价格水平的市场意义
   - POC点周围成交量的分布特征
   - POC点对价格走势的潜在影响

2. **成交密集区域识别**：
   - 识别高成交量集中的关键价格区间
   - 分析这些区域的市场作用和支撑/阻力潜力
   - 评估这些区域对流动性管理的价值

3. **市场结构洞察**：
   - 成交量分布形态反映的市场特征（单峰、双峰、均匀等）
   - 通过VP数据揭示的买卖力量对比
   - 价格发现过程中的关键节点和转折点

4. **流动性做市策略**：
   - 基于VP数据的最佳流动性部署区域建议
   - 风险控制点和价格水平选择
   - 与传统支撑阻力区域的协同策略

**输出格式要求**：
请严格按照以下JSON格式输出分析结果：
{{
  "poc分析": {{
    "poc价格": "POC点具体价格",
    "市场意义": "POC点在当前市场环境中的重要性分析",
    "支撑阻力强度": "作为支撑或阻力强度的评估（0-100分）",
    "周围特征": "POC点周围成交量分布的详细特征"
  }},
  "成交密集区域": [
    {{
      "价格区间": "具体的价格范围",
      "成交量": "该区间的总成交量",
      "市场作用": "该区域在市场中的具体作用",
      "做市适用性": "该区域适合做市活动的程度评估",
      "特征描述": "该区域的详细特征描述"
    }}
  ],
  "市场结构洞察": {{
    "分布形态": "成交量分布的整体形态特征",
    "买卖力量": "通过VP数据反映的买卖力量对比",
    "关键节点": "重要的价格转折点或支撑阻力点",
    "市场情绪": "成交量分布反映的市场参与情绪"
  }},
  "流动性做市建议": {{
    "最佳部署区域": "推荐的流动性部署价格区间",
    "风险控制点": "关键的风险控制价格水平",
    "策略要点": "具体的做市策略要点和注意事项",
    "与静态分析协同": "与静态支撑阻力分析的协同建议"
  }}
}}

请确保分析结果客观专业，基于Volume Profile数据提供深入的市场洞察。"""

        return prompt

    def _format_price_levels_data(self, price_levels_data: list[dict[str, str]]) -> str:
        """格式化价格水平数据。

        Args:
            price_levels_data: 价格水平数据列表

        Returns:
            格式化的字符串
        """
        if not price_levels_data:
            return "无价格水平数据"

        formatted_lines = []
        for i, level in enumerate(price_levels_data, 1):
            line = (
                f"  {i:2d}. {level['价格']} | "
                f"成交量: {level['成交量']} | "
                f"区间: {level['价格区间']}"
            )
            formatted_lines.append(line)

        return "\n".join(formatted_lines)

    def _get_price_range_description(self, price: float) -> str:
        """获取价格区间描述。

        Args:
            price: 价格值

        Returns:
            价格区间描述
        """
        if price >= 100000:
            return "十万价位"
        elif price >= 10000:
            return "万价位"
        elif price >= 1000:
            return "千价位"
        elif price >= 100:
            return "百价位"
        else:
            return "十价位以下"

    def _make_api_request(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        """发起DeepSeek API请求。

        Args:
            system_prompt: 系统提示词
            user_prompt: 用户提示词

        Returns:
            API响应数据
        """
        request_data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }

        response = self.client.post("/chat/completions", json=request_data)
        response.raise_for_status()
        return response.json()  # type: ignore

    def _parse_vp_analysis_response(
        self, response_data: dict[str, Any], symbol: str
    ) -> dict[str, Any]:
        """解析Volume Profile分析响应。

        Args:
            response_data: 原始API响应
            symbol: 交易符号

        Returns:
            结构化的分析结果
        """
        try:
            content = response_data["choices"][0]["message"]["content"]

            # 尝试从响应中提取JSON
            json_start = content.find("{")
            json_end = content.rfind("}") + 1

            if json_start != -1 and json_end > json_start:
                json_str = content[json_start:json_end]
                try:
                    analysis_json = json.loads(json_str)
                    return {
                        "symbol": symbol,
                        "analysis_type": "volume_profile",
                        "raw_content": content,
                        "structured_analysis": analysis_json,
                        "status": "success",
                        "timestamp": None,  # 将由调用者设置
                    }
                except json.JSONDecodeError:
                    logger.warning(
                        "Failed to parse JSON from DeepSeek VP response, using raw content"
                    )

            # 回退：返回原始内容
            return {
                "symbol": symbol,
                "analysis_type": "volume_profile",
                "raw_content": content,
                "structured_analysis": None,
                "status": "success",
                "timestamp": None,  # 将由调用者设置
            }

        except Exception as e:
            logger.error(f"Failed to parse DeepSeek VP analysis response: {e}")
            return self._create_error_analysis(symbol, f"解析失败: {str(e)}")

    def _create_error_analysis(self, symbol: str, error_message: str) -> dict[str, Any]:
        """创建错误分析结果。

        Args:
            symbol: 交易符号
            error_message: 错误描述

        Returns:
            错误分析结果
        """
        return {
            "symbol": symbol,
            "analysis_type": "volume_profile",
            "raw_content": None,
            "structured_analysis": None,
            "status": "error",
            "error": error_message,
            "timestamp": None,  # 将由调用者设置
        }

    def close(self) -> None:
        """关闭HTTP客户端。"""
        self.client.close()
        logger.info("DeepSeekVPAnalyzer HTTP client closed")


def _print_poc_analysis(poc: dict[str, Any]) -> None:
    """打印POC分析结果。

    Args:
        poc: POC分析数据
    """
    print("\n🎯 POC点分析:")
    print(f"   POC价格: {poc.get('poc价格', 'N/A')}")
    market_meaning = poc.get('市场意义', 'N/A')
    print(f"   市场意义: {market_meaning[:100]}{'...' if len(market_meaning) > 100 else ''}")
    print(f"   支撑阻力强度: {poc.get('支撑阻力强度', 'N/A')}")
    surrounding_features = poc.get('周围特征', 'N/A')
    print(f"   周围特征: {surrounding_features[:80]}{'...' if len(surrounding_features) > 80 else ''}")


def _print_dense_areas(dense_areas: list[dict[str, Any]]) -> None:
    """打印成交密集区域。

    Args:
        dense_areas: 成交密集区域列表
    """
    print("\n📊 成交密集区域:")
    for i, area in enumerate(dense_areas, 1):
        print(
            f"   区域 {i}: {area.get('价格区间', 'N/A')} | "
            f"成交量: {area.get('成交量', 'N/A')} | "
            f"做市适用性: {area.get('做市适用性', 'N/A')}"
        )
        features = area.get("特征描述")
        if features:
            print(f"          特征: {features[:80]}{'...' if len(features) > 80 else ''}")


def _print_market_insights(insights: dict[str, Any]) -> None:
    """打印市场结构洞察。

    Args:
        insights: 市场洞察数据
    """
    print("\n🔍 市场结构洞察:")
    distribution = insights.get('分布形态', 'N/A')
    print(f"   分布形态: {distribution[:80]}{'...' if len(distribution) > 80 else ''}")
    power_balance = insights.get('买卖力量', 'N/A')
    print(f"   买卖力量: {power_balance[:80]}{'...' if len(power_balance) > 80 else ''}")
    key_nodes = insights.get("关键节点")
    if key_nodes:
        print(f"   关键节点: {key_nodes[:80]}{'...' if len(key_nodes) > 80 else ''}")


def _print_liquidity_suggestions(suggestions: dict[str, Any]) -> None:
    """打印流动性做市建议。

    Args:
        suggestions: 做市建议数据
    """
    print("\n💡 流动性做市建议:")
    print(f"   最佳部署区域: {suggestions.get('最佳部署区域', 'N/A')}")
    print(f"   风险控制点: {suggestions.get('风险控制点', 'N/A')}")
    strategy_points = suggestions.get("策略要点")
    if strategy_points:
        print(f"   策略要点: {strategy_points[:100]}{'...' if len(strategy_points) > 100 else ''}")


def _print_raw_content(raw_content: str) -> None:
    """打印原始分析内容。

    Args:
        raw_content: 原始内容字符串
    """
    print("\n📋 原始分析内容:")
    for line in raw_content.split("\n")[:8]:  # 限制前8行
        if line.strip():
            print(f"   {line[:120]}{'...' if len(line) > 120 else ''}")


def print_deepseek_vp_analysis_results(results: dict[str, Any]) -> None:
    """打印DeepSeek Volume Profile分析结果。

    Args:
        results: Volume Profile分析结果
    """
    if results.get("status") != "success":
        print(
            f"\n❌ DeepSeek Volume Profile分析失败: {results.get('error', '未知错误')}"
        )
        return

    symbol = results.get("symbol", "UNKNOWN")
    print(f"\n=== {symbol} DeepSeek LLM Volume Profile 分析 ===")

    structured_analysis = results.get("structured_analysis")
    if structured_analysis:
        if "poc分析" in structured_analysis:
            _print_poc_analysis(structured_analysis["poc分析"])

        if "成交密集区域" in structured_analysis:
            _print_dense_areas(structured_analysis["成交密集区域"])

        if "市场结构洞察" in structured_analysis:
            _print_market_insights(structured_analysis["市场结构洞察"])

        if "流动性做市建议" in structured_analysis:
            _print_liquidity_suggestions(structured_analysis["流动性做市建议"])
    else:
        raw_content = results.get("raw_content")
        if raw_content:
            _print_raw_content(raw_content)

    print("\n" + "=" * 60)
