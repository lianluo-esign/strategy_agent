"""DeepSeek LLM analyzer for market structure analysis.

This module provides DeepSeek LLM integration specifically for analyzing
aggregated depth snapshot data to provide detailed support/resistance
analysis and market structure insights, not trading recommendations.
"""

import json
import logging
from decimal import Decimal
from typing import Any, Dict, List, Optional

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


class DeepSeekOrderBookAnalyzer:
    """DeepSeek LLM analyzer for order book and market structure analysis.

    This analyzer uses DeepSeek LLM to analyze aggregated order book data
    and provide detailed market structure insights, support/resistance levels,
    and liquidity analysis.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.deepseek.com/v1",
        model: str = "deepseek-chat",
        max_tokens: int = 4000,
        temperature: float = 0.1,
        timeout: int = 60*1000,
        max_retries: int = 3,
    ):
        """Initialize DeepSeek order book analyzer.

        Args:
            api_key: DeepSeek API key
            base_url: API base URL
            model: Model name
            max_tokens: Maximum tokens in response
            temperature: Temperature setting (0.0-1.0)
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
        """
        if not api_key:
            raise ValueError("DeepSeek API key is required")

        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries

        # Initialize synchronous HTTP client
        self.client = httpx.Client(
            base_url=base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
        )

        logger.info(
            f"Initialized DeepSeekOrderBookAnalyzer with model={model}, "
            f"max_tokens={max_tokens}, temperature={temperature}"
        )

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def analyze_order_book_with_llm(
        self,
        aggregated_bids: Dict[Decimal, Decimal],
        aggregated_asks: Dict[Decimal, Decimal],
        symbol: str = "BTCFDUSD",
    ) -> Dict[str, Any]:
        """Analyze aggregated order book data using DeepSeek LLM.

        Args:
            aggregated_bids: Dictionary of aggregated bid prices and volumes
            aggregated_asks: Dictionary of aggregated ask prices and volumes
            symbol: Trading symbol

        Returns:
            Dictionary containing LLM analysis results
        """
        try:
            logger.info(f"Starting DeepSeek LLM analysis for {symbol}")

            # Prepare system prompt for market structure analysis
            system_prompt = self._get_market_structure_system_prompt()

            # Prepare user prompt with order book data
            user_prompt = self._format_order_book_prompt(
                aggregated_bids, aggregated_asks, symbol
            )

            # Make synchronous API request
            response_data = self._make_api_request(system_prompt, user_prompt)

            # Parse and structure the analysis
            analysis_result = self._parse_llm_analysis(response_data, symbol)

            logger.info(f"DeepSeek LLM analysis completed for {symbol}")
            return analysis_result

        except Exception as e:
            logger.error(f"DeepSeek LLM analysis failed for {symbol}: {e}")
            return self._create_error_analysis(symbol, str(e))

    def _get_market_structure_system_prompt(self) -> str:
        """Get system prompt for market structure analysis."""
        return """你是一个专业的加密货币市场结构分析师，专门分析BTC-FDUSD现货市场的订单簿数据。

你的任务是分析聚合后的深度快照数据，识别关键的市场结构特征，包括：

1. **支撑/阻力层级分析**：
   - 识别重要的买盘支撑区域（高流动性买盘集中区）
   - 识别重要的卖盘阻力区域（高流动性卖盘集中区）
   - 评估每个层级的强度和可靠性

2. **流动性分布分析**：
   - 分析订单簿中的流动性分布特征
   - 识别流动性密集区和流动性稀疏区
   - 评估市场深度和流动性质量

3. **市场平衡分析**：
   - 评估买卖压力的平衡状态
   - 识别潜在的价格突破方向
   - 分析订单簿的不对称性

4. **关键价格水平**：
   - 识别对价格走势有重要影响的关键价位
   - 分析价格水平之间的相互关系
   - 评估价格水平的支撑/阻力转换概率

**重要**：
- 专注于市场结构分析，不要提供交易建议或买卖信号
- 基于聚合后的订单簿数据进行客观分析
- 提供详细的数据支撑和分析逻辑
- 使用中文输出分析结果

**输出格式**：
请按照以下结构组织你的分析结果：
{
  "支撑区域": [
    {
      "价格区间": "价格范围",
      "强度": "0-100的评分",
      "特征": "该区域的详细特征描述",
      "流动性": "流动性评估"
    }
  ],
  "阻力区域": [
    {
      "价格区间": "价格范围",
      "强度": "0-100的评分",
      "特征": "该区域的详细特征描述",
      "流动性": "流动性评估"
    }
  ],
  "市场平衡": {
    "状态": "买盘强势/卖盘强势/相对平衡",
    "分析": "详细的市场平衡分析"
  },
  "关键价位": [
    {
      "价格": "具体价格",
      "重要性": "该价格的重要性描述",
      "作用": "支撑/阻力/关键价位"
    }
  ],
  "流动性特征": {
    "分布": "流动性分布特征描述",
    "质量": "流动性质量评估",
    "风险点": "潜在的市场风险点"
  }
}"""

    def _format_order_book_prompt(
        self,
        aggregated_bids: Dict[Decimal, Decimal],
        aggregated_asks: Dict[Decimal, Decimal],
        symbol: str,
    ) -> str:
        """Format order book data for LLM analysis.

        Args:
            aggregated_bids: Aggregated bid data
            aggregated_asks: Aggregated ask data
            symbol: Trading symbol

        Returns:
            Formatted prompt string
        """
        # Sort and select top levels for analysis
        sorted_bids = sorted(aggregated_bids.items(), key=lambda x: x[0], reverse=True)
        sorted_asks = sorted(aggregated_asks.items(), key=lambda x: x[0])

        # Select top 20 levels from each side for manageable analysis
        top_bids = sorted_bids[:20]
        top_asks = sorted_asks[:20]

        # Format order book data
        bid_data = []
        for price, volume in top_bids:
            bid_data.append({
                "价格": f"${float(price):,.2f}",
                "挂单量": f"{float(volume):.2f}",
                "价格等级": self._get_price_level(float(price))
            })

        ask_data = []
        for price, volume in top_asks:
            ask_data.append({
                "价格": f"${float(price):,.2f}",
                "挂单量": f"{float(volume):.2f}",
                "价格等级": self._get_price_level(float(price))
            })

        # Calculate summary statistics
        total_bid_volume = sum(float(v) for _, v in top_bids)
        total_ask_volume = sum(float(v) for _, v in top_asks)
        best_bid = float(top_bids[0][0]) if top_bids else 0
        best_ask = float(top_asks[0][0]) if top_asks else 0
        spread = best_ask - best_bid if best_ask > 0 and best_bid > 0 else 0

        prompt = f"""请分析以下{symbol}的聚合订单簿数据：

**市场概况**：
- 买盘总量：{total_bid_volume:.2f}
- 卖盘总量：{total_ask_volume:.2f}
- 最优买价：${best_bid:,.2f}
- 最优卖价：${best_ask:,.2f}
- 买卖价差：${spread:,.2f}

**买盘数据（前20档）**：
{json.dumps(bid_data, ensure_ascii=False, indent=2)}

**卖盘数据（前20档）**：
{json.dumps(ask_data, ensure_ascii=False, indent=2)}

**分析要求**：
请基于以上订单簿数据，详细分析市场结构特征，重点关注：
1. 重要的支撑区域和阻力区域
2. 流动性分布特征和潜在风险点
3. 市场买卖力量的平衡状态
4. 对价格走势可能有重要影响的关键价位

请提供客观的结构化分析，避免任何交易建议。"""

        return prompt

    def _get_price_level(self, price: float) -> str:
        """Get price level classification.

        Args:
            price: Price value

        Returns:
            Price level classification string
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

    def _make_api_request(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """Make synchronous API request to DeepSeek.

        Args:
            system_prompt: System prompt for the analysis
            user_prompt: User prompt with order book data

        Returns:
            API response data
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
        return response.json()

    def _parse_llm_analysis(self, response_data: Dict[str, Any], symbol: str) -> Dict[str, Any]:
        """Parse LLM analysis response.

        Args:
            response_data: Raw API response
            symbol: Trading symbol

        Returns:
            Structured analysis result
        """
        try:
            content = response_data["choices"][0]["message"]["content"]

            # Try to extract JSON from the response
            json_start = content.find("{")
            json_end = content.rfind("}") + 1

            if json_start != -1 and json_end > json_start:
                json_str = content[json_start:json_end]
                try:
                    analysis_json = json.loads(json_str)
                    return {
                        "symbol": symbol,
                        "analysis_type": "market_structure",
                        "raw_content": content,
                        "structured_analysis": analysis_json,
                        "status": "success",
                        "timestamp": None  # Will be set by caller
                    }
                except json.JSONDecodeError:
                    logger.warning("Failed to parse JSON from LLM response, using raw content")

            # Fallback: return raw content
            return {
                "symbol": symbol,
                "analysis_type": "market_structure",
                "raw_content": content,
                "structured_analysis": None,
                "status": "success",
                "timestamp": None  # Will be set by caller
            }

        except Exception as e:
            logger.error(f"Failed to parse LLM analysis response: {e}")
            return self._create_error_analysis(symbol, f"解析失败: {str(e)}")

    def _create_error_analysis(self, symbol: str, error_message: str) -> Dict[str, Any]:
        """Create error analysis result.

        Args:
            symbol: Trading symbol
            error_message: Error description

        Returns:
            Error analysis result
        """
        return {
            "symbol": symbol,
            "analysis_type": "market_structure",
            "raw_content": None,
            "structured_analysis": None,
            "status": "error",
            "error": error_message,
            "timestamp": None  # Will be set by caller
        }

    def close(self) -> None:
        """Close the HTTP client."""
        self.client.close()
        logger.info("DeepSeekOrderBookAnalyzer HTTP client closed")


def print_deepseek_analysis_results(results: Dict[str, Any]) -> None:
    """Print DeepSeek analysis results in a clean, organized format.

    Args:
        results: DeepSeek analysis results
    """
    if results.get("status") != "success":
        print(f"\n❌ DeepSeek分析失败: {results.get('error', '未知错误')}")
        return

    symbol = results.get("symbol", "UNKNOWN")
    print(f"\n=== {symbol} DeepSeek LLM 市场结构分析 ===")

    structured_analysis = results.get("structured_analysis")
    if structured_analysis:
        # Print structured analysis
        if "支撑区域" in structured_analysis:
            print("\n🟢 买盘支撑区域:")
            for i, support in enumerate(structured_analysis["支撑区域"], 1):
                print(f"  支撑 {i}: {support.get('价格区间', 'N/A')} | "
                      f"强度: {support.get('强度', 'N/A')} | "
                      f"特征: {support.get('特征', 'N/A')[:50]}{'...' if len(support.get('特征', '')) > 50 else ''}")

        if "阻力区域" in structured_analysis:
            print("\n🔻 卖盘阻力区域:")
            for i, resistance in enumerate(structured_analysis["阻力区域"], 1):
                print(f"  阻力 {i}: {resistance.get('价格区间', 'N/A')} | "
                      f"强度: {resistance.get('强度', 'N/A')} | "
                      f"特征: {resistance.get('特征', 'N/A')[:50]}{'...' if len(resistance.get('特征', '')) > 50 else ''}")

        if "市场平衡" in structured_analysis:
            balance = structured_analysis["市场平衡"]
            print(f"\n⚖️  市场平衡状态: {balance.get('状态', 'N/A')}")
            if balance.get('分析'):
                print(f"   分析: {balance['分析'][:100]}{'...' if len(balance['分析']) > 100 else ''}")

        if "关键价位" in structured_analysis:
            print("\n📍 关键价位:")
            for i, key_level in enumerate(structured_analysis["关键价位"], 1):
                print(f"  关键价位 {i}: ${key_level.get('价格', 'N/A')} | "
                      f"作用: {key_level.get('作用', 'N/A')} | "
                      f"重要性: {key_level.get('重要性', 'N/A')[:30]}{'...' if len(key_level.get('重要性', '')) > 30 else ''}")

        if "流动性特征" in structured_analysis:
            liquidity = structured_analysis["流动性特征"]
            print(f"\n💧 流动性特征:")
            if liquidity.get('分布'):
                print(f"   分布: {liquidity['分布'][:80]}{'...' if len(liquidity['分布']) > 80 else ''}")
            if liquidity.get('质量'):
                print(f"   质量: {liquidity['质量'][:80]}{'...' if len(liquidity['质量']) > 80 else ''}")
            if liquidity.get('风险点'):
                print(f"   风险点: {liquidity['风险点'][:80]}{'...' if len(liquidity['风险点']) > 80 else ''}")

    else:
        # Print raw content if structured analysis is not available
        raw_content = results.get("raw_content")
        if raw_content:
            print(f"\n📋 原始分析内容:")
            print(raw_content[:500] + "..." if len(raw_content) > 500 else raw_content)

    print("\n" + "="*50)