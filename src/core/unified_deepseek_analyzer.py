"""统一DeepSeek分析器 - 合并深度快照和交易数据的AI分析。

这个模块提供统一的DeepSeek LLM集成，同时分析：
1. 静态深度快照数据（订单簿）
2. 动态交易窗口数据（Volume Profile）
3. 输出综合的支撑阻力位分析决策
"""

import json
import logging
from decimal import Decimal
from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


class UnifiedDeepSeekAnalyzer:
    """统一DeepSeek分析器，合并深度快照和交易数据分析。

    这个分析器使用单次AI请求同时分析静态订单簿数据和动态交易数据，
    输出综合的支撑阻力位分析和做市策略建议。
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.deepseek.com/v1",
        model: str = "deepseek-chat",
        max_tokens: int = 6000,
        temperature: float = 0.1,
        timeout: int = 90,
        max_retries: int = 3,
    ):
        """初始化统一DeepSeek分析器。

        Args:
            api_key: DeepSeek API密钥
            base_url: API基础URL
            model: 模型名称
            max_tokens: 最大令牌数（增加以支持更复杂的分析）
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
            f"Initialized UnifiedDeepSeekAnalyzer with model={model}, "
            f"max_tokens={max_tokens}, temperature={temperature}"
        )

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def analyze_unified_market_data(
        self,
        aggregated_bids: dict[Decimal, Decimal],
        aggregated_asks: dict[Decimal, Decimal],
        vp_result: dict[str, Any],
        symbol: str = "BTCFDUSD",
    ) -> dict[str, Any]:
        """使用单次AI请求统一分析深度快照和Volume Profile数据。

        Args:
            aggregated_bids: 聚合后的买盘数据
            aggregated_asks: 聚合后的卖盘数据
            vp_result: Volume Profile分析结果
            symbol: 交易符号

        Returns:
            包含综合分析结果的字典
        """
        try:
            # 输入验证
            self._validate_analysis_inputs(aggregated_bids, aggregated_asks, vp_result, symbol)

            logger.info(f"Starting unified DeepSeek analysis for {symbol}")

            # 准备系统提示词
            system_prompt = self._get_unified_analysis_system_prompt()

            # 准备用户提示词和综合数据
            user_prompt = self._create_unified_analysis_prompt(
                aggregated_bids, aggregated_asks, vp_result, symbol
            )

            # 发起API请求
            response_data = self._make_api_request(system_prompt, user_prompt)

            # 解析和结构化分析结果
            analysis_result = self._parse_unified_analysis_response(response_data, symbol)

            logger.info(f"Unified DeepSeek analysis completed for {symbol}")
            return analysis_result

        except ValueError as e:
            logger.error(f"Input validation failed for unified analysis: {e}")
            return self._create_error_analysis(symbol, f"输入验证失败: {str(e)}")
        except Exception as e:
            logger.error(f"Unified DeepSeek analysis failed: {e}")
            return self._create_error_analysis(symbol, str(e))

    def _validate_analysis_inputs(
        self,
        aggregated_bids: dict[Decimal, Decimal],
        aggregated_asks: dict[Decimal, Decimal],
        vp_result: dict[str, Any],
        symbol: str,
    ) -> None:
        """验证分析输入参数。

        Args:
            aggregated_bids: 聚合后的买盘数据
            aggregated_asks: 聚合后的卖盘数据
            vp_result: Volume Profile分析结果
            symbol: 交易符号

        Raises:
            ValueError: 当输入参数无效时
        """
        # 验证交易符号
        if not isinstance(symbol, str) or not symbol.strip():
            raise ValueError("Symbol must be a non-empty string")

        # 验证订单簿数据
        if not isinstance(aggregated_bids, dict):
            raise ValueError("Aggregated bids must be a dictionary")

        if not isinstance(aggregated_asks, dict):
            raise ValueError("Aggregated asks must be a dictionary")

        # 验证Volume Profile结果
        if not isinstance(vp_result, dict):
            raise ValueError("VP result must be a dictionary")

        if "vp_data" not in vp_result:
            raise ValueError("VP result must contain 'vp_data' field")

        # 验证价格和成交量数据类型
        for price, volume in aggregated_bids.items():
            if not isinstance(price, Decimal) or not isinstance(volume, Decimal):
                raise ValueError("Bid prices and volumes must be Decimal instances")
            if volume < 0:
                raise ValueError("Bid volumes cannot be negative")

        for price, volume in aggregated_asks.items():
            if not isinstance(price, Decimal) or not isinstance(volume, Decimal):
                raise ValueError("Ask prices and volumes must be Decimal instances")
            if volume < 0:
                raise ValueError("Ask volumes cannot be negative")

        # 验证买卖价顺序
        if aggregated_bids and aggregated_asks:
            max_bid = max(aggregated_bids.keys())
            min_ask = min(aggregated_asks.keys())
            if max_bid >= min_ask:
                logger.warning(f"Potential cross-spread detected: max_bid={max_bid}, min_ask={min_ask}")

        # 验证VP数据
        vp_data = vp_result.get("vp_data", {})
        if isinstance(vp_data, dict):
            for price, volume in vp_data.items():
                if not isinstance(price, (Decimal, int, float, str)) or not isinstance(volume, (Decimal, int, float, str)):
                    raise ValueError("VP data must contain numeric prices and volumes")

    def _get_unified_analysis_system_prompt(self) -> str:
        """获取统一分析的系统提示词。"""
        return """你是一个专业的加密货币市场分析师，专门为BTC-FDUSD高频做市策略提供支撑阻力位分析。

你的核心任务是综合分析静态订单簿深度数据和动态交易流数据，为现货高频做市策略提供：

1. **短期支撑位识别**：适合入场的价格位置
2. **短期阻力位识别**：适合退出的价格位置
3. **集中流动性供应区域**：最优的做市布局区间

**分析原则**：
- 结合静态深度（订单簿）和动态数据（成交量分布）进行综合判断
- 重点关注短期价格行为和流动性特征
- 识别高概率的支撑阻力位，为做市策略提供入场和退出信号
- 评估流动性集中区域，优化资金部署效率

**数据类型理解**：
1. **深度快照数据**：反映当前市场的挂单意愿和潜在支撑阻力
2. **Volume Profile数据**：反映24小时内实际成交的价格分布和市场共识
3. **综合分析**：结合两种数据识别更可靠的支撑阻力位

**输出要求**：
- 提供具体的入场和退出价格建议
- 识别适合集中流动性供应的区域
- 评估每个价格水平的可靠性
- 避免模糊表述，提供可操作的分析结果

**重要提醒**：
这是为高频做市策略提供的市场结构分析，专注于识别短期交易机会和流动性部署策略。
"""

    def _create_unified_analysis_prompt(
        self,
        aggregated_bids: dict[Decimal, Decimal],
        aggregated_asks: dict[Decimal, Decimal],
        vp_result: dict[str, Any],
        symbol: str,
    ) -> str:
        """创建统一分析的用户提示词。

        Args:
            aggregated_bids: 聚合买盘数据
            aggregated_asks: 聚合卖盘数据
            vp_result: Volume Profile分析结果
            symbol: 交易符号

        Returns:
            格式化的提示词字符串
        """
        # 处理深度快照数据
        sorted_bids = sorted(aggregated_bids.items(), key=lambda x: x[0], reverse=True)
        sorted_asks = sorted(aggregated_asks.items(), key=lambda x: x[0])
        top_bids = sorted_bids[:15]  # 前15档买盘
        top_asks = sorted_asks[:15]  # 前15档卖盘

        # 格式化订单簿数据
        order_book_data = {
            "买盘数据": [],
            "卖盘数据": []
        }

        for price, volume in top_bids:
            order_book_data["买盘数据"].append({
                "价格": f"${float(price):,.2f}",
                "挂单量": f"{float(volume):.2f}",
                "价格等级": self._get_price_level(float(price))
            })

        for price, volume in top_asks:
            order_book_data["卖盘数据"].append({
                "价格": f"${float(price):,.2f}",
                "挂单量": f"{float(volume):.2f}",
                "价格等级": self._get_price_level(float(price))
            })

        # 计算订单簿统计信息
        total_bid_volume = sum(float(v) for _, v in top_bids)
        total_ask_volume = sum(float(v) for _, v in top_asks)
        best_bid = float(top_bids[0][0]) if top_bids else 0
        best_ask = float(top_asks[0][0]) if top_asks else 0
        spread = best_ask - best_bid if best_ask > 0 and best_bid > 0 else 0

        # 处理Volume Profile数据
        vp_data = vp_result.get("vp_data", {})
        poc_analysis = vp_result.get("poc_analysis", {})
        sorted_levels = sorted(vp_data.items(), key=lambda x: x[1], reverse=True)
        top_vp_levels = sorted_levels[:10]  # 前10个最高成交量价格水平

        # 格式化Volume Profile数据
        vp_levels_data = []
        for price, volume in top_vp_levels:
            vp_levels_data.append({
                "价格": f"${float(price):,.2f}",
                "成交量": f"{float(volume):.2f}",
                "价格区间": self._get_price_level(float(price))
            })

        # 计算VP统计信息
        total_volume = vp_result.get("total_volume", 0)
        poc_price = poc_analysis.get("poc_price", 0)
        poc_volume = poc_analysis.get("poc_volume", 0)
        value_area_high = poc_analysis.get("value_area_high", 0)
        value_area_low = poc_analysis.get("value_area_low", 0)

        prompt = f"""请为{symbol}高频做市策略进行综合市场分析：

**市场概况**：
- 买盘总量：{total_bid_volume:.2f}
- 卖盘总量：{total_ask_volume:.2f}
- 最优买价：${best_bid:,.2f}
- 最优卖价：${best_ask:,.2f}
- 买卖价差：${spread:,.2f}
- 24小时总成交量：{total_volume:.2f}

**深度快照数据（静态订单簿）**：
{json.dumps(order_book_data, ensure_ascii=False, indent=2)}

**Volume Profile数据（动态成交分布）**：
- POC点价格：${float(poc_price):,.2f}
- POC点成交量：{poc_volume:.2f}
- 价值区间：${float(value_area_low):,.2f} - ${float(value_area_high):,.2f}

**顶级成交量价格水平**：
{json.dumps(vp_levels_data, ensure_ascii=False, indent=2)}

**分析任务**：
请基于以上综合数据，为高频做市策略提供以下分析：

1. **短期支撑位分析（入场机会）**：
   - 识别最强的3个支撑位价格
   - 评估每个支撑位的可靠性（0-100分）
   - 分析支撑位形成的原因（订单簿支撑/成交量共识）
   - 推荐最佳的入场价格区间

2. **短期阻力位分析（退出目标）**：
   - 识别最强的3个阻力位价格
   - 评估每个阻力位的可靠性（0-100分）
   - 分析阻力位形成的原因（订单簿阻力/历史成交压力）
   - 推荐最佳的退出价格区间

3. **集中流动性供应区域**：
   - 识别最适合部署集中流动性的价格区间
   - 分析该区域的市场特征和优势
   - 评估流动性的安全性和收益潜力
   - 提供具体的流动性部署建议

4. **做市策略要点**：
   - 当前市场环境下做市的主要机会
   - 需要重点关注的风险点
   - 最优的仓位管理建议
   - 入场和退出的时机把握

**输出格式要求**：
请严格按照以下JSON格式输出分析结果：
{{
  "短期支撑位": [
    {{
      "价格": "具体支撑价格",
      "可靠性评分": "0-100分",
      "形成原因": "订单簿支撑/成交量共识/综合因素",
      "推荐入场区间": "建议的入场价格范围",
      "特征描述": "该支撑位的详细特征"
    }}
  ],
  "短期阻力位": [
    {{
      "价格": "具体阻力价格",
      "可靠性评分": "0-100分",
      "形成原因": "订单簿阻力/历史成交压力/综合因素",
      "推荐退出区间": "建议的退出价格范围",
      "特征描述": "该阻力位的详细特征"
    }}
  ],
  "集中流动性供应区域": {{
    "最佳价格区间": "推荐的主要流动性部署区间",
    "备选区间": ["备选的价格区间1", "备选的价格区间2"],
    "市场特征": "该区域的市场特征和优势",
    "安全性评估": "流动性安全性分析",
    "收益潜力": "预期收益潜力评估"
  }},
  "做市策略要点": {{
    "主要机会": "当前市场环境下的主要做市机会",
    "风险控制": "需要重点关注的风险点",
    "仓位管理": "最优的仓位管理建议",
    "时机把握": "入场和退出的时机建议",
    "策略总结": "简洁的策略总结"
  }}
}}

请确保分析结果具体、可操作，直接服务于高频做市策略的决策需求。"""

        return prompt

    def _get_price_level(self, price: float) -> str:
        """获取价格等级分类。

        Args:
            price: 价格值

        Returns:
            价格等级分类字符串
        """
        # 输入验证
        if not isinstance(price, (int, float)):
            raise ValueError(f"Price must be a number, got {type(price)}")

        if price < 0:
            raise ValueError(f"Price cannot be negative: {price}")

        # 处理极值
        if price > 1e9:  # 10亿以上
            return "十亿价位以上"
        elif price >= 100000:
            return "十万价位"
        elif price >= 10000:
            return "万价位"
        elif price >= 1000:
            return "千价位"
        elif price >= 100:
            return "百价位"
        elif price >= 0:
            return "十价位以下"
        else:
            return "无效价位"

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
        return response.json()

    def _parse_unified_analysis_response(
        self, response_data: dict[str, Any], symbol: str
    ) -> dict[str, Any]:
        """解析统一分析响应。

        Args:
            response_data: 原始API响应
            symbol: 交易符号

        Returns:
            结构化的分析结果
        """
        try:
            # 验证响应数据结构
            if not isinstance(response_data, dict):
                raise ValueError("Response data is not a dictionary")

            if "choices" not in response_data or not response_data["choices"]:
                raise ValueError("No choices found in response data")

            if not isinstance(response_data["choices"], list) or len(response_data["choices"]) == 0:
                raise ValueError("Invalid choices format in response data")

            choice = response_data["choices"][0]
            if "message" not in choice or "content" not in choice["message"]:
                raise ValueError("Invalid message format in response choice")

            content = choice["message"]["content"]
            if not isinstance(content, str) or not content.strip():
                raise ValueError("Empty or invalid content in response")

            # 尝试从响应中提取JSON
            json_start = content.find("{")
            json_end = content.rfind("}") + 1

            if json_start != -1 and json_end > json_start and json_end <= len(content):
                json_str = content[json_start:json_end]
                try:
                    analysis_json = json.loads(json_str)
                    # 验证解析后的JSON包含必要字段
                    if not isinstance(analysis_json, dict):
                        raise ValueError("Parsed JSON is not a dictionary")

                    return {
                        "symbol": symbol,
                        "analysis_type": "unified_market_analysis",
                        "raw_content": content,
                        "structured_analysis": analysis_json,
                        "status": "success",
                        "timestamp": None,  # 将由调用者设置
                    }
                except json.JSONDecodeError as e:
                    logger.warning(
                        f"Failed to parse JSON from unified analysis response: {e}. Using raw content."
                    )

            # 回退：返回原始内容
            logger.info("Using raw content as structured analysis parsing failed")
            return {
                "symbol": symbol,
                "analysis_type": "unified_market_analysis",
                "raw_content": content,
                "structured_analysis": None,
                "status": "success",
                "timestamp": None,  # 将由调用者设置
            }

        except (KeyError, IndexError, TypeError, ValueError) as e:
            logger.error(f"Failed to parse unified analysis response due to invalid structure: {e}")
            return self._create_error_analysis(symbol, f"响应结构解析失败: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error parsing unified analysis response: {e}")
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
            "analysis_type": "unified_market_analysis",
            "raw_content": None,
            "structured_analysis": None,
            "status": "error",
            "error": error_message,
            "timestamp": None,  # 将由调用者设置
        }

    def close(self) -> None:
        """关闭HTTP客户端。"""
        self.client.close()
        logger.info("UnifiedDeepSeekAnalyzer HTTP client closed")


def print_unified_analysis_results(results: dict[str, Any]) -> None:
    """打印统一分析结果。

    Args:
        results: 统一分析结果
    """
    if results.get("status") != "success":
        print(f"\n❌ 统一AI分析失败: {results.get('error', '未知错误')}")
        return

    symbol = results.get("symbol", "UNKNOWN")
    print(f"\n=== {symbol} 统一AI分析结果 - 高频做市策略 ===")

    structured_analysis = results.get("structured_analysis")
    if structured_analysis:
        # 打印短期支撑位
        if "短期支撑位" in structured_analysis:
            print("\n🟢 短期支撑位（入场机会）:")
            for i, support in enumerate(structured_analysis["短期支撑位"], 1):
                print(
                    f"  支撑位 {i}: ${support.get('价格', 'N/A')} | "
                    f"可靠性: {support.get('可靠性评分', 'N/A')}/100 | "
                    f"入场区间: {support.get('推荐入场区间', 'N/A')}"
                )
                reason = support.get('形成原因', 'N/A')
                print(f"           原因: {reason[:60]}{'...' if len(reason) > 60 else ''}")

        # 打印短期阻力位
        if "短期阻力位" in structured_analysis:
            print("\n🔻 短期阻力位（退出目标）:")
            for i, resistance in enumerate(structured_analysis["短期阻力位"], 1):
                print(
                    f"  阻力位 {i}: ${resistance.get('价格', 'N/A')} | "
                    f"可靠性: {resistance.get('可靠性评分', 'N/A')}/100 | "
                    f"退出区间: {resistance.get('推荐退出区间', 'N/A')}"
                )
                reason = resistance.get('形成原因', 'N/A')
                print(f"           原因: {reason[:60]}{'...' if len(reason) > 60 else ''}")

        # 打印流动性供应区域
        if "集中流动性供应区域" in structured_analysis:
            liquidity = structured_analysis["集中流动性供应区域"]
            print("\n💰 集中流动性供应区域:")
            print(f"  最佳区间: {liquidity.get('最佳价格区间', 'N/A')}")
            backup_zones = liquidity.get('备选区间', [])
            if backup_zones:
                print(f"  备选区间: {', '.join(backup_zones)}")
            print(f"  市场特征: {liquidity.get('市场特征', 'N/A')[:80]}{'...' if len(liquidity.get('市场特征', '')) > 80 else ''}")
            print(f"  安全性: {liquidity.get('安全性评估', 'N/A')[:60]}{'...' if len(liquidity.get('安全性评估', '')) > 60 else ''}")

        # 打印做市策略要点
        if "做市策略要点" in structured_analysis:
            strategy = structured_analysis["做市策略要点"]
            print("\n📋 做市策略要点:")
            print(f"  主要机会: {strategy.get('主要机会', 'N/A')[:80]}{'...' if len(strategy.get('主要机会', '')) > 80 else ''}")
            print(f"  风险控制: {strategy.get('风险控制', 'N/A')[:80]}{'...' if len(strategy.get('风险控制', '')) > 80 else ''}")
            print(f"  策略总结: {strategy.get('策略总结', 'N/A')[:100]}{'...' if len(strategy.get('策略总结', '')) > 100 else ''}")

    else:
        # 打印原始内容
        raw_content = results.get("raw_content")
        if raw_content:
            print("\n📋 原始分析内容:")
            for line in raw_content.split("\n")[:10]:  # 限制前10行
                if line.strip():
                    print(f"   {line[:120]}{'...' if len(line) > 120 else ''}")

    print("\n" + "=" * 70)
