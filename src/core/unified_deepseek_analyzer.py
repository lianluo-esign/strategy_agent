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
        return """你是一个专业的加密货币市场分析师，专门为BTC-FDUSD高频做市策略提供市场分析。

你的核心任务是分析静态订单簿深度数据和动态交易流数据，生成具体的交易参数。

**分析任务**：
1. 分析订单簿深度数据和Volume Profile数据
2. 识别市场支撑阻力位和流动性特征
3. 基于市场分析生成具体的交易参数

**输出要求**：
你必须且只能返回标准的三字段JSON格式，不得包含任何其他解释、分析或说明：

```json
{
    "grid_delta": 2.0,
    "grid_quantity": 0.001,
    "active_side": "Buy"
}
```

**参数说明**：
- grid_delta: 交易价差，范围0.1-100.0，基于支撑阻力位间距和市场波动性确定
- grid_quantity: 挂单量，范围0.0001-10.0，基于流动性和风险评估确定
- active_side: 交易方向，只能是"Buy"或"Sell"，基于市场趋势分析确定

**重要**：
1. 只返回上述三字段的JSON，不要包含任何其他内容
2. 不要添加解释、分析过程或其他文字说明
3. 确保所有参数都在指定范围内
4. 确保JSON格式完全正确
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

        prompt = f"""分析{symbol}市场数据并生成交易参数：

**市场数据**：
- 最优买价：${best_bid:,.2f}
- 最优卖价：${best_ask:,.2f}
- 买卖价差：${spread:,.2f}
- 买盘总量：{total_bid_volume:.2f}
- 卖盘总量：{total_ask_volume:.2f}
- 24小时成交量：{total_volume:.2f}

**订单簿前5档**：
{json.dumps(order_book_data, ensure_ascii=False, indent=2)[:500]}

**Volume Profile关键数据**：
- POC价格：${float(poc_price):,.2f}
- POC成交量：{poc_volume:.2f}
- 价值区间：${float(value_area_low):,.2f} - ${float(value_area_high):,.2f}

**前5大成交量价位**：
{json.dumps(vp_levels_data[:5], ensure_ascii=False, indent=2)}

**分析要求**：
基于以上数据，分析市场结构和流动性分布，生成高频做市交易参数。

请直接返回标准的JSON格式：
{{
    "grid_delta": 数值,
    "grid_quantity": 数值,
    "active_side": "Buy"或"Sell"
}}

参数范围：
- grid_delta: 0.1-100.0 (基于支撑阻力位间距)
- grid_quantity: 0.0001-10.0 (基于流动性水平)
- active_side: "Buy"或"Sell" (基于市场趋势)"""

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

            # 使用结果验证器提取和验证交易参数
            from .result_validator import result_validator

            try:
                # 创建临时分析结果用于验证
                temp_analysis_result = {
                    "status": "success",
                    "raw_content": content,
                    "symbol": symbol
                }

                # 验证并提取交易参数
                trading_params = result_validator.validate_and_extract_trading_params(temp_analysis_result)

                logger.info(f"Successfully extracted trading parameters: {trading_params}")

                return {
                    "symbol": symbol,
                    "analysis_type": "unified_market_analysis",
                    "raw_content": content,
                    "trading_params": trading_params,
                    "structured_analysis": trading_params,  # 保持向后兼容
                    "status": "success",
                    "timestamp": None,  # 将由调用者设置
                }

            except Exception as validation_error:
                logger.warning(f"Failed to validate trading parameters: {validation_error}")
                # 返回原始内容但标记为验证失败
                return {
                    "symbol": symbol,
                    "analysis_type": "unified_market_analysis",
                    "raw_content": content,
                    "structured_analysis": None,
                    "validation_error": str(validation_error),
                    "status": "success",  # API调用成功，但验证失败
                    "timestamp": None,
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
