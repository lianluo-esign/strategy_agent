#!/usr/bin/env python3
"""
增强AI分析客户端 - 专门用于做多/做空/持有决策分析
"""

import json
import aiohttp
import asyncio
import logging
from datetime import datetime
from typing import Dict, Optional
import yaml
from enhanced_data_aggregator import EnhancedDataAggregator

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedAIClient:
    """增强AI分析客户端 - 专注于交易方向决策"""

    def __init__(self, config_path: str = "config/development.yaml"):
        """
        初始化AI分析客户端

        Args:
            config_path: 配置文件路径
        """
        self.config = self._load_config(config_path)
        self.data_aggregator = EnhancedDataAggregator(config_path)
        self.deepseek_config = self.config.get('analyzer', {}).get('deepseek', {})

    def _load_config(self, config_path: str) -> Dict:
        """加载配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"配置文件加载成功: {config_path}")
            return config
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            raise

    def generate_trading_decision_prompt(self, formatted_data: str) -> str:
        """
        生成交易决策分析提示词

        Args:
            formatted_data: 格式化的市场数据

        Returns:
            prompt: 完整的AI提示词
        """
        prompt = """You are an expert cryptocurrency trader specializing in directional trading analysis for BTC-FDUSD pair. Your task is to analyze the provided market data and determine whether to go LONG, SHORT, or HOLD.

## MARKET DATA FOR ANALYSIS
{formatted_data}

## ANALYSIS REQUIREMENTS

### 1. COMPREHENSIVE MARKET ANALYSIS
Analyze the following aspects:

**Depth Order Book Analysis (10-dollar precision):**
- Liquidity distribution across price levels
- Support and resistance strength
- Order flow imbalance
- Market depth concentration

**Recent 30-minute Price Action:**
- Short-term momentum and price direction
- Volume patterns and buy/sell pressure
- Microstructure signals
- Immediate market sentiment

**4-hour Historical Context:**
- Medium-term trend analysis
- Support/resistance levels from longer timeframe
- Volume accumulation/distribution patterns
- Trend sustainability assessment

### 2. DECISION MATRIX EVALUATION
Evaluate based on these criteria:

**LONG Signal Criteria:**
- Strong support at current or lower levels
- Buying pressure overwhelming selling pressure
- Positive momentum in both timeframes
- Depth shows buy-side liquidity dominance
- Price above key support levels

**SHORT Signal Criteria:**
- Strong resistance at current or higher levels
- Selling pressure overwhelming buying pressure
- Negative momentum in both timeframes
- Depth shows sell-side liquidity dominance
- Price below key resistance levels

**HOLD Signal Criteria:**
- Balanced market conditions
- Unclear directional bias
- Mixed signals across timeframes
- Range-bound market behavior
- High uncertainty or risk

### 3. RISK ASSESSMENT
For each potential trade direction, assess:
- Support/resistance proximity
- Volume confirmation strength
- Timeframe alignment
- Potential profit/loss ratio
- Execution risk and slippage

## OUTPUT FORMAT

You MUST respond with a valid JSON object ONLY. No other text, explanations, or formatting outside the JSON structure.

Your JSON response must have exactly these properties:
- direction: "Buy", "Sell", or "Hold"
- lower_bound: numeric value (float or integer) representing the lower bound of the trading range
- upper_bound: numeric value (float or integer) representing the upper bound of the trading range

Example format:
```json
{
  "direction": "Buy",
  "lower_bound": 110000.0,
  "upper_bound": 120000.0
}
```

## DECISION GUIDELINES

1. **For "Buy" direction:**
   - lower_bound should be a support level or entry price
   - upper_bound should be a target or resistance level
   - Ensure lower_bound < upper_bound

2. **For "Sell" direction:**
   - upper_bound should be a resistance level or entry price
   - lower_bound should be a target or support level
   - Ensure lower_bound < upper_bound

3. **For "Hold" direction:**
   - Set the current price range as the bounds
   - Use support/resistance levels to define the range

## ANALYSIS PROCESS

1. Identify current price level from the market data
2. Analyze support and resistance levels from depth data
3. Evaluate momentum from 30-minute and 4-hour data
4. Determine directional bias (Buy/Sell/Hold)
5. Identify optimal trading range:
   - For Buy: Support level (lower_bound) to target resistance (upper_bound)
   - For Sell: Resistance level (upper_bound) to target support (lower_bound)
   - For Hold: Current trading range with clear bounds

## CRITICAL REQUIREMENTS

1. **JSON ONLY**: Respond with valid JSON only, no additional text
2. **NUMERIC VALUES**: Use float or integer values for bounds
3. **LOGICAL RANGES**: Ensure lower_bound < upper_bound
4. **MARKET RELEVANCE**: Bounds should reflect actual support/resistance levels from the data
5. **PRICE PRECISION**: Use appropriate decimal places for cryptocurrency prices

Please analyze the provided market data and respond with a valid JSON object containing your trading decision and optimal trading range.

Remember: Your analysis should prioritize capital preservation and risk management. If the market condition is unclear, recommend "Hold" with a reasonable trading range."""

        return prompt

    async def call_deepseek_api(self, prompt: str) -> Optional[str]:
        """
        调用DeepSeek API进行分析

        Args:
            prompt: 分析提示词

        Returns:
            response: AI分析结果
        """
        api_key = self.deepseek_config.get('api_key')
        if not api_key:
            logger.error("DeepSeek API密钥未配置")
            return None

        base_url = self.deepseek_config.get('base_url', 'https://api.deepseek.com/v1')
        model = self.deepseek_config.get('model', 'deepseek-chat')
        max_tokens = self.deepseek_config.get('max_tokens', 6000)
        temperature = self.deepseek_config.get('temperature', 0.1)
        timeout = self.deepseek_config.get('timeout', 90)

        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }

        payload = {
            'model': model,
            'messages': [
                {
                    'role': 'system',
                    'content': 'You are an expert cryptocurrency trader specializing in directional analysis. Provide clear, actionable trading recommendations with comprehensive risk management. Always prioritize capital preservation.'
                },
                {
                    'role': 'user',
                    'content': prompt
                }
            ],
            'temperature': temperature,
            'max_tokens': max_tokens,
            'stream': False
        }

        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
                logger.info("发送交易决策分析请求到DeepSeek API...")

                async with session.post(f'{base_url}/chat/completions', headers=headers, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        content = result.get('choices', [{}])[0].get('message', {}).get('content', '')

                        if content:
                            logger.info("DeepSeek API调用成功")
                            return content
                        else:
                            logger.error("API响应内容为空")
                            return None
                    else:
                        error_text = await response.text()
                        logger.error(f"DeepSeek API错误: {response.status} - {error_text}")
                        return None

        except asyncio.TimeoutError:
            logger.error("DeepSeek API请求超时")
            return None
        except Exception as e:
            logger.error(f"DeepSeek API请求异常: {e}")
            return None

    async def analyze_trading_decision(self) -> Optional[str]:
        """
        执行完整的交易决策分析

        Returns:
            analysis_result: AI分析结果
        """
        try:
            logger.info("开始交易决策分析流程...")

            # 1. 获取和处理深度数据
            logger.info("步骤1: 获取和聚合深度数据")
            depth_str = self.data_aggregator.redis.get('depth_snapshot_5000')
            if not depth_str:
                logger.error("没有可用的深度数据")
                return None

            depth_data = json.loads(depth_str)
            aggregated_depth = self.data_aggregator.aggregate_depth_data(depth_data)

            # 2. 获取最近30分钟交易数据
            logger.info("步骤2: 获取最近30分钟交易数据")
            recent_trades = self.data_aggregator.get_recent_trades_data(minutes=30)
            if not recent_trades:
                logger.error("没有可用的最近交易数据")
                return None

            # 3. 聚合4小时数据
            logger.info("步骤3: 聚合4小时数据")
            all_trades = self.data_aggregator.get_recent_trades_data(minutes=2880)
            aggregated_4h = self.data_aggregator.aggregate_4h_data(all_trades)

            # 4. 格式化数据
            logger.info("步骤4: 格式化数据")
            formatted_data = self.data_aggregator.format_data_for_ai(
                recent_trades, aggregated_depth, aggregated_4h
            )

            # 保存格式化数据
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            data_filename = f"trading_analysis_data_{timestamp}.txt"
            self.data_aggregator.save_formatted_data(formatted_data, data_filename)

            # 5. 生成AI提示词
            logger.info("步骤5: 生成交易决策提示词")
            prompt = self.generate_trading_decision_prompt(formatted_data)

            # 保存提示词
            prompt_filename = f"trading_decision_prompt_{timestamp}.txt"
            with open(prompt_filename, 'w', encoding='utf-8') as f:
                f.write(prompt)
            logger.info(f"交易决策提示词已保存到: {prompt_filename}")

            # 6. 调用AI分析
            logger.info("步骤6: 调用DeepSeek AI进行交易决策分析")
            analysis_result = await self.call_deepseek_api(prompt)

            if analysis_result:
                # 保存分析结果
                result_filename = f"trading_decision_analysis_{timestamp}.txt"
                with open(result_filename, 'w', encoding='utf-8') as f:
                    f.write(analysis_result)
                logger.info(f"交易决策分析结果已保存到: {result_filename}")

                return analysis_result
            else:
                logger.error("AI分析失败")
                return None

        except Exception as e:
            logger.error(f"交易决策分析流程失败: {e}")
            return None

    def print_analysis_result(self, result: str):
        """
        打印分析结果

        Args:
            result: 分析结果文本
        """
        if not result:
            print("没有分析结果可显示")
            return

        print("\n" + "="*80)
        print("🤖 DEEPSEEK AI 交易决策分析结果")
        print("="*80)

        # 尝试解析为JSON并美化显示
        try:
            import json
            decision_data = json.loads(result.strip())

            print("📋 JSON格式分析结果:")
            print(json.dumps(decision_data, indent=2, ensure_ascii=False))

        except json.JSONDecodeError:
            print("📋 原始分析结果:")
            print(result)

        print("="*80)

    def extract_decision_summary(self, result: str) -> Dict:
        """
        提取决策摘要 - 从JSON格式解析

        Args:
            result: AI分析结果（JSON格式）

        Returns:
            summary: 决策摘要
        """
        try:
            # 首先尝试提取markdown代码块中的JSON
            json_content = self._extract_json_from_markdown(result)

            # 如果没有找到markdown中的JSON，尝试直接解析
            if json_content is None:
                json_content = result.strip()

            # 尝试解析JSON
            import json
            decision_data = json.loads(json_content)

            # 验证必需字段
            if not all(key in decision_data for key in ['direction', 'lower_bound', 'upper_bound']):
                logger.warning("JSON结果缺少必需字段")
                return self._fallback_extraction(result)

            # 转换为标准格式
            summary = {
                'direction': decision_data['direction'],
                'lower_bound': decision_data['lower_bound'],
                'upper_bound': decision_data['upper_bound'],
                'trading_range': decision_data['upper_bound'] - decision_data['lower_bound']
            }

            # 验证逻辑合理性
            if summary['lower_bound'] >= summary['upper_bound']:
                logger.warning("交易区间逻辑错误：下界大于等于上界")
                summary['valid'] = False
            else:
                summary['valid'] = True

            # 添加方向说明
            direction_mapping = {
                'Buy': '做多',
                'Sell': '做空',
                'Hold': '持有'
            }
            summary['direction_cn'] = direction_mapping.get(summary['direction'], summary['direction'])

            logger.info(f"成功解析JSON决策结果: {summary['direction']} [{summary['lower_bound']}, {summary['upper_bound']}]")
            return summary

        except json.JSONDecodeError as e:
            logger.warning(f"JSON解析失败: {e}")
            return self._fallback_extraction(result)
        except Exception as e:
            logger.warning(f"提取决策摘要失败: {e}")
            return {'direction': 'UNKNOWN', 'error': str(e)}

    def _extract_json_from_markdown(self, result: str) -> str:
        """
        从markdown代码块中提取JSON内容

        Args:
            result: 包含markdown代码块的文本

        Returns:
            json_content: 提取的JSON字符串，如果没有找到则返回None
        """
        try:
            import re

            # 尝试匹配```json...```格式（更宽松的模式）
            json_pattern = r'```(?:json)?\s*\n?\s*\{.*?\}\s*\n?```'
            json_match = re.search(json_pattern, result, re.DOTALL | re.IGNORECASE)

            if json_match:
                # 提取{}内的内容
                full_match = json_match.group(0)
                # 使用更精确的提取来获取JSON部分
                json_inner_pattern = r'\{.*?\}'
                json_content_match = re.search(json_inner_pattern, full_match, re.DOTALL)

                if json_content_match:
                    json_content = json_content_match.group(0).strip()
                    logger.info("成功从markdown代码块中提取JSON")
                    return json_content

            # 尝试匹配没有代码块标记的JSON（纯JSON）
            json_pattern_pure = r'^\s*\{.*\}\s*$'
            pure_match = re.search(json_pattern_pure, result, re.DOTALL)

            if pure_match:
                json_content = pure_match.group(0).strip()
                logger.info("检测到纯JSON格式")
                return json_content

            # 最后尝试：在任意文本中查找JSON结构
            json_any_pattern = r'\{[^{}]*"direction"[^{}]*"lower_bound"[^{}]*"upper_bound"[^{}]*\}'
            any_match = re.search(json_any_pattern, result, re.DOTALL)

            if any_match:
                json_content = any_match.group(0).strip()
                logger.info("在文本中找到JSON结构")
                return json_content

            logger.info("未找到markdown代码块或纯JSON格式")
            return None

        except Exception as e:
            logger.warning(f"从markdown提取JSON失败: {e}")
            return None

    def _fallback_extraction(self, result: str) -> Dict:
        """
        回退提取方法 - 用于JSON解析失败时

        Args:
            result: AI分析结果文本

        Returns:
            summary: 决策摘要
        """
        try:
            # 尝试从文本中提取JSON
            import re
            json_pattern = r'\{[^{}]*"direction"\s*:\s*"[^"]*"[^{}]*"lower_bound"\s*:\s*[\d.]+[^{}]*"upper_bound"\s*:\s*[\d.]+[^{}]*\}'
            json_match = re.search(json_pattern, result, re.DOTALL)

            if json_match:
                try:
                    import json
                    decision_data = json.loads(json_match.group())
                    return {
                        'direction': decision_data.get('direction', 'UNKNOWN'),
                        'lower_bound': decision_data.get('lower_bound', 0),
                        'upper_bound': decision_data.get('upper_bound', 0),
                        'valid': True,
                        'extracted_from': 'text_fallback'
                    }
                except:
                    pass

            # 最后回退：尝试从文本中提取关键词
            summary = {'direction': 'UNKNOWN'}

            if 'BUY' in result.upper() or 'Buy' in result:
                summary['direction'] = 'Buy'
            elif 'SELL' in result.upper() or 'Sell' in result:
                summary['direction'] = 'Sell'
            elif 'HOLD' in result.upper() or 'Hold' in result:
                summary['direction'] = 'Hold'

            summary['valid'] = False
            summary['extracted_from'] = 'text_only'

            return summary

        except Exception as e:
            logger.error(f"回退提取失败: {e}")
            return {'direction': 'UNKNOWN', 'error': str(e), 'valid': False}

async def main():
    """主函数 - 演示交易决策分析"""
    logger.info("启动交易决策分析演示")

    try:
        # 创建AI分析客户端
        client = EnhancedAIClient()

        # 执行交易决策分析
        print("🚀 开始交易决策分析...")
        analysis_result = await client.analyze_trading_decision()

        if analysis_result:
            # 打印分析结果
            client.print_analysis_result(analysis_result)

            # 提取并显示决策摘要
            summary = client.extract_decision_summary(analysis_result)

            print("\n" + "="*60)
            print("📋 决策摘要")
            print("="*60)

            # 显示JSON格式的决策结果
            if summary.get('valid', False):
                direction_emoji = {
                    'Buy': '📈 做多',
                    'Sell': '📉 做空',
                    'Hold': '⏸️ 持有'
                }.get(summary.get('direction'), f'❓ {summary.get("direction")}')

                print(f"🎯 推荐操作: {direction_emoji}")
                print(f"📊 交易区间: ${summary.get('lower_bound', 0):,.2f} - ${summary.get('upper_bound', 0):,.2f}")
                print(f"📏 区间宽度: ${summary.get('trading_range', 0):,.2f}")

                if 'direction_cn' in summary:
                    print(f"🌐 中文方向: {summary['direction_cn']}")

                print(f"✅ 数据格式: JSON (有效)")

            else:
                print(f"🎯 推荐操作: {summary.get('direction', 'UNKNOWN')}")
                print(f"❌ 数据解析失败，请检查原始结果")
                if 'error' in summary:
                    print(f"🔍 错误信息: {summary['error']}")

            print(f"\n✅ 交易决策分析完成!")
            print(f"📄 详细分析结果已保存到文件")
            print(f"🔍 基于深度数据(10美元精度) + 30分钟详细数据 + 4小时趋势分析")
            print(f"📋 分析格式: JSON标准化输出")

        else:
            print("❌ 交易决策分析失败，请检查配置和数据")

    except Exception as e:
        logger.error(f"交易决策分析演示失败: {e}")
        print(f"❌ 程序异常: {e}")

if __name__ == "__main__":
    asyncio.run(main())