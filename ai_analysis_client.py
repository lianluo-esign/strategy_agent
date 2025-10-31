#!/usr/bin/env python3
"""
AI分析客户端 - 将聚合的数据发送给DeepSeek进行分析
"""

import json
import aiohttp
import asyncio
import logging
from datetime import datetime
from typing import Dict, Optional, List
import yaml
from data_aggregator import DataAggregator

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AIAnalysisClient:
    """DeepSeek AI分析客户端"""

    def __init__(self, config_path: str = "config/development.yaml"):
        """
        初始化AI分析客户端

        Args:
            config_path: 配置文件路径
        """
        self.config = self._load_config(config_path)
        self.data_aggregator = DataAggregator(config_path)
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

    def generate_liquidity_analysis_prompt(self, formatted_data: str) -> str:
        """
        生成流动性分析提示词

        Args:
            formatted_data: 格式化的市场数据

        Returns:
            prompt: 完整的AI提示词
        """
        prompt = f"""You are an expert cryptocurrency quantitative trader specializing in liquidity analysis and high-frequency arbitrage strategies for BTC-FDUSD trading pair.

## TASK OVERVIEW
Analyze the provided market data to identify high-density liquidity zones that contain 70% of trading volume and provide specific long-only arbitrage trading recommendations.

## MARKET DATA
{formatted_data}

## ANALYSIS REQUIREMENTS

### 1. LIQUIDITY DENSITY ANALYSIS
Identify the 70% volume concentration zone:
- Calculate cumulative volume distribution across price levels
- Determine the price range containing 70% of total trading volume
- Analyze buy/sell pressure distribution within the zone
- Assess liquidity depth and stability

### 2. TECHNICAL ANALYSIS
Evaluate market structure:
- Support and resistance levels based on volume clusters
- Price momentum and trend direction
- Volatility and risk assessment
- Market sentiment indicators

### 3. TRADING OPPORTUNITY ASSESSMENT (LONG-ONLY)
Provide specific trading recommendations:
- Optimal entry price levels near liquidity support
- Stop-loss levels (0.5% below support)
- Take-profit targets (0.2% above zone center)
- Position sizing recommendations
- Expected win rate and risk-reward ratio
- Holding period expectation (5-30 minutes)

### 4. RISK MANAGEMENT
Assess potential risks:
- Liquidity risk and slippage potential
- Market volatility impact
- Stop-loss reliability
- Position size recommendations (max 10% of capital)

## OUTPUT FORMAT

Please provide analysis in the following structure:

### LIQUIDITY ZONE IDENTIFICATION
**Primary Liquidity Zone (70% Volume Coverage):**
- Center Price: $XXXXX.XX
- Price Range: [$XXXXX.XX, $XXXXX.XX]
- Zone Width: $XXX.XX (X.XX%)
- Coverage Volume: X.XXX BTC (70% of total)
- Buy Ratio: XX.X%
- Sell Ratio: XX.X%
- Density Score: XXX
- Support Strength: X.XX/10
- Resistance Strength: X.XX/10

### TRADING RECOMMENDATION (LONG-ONLY)
**Entry Strategy:**
- Optimal Entry Price: $XXXXX.XX
- Entry Reason: [Specific market condition]
- Confidence Level: XX%

**Risk Management:**
- Stop Loss: $XXXXX.XX (X.XX% below entry)
- Take Profit: $XXXXX.XX (X.XX% above entry)
- Risk/Reward Ratio: 1:X.X
- Maximum Position Size: X% of capital

**Expected Performance:**
- Win Rate Probability: XX%
- Average Holding Period: X-X minutes
- Expected Return per Trade: X.XX%

### MARKET STRUCTURE ANALYSIS
**Current Market State:**
- Trend Direction: [Uptrend/Downterm/Sideways]
- Volatility Level: [Low/Medium/High]
- Liquidity Status: [Sufficient/Tight/Imbalanced]
- Market Sentiment: [Bullish/Bearish/Neutral]

**Key Levels:**
- Strong Support: $XXXXX.XX
- Immediate Resistance: $XXXXX.XX
- Liquidity Vacuum Areas: [$XXXXX.XX-$XXXXX.XX]

### EXECUTION CONSIDERATIONS
**Optimal Timing:**
- Best trading window: [Time periods]
- Market conditions to avoid: [Specific scenarios]
- Volume confirmation requirements: [Minimum thresholds]

**Slippage Analysis:**
- Expected slippage: X.XX%
- Liquidity depth at entry: X.XXX BTC
- Order execution strategy: [Market/Limit/Hybrid]

### RISK WARNING
Please include specific risk factors and contraindications for this trade setup.

## ANALYSIS GUIDELINES
1. Focus on data-driven insights from the provided market information
2. Prioritize capital preservation over high returns
3. Consider the short-term nature (5-30 minutes) of the strategy
4. Emphasize liquidity and execution quality
5. Provide specific, actionable price levels
6. Include clear exit criteria for both profit and loss scenarios

Please analyze the provided data and deliver a comprehensive trading recommendation focused on liquidity density-based long-only arbitrage opportunities."""

        return prompt

    def generate_signal_evaluation_prompt(self, current_signal: Dict, market_context: Dict) -> str:
        """
        生成信号评估提示词

        Args:
            current_signal: 当前交易信号
            market_context: 市场上下文

        Returns:
            prompt: 信号评估提示词
        """
        prompt = f"""You are a senior risk manager reviewing high-frequency trading signals for BTC-FDUSD.

## CURRENT TRADING SIGNAL
```json
{json.dumps(current_signal, indent=2)}
```

## MARKET CONTEXT
```json
{json.dumps(market_context, indent=2)}
```

## EVALUATION CRITERIA
Please assess this trading signal based on:

1. **Signal Quality** (1-10 scale):
   - Technical analysis validity
   - Liquidity support strength
   - Risk/reward ratio attractiveness
   - Market timing appropriateness

2. **Risk Assessment**:
   - Maximum downside potential
   - Stop-loss reliability
   - Position size appropriateness
   - Market condition compatibility

3. **Execution Considerations**:
   - Expected slippage impact
   - Liquidity depth adequacy
   - Optimal order type recommendation
   - Timing optimization suggestions

4. **Adjustment Recommendations**:
   - Entry price optimization
   - Stop-loss level adjustment
   - Take-profit target refinement
   - Position size modification

## OUTPUT FORMAT
Provide your evaluation in this structure:

### SIGNAL QUALITY SCORE: X/10
**Strengths:**
- [List specific strengths]

**Concerns:**
- [List specific concerns]

### RISK ASSESSMENT
**Risk Level:** [Low/Medium/High]
**Maximum Expected Loss:** X.XX%
**Probability of Stop Loss Hit:** XX%

### RECOMMENDATIONS
**Optimized Parameters:**
- Entry Price: $XXXXX.XX (adjustment: ±$X.XX)
- Stop Loss: $XXXXX.XX (adjustment: ±$X.XX)
- Take Profit: $XXXXX.XX (adjustment: ±$X.XX)
- Position Size: X% (adjustment: ±X%)

**Execution Strategy:**
- Order Type: [Market/Limit/Stop-Limit]
- Timing: [Immediate/Wait for specific condition]
- Volume Splitting: [Single order/Multiple orders]

**Overall Recommendation:** [EXECUTE/MODIFY/CANCEL]
**Confidence in Recommendation:** XX%

Please provide specific, actionable recommendations with clear reasoning."""

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
                    'content': 'You are an expert cryptocurrency quantitative trader specializing in liquidity analysis and high-frequency arbitrage strategies. Provide data-driven, specific trading recommendations with clear risk management parameters.'
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
                logger.info("发送请求到DeepSeek API...")

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

    async def analyze_market_data(self, trades_limit: int = 200) -> Optional[str]:
        """
        完整的市场数据分析流程

        Args:
            trades_limit: 交易数据限制条数

        Returns:
            analysis_result: AI分析结果
        """
        try:
            logger.info("开始市场数据分析流程...")

            # 1. 获取和聚合数据
            logger.info("步骤1: 获取市场数据")
            trades_data = self.data_aggregator.get_trades_window_data(limit=trades_limit)
            depth_data = self.data_aggregator.get_depth_snapshot_data()

            if not trades_data and not depth_data:
                logger.error("没有可用的市场数据")
                return None

            # 2. 数据聚合
            logger.info("步骤2: 聚合数据")
            trades_aggregated = self.data_aggregator.aggregate_trades_data(trades_data)
            depth_aggregated = self.data_aggregator.aggregate_depth_data(depth_data)

            # 3. 格式化数据
            logger.info("步骤3: 格式化数据")
            formatted_data = self.data_aggregator.format_for_ai_analysis(trades_aggregated, depth_aggregated)

            # 保存格式化数据
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            data_filename = f"market_data_{timestamp}.txt"
            self.data_aggregator.save_formatted_data(formatted_data, data_filename)

            # 4. 生成AI提示词
            logger.info("步骤4: 生成AI分析提示词")
            prompt = self.generate_liquidity_analysis_prompt(formatted_data)

            # 保存提示词
            prompt_filename = f"ai_prompt_{timestamp}.txt"
            with open(prompt_filename, 'w', encoding='utf-8') as f:
                f.write(prompt)
            logger.info(f"AI提示词已保存到: {prompt_filename}")

            # 5. 调用AI分析
            logger.info("步骤5: 调用DeepSeek AI进行分析")
            analysis_result = await self.call_deepseek_api(prompt)

            if analysis_result:
                # 保存分析结果
                result_filename = f"ai_analysis_{timestamp}.txt"
                with open(result_filename, 'w', encoding='utf-8') as f:
                    f.write(analysis_result)
                logger.info(f"AI分析结果已保存到: {result_filename}")

                return analysis_result
            else:
                logger.error("AI分析失败")
                return None

        except Exception as e:
            logger.error(f"市场数据分析流程失败: {e}")
            return None

    async def evaluate_trading_signal(self, signal: Dict, market_context: Dict) -> Optional[str]:
        """
        评估交易信号

        Args:
            signal: 交易信号
            market_context: 市场上下文

        Returns:
            evaluation_result: 评估结果
        """
        try:
            logger.info("开始交易信号评估...")

            # 生成评估提示词
            prompt = self.generate_signal_evaluation_prompt(signal, market_context)

            # 调用AI评估
            evaluation_result = await self.call_deepseek_api(prompt)

            if evaluation_result:
                # 保存评估结果
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"signal_evaluation_{timestamp}.txt"
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(evaluation_result)
                logger.info(f"信号评估结果已保存到: {filename}")

                return evaluation_result
            else:
                logger.error("信号评估失败")
                return None

        except Exception as e:
            logger.error(f"交易信号评估失败: {e}")
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
        print("🤖 DEEPSEEK AI 分析结果")
        print("="*80)
        print(result)
        print("="*80)

async def main():
    """主函数 - 演示AI分析功能"""
    logger.info("启动AI分析客户端演示")

    try:
        # 创建AI分析客户端
        client = AIAnalysisClient()

        # 执行完整的市场数据分析
        print("🚀 开始市场数据分析...")
        analysis_result = await client.analyze_market_data(trades_limit=150)

        if analysis_result:
            # 打印分析结果
            client.print_analysis_result(analysis_result)

            # 保存结果摘要
            print(f"\n✅ 分析完成!")
            print(f"📄 分析结果已保存到文件")
            print(f"📊 基于最近150分钟的交易数据和深度订单簿")
            print(f"🎯 专注于70%流动性密度区域识别")
            print(f"📈 只做多的高频套利策略建议")

        else:
            print("❌ 分析失败，请检查配置和数据")

    except Exception as e:
        logger.error(f"AI分析客户端演示失败: {e}")
        print(f"❌ 程序异常: {e}")

if __name__ == "__main__":
    asyncio.run(main())