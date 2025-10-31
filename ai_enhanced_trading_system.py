#!/usr/bin/env python3
"""
AI增强的高频套利交易系统
整合DeepSeek AI分析，实现智能流动性密度交易
"""

import json
import redis
import numpy as np
import pandas as pd
import asyncio
import aiohttp
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from liquidity_density_backtesting_framework import (
    LiquidityZone, TradingSignal, LiquidityDensityAnalyzer,
    HighFrequencyArbitrageStrategy
)
from deepseek_liquidity_analysis_prompt import LiquidityAnalysisPromptGenerator

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class AIAnalysisResult:
    """AI分析结果"""
    primary_zone: LiquidityZone
    entry_price: float
    stop_loss: float
    take_profit: float
    confidence: float
    expected_win_rate: float
    risk_reward_ratio: float
    position_size_percent: float
    reasoning: str

@dataclass
class MarketContext:
    """市场上下文"""
    current_price: float
    price_change_1h: float
    price_change_24h: float
    volume_24h: float
    volatility: float
    market_sentiment: str

class DeepSeekAIClient:
    """DeepSeek AI客户端"""

    def __init__(self, api_key: str, base_url: str = "https://api.deepseek.com/v1"):
        self.api_key = api_key
        self.base_url = base_url
        self.session = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def analyze_liquidity_zones(self, prompt: str) -> str:
        """调用DeepSeek API进行流动性分析"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "deepseek-chat",
            "messages": [
                {
                    "role": "system",
                    "content": "你是一个专业的加密货币量化交易分析师，专注于流动性分析和高频交易策略。"
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.1,
            "max_tokens": 4000
        }

        try:
            async with self.session.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return result["choices"][0]["message"]["content"]
                else:
                    error_text = await response.text()
                    logger.error(f"DeepSeek API error: {response.status} - {error_text}")
                    return None

        except Exception as e:
            logger.error(f"DeepSeek API request failed: {e}")
            return None

class AIEnhancedTradingSystem:
    """AI增强的交易系统"""

    def __init__(self, redis_client: redis.Redis, deepseek_api_key: str):
        self.redis = redis_client
        self.deepseek_client = DeepSeekAIClient(deepseek_api_key)
        self.prompt_generator = LiquidityAnalysisPromptGenerator()
        self.liquidity_analyzer = LiquidityDensityAnalyzer(redis_client)
        self.strategy = HighFrequencyArbitrageStrategy()

        # 交易状态
        self.current_position = 0
        self.entry_price = None
        self.unrealized_pnl = 0
        self.realized_pnl = 0
        self.trade_history = []

        # 配置参数
        self.max_position_size = 0.1  # 最大仓位比例
        self.risk_per_trade = 0.01     # 单笔交易风险
        self.min_confidence = 0.7       # 最小置信度

    async def run_trading_loop(self):
        """运行主交易循环"""
        logger.info("启动AI增强高频交易系统")

        async with self.deepseek_client:
            while True:
                try:
                    # 1. 获取市场数据
                    market_data = await self._get_market_data()
                    if not market_data:
                        await asyncio.sleep(60)
                        continue

                    # 2. 执行AI分析
                    ai_analysis = await self._perform_ai_analysis(market_data)
                    if not ai_analysis:
                        await asyncio.sleep(60)
                        continue

                    # 3. 生成交易信号
                    signal = self._generate_enhanced_signal(ai_analysis, market_data)

                    # 4. 执行交易
                    await self._execute_trading_signal(signal)

                    # 5. 更新持仓状态
                    await self._update_position_status(market_data)

                    # 6. 记录交易日志
                    self._log_trading_status(signal, market_data, ai_analysis)

                    await asyncio.sleep(30)  # 30秒检查一次

                except Exception as e:
                    logger.error(f"交易循环错误: {e}")
                    await asyncio.sleep(60)

    async def _get_market_data(self) -> Optional[Dict]:
        """获取市场数据"""
        try:
            # 从Redis获取数据
            trades_data = self._get_recent_trades()
            depth_data = self._get_depth_snapshot()

            if not trades_data or not depth_data:
                return None

            # 获取当前价格
            current_price = self._get_current_price(trades_data[-1])

            # 计算价格变化
            price_1h_ago = self._get_price_1h_ago(trades_data)
            price_24h_ago = self._get_price_24h_ago(trades_data)

            # 计算成交量
            volume_24h = sum(
                item['total_volume']
                for minute_data in trades_data[-1440:]  # 最近24小时
                for item in minute_data.get('price_levels', {}).values()
            )

            # 计算波动率
            prices = [self._get_current_price(data) for data in trades_data[-60:]]  # 最近1小时
            volatility = np.std(prices) / np.mean(prices) if prices else 0

            return {
                'trades_data': trades_data,
                'depth_data': depth_data,
                'current_price': current_price,
                'price_change_1h': (current_price - price_1h_ago) / price_1h_ago if price_1h_ago else 0,
                'price_change_24h': (current_price - price_24h_ago) / price_24h_ago if price_24h_ago else 0,
                'volume_24h': volume_24h,
                'volatility': volatility,
                'timestamp': datetime.now()
            }

        except Exception as e:
            logger.error(f"获取市场数据失败: {e}")
            return None

    def _get_recent_trades(self) -> List[Dict]:
        """获取最近的交易数据"""
        try:
            trades = []
            window_size = min(120, self.redis.llen('trades_window'))  # 获取最近2小时数据

            for i in range(window_size):
                data_str = self.redis.lindex('trades_window', -i-1)  # 从最新的开始
                if data_str:
                    data = json.loads(data_str)
                    trades.append(data)

            return trades[::-1]  # 按时间正序排列

        except Exception as e:
            logger.error(f"获取交易数据失败: {e}")
            return []

    def _get_depth_snapshot(self) -> Optional[Dict]:
        """获取深度快照"""
        try:
            depth_str = self.redis.get('depth_snapshot_5000')
            if depth_str:
                return json.loads(depth_str)
            return None

        except Exception as e:
            logger.error(f"获取深度快照失败: {e}")
            return None

    def _get_current_price(self, minute_data: Dict) -> float:
        """获取当前价格"""
        price_levels = minute_data.get('price_levels', {})
        if not price_levels:
            return 0.0

        # 返回成交量最大的价格
        max_volume_price = max(
            price_levels.keys(),
            key=lambda p: price_levels[p]['total_volume']
        )
        return float(max_volume_price)

    def _get_price_1h_ago(self, trades_data: List[Dict]) -> Optional[float]:
        """获取1小时前的价格"""
        if len(trades_data) >= 60:
            return self._get_current_price(trades_data[-60])
        return None

    def _get_price_24h_ago(self, trades_data: List[Dict]) -> Optional[float]:
        """获取24小时前的价格"""
        if len(trades_data) >= 1440:
            return self._get_current_price(trades_data[-1440])
        return None

    async def _perform_ai_analysis(self, market_data: Dict) -> Optional[AIAnalysisResult]:
        """执行AI分析"""
        try:
            # 生成AI提示词
            prompt = self.prompt_generator.generate_comprehensive_analysis_prompt(
                market_data['trades_data'][-100:],  # 使用最近100分钟数据
                market_data['depth_data']
            )

            # 调用DeepSeek API
            ai_response = await self.deepseek_client.analyze_liquidity_zones(prompt)
            if not ai_response:
                logger.error("AI分析失败")
                return None

            # 解析AI响应
            parsed_result = self._parse_ai_response(ai_response, market_data)
            if not parsed_result:
                logger.error("AI响应解析失败")
                return None

            logger.info(f"AI分析完成: 入场价={parsed_result.entry_price:.2f}, 置信度={parsed_result.confidence:.2f}")
            return parsed_result

        except Exception as e:
            logger.error(f"AI分析过程出错: {e}")
            return None

    def _parse_ai_response(self, ai_response: str, market_data: Dict) -> Optional[AIAnalysisResult]:
        """解析AI响应"""
        try:
            # 这里需要根据AI响应的实际格式进行解析
            # 由于AI响应是自然语言，需要提取关键信息

            # 示例解析逻辑 (实际实现需要更复杂的文本解析)
            lines = ai_response.split('\n')

            # 提取关键信息
            entry_price = None
            stop_loss = None
            take_profit = None
            confidence = 0.5
            expected_win_rate = 0.6
            risk_reward_ratio = 2.0

            # 简单的文本解析示例
            for line in lines:
                if '最佳入场价格:' in line:
                    try:
                        entry_price = float(line.split(':')[-1].strip().split()[0])
                    except:
                        pass
                elif '止损价格:' in line:
                    try:
                        stop_loss = float(line.split(':')[-1].strip().split()[0])
                    except:
                        pass
                elif '止盈价格:' in line:
                    try:
                        take_profit = float(line.split(':')[-1].strip().split()[0])
                    except:
                        pass
                elif '预期胜率:' in line:
                    try:
                        expected_win_rate = float(line.split(':')[-1].strip().replace('%', '')) / 100
                    except:
                        pass

            # 如果解析失败，使用默认值
            if not all([entry_price, stop_loss, take_profit]):
                current_price = market_data['current_price']
                entry_price = current_price * 0.999
                stop_loss = current_price * 0.995
                take_profit = current_price * 1.002
                confidence = 0.6

            # 创建流动性区域 (简化版)
            liquidity_zone = LiquidityZone(
                price_center=entry_price,
                price_range=(stop_loss, take_profit),
                total_volume=1.0,
                buy_ratio=0.6,
                sell_ratio=0.4,
                trade_count=100,
                density_score=100,
                support_strength=0.7,
                resistance_strength=0.3
            )

            return AIAnalysisResult(
                primary_zone=liquidity_zone,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                confidence=confidence,
                expected_win_rate=expected_win_rate,
                risk_reward_ratio=risk_reward_ratio,
                position_size_percent=5.0,
                reasoning=ai_response[:200] + "..."
            )

        except Exception as e:
            logger.error(f"解析AI响应失败: {e}")
            return None

    def _generate_enhanced_signal(self, ai_analysis: AIAnalysisResult, market_data: Dict) -> TradingSignal:
        """基于AI分析生成增强的交易信号"""
        current_price = market_data['current_price']

        # 只做多策略
        if self.current_position == 0 and ai_analysis.confidence >= self.min_confidence:
            if current_price <= ai_analysis.entry_price and current_price > ai_analysis.stop_loss:
                # 计算仓位大小
                position_size = min(
                    ai_analysis.position_size_percent / 100,
                    self.max_position_size
                )

                return TradingSignal(
                    timestamp=datetime.now(),
                    action='BUY',
                    price=current_price,
                    quantity=position_size,
                    confidence=ai_analysis.confidence,
                    reason=f"AI建议做多: {ai_analysis.reasoning[:50]}...",
                    liquidity_zone=ai_analysis.primary_zone
                )

        # 检查卖出条件
        elif self.current_position > 0:
            sell_conditions = [
                current_price >= self.take_profit or current_price >= ai_analysis.take_profit,
                current_price <= self.stop_loss or current_price <= ai_analysis.stop_loss,
                ai_analysis.confidence < 0.3  # AI置信度降低
            ]

            if any(sell_conditions):
                return TradingSignal(
                    timestamp=datetime.now(),
                    action='SELL',
                    price=current_price,
                    quantity=self.current_position,
                    confidence=1.0,
                    reason="达到止盈/止损条件或AI建议平仓"
                )

        return TradingSignal(
            timestamp=datetime.now(),
            action='HOLD',
            price=current_price,
            quantity=0,
            confidence=0,
            reason="无交易信号"
        )

    async def _execute_trading_signal(self, signal: TradingSignal):
        """执行交易信号"""
        if signal.action == 'BUY':
            # 执行买入逻辑
            self.current_position = signal.quantity
            self.entry_price = signal.price
            self.stop_loss = signal.liquidity_zone.price_range[0] * 0.995
            self.take_profit = signal.liquidity_zone.price_center * 1.002

            logger.info(f"执行买入: {signal.quantity:.6f} @ {signal.price:.2f}")

        elif signal.action == 'SELL':
            # 执行卖出逻辑
            if self.current_position > 0:
                pnl = (signal.price - self.entry_price) * self.current_position
                self.realized_pnl += pnl

                trade_record = {
                    'entry_price': self.entry_price,
                    'exit_price': signal.price,
                    'quantity': self.current_position,
                    'pnl': pnl,
                    'entry_time': self.entry_time,
                    'exit_time': datetime.now()
                }
                self.trade_history.append(trade_record)

                logger.info(f"执行卖出: {self.current_position:.6f} @ {signal.price:.2f}, 盈亏: {pnl:.2f}")

                self.current_position = 0
                self.entry_price = None

    async def _update_position_status(self, market_data: Dict):
        """更新持仓状态"""
        if self.current_position > 0:
            current_price = market_data['current_price']
            self.unrealized_pnl = (current_price - self.entry_price) * self.current_position

    def _log_trading_status(self, signal: TradingSignal, market_data: Dict, ai_analysis: AIAnalysisResult):
        """记录交易状态"""
        status = {
            'timestamp': datetime.now().isoformat(),
            'signal_action': signal.action,
            'signal_price': signal.price,
            'signal_confidence': signal.confidence,
            'current_position': self.current_position,
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl,
            'total_trades': len(self.trade_history),
            'market_price': market_data['current_price'],
            'ai_confidence': ai_analysis.confidence if ai_analysis else 0
        }

        logger.info(f"交易状态: {json.dumps(status, indent=2)}")

        # 保存到Redis
        self.redis.lpush('trading_status_log', json.dumps(status, default=str))
        self.redis.ltrim('trading_status_log', 0, 999)  # 保留最近1000条记录

async def main():
    """主函数"""
    # 配置参数
    REDIS_HOST = 'localhost'
    REDIS_PORT = 6379
    DEEPSEEK_API_KEY = 'your_deepseek_api_key_here'  # 替换为实际的API密钥

    # 连接Redis
    redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

    # 创建交易系统
    trading_system = AIEnhancedTradingSystem(redis_client, DEEPSEEK_API_KEY)

    # 运行交易系统
    try:
        await trading_system.run_trading_loop()
    except KeyboardInterrupt:
        logger.info("交易系统停止")
    except Exception as e:
        logger.error(f"交易系统异常: {e}")

if __name__ == "__main__":
    asyncio.run(main())