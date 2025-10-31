#!/usr/bin/env python3
"""
高频流动性密度分析回测框架
用于分析70%流动性密集区域并支持做多套利策略
"""

import json
import redis
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from collections import defaultdict, deque
import asyncio
import statistics

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class LiquidityZone:
    """流动性区域数据结构"""
    price_center: float
    price_range: Tuple[float, float]
    total_volume: float
    buy_ratio: float
    sell_ratio: float
    trade_count: int
    density_score: float
    support_strength: float
    resistance_strength: float

@dataclass
class TradingSignal:
    """交易信号数据结构"""
    timestamp: datetime
    action: str  # 'BUY', 'SELL', 'HOLD'
    price: float
    quantity: float
    confidence: float
    reason: str
    liquidity_zone: Optional[LiquidityZone] = None

class LiquidityDensityAnalyzer:
    """流动性密度分析器"""

    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.price_history = deque(maxlen=1000)
        self.volume_profile = defaultdict(float)
        self.support_resistance_levels = []

    def analyze_liquidity_density(self, trades_window_data: List[Dict]) -> List[LiquidityZone]:
        """分析流动性密度并识别70%密集区域"""

        # 1. 构建成交量分布
        volume_distribution = defaultdict(lambda: {'buy_volume': 0, 'sell_volume': 0, 'trade_count': 0})

        for minute_data in trades_window_data:
            price_levels = minute_data.get('price_levels', {})
            for price_str, level_data in price_levels.items():
                price = float(price_str)
                volume_distribution[price]['buy_volume'] += level_data.get('buy_volume', 0)
                volume_distribution[price]['sell_volume'] += level_data.get('sell_volume', 0)
                volume_distribution[price]['trade_count'] += level_data.get('trade_count', 0)

        # 2. 计算总成交量并排序
        total_volume = sum(
            data['buy_volume'] + data['sell_volume']
            for data in volume_distribution.values()
        )

        sorted_prices = sorted(volume_distribution.keys())
        cumulative_volume = 0
        liquidity_zones = []

        # 3. 识别70%流动性密集区域
        target_volume = total_volume * 0.7

        for i, price in enumerate(sorted_prices):
            data = volume_distribution[price]
            price_volume = data['buy_volume'] + data['sell_volume']
            cumulative_volume += price_volume

            if cumulative_volume >= target_volume:
                # 找到了70%流动性的边界
                start_price = sorted_prices[0]
                end_price = price

                # 计算区域统计信息
                zone_prices = [p for p in sorted_prices if start_price <= p <= end_price]
                zone_volume = sum(
                    volume_distribution[p]['buy_volume'] + volume_distribution[p]['sell_volume']
                    for p in zone_prices
                )

                if zone_prices:
                    zone = LiquidityZone(
                        price_center=statistics.mean(zone_prices),
                        price_range=(min(zone_prices), max(zone_prices)),
                        total_volume=zone_volume,
                        buy_ratio=sum(volume_distribution[p]['buy_volume'] for p in zone_prices) / zone_volume,
                        sell_ratio=sum(volume_distribution[p]['sell_volume'] for p in zone_prices) / zone_volume,
                        trade_count=sum(volume_distribution[p]['trade_count'] for p in zone_prices),
                        density_score=zone_volume / (max(zone_prices) - min(zone_prices)) if max(zone_prices) != min(zone_prices) else 0,
                        support_strength=self._calculate_support_strength(zone_prices, volume_distribution),
                        resistance_strength=self._calculate_resistance_strength(zone_prices, volume_distribution)
                    )
                    liquidity_zones.append(zone)
                break

        return liquidity_zones

    def _calculate_support_strength(self, prices: List[float], volume_data: Dict) -> float:
        """计算支撑强度"""
        support_score = 0
        for price in prices:
            data = volume_data[price]
            if data['buy_volume'] > data['sell_volume']:
                support_score += (data['buy_volume'] - data['sell_volume']) * data['trade_count']
        return support_score

    def _calculate_resistance_strength(self, prices: List[float], volume_data: Dict) -> float:
        """计算阻力强度"""
        resistance_score = 0
        for price in prices:
            data = volume_data[price]
            if data['sell_volume'] > data['buy_volume']:
                resistance_score += (data['sell_volume'] - data['buy_volume']) * data['trade_count']
        return resistance_score

class HighFrequencyArbitrageStrategy:
    """高频套利策略 (只做多)"""

    def __init__(self, risk_per_trade: float = 0.01, min_profit_target: float = 0.001):
        self.risk_per_trade = risk_per_trade
        self.min_profit_target = min_profit_target
        self.position = None
        self.entry_price = None
        self.stop_loss = None
        self.take_profit = None

    def generate_signal(self, current_price: float, liquidity_zones: List[LiquidityZone],
                       market_data: Dict) -> TradingSignal:
        """生成交易信号 (只做多策略)"""

        if not liquidity_zones:
            return TradingSignal(
                timestamp=datetime.now(),
                action='HOLD',
                price=current_price,
                quantity=0,
                confidence=0,
                reason='No liquidity zones detected'
            )

        # 选择最强的流动性区域
        best_zone = max(liquidity_zones, key=lambda z: z.density_score)

        # 计算价格相对于流动性区域的位置
        zone_center = best_zone.price_center
        zone_lower = best_zone.price_range[0]
        zone_upper = best_zone.price_range[1]

        # 买入信号条件
        buy_conditions = [
            current_price < zone_center,  # 价格低于区域中心
            current_price > zone_lower,   # 价格高于区域下沿
            best_zone.buy_ratio > 0.6,    # 买盘占比超过60%
            best_zone.support_strength > best_zone.resistance_strength,  # 支撑强于阻力
            best_zone.density_score > 100  # 流动性密度足够高
        ]

        if all(buy_conditions) and self.position is None:
            # 计算仓位大小
            entry_price = current_price
            stop_loss_price = zone_lower * 0.995  # 区域下沿下方0.5%
            take_profit_price = zone_center * 1.002  # 区域中心上方0.2%

            risk_amount = entry_price - stop_loss_price
            position_size = (self.risk_per_trade * entry_price) / risk_amount

            return TradingSignal(
                timestamp=datetime.now(),
                action='BUY',
                price=entry_price,
                quantity=position_size,
                confidence=min(sum(buy_conditions) / len(buy_conditions), 1.0),
                reason=f'BUY: Price near liquidity support zone (center: {zone_center:.2f})',
                liquidity_zone=best_zone
            )

        # 卖出信号条件
        if self.position is not None:
            sell_conditions = [
                current_price >= self.take_profit,  # 达到止盈目标
                current_price <= self.stop_loss,    # 触及止损
                current_price > zone_upper          # 价格突破区域上沿
            ]

            if any(sell_conditions):
                return TradingSignal(
                    timestamp=datetime.now(),
                    action='SELL',
                    price=current_price,
                    quantity=self.position,
                    confidence=1.0,
                    reason=f'SELL: {"Take profit" if current_price >= self.take_profit else "Stop loss" if current_price <= self.stop_loss else "Zone breakout"}'
                )

        return TradingSignal(
            timestamp=datetime.now(),
            action='HOLD',
            price=current_price,
            quantity=0,
            confidence=0,
            reason='No trading conditions met'
        )

class BacktestEngine:
    """回测引擎"""

    def __init__(self, redis_client: redis.Redis, initial_capital: float = 10000):
        self.redis = redis_client
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.position = 0
        self.trades = []
        self.equity_curve = []

    async def run_backtest(self, start_time: datetime, end_time: datetime) -> Dict:
        """运行回测"""

        analyzer = LiquidityDensityAnalyzer(self.redis)
        strategy = HighFrequencyArbitrageStrategy()

        # 获取历史数据
        trades_data = await self._get_historical_trades(start_time, end_time)

        for i, minute_data in enumerate(trades_data):
            current_time = datetime.fromisoformat(minute_data['timestamp'].replace('Z', '+00:00'))
            current_price = self._get_current_price(minute_data)

            # 分析流动性密度 (使用滑动窗口)
            window_data = trades_data[max(0, i-60):i+1]  # 60分钟窗口
            liquidity_zones = analyzer.analyze_liquidity_density(window_data)

            # 生成交易信号
            signal = strategy.generate_signal(current_price, liquidity_zones, minute_data)

            # 执行交易
            if signal.action == 'BUY' and self.position == 0:
                self._execute_buy(signal)
            elif signal.action == 'SELL' and self.position > 0:
                self._execute_sell(signal)

            # 更新权益曲线
            current_equity = self.current_capital + (self.position * current_price if self.position > 0 else 0)
            self.equity_curve.append({
                'timestamp': current_time,
                'equity': current_equity,
                'price': current_price,
                'position': self.position
            })

        return self._calculate_performance_metrics()

    async def _get_historical_trades(self, start_time: datetime, end_time: datetime) -> List[Dict]:
        """从Redis获取历史交易数据"""
        # 模拟获取历史数据，实际实现需要根据Redis结构调整
        trades = []

        # 从trades_window获取数据
        window_size = self.redis.llen('trades_window')
        for i in range(window_size):
            data_str = self.redis.lindex('trades_window', i)
            if data_str:
                data = json.loads(data_str)
                timestamp = datetime.fromisoformat(data['timestamp'].replace('Z', '+00:00'))
                if start_time <= timestamp <= end_time:
                    trades.append(data)

        return trades

    def _get_current_price(self, minute_data: Dict) -> float:
        """从分钟数据中获取当前价格"""
        price_levels = minute_data.get('price_levels', {})
        if not price_levels:
            return 0.0

        # 返回成交量最大的价格作为代表性价格
        max_volume_price = max(
            price_levels.keys(),
            key=lambda p: price_levels[p]['total_volume']
        )
        return float(max_volume_price)

    def _execute_buy(self, signal: TradingSignal):
        """执行买入交易"""
        cost = signal.price * signal.quantity
        if cost <= self.current_capital:
            self.current_capital -= cost
            self.position = signal.quantity
            self.trades.append({
                'timestamp': signal.timestamp,
                'action': 'BUY',
                'price': signal.price,
                'quantity': signal.quantity,
                'reason': signal.reason
            })
            logger.info(f"BUY: {signal.quantity:.6f} @ {signal.price:.2f} - {signal.reason}")

    def _execute_sell(self, signal: TradingSignal):
        """执行卖出交易"""
        if self.position > 0:
            revenue = signal.price * self.position
            self.current_capital += revenue
            self.trades.append({
                'timestamp': signal.timestamp,
                'action': 'SELL',
                'price': signal.price,
                'quantity': self.position,
                'reason': signal.reason
            })
            logger.info(f"SELL: {self.position:.6f} @ {signal.price:.2f} - {signal.reason}")
            self.position = 0

    def _calculate_performance_metrics(self) -> Dict:
        """计算性能指标"""
        if not self.equity_curve:
            return {}

        equity_values = [e['equity'] for e in self.equity_curve]
        total_return = (self.current_capital - self.initial_capital) / self.initial_capital

        # 计算最大回撤
        peak = equity_values[0]
        max_drawdown = 0
        for equity in equity_values:
            if equity > peak:
                peak = equity
            drawdown = (peak - equity) / peak
            if drawdown > max_drawdown:
                max_drawdown = drawdown

        # 计算夏普比率
        returns = np.diff(equity_values) / equity_values[:-1]
        sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0

        # 胜率统计
        winning_trades = sum(1 for i in range(0, len(self.trades), 2)
                           if i+1 < len(self.trades) and
                           self.trades[i+1]['price'] > self.trades[i]['price'])
        total_trades = len(self.trades) // 2
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        return {
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'win_rate': win_rate,
            'total_trades': total_trades,
            'final_capital': self.current_capital,
            'equity_curve': self.equity_curve,
            'trades': self.trades
        }

async def main():
    """主函数"""
    # 连接Redis
    redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)

    # 创建回测引擎
    engine = BacktestEngine(redis_client, initial_capital=10000)

    # 设置回测时间范围 (最近48小时)
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=48)

    # 运行回测
    logger.info(f"开始回测: {start_time} 到 {end_time}")
    results = await engine.run_backtest(start_time, end_time)

    # 输出结果
    print("\n=== 回测结果 ===")
    print(f"总收益率: {results['total_return']:.2%}")
    print(f"最大回撤: {results['max_drawdown']:.2%}")
    print(f"夏普比率: {results['sharpe_ratio']:.3f}")
    print(f"胜率: {results['win_rate']:.2%}")
    print(f"总交易次数: {results['total_trades']}")
    print(f"最终资金: ${results['final_capital']:.2f}")

    # 保存详细结果
    with open('backtest_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    logger.info("回测完成，结果已保存到 backtest_results.json")

if __name__ == "__main__":
    asyncio.run(main())