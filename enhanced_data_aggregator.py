#!/usr/bin/env python3
"""
增强数据聚合器 - 精细化处理trades_window和depth_snapshot_5000数据
按10美元精度聚合深度数据，生成长短周期结合的数据集
"""

import json
import redis
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from collections import defaultdict
import yaml

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedDataAggregator:
    """增强数据聚合器类"""

    def __init__(self, config_path: str = "config/development.yaml"):
        """
        初始化数据聚合器

        Args:
            config_path: 配置文件路径
        """
        self.config = self._load_config(config_path)
        self.redis = self._connect_redis()
        self.price_precision = 10.0  # 10美元精度

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

    def _connect_redis(self) -> redis.Redis:
        """连接Redis"""
        try:
            redis_config = self.config.get('redis', {})
            r = redis.Redis(
                host=redis_config.get('host', 'localhost'),
                port=redis_config.get('port', 6379),
                db=redis_config.get('db', 0),
                decode_responses=redis_config.get('decode_responses', True),
                socket_timeout=redis_config.get('socket_timeout', 5),
                socket_connect_timeout=redis_config.get('socket_connect_timeout', 5)
            )
            # 测试连接
            r.ping()
            logger.info("Redis连接成功")
            return r
        except Exception as e:
            logger.error(f"Redis连接失败: {e}")
            raise

    def _floor_to_precision(self, price: float, precision: float) -> float:
        """
        按照指定精度向下取整价格

        Args:
            price: 原始价格
            precision: 精度（如10.0表示10美元）

        Returns:
            floored_price: 向下取整后的价格
        """
        return np.floor(price / precision) * precision

    def aggregate_depth_data(self, depth_data: Dict) -> Dict:
        """
        按照10美元精度聚合深度数据

        Args:
            depth_data: 原始深度数据

        Returns:
            aggregated_depth: 聚合后的深度数据
        """
        if not depth_data:
            return {}

        logger.info("开始按10美元精度聚合深度数据")

        # 获取原始数据
        asks = depth_data.get('asks', [])
        bids = depth_data.get('bids', [])

        # 聚合卖单 (asks)
        aggregated_asks = self._aggregate_order_book_side(asks, 'asks', self.price_precision)

        # 聚合买单 (bids)
        aggregated_bids = self._aggregate_order_book_side(bids, 'bids', self.price_precision)

        # 计算统计信息
        ask_stats = self._calculate_side_statistics(aggregated_asks)
        bid_stats = self._calculate_side_statistics(aggregated_bids)

        # 找到最佳价格
        best_ask = min(aggregated_asks.keys()) if aggregated_asks else 0
        best_bid = max(aggregated_bids.keys()) if aggregated_bids else 0
        spread = best_ask - best_bid if best_ask > 0 and best_bid > 0 else 0

        aggregated_depth = {
            'symbol': depth_data.get('symbol', 'BTCFDUSD'),
            'timestamp': depth_data.get('timestamp'),
            'aggregation_precision': self.price_precision,
            'price_levels': {
                'total_levels': len(aggregated_asks) + len(aggregated_bids),
                'ask_levels': len(aggregated_asks),
                'bid_levels': len(aggregated_bids)
            },
            'aggregated_asks': dict(sorted(aggregated_asks.items())),
            'aggregated_bids': dict(sorted(aggregated_bids.items(), reverse=True)),
            'market_summary': {
                'best_ask': best_ask,
                'best_bid': best_bid,
                'spread': spread,
                'spread_percentage': (spread / best_ask * 100) if best_ask > 0 else 0,
                'ask_stats': ask_stats,
                'bid_stats': bid_stats
            }
        }

        logger.info(f"深度数据聚合完成: {len(aggregated_asks)}个ask价格档位, {len(aggregated_bids)}个bid价格档位")
        return aggregated_depth

    def _aggregate_order_book_side(self, orders: List, side: str, precision: float) -> Dict[float, float]:
        """
        聚合订单簿的一侧

        Args:
            orders: 订单列表 [[price, quantity], ...]
            side: 'asks' 或 'bids'
            precision: 价格精度

        Returns:
            aggregated_orders: 聚合后的订单 {price: quantity}
        """
        aggregated = defaultdict(float)

        for price, quantity in orders:
            try:
                price = float(price)
                quantity = float(quantity)

                # 按10美元精度向下取整
                floored_price = self._floor_to_precision(price, precision)

                # 累加到对应价格档位
                aggregated[floored_price] += quantity

            except (ValueError, TypeError):
                continue

        return dict(aggregated)

    def _calculate_side_statistics(self, aggregated_orders: Dict[float, float]) -> Dict:
        """
        计算聚合订单的统计信息

        Args:
            aggregated_orders: 聚合后的订单数据

        Returns:
            stats: 统计信息
        """
        if not aggregated_orders:
            return {}

        prices = list(aggregated_orders.keys())
        quantities = list(aggregated_orders.values())

        return {
            'total_quantity': sum(quantities),
            'avg_quantity': np.mean(quantities),
            'max_quantity': max(quantities),
            'min_quantity': min(quantities),
            'price_range': {
                'min': min(prices),
                'max': max(prices),
                'span': max(prices) - min(prices)
            },
            'concentration': {
                'top_3_levels_ratio': sum(sorted(quantities, reverse=True)[:3]) / sum(quantities),
                'top_5_levels_ratio': sum(sorted(quantities, reverse=True)[:5]) / sum(quantities),
                'top_10_levels_ratio': sum(sorted(quantities, reverse=True)[:10]) / sum(quantities)
            }
        }

    def get_recent_trades_data(self, minutes: int = 30) -> List[Dict]:
        """
        获取最近N分钟的交易数据

        Args:
            minutes: 分钟数

        Returns:
            trades_data: 交易数据列表
        """
        try:
            window_size = self.redis.llen('trades_window')
            if window_size == 0:
                logger.warning("trades_window为空")
                return []

            # 获取最近N分钟数据
            limit = min(minutes, window_size)
            trades_data = []

            for i in range(limit):
                data_str = self.redis.lindex('trades_window', -i-1)
                if data_str:
                    try:
                        data = json.loads(data_str)
                        trades_data.append(data)
                    except json.JSONDecodeError:
                        continue

            # 按时间正序排列
            trades_data.reverse()
            logger.info(f"获取最近{len(trades_data)}分钟交易数据")
            return trades_data

        except Exception as e:
            logger.error(f"获取交易数据失败: {e}")
            return []

    def aggregate_4h_data(self, trades_data: List[Dict]) -> List[Dict]:
        """
        聚合4小时数据

        Args:
            trades_data: 交易数据列表

        Returns:
            aggregated_4h: 4小时聚合数据列表
        """
        if not trades_data:
            return []

        logger.info("开始聚合4小时数据")

        # 按时间分组，每4小时一组
        grouped_data = defaultdict(list)

        for minute_data in trades_data:
            try:
                timestamp_str = minute_data.get('timestamp', '')
                timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))

                # 计算这个时间点属于哪个4小时段
                base_time = timestamp.replace(
                    hour=(timestamp.hour // 4) * 4,
                    minute=0,
                    second=0,
                    microsecond=0
                )

                grouped_data[base_time].append(minute_data)

            except Exception as e:
                logger.warning(f"解析时间戳失败: {e}")
                continue

        # 聚合每个4小时段
        aggregated_4h = []

        for period_start, period_data in sorted(grouped_data.items()):
            aggregated = self._aggregate_4h_period(period_start, period_data)
            if aggregated:
                aggregated_4h.append(aggregated)

        logger.info(f"4小时数据聚合完成，共{len(aggregated_4h)}个时间段")
        return aggregated_4h

    def _aggregate_4h_period(self, period_start: datetime, period_data: List[Dict]) -> Optional[Dict]:
        """
        聚合单个4小时段的数据

        Args:
            period_start: 时间段开始时间
            period_data: 时间段内的数据

        Returns:
            aggregated_period: 聚合后的时间段数据
        """
        if not period_data:
            return None

        # 收集所有价格水平的数据
        all_prices = []
        all_volumes = []
        buy_volumes = []
        sell_volumes = []
        trade_counts = []
        deltas = []

        for minute_data in period_data:
            price_levels = minute_data.get('price_levels', {})

            for price_str, level_data in price_levels.items():
                try:
                    price = float(price_str)
                    total_volume = level_data.get('total_volume', 0)
                    buy_volume = level_data.get('buy_volume', 0)
                    sell_volume = level_data.get('sell_volume', 0)
                    trade_count = level_data.get('trade_count', 0)
                    delta = level_data.get('delta', 0)

                    all_prices.extend([price] * trade_count)
                    all_volumes.append(total_volume)
                    buy_volumes.append(buy_volume)
                    sell_volumes.append(sell_volume)
                    trade_counts.append(trade_count)
                    deltas.append(delta)

                except (ValueError, KeyError):
                    continue

        if not all_prices:
            return None

        # 计算4小时聚合指标
        period_end = period_start + timedelta(hours=4)

        aggregated_period = {
            'period_start': period_start.isoformat(),
            'period_end': period_end.isoformat(),
            'duration_hours': 4,
            'price_analysis': {
                'high': max(all_prices),
                'low': min(all_prices),
                'open': all_prices[0] if all_prices else 0,
                'close': all_prices[-1] if all_prices else 0,
                'avg_price': np.mean(all_prices),
                'price_change': (all_prices[-1] - all_prices[0]) / all_prices[0] if all_prices[0] > 0 else 0,
                'price_range': max(all_prices) - min(all_prices)
            },
            'volume_analysis': {
                'total_volume': sum(all_volumes),
                'total_buy_volume': sum(buy_volumes),
                'total_sell_volume': sum(sell_volumes),
                'total_trades': sum(trade_counts),
                'buy_ratio': sum(buy_volumes) / sum(all_volumes) if sum(all_volumes) > 0 else 0,
                'sell_ratio': sum(sell_volumes) / sum(all_volumes) if sum(all_volumes) > 0 else 0,
                'avg_volume_per_price': np.mean(all_volumes) if all_volumes else 0,
                'avg_trades_per_price': np.mean(trade_counts) if trade_counts else 0
            },
            'delta_analysis': {
                'total_delta': sum(deltas),
                'avg_delta': np.mean(deltas) if deltas else 0,
                'positive_periods': sum(1 for d in deltas if d > 0),
                'negative_periods': sum(1 for d in deltas if d < 0),
                'neutral_periods': sum(1 for d in deltas if d == 0),
                'delta_strength': abs(sum(deltas)) / sum(all_volumes) if sum(all_volumes) > 0 else 0
            },
            'data_points': len(period_data),
            'price_levels_count': len(set(all_prices))
        }

        return aggregated_period

    def format_data_for_ai(self, recent_trades: List[Dict], depth_aggregated: Dict,
                          aggregated_4h: List[Dict]) -> str:
        """
        将数据格式化为适合AI分析的文本

        Args:
            recent_trades: 最近30分钟交易数据
            depth_aggregated: 聚合后的深度数据
            aggregated_4h: 4小时聚合数据

        Returns:
            formatted_text: 格式化文本
        """
        formatted_text = []

        # 标题和概述
        formatted_text.append("="*80)
        formatted_text.append("BTC-FDUSD 多周期市场分析数据")
        formatted_text.append("="*80)
        formatted_text.append(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        formatted_text.append("")

        # 第一部分: 聚合深度数据 (10美元精度)
        formatted_text.append("📚 深度订单簿分析 (10美元精度聚合)")
        formatted_text.append("-"*60)

        if depth_aggregated:
            market_summary = depth_aggregated.get('market_summary', {})

            formatted_text.append(f"最佳卖价: ${market_summary.get('best_ask', 0):,.2f}")
            formatted_text.append(f"最佳买价: ${market_summary.get('best_bid', 0):,.2f}")
            formatted_text.append(f"价差: ${market_summary.get('spread', 0):,.2f} ({market_summary.get('spread_percentage', 0):.4f}%)")
            formatted_text.append(f"聚合精度: ${depth_aggregated.get('aggregation_precision', 10):.0f}")
            formatted_text.append("")

            # 显示关键ask价格档位
            formatted_text.append("📉 卖方深度 (Asks) - 关键价格档位:")
            asks = depth_aggregated.get('aggregated_asks', {})
            top_asks = sorted(asks.items(), key=lambda x: x[0])[:15]  # 显示最低15个ask价格

            for price, quantity in top_asks:
                formatted_text.append(f"  ${price:,.0f}: {quantity:.4f} BTC")

            formatted_text.append("")

            # 显示关键bid价格档位
            formatted_text.append("📈 买方深度 (Bids) - 关键价格档位:")
            bids = depth_aggregated.get('aggregated_bids', {})
            top_bids = sorted(bids.items(), key=lambda x: x[0], reverse=True)[:15]  # 显示最高15个bid价格

            for price, quantity in top_bids:
                formatted_text.append(f"  ${price:,.0f}: {quantity:.4f} BTC")

            formatted_text.append("")

            # 深度统计
            ask_stats = market_summary.get('ask_stats', {})
            bid_stats = market_summary.get('bid_stats', {})

            formatted_text.append("📊 深度统计:")
            formatted_text.append(f"  卖方总挂单量: {ask_stats.get('total_quantity', 0):.4f} BTC")
            formatted_text.append(f"  买方总挂单量: {bid_stats.get('total_quantity', 0):.4f} BTC")
            formatted_text.append(f"  卖方集中度(前3档): {ask_stats.get('concentration', {}).get('top_3_levels_ratio', 0):.1%}")
            formatted_text.append(f"  买方集中度(前3档): {bid_stats.get('concentration', {}).get('top_3_levels_ratio', 0):.1%}")

        formatted_text.append("")

        # 第二部分: 最近30分钟详细数据
        formatted_text.append("📈 最近30分钟交易流详细数据")
        formatted_text.append("-"*60)

        if recent_trades:
            # 计算整体统计
            total_volume = 0
            total_trades = 0
            buy_volume = 0
            sell_volume = 0
            all_prices = []

            for minute_data in recent_trades:
                price_levels = minute_data.get('price_levels', {})
                for level_data in price_levels.values():
                    total_volume += level_data.get('total_volume', 0)
                    total_trades += level_data.get('trade_count', 0)
                    buy_volume += level_data.get('buy_volume', 0)
                    sell_volume += level_data.get('sell_volume', 0)

            # 显示每分钟数据
            formatted_text.append(f"最近30分钟数据 ({len(recent_trades)}分钟):")
            formatted_text.append(f"总成交量: {total_volume:.4f} BTC")
            formatted_text.append(f"总交易次数: {total_trades}")
            formatted_text.append(f"买盘占比: {buy_volume/total_volume:.1%}" if total_volume > 0 else "买盘占比: N/A")
            formatted_text.append("")

            # 显示每分钟的关键信息
            for i, minute_data in enumerate(recent_trades):
                timestamp = minute_data.get('timestamp', '')
                price_levels = minute_data.get('price_levels', {})

                if price_levels:
                    # 找到成交量最大的价格
                    max_volume_price = max(
                        price_levels.keys(),
                        key=lambda p: price_levels[p]['total_volume']
                    )

                    max_volume_data = price_levels[max_volume_price]

                    formatted_text.append(f"分钟 {i+1} ({timestamp[-8:]}):")
                    formatted_text.append(f"  主力价格: ${float(max_volume_price):,.2f}")
                    formatted_text.append(f"  成交量: {max_volume_data.get('total_volume', 0):.4f} BTC")
                    formatted_text.append(f"  买盘: {max_volume_data.get('buy_volume', 0):.4f} BTC")
                    formatted_text.append(f"  卖盘: {max_volume_data.get('sell_volume', 0):.4f} BTC")
                    formatted_text.append(f"  净流入: {max_volume_data.get('delta', 0):+.4f} BTC")
                    formatted_text.append(f"  交易次数: {max_volume_data.get('trade_count', 0)}")
                    formatted_text.append("")

        formatted_text.append("")

        # 第三部分: 4小时聚合数据
        formatted_text.append("📊 4小时周期聚合数据")
        formatted_text.append("-"*60)

        if aggregated_4h:
            for i, period_data in enumerate(aggregated_4h, 1):
                price_analysis = period_data.get('price_analysis', {})
                volume_analysis = period_data.get('volume_analysis', {})
                delta_analysis = period_data.get('delta_analysis', {})

                formatted_text.append(f"4小时段 {i}:")
                formatted_text.append(f"  时间: {period_data.get('period_start', '')[:19]} - {period_data.get('period_end', '')[:19]}")
                formatted_text.append(f"  价格区间: ${price_analysis.get('low', 0):,.2f} - ${price_analysis.get('high', 0):,.2f}")
                formatted_text.append(f"  开盘价: ${price_analysis.get('open', 0):,.2f}")
                formatted_text.append(f"  收盘价: ${price_analysis.get('close', 0):,.2f}")
                formatted_text.append(f"  价格变化: {price_analysis.get('price_change', 0):+.2%}")
                formatted_text.append(f"  总成交量: {volume_analysis.get('total_volume', 0):.4f} BTC")
                formatted_text.append(f"  买盘占比: {volume_analysis.get('buy_ratio', 0):.1%}")
                formatted_text.append(f"  总交易次数: {volume_analysis.get('total_trades', 0)}")
                formatted_text.append(f"  净流量: {delta_analysis.get('total_delta', 0):+.4f} BTC")
                formatted_text.append(f"  流动性强度: {delta_analysis.get('delta_strength', 0):.4f}")
                formatted_text.append("")

        formatted_text.append("="*80)
        formatted_text.append("数据分析完成 - 请基于以上信息进行交易决策分析 (做多/做空/持有)")
        formatted_text.append("="*80)

        return "\n".join(formatted_text)

    def save_formatted_data(self, formatted_text: str, filename: str = None) -> str:
        """
        保存格式化数据到文件

        Args:
            formatted_text: 格式化文本
            filename: 文件名

        Returns:
            filepath: 保存的文件路径
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"enhanced_market_analysis_{timestamp}.txt"

        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(formatted_text)
            logger.info(f"增强格式化数据已保存到: {filename}")
            return filename
        except Exception as e:
            logger.error(f"保存文件失败: {e}")
            raise

def main():
    """主函数 - 演示增强数据聚合功能"""
    logger.info("开始增强数据聚合演示")

    try:
        # 创建增强数据聚合器
        aggregator = EnhancedDataAggregator()

        # 1. 获取和处理深度数据
        logger.info("步骤1: 获取和聚合深度数据")
        depth_str = aggregator.redis.get('depth_snapshot_5000')
        if depth_str:
            depth_data = json.loads(depth_str)
            aggregated_depth = aggregator.aggregate_depth_data(depth_data)
            print("✅ 深度数据聚合完成")
        else:
            print("❌ 没有深度数据")
            aggregated_depth = {}

        # 2. 获取最近30分钟交易数据
        logger.info("步骤2: 获取最近30分钟交易数据")
        recent_trades = aggregator.get_recent_trades_data(minutes=30)
        print(f"✅ 获取最近{len(recent_trades)}分钟交易数据")

        # 3. 聚合4小时数据
        logger.info("步骤3: 聚合4小时数据")
        all_trades = aggregator.get_recent_trades_data(minutes=2880)  # 48小时数据用于4小时聚合
        aggregated_4h = aggregator.aggregate_4h_data(all_trades)
        print(f"✅ 4小时数据聚合完成，共{len(aggregated_4h)}个时间段")

        # 4. 格式化数据
        logger.info("步骤4: 格式化数据")
        formatted_text = aggregator.format_data_for_ai(recent_trades, aggregated_depth, aggregated_4h)

        # 5. 保存数据
        filename = aggregator.save_formatted_data(formatted_text)

        # 6. 显示摘要
        print("\n" + "="*60)
        print("增强数据聚合完成!")
        print("="*60)
        print(f"深度数据: {'已聚合' if aggregated_depth else '未获取'}")
        print(f"最近30分钟: {len(recent_trades)} 分钟")
        print(f"4小时数据: {len(aggregated_4h)} 个时间段")
        print(f"格式化文件: {filename}")
        print("\n格式化文本预览:")
        print("-"*40)
        print(formatted_text[:2000] + "..." if len(formatted_text) > 2000 else formatted_text)

    except Exception as e:
        logger.error(f"增强数据聚合失败: {e}")
        raise

if __name__ == "__main__":
    main()