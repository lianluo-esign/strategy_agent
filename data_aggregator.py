#!/usr/bin/env python3
"""
数据聚合器 - 处理trades_window和depth_snapshot_5000数据
将Redis数据聚合成适合大模型阅读的文本格式
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

class DataAggregator:
    """数据聚合器类"""

    def __init__(self, config_path: str = "config/development.yaml"):
        """
        初始化数据聚合器

        Args:
            config_path: 配置文件路径
        """
        self.config = self._load_config(config_path)
        self.redis = self._connect_redis()

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

    def get_trades_window_data(self, limit: Optional[int] = None) -> List[Dict]:
        """
        获取trades_window数据

        Args:
            limit: 获取的数据条数限制

        Returns:
            trades_data: 交易数据列表
        """
        try:
            window_size = self.redis.llen('trades_window')
            if window_size == 0:
                logger.warning("trades_window为空")
                return []

            # 如果没有指定limit，获取所有数据
            if limit is None:
                limit = window_size
            else:
                limit = min(limit, window_size)

            logger.info(f"获取trades_window数据: {limit}/{window_size}条")

            trades_data = []
            for i in range(limit):
                data_str = self.redis.lindex('trades_window', -i-1)  # 从最新的开始
                if data_str:
                    try:
                        data = json.loads(data_str)
                        trades_data.append(data)
                    except json.JSONDecodeError as e:
                        logger.warning(f"解析交易数据失败: {e}")
                        continue

            # 按时间正序排列
            trades_data.reverse()
            logger.info(f"成功获取{len(trades_data)}条交易数据")
            return trades_data

        except Exception as e:
            logger.error(f"获取trades_window数据失败: {e}")
            return []

    def get_depth_snapshot_data(self) -> Optional[Dict]:
        """
        获取depth_snapshot_5000数据

        Returns:
            depth_data: 深度数据字典
        """
        try:
            depth_str = self.redis.get('depth_snapshot_5000')
            if not depth_str:
                logger.warning("depth_snapshot_5000为空")
                return None

            depth_data = json.loads(depth_str)
            logger.info("获取depth_snapshot_5000数据成功")
            return depth_data

        except Exception as e:
            logger.error(f"获取depth_snapshot_5000数据失败: {e}")
            return None

    def aggregate_trades_data(self, trades_data: List[Dict]) -> Dict:
        """
        聚合交易数据，生成统计信息

        Args:
            trades_data: 原始交易数据列表

        Returns:
            aggregated_data: 聚合后的数据字典
        """
        if not trades_data:
            return {}

        logger.info(f"开始聚合{len(trades_data)}条交易数据")

        # 基础统计
        total_volume = 0
        total_trades = 0
        buy_volume = 0
        sell_volume = 0

        # 价格统计
        all_prices = []
        all_volumes = []

        # 时间统计
        timestamps = []

        # 每分钟统计
        minute_stats = []

        # 价格水平聚合
        price_level_stats = defaultdict(lambda: {
            'total_volume': 0,
            'buy_volume': 0,
            'sell_volume': 0,
            'trade_count': 0,
            'delta': 0,
            'first_seen': None,
            'last_seen': None
        })

        for minute_data in trades_data:
            timestamp_str = minute_data.get('timestamp', '')
            price_levels = minute_data.get('price_levels', {})

            try:
                timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                timestamps.append(timestamp)
            except:
                continue

            minute_total_volume = 0
            minute_trade_count = 0
            minute_buy_volume = 0
            minute_sell_volume = 0

            for price_str, level_data in price_levels.items():
                try:
                    price = float(price_str)
                    level_buy_volume = level_data.get('buy_volume', 0)
                    level_sell_volume = level_data.get('sell_volume', 0)
                    level_total_volume = level_data.get('total_volume', 0)
                    level_trade_count = level_data.get('trade_count', 0)
                    level_delta = level_data.get('delta', 0)

                    # 累加到总体统计
                    total_volume += level_total_volume
                    total_trades += level_trade_count
                    buy_volume += level_buy_volume
                    sell_volume += level_sell_volume

                    # 记录价格和成交量
                    all_prices.extend([price] * level_trade_count)
                    all_volumes.append(level_total_volume)

                    # 分钟统计
                    minute_total_volume += level_total_volume
                    minute_trade_count += level_trade_count
                    minute_buy_volume += level_buy_volume
                    minute_sell_volume += level_sell_volume

                    # 价格水平统计
                    price_stats = price_level_stats[price]
                    price_stats['total_volume'] += level_total_volume
                    price_stats['buy_volume'] += level_buy_volume
                    price_stats['sell_volume'] += level_sell_volume
                    price_stats['trade_count'] += level_trade_count
                    price_stats['delta'] += level_delta

                    if price_stats['first_seen'] is None:
                        price_stats['first_seen'] = timestamp
                    price_stats['last_seen'] = timestamp

                except (ValueError, KeyError) as e:
                    logger.warning(f"解析价格水平数据失败: {e}")
                    continue

            # 保存分钟统计
            if minute_total_volume > 0:
                minute_stats.append({
                    'timestamp': timestamp_str,
                    'total_volume': minute_total_volume,
                    'buy_volume': minute_buy_volume,
                    'sell_volume': minute_sell_volume,
                    'trade_count': minute_trade_count,
                    'price_levels_count': len(price_levels)
                })

        # 计算衍生统计
        current_price = np.mean(all_prices) if all_prices else 0
        price_std = np.std(all_prices) if all_prices else 0
        price_range = (max(all_prices) - min(all_prices)) if all_prices else 0

        # 计算成交量最大的价格水平
        top_price_levels = sorted(
            price_level_stats.items(),
            key=lambda x: x[1]['total_volume'],
            reverse=True
        )[:10]

        # 计算买卖比率
        buy_ratio = buy_volume / total_volume if total_volume > 0 else 0
        sell_ratio = sell_volume / total_volume if total_volume > 0 else 0

        # 时间范围
        time_range = {
            'start': min(timestamps) if timestamps else None,
            'end': max(timestamps) if timestamps else None,
            'duration_hours': 0
        }

        if time_range['start'] and time_range['end']:
            time_range['duration_hours'] = (time_range['end'] - time_range['start']).total_seconds() / 3600

        aggregated_data = {
            'summary': {
                'total_minutes': len(trades_data),
                'total_volume_btc': round(total_volume, 4),
                'total_trades': total_trades,
                'buy_volume_btc': round(buy_volume, 4),
                'sell_volume_btc': round(sell_volume, 4),
                'buy_ratio': round(buy_ratio, 3),
                'sell_ratio': round(sell_ratio, 3),
                'current_price': round(current_price, 2),
                'price_std': round(price_std, 2),
                'price_range': round(price_range, 2)
            },
            'time_analysis': {
                'start_time': time_range['start'].isoformat() if time_range['start'] else None,
                'end_time': time_range['end'].isoformat() if time_range['end'] else None,
                'duration_hours': round(time_range['duration_hours'], 2)
            },
            'price_analysis': {
                'min_price': round(min(all_prices), 2) if all_prices else 0,
                'max_price': round(max(all_prices), 2) if all_prices else 0,
                'avg_price': round(current_price, 2),
                'median_price': round(np.median(all_prices), 2) if all_prices else 0,
                'price_levels_analyzed': len(price_level_stats)
            },
            'volume_analysis': {
                'avg_volume_per_minute': round(total_volume / len(trades_data), 4) if trades_data else 0,
                'avg_trades_per_minute': round(total_trades / len(trades_data), 1) if trades_data else 0,
                'volume_distribution': self._calculate_volume_distribution(price_level_stats)
            },
            'top_price_levels': [
                {
                    'price': round(price, 2),
                    'total_volume': round(stats['total_volume'], 4),
                    'buy_volume': round(stats['buy_volume'], 4),
                    'sell_volume': round(stats['sell_volume'], 4),
                    'trade_count': stats['trade_count'],
                    'buy_ratio': round(stats['buy_volume'] / stats['total_volume'], 3) if stats['total_volume'] > 0 else 0,
                    'delta': round(stats['delta'], 4)
                }
                for price, stats in top_price_levels
            ],
            'minute_statistics': minute_stats[:20],  # 最近20分钟数据
            'price_level_count': len(price_level_stats)
        }

        logger.info("交易数据聚合完成")
        return aggregated_data

    def _calculate_volume_distribution(self, price_level_stats: Dict) -> Dict:
        """计算成交量分布统计"""
        if not price_level_stats:
            return {}

        volumes = [stats['total_volume'] for stats in price_level_stats.values()]

        return {
            'total_levels': len(volumes),
            'max_volume': round(max(volumes), 4),
            'min_volume': round(min(volumes), 4),
            'avg_volume': round(np.mean(volumes), 4),
            'median_volume': round(np.median(volumes), 4),
            'volume_std': round(np.std(volumes), 4)
        }

    def aggregate_depth_data(self, depth_data: Dict) -> Dict:
        """
        聚合深度数据

        Args:
            depth_data: 原始深度数据

        Returns:
            aggregated_depth: 聚合后的深度数据
        """
        if not depth_data:
            return {}

        logger.info("开始聚合深度数据")

        asks = depth_data.get('asks', [])
        bids = depth_data.get('bids', [])

        # 处理卖单
        ask_analysis = self._analyze_order_book_side(asks, 'asks')

        # 处理买单
        bid_analysis = self._analyze_order_book_side(bids, 'bids')

        # 计算价差
        best_ask = ask_analysis['best_price'] if ask_analysis else 0
        best_bid = bid_analysis['best_price'] if bid_analysis else 0
        spread = best_ask - best_bid
        spread_percentage = (spread / best_ask) * 100 if best_ask > 0 else 0

        # 计算深度加权平均价格
        total_ask_value = ask_analysis['total_value'] if ask_analysis else 0
        total_bid_value = bid_analysis['total_value'] if bid_analysis else 0
        total_ask_volume = ask_analysis['total_volume'] if ask_analysis else 0
        total_bid_volume = bid_analysis['total_volume'] if bid_analysis else 0

        vwap_ask = total_ask_value / total_ask_volume if total_ask_volume > 0 else 0
        vwap_bid = total_bid_value / total_bid_volume if total_bid_volume > 0 else 0

        aggregated_depth = {
            'symbol': depth_data.get('symbol', 'BTCFDUSD'),
            'timestamp': depth_data.get('timestamp'),
            'spread_analysis': {
                'best_ask': round(best_ask, 2),
                'best_bid': round(best_bid, 2),
                'spread': round(spread, 2),
                'spread_percentage': round(spread_percentage, 4)
            },
            'ask_analysis': ask_analysis,
            'bid_analysis': bid_analysis,
            'market_depth': {
                'vwap_ask': round(vwap_ask, 2),
                'vwap_bid': round(vwap_bid, 2),
                'mid_price': round((vwap_ask + vwap_bid) / 2, 2),
                'total_depth_value': round(total_ask_value + total_bid_value, 2)
            }
        }

        logger.info("深度数据聚合完成")
        return aggregated_depth

    def _analyze_order_book_side(self, orders: List, side: str) -> Dict:
        """
        分析订单簿一侧

        Args:
            orders: 订单列表 [[price, quantity], ...]
            side: 'asks' 或 'bids'

        Returns:
            analysis: 分析结果
        """
        if not orders:
            return {}

        # 转换为DataFrame便于分析
        df = pd.DataFrame(orders, columns=['price', 'quantity'])
        df['price'] = pd.to_numeric(df['price'])
        df['quantity'] = pd.to_numeric(df['quantity'])
        df['value'] = df['price'] * df['quantity']

        # 计算累积成交量
        df['cumulative_quantity'] = df['quantity'].cumsum()
        df['cumulative_value'] = df['value'].cumsum()

        # 分析前10档
        top_10 = df.head(10)

        analysis = {
            'total_levels': len(df),
            'best_price': round(df.iloc[0]['price'], 2),
            'best_quantity': round(df.iloc[0]['quantity'], 4),
            'total_volume': round(df['quantity'].sum(), 4),
            'total_value': round(df['value'].sum(), 2),
            'top_10_analysis': {
                'volume': round(top_10['quantity'].sum(), 4),
                'value': round(top_10['value'].sum(), 2),
                'volume_ratio': round(top_10['quantity'].sum() / df['quantity'].sum(), 3),
                'price_range': {
                    'min': round(top_10['price'].min(), 2),
                    'max': round(top_10['price'].max(), 2)
                }
            },
            'concentration_analysis': {
                'top_3_volume_ratio': round(df.head(3)['quantity'].sum() / df['quantity'].sum(), 3),
                'top_5_volume_ratio': round(df.head(5)['quantity'].sum() / df['quantity'].sum(), 3),
                'top_10_volume_ratio': round(top_10['quantity'].sum() / df['quantity'].sum(), 3)
            }
        }

        return analysis

    def format_for_ai_analysis(self, trades_aggregated: Dict, depth_aggregated: Dict) -> str:
        """
        将聚合数据格式化为适合AI分析的文本格式

        Args:
            trades_aggregated: 聚合后的交易数据
            depth_aggregated: 聚合后的深度数据

        Returns:
            formatted_text: 格式化后的文本
        """
        if not trades_aggregated and not depth_aggregated:
            return "无可用数据进行分析"

        formatted_text = []

        # 添加标题
        formatted_text.append("=" * 80)
        formatted_text.append("BTC-FDUSD 市场数据分析报告")
        formatted_text.append("=" * 80)
        formatted_text.append("")

        # 交易数据分析
        if trades_aggregated:
            formatted_text.append("📊 交易流数据分析 (Trades Flow Analysis)")
            formatted_text.append("-" * 50)

            summary = trades_aggregated.get('summary', {})
            formatted_text.append(f"数据时间范围: {trades_aggregated.get('time_analysis', {}).get('duration_hours', 0):.1f} 小时")
            formatted_text.append(f"分析分钟数: {summary.get('total_minutes', 0)} 分钟")
            formatted_text.append(f"总成交量: {summary.get('total_volume_btc', 0)} BTC")
            formatted_text.append(f"总交易次数: {summary.get('total_trades', 0)} 次")
            formatted_text.append(f"买盘成交量: {summary.get('buy_volume_btc', 0)} BTC ({summary.get('buy_ratio', 0):.1%})")
            formatted_text.append(f"卖盘成交量: {summary.get('sell_volume_btc', 0)} BTC ({summary.get('sell_ratio', 0):.1%})")
            formatted_text.append(f"当前价格: ${summary.get('current_price', 0):,.2f}")
            formatted_text.append(f"价格标准差: ${summary.get('price_std', 0):,.2f}")
            formatted_text.append(f"价格区间: ${summary.get('price_range', 0):,.2f}")
            formatted_text.append("")

            # 价格分析
            price_analysis = trades_aggregated.get('price_analysis', {})
            formatted_text.append("📈 价格分析:")
            formatted_text.append(f"  最低价: ${price_analysis.get('min_price', 0):,.2f}")
            formatted_text.append(f"  最高价: ${price_analysis.get('max_price', 0):,.2f}")
            formatted_text.append(f"  平均价: ${price_analysis.get('avg_price', 0):,.2f}")
            formatted_text.append(f"  中位数价: ${price_analysis.get('median_price', 0):,.2f}")
            formatted_text.append(f"  分析价格水平数: {price_analysis.get('price_levels_analyzed', 0)}")
            formatted_text.append("")

            # 成交量分析
            volume_analysis = trades_aggregated.get('volume_analysis', {})
            formatted_text.append("📊 成交量分析:")
            formatted_text.append(f"  平均每分钟成交量: {volume_analysis.get('avg_volume_per_minute', 0)} BTC")
            formatted_text.append(f"  平均每分钟交易次数: {volume_analysis.get('avg_trades_per_minute', 0)} 次")

            volume_dist = volume_analysis.get('volume_distribution', {})
            formatted_text.append(f"  价格水平数: {volume_dist.get('total_levels', 0)}")
            formatted_text.append(f"  最大单价格成交量: {volume_dist.get('max_volume', 0)} BTC")
            formatted_text.append(f"  平均成交量: {volume_dist.get('avg_volume', 0)} BTC")
            formatted_text.append("")

            # 热门价格水平
            formatted_text.append("🔥 热门价格水平 (Top Price Levels):")
            for i, level in enumerate(trades_aggregated.get('top_price_levels', [])[:5], 1):
                formatted_text.append(f"  {i}. 价格: ${level['price']:,.2f}")
                formatted_text.append(f"     成交量: {level['total_volume']} BTC")
                formatted_text.append(f"     买盘: {level['buy_volume']} BTC ({level['buy_ratio']:.1%})")
                formatted_text.append(f"     卖盘: {level['sell_volume']} BTC")
                formatted_text.append(f"     交易次数: {level['trade_count']} 次")
                formatted_text.append(f"     净流入: {level['delta']:+.4f} BTC")
                formatted_text.append("")

        # 深度数据分析
        if depth_aggregated:
            formatted_text.append("📚 深度订单簿分析 (Order Book Depth Analysis)")
            formatted_text.append("-" * 50)

            spread_analysis = depth_aggregated.get('spread_analysis', {})
            formatted_text.append(f"交易对: {depth_aggregated.get('symbol', 'BTCFDUSD')}")
            formatted_text.append(f"数据时间: {depth_aggregated.get('timestamp', 'Unknown')}")
            formatted_text.append(f"最佳卖价: ${spread_analysis.get('best_ask', 0):,.2f}")
            formatted_text.append(f"最佳买价: ${spread_analysis.get('best_bid', 0):,.2f}")
            formatted_text.append(f"价差: ${spread_analysis.get('spread', 0):,.2f} ({spread_analysis.get('spread_percentage', 0):.4f}%)")
            formatted_text.append("")

            market_depth = depth_aggregated.get('market_depth', {})
            formatted_text.append("🎯 市场深度:")
            formatted_text.append(f"  卖方VWAP: ${market_depth.get('vwap_ask', 0):,.2f}")
            formatted_text.append(f"  买方VWAP: ${market_depth.get('vwap_bid', 0):,.2f}")
            formatted_text.append(f"  中间价: ${market_depth.get('mid_price', 0):,.2f}")
            formatted_text.append(f"  总深度价值: ${market_depth.get('total_depth_value', 0):,.2f}")
            formatted_text.append("")

            # 卖方分析
            ask_analysis = depth_aggregated.get('ask_analysis', {})
            if ask_analysis:
                formatted_text.append("📉 卖方深度 (Asks):")
                formatted_text.append(f"  总档位数: {ask_analysis.get('total_levels', 0)}")
                formatted_text.append(f"  总卖量: {ask_analysis.get('total_volume', 0)} BTC")
                formatted_text.append(f"  总价值: ${ask_analysis.get('total_value', 0):,.2f}")

                top_10 = ask_analysis.get('top_10_analysis', {})
                formatted_text.append(f"  前10档卖量: {top_10.get('volume', 0)} BTC ({top_10.get('volume_ratio', 0):.1%})")
                formatted_text.append(f"  前10档价值: ${top_10.get('value', 0):,.2f}")

                concentration = ask_analysis.get('concentration_analysis', {})
                formatted_text.append(f"  卖盘集中度: 前3档{concentration.get('top_3_volume_ratio', 0):.1%}, 前5档{concentration.get('top_5_volume_ratio', 0):.1%}")
                formatted_text.append("")

            # 买方分析
            bid_analysis = depth_aggregated.get('bid_analysis', {})
            if bid_analysis:
                formatted_text.append("📈 买方深度 (Bids):")
                formatted_text.append(f"  总档位数: {bid_analysis.get('total_levels', 0)}")
                formatted_text.append(f"  总买量: {bid_analysis.get('total_volume', 0)} BTC")
                formatted_text.append(f"  总价值: ${bid_analysis.get('total_value', 0):,.2f}")

                top_10 = bid_analysis.get('top_10_analysis', {})
                formatted_text.append(f"  前10档买量: {top_10.get('volume', 0)} BTC ({top_10.get('volume_ratio', 0):.1%})")
                formatted_text.append(f"  前10档价值: ${top_10.get('value', 0):,.2f}")

                concentration = bid_analysis.get('concentration_analysis', {})
                formatted_text.append(f"  买盘集中度: 前3档{concentration.get('top_3_volume_ratio', 0):.1%}, 前5档{concentration.get('top_5_volume_ratio', 0):.1%}")
                formatted_text.append("")

        # 添加分析结论
        formatted_text.append("=" * 80)
        formatted_text.append("数据分析完成 - 请基于以上信息进行流动性密度和交易机会分析")
        formatted_text.append("=" * 80)

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
            filename = f"market_analysis_{timestamp}.txt"

        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(formatted_text)
            logger.info(f"格式化数据已保存到: {filename}")
            return filename
        except Exception as e:
            logger.error(f"保存文件失败: {e}")
            raise

def main():
    """主函数 - 演示数据聚合功能"""
    logger.info("开始数据聚合演示")

    try:
        # 创建数据聚合器
        aggregator = DataAggregator()

        # 获取原始数据
        logger.info("获取原始数据...")
        trades_data = aggregator.get_trades_window_data(limit=100)  # 获取最近100分钟数据
        depth_data = aggregator.get_depth_snapshot_data()

        if not trades_data and not depth_data:
            logger.error("没有可用的数据进行分析")
            return

        # 聚合数据
        logger.info("聚合交易数据...")
        trades_aggregated = aggregator.aggregate_trades_data(trades_data)

        logger.info("聚合深度数据...")
        depth_aggregated = aggregator.aggregate_depth_data(depth_data)

        # 格式化为AI可读文本
        logger.info("格式化数据...")
        formatted_text = aggregator.format_for_ai_analysis(trades_aggregated, depth_aggregated)

        # 保存到文件
        filename = aggregator.save_formatted_data(formatted_text)

        # 打印摘要
        print("\n" + "="*60)
        print("数据聚合完成!")
        print("="*60)
        print(f"交易数据: {len(trades_data)} 分钟")
        print(f"深度数据: {'已获取' if depth_data else '未获取'}")
        print(f"格式化文件: {filename}")
        print("\n格式化文本预览:")
        print("-"*40)
        print(formatted_text[:1000] + "..." if len(formatted_text) > 1000 else formatted_text)

    except Exception as e:
        logger.error(f"数据聚合失败: {e}")
        raise

if __name__ == "__main__":
    main()