"""静态订单簿分析器 - 深度快照数据分析。

该模块提供深度订单簿数据的获取和聚合分析功能：
1. 从Redis获取depth_snapshot_5000数据
2. 按10美元精度聚合订单簿数据
3. 识别支撑阻力位和流动性墙
4. 为贝叶斯分析提供静态证据
"""

import logging
from collections import defaultdict
from decimal import Decimal
from typing import Any

logger = logging.getLogger(__name__)

# 聚合精度常量
ORDERBOOK_AGGREGATION_PRECISION = Decimal("10.0")  # 10美元聚合精度
TOP_LIQUIDITY_LEVELS = 20  # 显示的顶级流动性水平数量


class StaticOrderBookAnalyzer:
    """静态订单簿分析器，用于分析深度快照数据。

    该分析器处理5000层深度快照，生成聚合的订单簿视图，
    并识别关键的支撑阻力位和流动性集中区域。
    """

    def __init__(self, aggregation_precision: float = 10.0):
        """初始化静态订单簿分析器。

        Args:
            aggregation_precision: 订单簿价格聚合精度（美元）
        """
        self.aggregation_precision = Decimal(str(aggregation_precision))

        logger.info(
            f"Initialized StaticOrderBookAnalyzer with precision=${aggregation_precision}"
        )

    def analyze_order_book(
        self,
        depth_snapshot: Any,
        symbol: str = "BTCFDUSD"
    ) -> dict[str, Any]:
        """分析订单簿数据。

        Args:
            depth_snapshot: 深度快照数据
            symbol: 交易符号

        Returns:
            包含订单簿分析结果的字典
        """
        logger.info(f"Starting order book analysis for {symbol}")

        if not depth_snapshot:
            logger.warning("No depth snapshot data available for analysis")
            return self._create_empty_result(symbol)

        try:
            # 聚合订单簿数据
            aggregated_bids, aggregated_asks = self._aggregate_order_book_depth(
                depth_snapshot.bids, depth_snapshot.asks
            )

            if not aggregated_bids and not aggregated_asks:
                logger.warning("No order book data after aggregation")
                return self._create_empty_result(symbol)

            # 分析流动性特征
            liquidity_analysis = self._analyze_liquidity_features(
                aggregated_bids, aggregated_asks
            )

            # 识别关键价格水平
            key_levels = self._identify_key_price_levels(
                aggregated_bids, aggregated_asks
            )

            # 计算订单簿不平衡度
            imbalance_metrics = self._calculate_order_book_imbalance(
                aggregated_bids, aggregated_asks
            )

            result = {
                "symbol": symbol,
                "timestamp": depth_snapshot.timestamp,
                "analysis_type": "static_order_book",
                "aggregation_precision": float(self.aggregation_precision),
                "aggregated_bids": aggregated_bids,
                "aggregated_asks": aggregated_asks,
                "liquidity_analysis": liquidity_analysis,
                "key_levels": key_levels,
                "imbalance_metrics": imbalance_metrics,
                "status": "success",
            }

            logger.info(
                f"Order book analysis completed: {len(aggregated_bids)} bid levels, "
                f"{len(aggregated_asks)} ask levels"
            )

            return result

        except Exception as e:
            logger.error(f"Order book analysis failed: {e}")
            return self._create_error_result(symbol, str(e))

    def _aggregate_order_book_depth(
        self,
        raw_bids: list[tuple[str, str]],
        raw_asks: list[tuple[str, str]]
    ) -> tuple[dict[str, Decimal], dict[str, Decimal]]:
        """聚合订单簿深度数据。

        Args:
            raw_bids: 原始买单数据 [(price, quantity), ...]
            raw_asks: 原始卖单数据 [(price, quantity), ...]

        Returns:
            聚合后的买单和卖单字典
        """
        aggregated_bids: defaultdict[Decimal, Decimal] = defaultdict(Decimal)
        aggregated_asks: defaultdict[Decimal, Decimal] = defaultdict(Decimal)

        # 聚合买单
        for price_str, quantity_str in raw_bids:
            try:
                price = Decimal(str(price_str))
                quantity = Decimal(str(quantity_str))

                if quantity > 0:
                    # 按10美元精度对齐价格
                    aligned_price = self._align_price_to_precision(price)
                    aggregated_bids[aligned_price] += quantity
            except (ValueError, TypeError) as e:
                logger.debug(f"Skipping invalid bid data: {e}")
                continue

        # 聚合卖单
        for price_str, quantity_str in raw_asks:
            try:
                price = Decimal(str(price_str))
                quantity = Decimal(str(quantity_str))

                if quantity > 0:
                    # 按10美元精度对齐价格
                    aligned_price = self._align_price_to_precision(price)
                    aggregated_asks[aligned_price] += quantity
            except (ValueError, TypeError) as e:
                logger.debug(f"Skipping invalid ask data: {e}")
                continue

        logger.info(
            f"Aggregated order book: {len(raw_bids)} bids -> {len(aggregated_bids)} levels, "
            f"{len(raw_asks)} asks -> {len(aggregated_asks)} levels"
        )

        return dict(aggregated_bids), dict(aggregated_asks)

    def _align_price_to_precision(self, price: Decimal) -> Decimal:
        """将价格对齐到聚合精度。

        Args:
            price: 原始价格

        Returns:
            对齐后的价格
        """
        # 向下对齐到聚合精度
        aligned = (price // self.aggregation_precision) * self.aggregation_precision
        return aligned

    def _analyze_liquidity_features(
        self,
        aggregated_bids: dict[str, Decimal],
        aggregated_asks: dict[str, Decimal]
    ) -> dict[str, Any]:
        """分析流动性特征。

        Args:
            aggregated_bids: 聚合后的买单数据
            aggregated_asks: 聚合后的卖单数据

        Returns:
            流动性特征分析结果
        """
        # 买单流动性统计
        bid_volumes = list(aggregated_bids.values())
        bid_total_volume = sum(bid_volumes)
        bid_max_volume = max(bid_volumes) if bid_volumes else Decimal('0')
        bid_avg_volume = bid_total_volume / len(bid_volumes) if bid_volumes else Decimal('0')

        # 卖单流动性统计
        ask_volumes = list(aggregated_asks.values())
        ask_total_volume = sum(ask_volumes)
        ask_max_volume = max(ask_volumes) if ask_volumes else Decimal('0')
        ask_avg_volume = ask_total_volume / len(ask_volumes) if ask_volumes else Decimal('0')

        # 计算流动性集中度（最大量/平均量）
        bid_concentration = float(bid_max_volume / bid_avg_volume) if bid_avg_volume > 0 else 0
        ask_concentration = float(ask_max_volume / ask_avg_volume) if ask_avg_volume > 0 else 0

        # 计算总流动性
        total_liquidity = float(bid_total_volume + ask_total_volume)

        return {
            "bid_side": {
                "total_volume": float(bid_total_volume),
                "max_volume": float(bid_max_volume),
                "avg_volume": float(bid_avg_volume),
                "concentration": bid_concentration,
                "levels_count": len(aggregated_bids)
            },
            "ask_side": {
                "total_volume": float(ask_total_volume),
                "max_volume": float(ask_max_volume),
                "avg_volume": float(ask_avg_volume),
                "concentration": ask_concentration,
                "levels_count": len(aggregated_asks)
            },
            "total_liquidity": total_liquidity,
            "bid_ask_ratio": float(bid_total_volume / ask_total_volume) if ask_total_volume > 0 else float('inf')
        }

    def _identify_key_price_levels(
        self,
        aggregated_bids: dict[str, Decimal],
        aggregated_asks: dict[str, Decimal]
    ) -> dict[str, Any]:
        """识别关键价格水平。

        Args:
            aggregated_bids: 聚合后的买单数据
            aggregated_asks: 聚合后的卖单数据

        Returns:
            关键价格水平分析结果
        """
        # 找出最大的买单和卖单（支撑阻力位）
        best_bid = max(aggregated_bids.items(), key=lambda x: x[1]) if aggregated_bids else None
        best_ask = min(aggregated_asks.items(), key=lambda x: x[0]) if aggregated_asks else None

        # 找出最大的卖单（最强阻力）
        strongest_ask = max(aggregated_asks.items(), key=lambda x: x[1]) if aggregated_asks else None

        # 找出最大的买单（最强支撑）
        strongest_bid = max(aggregated_bids.items(), key=lambda x: x[1]) if aggregated_bids else None

        # 获取顶级流动性水平
        top_bids = sorted(
            aggregated_bids.items(),
            key=lambda x: x[1],
            reverse=True
        )[:TOP_LIQUIDITY_LEVELS]

        top_asks = sorted(
            aggregated_asks.items(),
            key=lambda x: x[1],
            reverse=True
        )[:TOP_LIQUIDITY_LEVELS]

        return {
            "best_bid": {
                "price": float(best_bid[0]) if best_bid else None,
                "volume": float(best_bid[1]) if best_bid else 0
            },
            "best_ask": {
                "price": float(best_ask[0]) if best_ask else None,
                "volume": float(best_ask[1]) if best_ask else 0
            },
            "strongest_support": {
                "price": float(strongest_bid[0]) if strongest_bid else None,
                "volume": float(strongest_bid[1]) if strongest_bid else 0
            },
            "strongest_resistance": {
                "price": float(strongest_ask[0]) if strongest_ask else None,
                "volume": float(strongest_ask[1]) if strongest_ask else 0
            },
            "top_liquidities": {
                "bid_levels": [
                    {"price": float(price), "volume": float(volume)}
                    for price, volume in top_bids
                ],
                "ask_levels": [
                    {"price": float(price), "volume": float(volume)}
                    for price, volume in top_asks
                ]
            }
        }

    def _calculate_order_book_imbalance(
        self,
        aggregated_bids: dict[str, Decimal],
        aggregated_asks: dict[str, Decimal]
    ) -> dict[str, Any]:
        """计算订单簿不平衡度指标。

        Args:
            aggregated_bids: 聚合后的买单数据
            aggregated_asks: 聚合后的卖单数据

        Returns:
            订单簿不平衡度指标
        """
        # 计算买卖总量
        bid_total = sum(aggregated_bids.values())
        ask_total = sum(aggregated_asks.values())
        total_volume = bid_total + ask_total

        if total_volume == 0:
            return {
                "bid_ask_ratio": 1.0,
                "bid_percentage": 0.5,
                "ask_percentage": 0.5,
                "imbalance_strength": 0.0,
                "direction": "neutral"
            }

        # 计算买卖比例
        bid_ask_ratio = float(bid_total / ask_total) if ask_total > 0 else float('inf')
        bid_percentage = float(bid_total / total_volume)
        ask_percentage = float(ask_total / total_volume)

        # 计算不平衡强度 (0-1, 1表示完全不平衡)
        imbalance_strength = abs(bid_percentage - 0.5) * 2

        # 确定方向
        if bid_percentage > 0.6:
            direction = "bullish"
        elif ask_percentage > 0.6:
            direction = "bearish"
        else:
            direction = "neutral"

        return {
            "bid_ask_ratio": bid_ask_ratio,
            "bid_percentage": bid_percentage,
            "ask_percentage": ask_percentage,
            "imbalance_strength": imbalance_strength,
            "direction": direction
        }

    def _create_empty_result(self, symbol: str) -> dict[str, Any]:
        """创建空的分析结果。

        Args:
            symbol: 交易符号

        Returns:
            空的分析结果字典
        """
        return {
            "symbol": symbol,
            "analysis_type": "static_order_book",
            "status": "no_data",
            "aggregated_bids": {},
            "aggregated_asks": {},
            "liquidity_analysis": {},
            "key_levels": {},
            "imbalance_metrics": {}
        }

    def _create_error_result(self, symbol: str, error_message: str) -> dict[str, Any]:
        """创建错误分析结果。

        Args:
            symbol: 交易符号
            error_message: 错误消息

        Returns:
            错误分析结果字典
        """
        return {
            "symbol": symbol,
            "analysis_type": "static_order_book",
            "status": "error",
            "error": error_message,
            "aggregated_bids": {},
            "aggregated_asks": {},
            "liquidity_analysis": {},
            "key_levels": {},
            "imbalance_metrics": {}
        }
