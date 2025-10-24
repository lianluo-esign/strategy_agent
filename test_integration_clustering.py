#!/usr/bin/env python3
"""Integration test for sklearn clustering in the actual analyzer system."""

import matplotlib.pyplot as plt
import numpy as np
from decimal import Decimal

from src.core.models import DepthLevel, DepthSnapshot
from src.core.sklearn_cluster_analyzer import SklearnClusterAnalyzer, print_clustering_results


def create_test_order_book():
    """Create realistic order book data for testing."""
    base_price = 70000.0

    # Create concentrated liquidity clusters
    bids = []
    asks = []

    # Bid clusters (support levels)
    bid_clusters = [
        {"center": base_price - 50, "spread": 20, "volume": 15.0},   # Strong support
        {"center": base_price - 120, "spread": 30, "volume": 8.0},  # Medium support
        {"center": base_price - 200, "spread": 25, "volume": 5.0},  # Weak support
    ]

    for cluster in bid_clusters:
        center = cluster["center"]
        spread = cluster["spread"]
        volume = cluster["volume"]

        # Create multiple orders around the cluster center
        for i in range(10):
            price_offset = np.random.uniform(-spread/2, spread/2)
            price = center + price_offset
            order_volume = volume * np.random.uniform(0.5, 1.5)

            bids.append(DepthLevel(
                price=Decimal(f"{price:.2f}"),
                quantity=Decimal(f"{order_volume:.4f}")
            ))

    # Ask clusters (resistance levels)
    ask_clusters = [
        {"center": base_price + 30, "spread": 15, "volume": 12.0},   # Strong resistance
        {"center": base_price + 80, "spread": 25, "volume": 6.0},   # Medium resistance
        {"center": base_price + 150, "spread": 20, "volume": 4.0},   # Weak resistance
    ]

    for cluster in ask_clusters:
        center = cluster["center"]
        spread = cluster["spread"]
        volume = cluster["volume"]

        for i in range(8):
            price_offset = np.random.uniform(-spread/2, spread/2)
            price = center + price_offset
            order_volume = volume * np.random.uniform(0.5, 1.5)

            asks.append(DepthLevel(
                price=Decimal(f"{price:.2f}"),
                quantity=Decimal(f"{order_volume:.4f}")
            ))

    # Sort by price
    bids.sort(key=lambda x: x.price, reverse=True)
    asks.sort(key=lambda x: x.price)

    return bids, asks


def main():
    """Main integration test."""
    print("=" * 60)
    print("SKLEARN聚类分析集成测试")
    print("=" * 60)

    # Create test data with clear liquidity clusters
    print("创建包含明显流动性集群的测试订单簿...")
    bids, asks = create_test_order_book()

    # Create depth snapshot
    snapshot = DepthSnapshot(
        symbol="BTCFDUSD",
        timestamp=pd.Timestamp.now(),
        bids=bids,
        asks=asks
    )

    print(f"订单簿数据: {len(bids)}个买盘, {len(asks)}个卖盘")
    print(f"价格范围: ${float(bids[-1].price):.2f} - ${float(asks[-1].price):.2f}")

    # Initialize and run clustering analysis
    print("\n初始化Sklearn聚类分析器...")
    analyzer = SklearnClusterAnalyzer(
        min_samples=3,
        eps_multiplier=0.015,  # Smaller for better cluster detection
        max_clusters=6,
        volume_weight=1.5
    )

    print("执行聚类分析...")
    results = analyzer.analyze_order_book_clustering(snapshot)

    # Display results in specified format
    print_clustering_results(results)

    # Additional insights
    print(f"\n=== 聚类分析洞察 ===")
    liquidity_peaks = results['liquidity_peaks']

    if liquidity_peaks:
        print(f"识别出 {len(liquidity_peaks)} 个流动性峰值")

        # Analyze dominant sides
        bid_peaks = [p for p in liquidity_peaks if p['dominant_side'] == 'bid']
        ask_peaks = [p for p in liquidity_peaks if p['dominant_side'] == 'ask']

        print(f"买盘峰值: {len(bid_peaks)}个")
        print(f"卖盘峰值: {len(ask_peaks)}个")

        if len(bid_peaks) > len(ask_peaks):
            print("市场结构: 买盘支撑较强，可能为上涨趋势")
        elif len(ask_peaks) > len(bid_peaks):
            print("市场结构: 卖盘阻力较强，可能为下跌趋势")
        else:
            print("市场结构: 买卖力量相对平衡")

        # Find strongest peaks
        strongest_peak = max(liquidity_peaks, key=lambda x: x['total_volume'])
        print(f"最强峰值: ${strongest_peak['center_price']:.2f}, "
              f"总量: {strongest_peak['total_volume']:.0f}, "
              f"方向: {strongest_peak['dominant_side']}")

    # Clustering quality assessment
    silhouette = results['silhouette_score']
    if silhouette > 0.7:
        print("聚类质量: 优秀 (轮廓系数 > 0.7)")
    elif silhouette > 0.5:
        print("聚类质量: 良好 (轮廓系数 > 0.5)")
    elif silhouette > 0.25:
        print("聚类质量: 一般 (轮廓系数 > 0.25)")
    else:
        print("聚类质量: 较差 (轮廓系数 ≤ 0.25)")

    print(f"轮廓系数: {silhouette:.3f}")
    print(f"最优聚类数: {results['optimal_clusters']}")

    # Create visualization
    print("\n生成可视化图表...")
    try:
        from src.core.sklearn_cluster_analyzer import ClusterVisualizer
        visualizer = ClusterVisualizer()
        visualizer.plot_clustering_results(results, save_path='sklearn_clustering_integration_test.png')
        print("图表已保存为 sklearn_clustering_integration_test.png")
    except Exception as e:
        print(f"可视化生成失败: {e}")

    print("\n✅ Sklearn聚类分析集成测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    import pandas as pd

    # Set matplotlib backend
    try:
        import matplotlib
        matplotlib.use('Agg')
    except ImportError:
        pass

    main()