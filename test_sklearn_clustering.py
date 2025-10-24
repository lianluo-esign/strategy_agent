#!/usr/bin/env python3
"""Test script for sklearn clustering analysis."""

import matplotlib.pyplot as plt
import numpy as np
from decimal import Decimal

from src.core.models import DepthLevel, DepthSnapshot
from src.core.sklearn_cluster_analyzer import (
    SklearnClusterAnalyzer,
    ClusterVisualizer,
    print_clustering_results
)


def create_realistic_btc_order_book():
    """Create realistic BTC-FDUSD order book data for testing."""
    # Simulate BTC price around $70,000 with realistic spreads
    base_price = 70000.0

    # Create bid levels (descending prices)
    bids = []
    for i in range(20):  # 20 bid levels
        price = base_price - (i * 10) - np.random.uniform(-5, 5)
        # Higher volume closer to the current price
        volume = np.random.uniform(0.5, 5.0) * np.exp(-i * 0.1)
        # Add some clustering effect
        if i in [2, 3, 4]:  # Create a liquidity cluster
            volume *= 3.0
        if i in [8, 9, 10]:  # Another liquidity cluster
            volume *= 2.5

        bids.append(DepthLevel(price=Decimal(f"{price:.2f}"),
                              quantity=Decimal(f"{volume:.4f}")))

    # Create ask levels (ascending prices)
    asks = []
    for i in range(20):  # 20 ask levels
        price = base_price + 10 + (i * 10) + np.random.uniform(-5, 5)
        # Higher volume closer to the current price
        volume = np.random.uniform(0.5, 5.0) * np.exp(-i * 0.1)
        # Add clustering effect
        if i in [1, 2, 3]:  # Create a liquidity cluster
            volume *= 3.5
        if i in [7, 8, 9]:  # Another liquidity cluster
            volume *= 2.0

        asks.append(DepthLevel(price=Decimal(f"{price:.2f}"),
                              quantity=Decimal(f"{volume:.4f}")))

    return bids, asks


def main():
    """Main function to test sklearn clustering analysis."""
    print("=" * 60)
    print("SKLEARN聚类分析测试")
    print("=" * 60)

    # Create realistic order book data
    print("生成模拟BTC-FDUSD订单簿数据...")
    bids, asks = create_realistic_btc_order_book()

    # Create depth snapshot
    snapshot = DepthSnapshot(
        symbol="BTCFDUSD",
        timestamp=pd.Timestamp.now(),
        bids=bids,
        asks=asks
    )

    print(f"订单簿数据: {len(bids)}个买盘, {len(asks)}个卖盘")
    print(f"价格范围: ${float(bids[-1].price):.2f} - ${float(asks[-1].price):.2f}")

    # Initialize cluster analyzer
    print("\n初始化聚类分析器...")
    analyzer = SklearnClusterAnalyzer(
        min_samples=3,
        eps_multiplier=0.02,
        max_clusters=8,
        volume_weight=2.0
    )

    # Perform clustering analysis
    print("执行聚类分析...")
    results = analyzer.analyze_order_book_clustering(snapshot)

    # Print results in the specified format
    print_clustering_results(results)

    # Additional analysis insights
    print(f"\n分析洞察:")
    print(f"- 总共识别出 {len(results['liquidity_peaks'])} 个流动性峰值")
    print(f"- 聚类质量评估: 轮廓系数 {results['silhouette_score']:.3f}")

    if results['silhouette_score'] > 0.5:
        print("- 聚类质量: 优秀")
    elif results['silhouette_score'] > 0.25:
        print("- 聚类质量: 良好")
    else:
        print("- 聚类质量: 需要改进")

    # Print dominant sides analysis
    bid_dominant = sum(1 for peak in results['liquidity_peaks']
                      if peak['dominant_side'] == 'bid')
    ask_dominant = sum(1 for peak in results['liquidity_peaks']
                      if peak['dominant_side'] == 'ask')

    print(f"- 买盘主导峰值: {bid_dominant}个")
    print(f"- 卖盘主导峰值: {ask_dominant}个")

    if bid_dominant > ask_dominant:
        print("- 市场情绪: 偏向买盘支撑")
    elif ask_dominant > bid_dominant:
        print("- 市场情绪: 偏向卖盘阻力")
    else:
        print("- 市场情绪: 相对平衡")

    # Create visualization
    print("\n生成可视化图表...")
    visualizer = ClusterVisualizer()
    visualizer.plot_clustering_results(results)

    # Save the plot
    try:
        plt.savefig('clustering_analysis_results.png', dpi=300, bbox_inches='tight')
        print("图表已保存为 clustering_analysis_results.png")
    except Exception as e:
        print(f"保存图表时出错: {e}")

    print("\n聚类分析完成！")
    print("=" * 60)


if __name__ == "__main__":
    import pandas as pd

    # Set matplotlib to use a backend that works in headless mode if needed
    import matplotlib
    matplotlib.use('Agg')  # Uncomment if running in headless environment

    main()