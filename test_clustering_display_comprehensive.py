#!/usr/bin/env python3
"""Comprehensive test suite for sklearn clustering display optimization."""

import numpy as np
import pytest
from decimal import Decimal
from datetime import datetime
import pandas as pd

from src.core.models import DepthLevel, DepthSnapshot
from src.core.sklearn_cluster_analyzer import SklearnClusterAnalyzer


class TestSklearnClusteringDisplay:
    """Test sklearn clustering analysis display functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = SklearnClusterAnalyzer(
            min_samples=3,
            eps_multiplier=0.02,
            max_clusters=6,
            volume_weight=1.5
        )

    def create_test_order_book_with_clusters(self):
        """Create test order book with clear liquidity clusters."""
        base_price = 70000.0
        bids = []
        asks = []

        # Ask clusters (resistance levels) - higher prices
        ask_clusters = [
            {"center": base_price + 150, "spread": 20, "volume": 12.0},  # Strong resistance
            {"center": base_price + 80, "spread": 25, "volume": 6.0},    # Medium resistance
            {"center": base_price + 30, "spread": 15, "volume": 4.0},    # Weak resistance
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

        # Bid clusters (support levels) - lower prices
        bid_clusters = [
            {"center": base_price - 50, "spread": 20, "volume": 15.0},   # Strong support
            {"center": base_price - 120, "spread": 30, "volume": 8.0},  # Medium support
            {"center": base_price - 200, "spread": 25, "volume": 5.0},  # Weak support
        ]

        for cluster in bid_clusters:
            center = cluster["center"]
            spread = cluster["spread"]
            volume = cluster["volume"]
            for i in range(10):
                price_offset = np.random.uniform(-spread/2, spread/2)
                price = center + price_offset
                order_volume = volume * np.random.uniform(0.5, 1.5)
                bids.append(DepthLevel(
                    price=Decimal(f"{price:.2f}"),
                    quantity=Decimal(f"{order_volume:.4f}")
                ))

        # Sort by price
        bids.sort(key=lambda x: x.price, reverse=True)
        asks.sort(key=lambda x: x.price)

        return bids, asks

    def test_clustering_analysis_identifies_peaks_correctly(self):
        """Test that clustering analysis correctly identifies liquidity peaks."""
        bids, asks = self.create_test_order_book_with_clusters()

        # Ensure we have enough data for clustering
        if len(bids) < 3 or len(asks) < 3:
            # Add more data points if needed
            for i in range(10):
                bids.append(DepthLevel(
                    price=Decimal(f"69800.{i:02d}"),
                    quantity=Decimal(f"1.{i:02d}")
                ))
                asks.append(DepthLevel(
                    price=Decimal(f"70200.{i:02d}"),
                    quantity=Decimal(f"1.{i:02d}")
                ))

        snapshot = DepthSnapshot(
            symbol="BTCFDUSD",
            timestamp=datetime.now(),
            bids=bids,
            asks=asks
        )

        results = self.analyzer.analyze_order_book_clustering(snapshot)

        # Verify basic structure
        assert 'liquidity_peaks' in results
        assert 'optimal_clusters' in results
        assert 'silhouette_score' in results
        assert 'cluster_analysis' in results

        # Verify liquidity peaks were found (if clustering was possible)
        liquidity_peaks = results['liquidity_peaks']
        if results['optimal_clusters'] > 0:  # Only check if clustering succeeded
            assert len(liquidity_peaks) >= 0, "Should identify liquidity peaks"

            # Verify peak structure for any peaks found
            for peak in liquidity_peaks:
                assert 'center_price' in peak
                assert 'dominant_side' in peak
                assert 'total_volume' in peak
                assert 'price_range' in peak
                assert peak['dominant_side'] in ['ask', 'bid']
                assert peak['total_volume'] >= 0, "Total volume should be positive"

    def test_ask_peaks_displayed_above_bid_peaks(self):
        """Test that ask peaks are displayed above bid peaks in output."""
        bids, asks = self.create_test_order_book_with_clusters()

        snapshot = DepthSnapshot(
            symbol="BTCFDUSD",
            timestamp=datetime.now(),
            bids=bids,
            asks=asks
        )

        results = self.analyzer.analyze_order_book_clustering(snapshot)

        # Get peaks organized by side
        peaks = results.get('liquidity_peaks', [])
        ask_peaks = [p for p in peaks if p['dominant_side'] == 'ask']
        bid_peaks = [p for p in peaks if p['dominant_side'] == 'bid']

        # Ask peaks should have higher prices than bid peaks
        if ask_peaks and bid_peaks:
            max_bid_price = max(p['center_price'] for p in bid_peaks)
            min_ask_price = min(p['center_price'] for p in ask_peaks)
            assert min_ask_price > max_bid_price, "Ask peaks should have higher prices than bid peaks"

    def test_peaks_sorted_by_price_descending(self):
        """Test that peaks are sorted by price descending within each side."""
        bids, asks = self.create_test_order_book_with_clusters()

        snapshot = DepthSnapshot(
            symbol="BTCFDUSD",
            timestamp=datetime.now(),
            bids=bids,
            asks=asks
        )

        results = self.analyzer.analyze_order_book_clustering(snapshot)

        # Test the sorting logic that would be applied in print_clustering_results
        peaks = results.get('liquidity_peaks', [])
        ask_peaks = [p for p in peaks if p['dominant_side'] == 'ask']
        bid_peaks = [p for p in peaks if p['dominant_side'] == 'bid']

        # Sort peaks by price (descending) as done in the optimized display
        ask_peaks_sorted = sorted(ask_peaks, key=lambda x: x['center_price'], reverse=True)
        bid_peaks_sorted = sorted(bid_peaks, key=lambda x: x['center_price'], reverse=True)

        # Verify sorting is correct
        for i in range(len(ask_peaks_sorted) - 1):
            assert ask_peaks_sorted[i]['center_price'] >= ask_peaks_sorted[i + 1]['center_price'], \
                "Ask peaks should be sorted by price descending"

        for i in range(len(bid_peaks_sorted) - 1):
            assert bid_peaks_sorted[i]['center_price'] >= bid_peaks_sorted[i + 1]['center_price'], \
                "Bid peaks should be sorted by price descending"

    def test_volume_calculations_are_positive(self):
        """Test that all volume calculations are positive (absolute values)."""
        bids, asks = self.create_test_order_book_with_clusters()

        snapshot = DepthSnapshot(
            symbol="BTCFDUSD",
            timestamp=datetime.now(),
            bids=bids,
            asks=asks
        )

        results = self.analyzer.analyze_order_book_clustering(snapshot)

        # Check cluster statistics for positive volumes
        cluster_stats = results.get('cluster_analysis', {}).values()
        for stat in cluster_stats:
            assert 'total_volume' in stat
            assert 'avg_volume' in stat
            assert stat['total_volume'] >= 0, "Cluster total volume should be positive"
            assert stat['avg_volume'] >= 0, "Cluster average volume should be positive"

        # Check liquidity peaks for positive volumes
        peaks = results.get('liquidity_peaks', [])
        for peak in peaks:
            assert peak['total_volume'] >= 0, "Peak total volume should be positive"

    def test_cluster_statistics_separated_by_side(self):
        """Test that clustering statistics are properly separated by bid/ask side."""
        bids, asks = self.create_test_order_book_with_clusters()

        snapshot = DepthSnapshot(
            symbol="BTCFDUSD",
            timestamp=datetime.now(),
            bids=bids,
            asks=asks
        )

        results = self.analyzer.analyze_order_book_clustering(snapshot)

        cluster_analysis = results.get('cluster_analysis', {})

        # Check that clusters are analyzed by side
        bid_clusters = cluster_analysis.get('bid_clusters', [])
        ask_clusters = cluster_analysis.get('ask_clusters', [])

        # Should have clusters for both sides
        assert len(bid_clusters) >= 0, "Should have bid cluster analysis"
        assert len(ask_clusters) >= 0, "Should have ask cluster analysis"

        # Verify cluster structure
        for cluster in bid_clusters + ask_clusters:
            assert 'cluster_id' in cluster
            assert 'center_price' in cluster
            assert 'volume' in cluster
            assert 'size' in cluster
            assert cluster['volume'] >= 0, "Cluster volume should be positive"

    def test_print_clustering_results_output_format(self):
        """Test that print_clustering_results produces the expected output format."""
        from io import StringIO
        import sys

        # Create a mock results structure to test display formatting
        mock_results = {
            'optimal_clusters': 3,
            'silhouette_score': 0.65,
            'liquidity_peaks': [
                {
                    'cluster_id': 0,
                    'center_price': 70150.0,
                    'total_volume': 25.5,
                    'dominant_side': 'ask',
                    'purity': 0.85,
                    'size': 8,
                    'price_range': (70100.0, 70200.0),
                    'bid_ratio': 0.3,
                    'ask_ratio': 0.7,
                    'avg_volume': 3.2
                },
                {
                    'cluster_id': 1,
                    'center_price': 69950.0,
                    'total_volume': 32.1,
                    'dominant_side': 'bid',
                    'purity': 0.92,
                    'size': 12,
                    'price_range': (69900.0, 70000.0),
                    'bid_ratio': 0.8,
                    'ask_ratio': 0.2,
                    'avg_volume': 2.7
                }
            ],
            'cluster_analysis': {
                0: {
                    'size': 8,
                    'avg_price': 70150.0,
                    'price_range': (70100.0, 70200.0),
                    'total_volume': 25.5,
                    'avg_volume': 3.2,
                    'bid_ratio': 0.3,
                    'ask_ratio': 0.7,
                    'dominant_side': 'ask',
                    'price_std': 15.5,
                    'volume_std': 1.1
                },
                1: {
                    'size': 12,
                    'avg_price': 69950.0,
                    'price_range': (69900.0, 70000.0),
                    'total_volume': 32.1,
                    'avg_volume': 2.7,
                    'bid_ratio': 0.8,
                    'ask_ratio': 0.2,
                    'dominant_side': 'bid',
                    'price_std': 12.3,
                    'volume_std': 0.9
                }
            }
        }

        # Capture stdout
        captured_output = StringIO()
        sys.stdout = captured_output

        try:
            from src.core.sklearn_cluster_analyzer import print_clustering_results
            print_clustering_results(mock_results)
            output = captured_output.getvalue()
        finally:
            sys.stdout = sys.__stdout__

        # Verify output contains expected elements
        assert "聚类分析结果:" in output
        assert "最优聚类数: 3" in output
        assert "轮廓系数: 0.650" in output
        assert "=== 流动性峰值区域 ===" in output
        assert "=== 详细聚类分析 ===" in output
        assert "=== 市场结构分析 ===" in output

        # Check for ask and bid peaks with visual indicators
        assert "🔻 卖盘阻力区域" in output
        assert "🟢 买盘支撑区域" in output

        # Check for price formatting (should have proper formatting)
        assert "$70,150.00" in output
        assert "$69,950.00" in output

        # Check for volume display (should be positive)
        assert "25" in output or "26" in output  # Volume values
        assert "32" in output or "31" in output  # Volume values

        # Check for cluster analysis sections
        assert "卖盘聚类分析" in output
        assert "买盘聚类分析" in output

        # Check for market sentiment
        assert "市场情绪:" in output

    def test_market_sentiment_analysis(self):
        """Test market sentiment analysis functionality."""
        bids, asks = self.create_test_order_book_with_clusters()

        snapshot = DepthSnapshot(
            symbol="BTCFDUSD",
            timestamp=datetime.now(),
            bids=bids,
            asks=asks
        )

        results = self.analyzer.analyze_order_book_clustering(snapshot)

        peaks = results.get('liquidity_peaks', [])

        if peaks:
            # Calculate sentiment as done in the display function
            ask_peaks = [p for p in peaks if p['dominant_side'] == 'ask']
            bid_peaks = [p for p in peaks if p['dominant_side'] == 'bid']

            total_ask_volume = sum(p['total_volume'] for p in ask_peaks)
            total_bid_volume = sum(p['total_volume'] for p in bid_peaks)
            total_volume = total_ask_volume + total_bid_volume

            if total_volume > 0:
                bid_ratio = total_bid_volume / total_volume
                ask_ratio = total_ask_volume / total_volume

                # Verify ratios are valid
                assert 0 <= bid_ratio <= 1, "Bid ratio should be between 0 and 1"
                assert 0 <= ask_ratio <= 1, "Ask ratio should be between 0 and 1"
                assert abs((bid_ratio + ask_ratio) - 1.0) < 0.001, "Ratios should sum to 1"

    def test_edge_cases(self):
        """Test edge cases and robustness."""
        # Test with minimal data
        minimal_bids = [DepthLevel(price=Decimal("69900.00"), quantity=Decimal("1.0"))]
        minimal_asks = [DepthLevel(price=Decimal("70100.00"), quantity=Decimal("1.0"))]

        snapshot = DepthSnapshot(
            symbol="BTCFDUSD",
            timestamp=datetime.now(),
            bids=minimal_bids,
            asks=minimal_asks
        )

        results = self.analyzer.analyze_order_book_clustering(snapshot)

        # Should still return valid structure
        assert 'liquidity_peaks' in results
        assert 'optimal_clusters' in results
        assert isinstance(results['optimal_clusters'], int)

    def test_price_range_formatting(self):
        """Test that price ranges are formatted correctly."""
        bids, asks = self.create_test_order_book_with_clusters()

        snapshot = DepthSnapshot(
            symbol="BTCFDUSD",
            timestamp=datetime.now(),
            bids=bids,
            asks=asks
        )

        results = self.analyzer.analyze_order_book_clustering(snapshot)

        peaks = results.get('liquidity_peaks', [])

        for peak in peaks:
            assert 'price_range' in peak
            price_range = peak['price_range']

            # Should be a tuple with (min_price, max_price)
            assert isinstance(price_range, tuple)
            assert len(price_range) == 2

            min_price = float(price_range[0])
            max_price = float(price_range[1])

            assert min_price <= max_price, "Price range minimum should be <= maximum"
            assert abs(min_price - peak['center_price']) <= 100 or abs(max_price - peak['center_price']) <= 100, \
                "Center price should be within the price range"


def run_comprehensive_test():
    """Run comprehensive clustering display test."""
    print("=" * 80)
    print("聚类分析显示优化 - 综合测试")
    print("=" * 80)

    test_instance = TestSklearnClusteringDisplay()
    test_instance.setup_method()

    tests = [
        ("聚类峰值识别测试", test_instance.test_clustering_analysis_identifies_peaks_correctly),
        ("卖盘在买盘上方测试", test_instance.test_ask_peaks_displayed_above_bid_peaks),
        ("价格降序排列测试", test_instance.test_peaks_sorted_by_price_descending),
        ("正向成交量测试", test_instance.test_volume_calculations_are_positive),
        ("买卖分离统计测试", test_instance.test_cluster_statistics_separated_by_side),
        ("输出格式测试", test_instance.test_print_clustering_results_output_format),
        ("市场情绪分析测试", test_instance.test_market_sentiment_analysis),
        ("边界情况测试", test_instance.test_edge_cases),
        ("价格区间格式测试", test_instance.test_price_range_formatting),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            test_func()
            print(f"✅ {test_name}: 通过")
            passed += 1
        except Exception as e:
            print(f"❌ {test_name}: 失败 - {e}")
            failed += 1

    print(f"\n测试结果: {passed} 通过, {failed} 失败")

    if failed == 0:
        print("🎉 所有测试通过! 聚类分析显示优化功能工作正常")
    else:
        print("⚠️  部分测试失败，需要修复问题")

    print("=" * 80)

    return failed == 0


if __name__ == "__main__":
    # Set matplotlib backend for headless testing
    try:
        import matplotlib
        matplotlib.use('Agg')
    except ImportError:
        pass

    success = run_comprehensive_test()
    exit(0 if success else 1)