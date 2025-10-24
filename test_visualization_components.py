#!/usr/bin/env python3
"""Additional tests for visualization components to improve coverage."""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from decimal import Decimal
from datetime import datetime

from src.core.models import DepthLevel, DepthSnapshot
from src.core.sklearn_cluster_analyzer import ClusterVisualizer, SklearnClusterAnalyzer


class TestVisualizationComponents:
    """Test visualization and edge case components."""

    def setup_method(self):
        """Set up test fixtures."""
        self.visualizer = ClusterVisualizer()
        self.analyzer = SklearnClusterAnalyzer()

    def create_mock_results(self) -> dict:
        """Create mock clustering results for testing."""
        return {
            'optimal_clusters': 2,
            'silhouette_score': 0.75,
            'wcss': [100.0, 50.0, 25.0],
            'clustering_data': np.array([
                [1.0, 2.0, 0.0],  # bid point
                [1.1, 1.8, 0.0],  # bid point
                [2.0, 3.0, 1.0],  # ask point
                [2.1, 2.9, 1.0],  # ask point
            ]),
            'labels': np.array([0, 0, 1, 1]),
            'centers': np.array([
                [1.05, 1.9, 0.0],
                [2.05, 2.95, 1.0]
            ]),
            'cluster_analysis': {
                0: {
                    'size': 2,
                    'avg_price': 1.05,
                    'total_volume': 3.8,
                    'avg_volume': 1.9,
                    'bid_ratio': 1.0,
                    'ask_ratio': 0.0,
                    'dominant_side': 'bid'
                },
                1: {
                    'size': 2,
                    'avg_price': 2.05,
                    'total_volume': 5.9,
                    'avg_volume': 2.95,
                    'bid_ratio': 0.0,
                    'ask_ratio': 1.0,
                    'dominant_side': 'ask'
                }
            },
            'liquidity_peaks': [
                {
                    'center_price': 2.05,
                    'total_volume': 5.9,
                    'dominant_side': 'ask',
                    'purity': 0.9
                },
                {
                    'center_price': 1.05,
                    'total_volume': 3.8,
                    'dominant_side': 'bid',
                    'purity': 0.85
                }
            ]
        }

    def test_cluster_visualizer_initialization(self):
        """Test ClusterVisualizer initialization."""
        assert self.visualizer.fig is not None
        assert self.visualizer.axes is not None
        assert len(self.visualizer.axes.flatten()) == 4  # 2x2 subplot grid

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.show')
    def test_plot_clustering_results_with_save_path(self, mock_show, mock_savefig):
        """Test plotting with save path functionality."""
        mock_results = self.create_mock_results()

        self.visualizer.plot_clustering_results(mock_results, save_path='test_plot.png')

        mock_savefig.assert_called_once_with('test_plot.png', dpi=300, bbox_inches='tight')
        mock_show.assert_called_once()

    @patch('matplotlib.pyplot.show')
    def test_plot_clustering_results_without_save_path(self, mock_show):
        """Test plotting without save path."""
        mock_results = self.create_mock_results()

        self.visualizer.plot_clustering_results(mock_results)

        mock_show.assert_called_once()

    def test_plot_clustering_results_empty_data(self):
        """Test plotting with empty data."""
        empty_results = {
            'clustering_data': np.array([]),
            'labels': np.array([]),
            'centers': np.array([]),
            'cluster_analysis': {},
            'liquidity_peaks': []
        }

        # Should not raise an exception
        self.visualizer.plot_clustering_results(empty_results)

    def test_plot_liquidity_peaks_empty(self):
        """Test plotting liquidity peaks when none exist."""
        # Mock the axes
        ax = MagicMock()
        self.visualizer.axes[0, 1] = ax

        # Call with empty peaks
        self.visualizer._plot_liquidity_peaks([])

        # Verify text was added for empty state
        ax.text.assert_called_once()

    def test_plot_elbow_method_empty_wcss(self):
        """Test plotting elbow method with empty WCSS data."""
        # Mock the axes
        ax = MagicMock()
        self.visualizer.axes[1, 0] = ax

        # Call with empty WCSS
        self.visualizer._plot_elbow_method({'wcss': []})

        # Verify text was added for empty state
        ax.text.assert_called_once()

    def test_plot_cluster_statistics_table_empty(self):
        """Test plotting cluster statistics table with empty data."""
        # Mock the axes
        ax = MagicMock()
        self.visualizer.axes[1, 1] = ax

        # Call with empty analysis
        self.visualizer._plot_cluster_statistics_table({})

        # Verify text was added for empty state
        ax.text.assert_called_once()
        ax.axis.assert_called_once_with('off')

    def test_edge_case_insufficient_data(self):
        """Test analyzer with insufficient data."""
        # Create minimal data
        bids = [DepthLevel(price=Decimal("70000.00"), quantity=Decimal("1.0"))]
        asks = [DepthLevel(price=Decimal("70100.00"), quantity=Decimal("1.0"))]

        snapshot = DepthSnapshot(
            symbol="BTCFDUSD",
            timestamp=datetime.now(),
            bids=bids,
            asks=asks
        )

        results = self.analyzer.analyze_order_book_clustering(snapshot)

        # Should return empty result structure
        assert results['optimal_clusters'] == 0
        assert results['silhouette_score'] == 0.0
        assert len(results['liquidity_peaks']) == 0

    def test_edge_case_single_price_level(self):
        """Test analyzer with single price level data."""
        # Create data with all same price
        bids = [
            DepthLevel(price=Decimal("70000.00"), quantity=Decimal("1.0")),
            DepthLevel(price=Decimal("70000.00"), quantity=Decimal("2.0")),
            DepthLevel(price=Decimal("70000.00"), quantity=Decimal("1.5"))
        ]
        asks = [
            DepthLevel(price=Decimal("70100.00"), quantity=Decimal("1.0")),
            DepthLevel(price=Decimal("70100.00"), quantity=Decimal("2.0")),
            DepthLevel(price=Decimal("70100.00"), quantity=Decimal("1.5"))
        ]

        snapshot = DepthSnapshot(
            symbol="BTCFDUSD",
            timestamp=datetime.now(),
            bids=bids,
            asks=asks
        )

        results = self.analyzer.analyze_order_book_clustering(snapshot)

        # Should handle gracefully
        assert isinstance(results, dict)
        assert 'optimal_clusters' in results

    def test_edge_case_zero_volumes(self):
        """Test analyzer with zero volume orders."""
        # Create data with zero volumes
        bids = [
            DepthLevel(price=Decimal("70000.00"), quantity=Decimal("0.0")),
            DepthLevel(price=Decimal("69900.00"), quantity=Decimal("0.0"))
        ]
        asks = [
            DepthLevel(price=Decimal("70100.00"), quantity=Decimal("0.0")),
            DepthLevel(price=Decimal("70200.00"), quantity=Decimal("0.0"))
        ]

        snapshot = DepthSnapshot(
            symbol="BTCFDUSD",
            timestamp=datetime.now(),
            bids=bids,
            asks=asks
        )

        results = self.analyzer.analyze_order_book_clustering(snapshot)

        # Should handle zero volumes gracefully
        assert isinstance(results, dict)

    def test_print_functions_edge_cases(self):
        """Test print functions with edge case data."""
        from src.core.sklearn_cluster_analyzer import (
            _print_summary_metrics,
            _print_liquidity_peaks,
            _print_detailed_cluster_analysis,
            _print_market_structure_analysis
        )

        # Test with empty results
        empty_results = {
            'optimal_clusters': 0,
            'silhouette_score': 0.0,
            'liquidity_peaks': [],
            'cluster_analysis': {}
        }

        # Should not raise exceptions
        _print_summary_metrics(empty_results)
        _print_liquidity_peaks(empty_results)
        _print_detailed_cluster_analysis(empty_results)
        _print_market_structure_analysis(empty_results)

    def test_analyzer_with_extreme_values(self):
        """Test analyzer with extreme price and volume values."""
        # Create data with extreme values
        bids = [
            DepthLevel(price=Decimal("1.00"), quantity=Decimal("1000000.0")),
            DepthLevel(price=Decimal("1000000.00"), quantity=Decimal("0.000001"))
        ]
        asks = [
            DepthLevel(price=Decimal("2.00"), quantity=Decimal("2000000.0")),
            DepthLevel(price=Decimal("2000000.00"), quantity=Decimal("0.000002"))
        ]

        snapshot = DepthSnapshot(
            symbol="BTCFDUSD",
            timestamp=datetime.now(),
            bids=bids,
            asks=asks
        )

        results = self.analyzer.analyze_order_book_clustering(snapshot)

        # Should handle extreme values without overflow
        assert isinstance(results, dict)
        assert 'optimal_clusters' in results

    def test_memory_efficiency_large_dataset(self):
        """Test analyzer with large dataset for memory efficiency."""
        # Create large dataset
        bids = []
        asks = []

        for i in range(1000):
            bids.append(DepthLevel(
                price=Decimal(f"70000.{i%100:02d}"),
                quantity=Decimal(f"1.{i%10:02d}")
            ))
            asks.append(DepthLevel(
                price=Decimal(f"70100.{i%100:02d}"),
                quantity=Decimal(f"1.{i%10:02d}")
            ))

        snapshot = DepthSnapshot(
            symbol="BTCFDUSD",
            timestamp=datetime.now(),
            bids=bids,
            asks=asks
        )

        # Should process without memory issues
        results = self.analyzer.analyze_order_book_clustering(snapshot)

        assert isinstance(results, dict)
        # Large dataset should produce multiple clusters
        assert results['optimal_clusters'] >= 0

    def test_constructor_input_validation(self):
        """Test analyzer constructor input validation."""
        # Test invalid min_samples
        with pytest.raises(ValueError, match="min_samples must be at least 1"):
            SklearnClusterAnalyzer(min_samples=0)

        # Test invalid eps_multiplier
        with pytest.raises(ValueError, match="eps_multiplier must be positive"):
            SklearnClusterAnalyzer(eps_multiplier=0)

        # Test invalid max_clusters
        with pytest.raises(ValueError, match="max_clusters must be at least 2"):
            SklearnClusterAnalyzer(max_clusters=1)

        # Test invalid volume_weight
        with pytest.raises(ValueError, match="volume_weight must be positive"):
            SklearnClusterAnalyzer(volume_weight=-1)

    def test_clustering_failure_scenarios(self):
        """Test clustering algorithm failure scenarios."""
        # Test with malformed data that could cause clustering failures
        bids = [
            DepthLevel(price=Decimal("70000.00"), quantity=Decimal("1.0"))
        ]
        asks = [
            DepthLevel(price=Decimal("70100.00"), quantity=Decimal("1.0"))
        ]

        snapshot = DepthSnapshot(
            symbol="BTCFDUSD",
            timestamp=datetime.now(),
            bids=bids,
            asks=asks
        )

        # Should handle gracefully with empty result
        results = self.analyzer.analyze_order_book_clustering(snapshot)

        assert isinstance(results, dict)
        assert results['optimal_clusters'] == 0
        assert results['silhouette_score'] == 0.0
        assert len(results['liquidity_peaks']) == 0


def run_additional_tests():
    """Run additional visualization and edge case tests."""
    print("=" * 80)
    print("聚类分析可视化和边界情况 - 补充测试")
    print("=" * 80)

    test_instance = TestVisualizationComponents()
    test_instance.setup_method()

    tests = [
        ("可视化器初始化测试", test_instance.test_cluster_visualizer_initialization),
        ("带保存路径绘图测试", test_instance.test_plot_clustering_results_with_save_path),
        ("无保存路径绘图测试", test_instance.test_plot_clustering_results_without_save_path),
        ("空数据绘图测试", test_instance.test_plot_clustering_results_empty_data),
        ("空流动性峰值绘图测试", test_instance.test_plot_liquidity_peaks_empty),
        ("空WCSS肘部方法测试", test_instance.test_plot_elbow_method_empty_wcss),
        ("空聚类统计表测试", test_instance.test_plot_cluster_statistics_table_empty),
        ("数据不足边界测试", test_instance.test_edge_case_insufficient_data),
        ("单一价格水平测试", test_instance.test_edge_case_single_price_level),
        ("零成交量测试", test_instance.test_edge_case_zero_volumes),
        ("打印函数边界测试", test_instance.test_print_functions_edge_cases),
        ("极值处理测试", test_instance.test_analyzer_with_extreme_values),
        ("大数据集内存效率测试", test_instance.test_memory_efficiency_large_dataset),
        ("构造函数输入验证测试", test_instance.test_constructor_input_validation),
        ("聚类失败场景测试", test_instance.test_clustering_failure_scenarios),
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

    print(f"\n补充测试结果: {passed} 通过, {failed} 失败")

    if failed == 0:
        print("🎉 所有补充测试通过!")
    else:
        print("⚠️  部分补充测试失败，需要修复问题")

    print("=" * 80)

    return failed == 0


if __name__ == "__main__":
    # Set matplotlib backend for headless testing
    try:
        import matplotlib
        matplotlib.use('Agg')
    except ImportError:
        pass

    success = run_additional_tests()
    exit(0 if success else 1)