"""Advanced clustering analysis using sklearn for order book liquidity detection.

This module provides sophisticated clustering algorithms to identify liquidity peaks
and support/resistance levels in order book data using machine learning approaches.
"""

import logging
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from decimal import Decimal
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from datetime import datetime

from .models import DepthLevel, DepthSnapshot

logger = logging.getLogger(__name__)


class SklearnClusterAnalyzer:
    """Advanced order book clustering analyzer using sklearn algorithms."""

    def __init__(
        self,
        min_samples: int = 3,
        eps_multiplier: float = 0.01,
        max_clusters: int = 10,
        volume_weight: float = 2.0,
    ):
        """
        Initialize the sklearn cluster analyzer.

        Args:
            min_samples: Minimum samples for DBSCAN clustering
            eps_multiplier: Multiplier for DBSCAN epsilon calculation
            max_clusters: Maximum number of clusters to test with K-means
            volume_weight: Weight factor for volume in clustering features
        """
        self.min_samples = min_samples
        self.eps_multiplier = eps_multiplier
        self.max_clusters = max_clusters
        self.volume_weight = volume_weight

        logger.info(
            f"Initialized SklearnClusterAnalyzer with min_samples={min_samples}, "
            f"eps_multiplier={eps_multiplier}, max_clusters={max_clusters}"
        )

    def analyze_order_book_clustering(
        self, snapshot: DepthSnapshot
    ) -> Dict[str, Any]:
        """
        Perform comprehensive clustering analysis on order book data.

        Args:
            snapshot: Depth snapshot containing bids and asks

        Returns:
            Dictionary containing clustering analysis results
        """
        logger.info(f"Starting clustering analysis for {snapshot.symbol}")

        # Step 1: Prepare data for clustering
        clustering_data, price_volume_map, scaler = self._prepare_clustering_data(snapshot)

        if len(clustering_data) < self.min_samples:
            logger.warning(f"Insufficient data points for clustering: {len(clustering_data)}")
            return self._create_empty_result()

        # Step 2: Find optimal number of clusters using K-means and elbow method
        optimal_k, wcss_history = self._find_optimal_clusters_kmeans(clustering_data)

        # Step 3: Perform final clustering using DBSCAN with adaptive parameters
        labels, centers, cluster_stats = self._perform_dbscan_clustering(
            clustering_data, optimal_k, scaler
        )

        # Step 4: Calculate silhouette score
        silhouette_avg = self._calculate_silhouette_score(clustering_data, labels)

        # Step 5: Identify liquidity peaks
        liquidity_peaks = self._identify_liquidity_peaks(
            clustering_data, labels, centers, cluster_stats
        )

        # Step 6: Generate detailed cluster analysis
        cluster_analysis = self._analyze_clusters_detailed(
            clustering_data, labels, cluster_stats
        )

        results = {
            "optimal_clusters": optimal_k,
            "silhouette_score": silhouette_avg,
            "wcss": wcss_history,
            "clustering_data": clustering_data,
            "labels": labels,
            "centers": centers,
            "cluster_analysis": cluster_analysis,
            "liquidity_peaks": liquidity_peaks,
            "price_volume_map": price_volume_map,
            "scaler": scaler,
        }

        logger.info(
            f"Clustering analysis completed: {optimal_k} clusters, "
            f"silhouette_score={silhouette_avg:.3f}, "
            f"{len(liquidity_peaks)} liquidity peaks identified"
        )

        return results

    def _prepare_clustering_data(
        self, snapshot: DepthSnapshot
    ) -> Tuple[np.ndarray, Dict[Tuple[float, int], Decimal], StandardScaler]:
        """Prepare order book data for clustering analysis."""
        data_points = []
        price_volume_map = {}

        # Process bids
        for i, bid in enumerate(snapshot.bids):
            # Feature engineering: [price, volume_weighted, side_indicator]
            price = float(bid.price)
            volume = float(bid.quantity)
            volume_weighted = volume * self.volume_weight

            data_points.append([price, volume_weighted, 0])  # 0 for bid side
            price_volume_map[(price, 0)] = bid.quantity

        # Process asks
        for i, ask in enumerate(snapshot.asks):
            price = float(ask.price)
            volume = float(ask.quantity)
            volume_weighted = volume * self.volume_weight

            data_points.append([price, volume_weighted, 1])  # 1 for ask side
            price_volume_map[(price, 1)] = ask.quantity

        clustering_data = np.array(data_points)

        # Standardize features for better clustering
        scaler = StandardScaler()
        clustering_data[:, :2] = scaler.fit_transform(clustering_data[:, :2])

        logger.debug(f"Prepared {len(clustering_data)} data points for clustering")

        return clustering_data, price_volume_map, scaler

    def _find_optimal_clusters_kmeans(
        self, data: np.ndarray
    ) -> Tuple[int, List[float]]:
        """Find optimal number of clusters using elbow method."""
        if len(data) < 4:
            return min(2, len(data) // 2), []

        wcss = []
        k_range = range(2, min(self.max_clusters + 1, len(data) // 2 + 1))

        for k in k_range:
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans.fit(data)
                wcss.append(kmeans.inertia_)
            except Exception as e:
                logger.warning(f"K-means failed for k={k}: {e}")
                break

        if not wcss:
            return 2, []

        # Find elbow point using second derivative method
        optimal_k = self._find_elbow_point(list(k_range), wcss)

        logger.debug(f"Optimal K found: {optimal_k}, WCSS history: {wcss[:3]}...")

        return optimal_k, wcss

    def _find_elbow_point(self, k_range: List[int], wcss: List[float]) -> int:
        """Find the elbow point in WCSS curve."""
        if len(wcss) < 3:
            return k_range[0]

        # Calculate second derivatives
        second_derivatives = []
        for i in range(1, len(wcss) - 1):
            d2 = wcss[i + 1] - 2 * wcss[i] + wcss[i - 1]
            second_derivatives.append(abs(d2))

        if not second_derivatives:
            return k_range[0]

        # Find maximum second derivative (elbow point)
        elbow_idx = np.argmax(second_derivatives) + 1
        return k_range[min(elbow_idx, len(k_range) - 1)]

    def _perform_dbscan_clustering(
        self, data: np.ndarray, optimal_k: int, scaler: StandardScaler
    ) -> Tuple[np.ndarray, np.ndarray, Dict[int, Dict[str, Any]]]:
        """Perform DBSCAN clustering with adaptive parameters."""
        # Calculate adaptive epsilon based on data density and optimal K
        data_range = np.max(data[:, 0]) - np.min(data[:, 0])
        eps = data_range / (optimal_k * 10) * self.eps_multiplier

        # Ensure minimum epsilon
        eps = max(eps, 0.1)

        logger.debug(f"Using DBSCAN with eps={eps:.4f}, min_samples={self.min_samples}")

        try:
            dbscan = DBSCAN(eps=eps, min_samples=self.min_samples)
            labels = dbscan.fit_predict(data)

            # If DBSCAN finds too few clusters, fall back to K-means
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            if n_clusters < 2:
                logger.warning("DBSCAN found insufficient clusters, falling back to K-means")
                kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(data)
                centers = kmeans.cluster_centers_
            else:
                # Calculate cluster centers for DBSCAN
                centers = self._calculate_dbscan_centers(data, labels)

        except Exception as e:
            logger.error(f"DBSCAN clustering failed: {e}, falling back to K-means")
            kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(data)
            centers = kmeans.cluster_centers_

        # Generate cluster statistics
        cluster_stats = self._generate_cluster_statistics(data, labels, scaler)

        return labels, centers, cluster_stats

    def _calculate_dbscan_centers(
        self, data: np.ndarray, labels: np.ndarray
    ) -> np.ndarray:
        """Calculate cluster centers for DBSCAN results."""
        unique_labels = set(labels)
        centers = []

        for label in unique_labels:
            if label == -1:  # Skip noise points
                continue
            cluster_points = data[labels == label]
            center = np.mean(cluster_points, axis=0)
            centers.append(center)

        return np.array(centers) if centers else np.array([])

    def _generate_cluster_statistics(
        self, data: np.ndarray, labels: np.ndarray, scaler: StandardScaler
    ) -> Dict[int, Dict[str, Any]]:
        """Generate detailed statistics for each cluster."""
        cluster_stats = {}
        unique_labels = set(labels)

        # Extract original data before standardization
        original_data = scaler.inverse_transform(data[:, :2])
        original_prices = original_data[:, 0]

        for label in unique_labels:
            if label == -1:  # Skip noise points
                continue

            cluster_mask = labels == label
            cluster_points = data[cluster_mask]
            cluster_original_prices = original_prices[cluster_mask]

            # Calculate statistics using original prices
            size = len(cluster_points)
            prices = cluster_original_prices
            volumes = cluster_points[:, 1]  # Volumes are already scaled appropriately
            sides = cluster_points[:, 2]

            stats = {
                "size": size,
                "avg_price": float(np.mean(prices)),
                "price_range": (float(np.min(prices)), float(np.max(prices))),
                "total_volume": float(np.sum(volumes)),
                "avg_volume": float(np.mean(volumes)),
                "bid_ratio": float(np.sum(sides == 0) / size),  # Ratio of bid points
                "ask_ratio": float(np.sum(sides == 1) / size),  # Ratio of ask points
                "dominant_side": "bid" if np.sum(sides == 0) > np.sum(sides == 1) else "ask",
                "price_std": float(np.std(prices)),
                "volume_std": float(np.std(volumes)),
            }

            cluster_stats[label] = stats

        return cluster_stats

    def _calculate_silhouette_score(
        self, data: np.ndarray, labels: np.ndarray
    ) -> float:
        """Calculate silhouette score for clustering quality."""
        try:
            # Filter out noise points for silhouette calculation
            valid_mask = labels != -1
            if np.sum(valid_mask) < 2 or len(set(labels[valid_mask])) < 2:
                return 0.0

            score = silhouette_score(data[valid_mask], labels[valid_mask])
            return float(score)
        except Exception as e:
            logger.warning(f"Failed to calculate silhouette score: {e}")
            return 0.0

    def _identify_liquidity_peaks(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        centers: np.ndarray,
        cluster_stats: Dict[int, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Identify liquidity peaks from clustering results."""
        liquidity_peaks = []

        for cluster_id, stats in cluster_stats.items():
            # Calculate peak purity (how concentrated the cluster is)
            purity = 1.0 - (stats["price_std"] / (stats["price_range"][1] - stats["price_range"][0] + 0.001))
            purity = max(0.0, min(1.0, purity))

            peak = {
                "cluster_id": cluster_id,
                "center_price": stats["avg_price"],
                "total_volume": stats["total_volume"],
                "dominant_side": stats["dominant_side"],
                "purity": purity,
                "size": stats["size"],
                "price_range": stats["price_range"],
                "bid_ratio": stats["bid_ratio"],
                "ask_ratio": stats["ask_ratio"],
            }

            liquidity_peaks.append(peak)

        # Sort by total volume (descending)
        liquidity_peaks.sort(key=lambda x: x["total_volume"], reverse=True)

        return liquidity_peaks

    def _analyze_clusters_detailed(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        cluster_stats: Dict[int, Dict[str, Any]],
    ) -> Dict[int, Dict[str, Any]]:
        """Generate detailed analysis for each cluster."""
        return cluster_stats

    def _create_empty_result(self) -> Dict[str, Any]:
        """Create empty result when clustering is not possible."""
        return {
            "optimal_clusters": 0,
            "silhouette_score": 0.0,
            "wcss": [],
            "clustering_data": np.array([]),
            "labels": np.array([]),
            "centers": np.array([]),
            "cluster_analysis": {},
            "liquidity_peaks": [],
            "price_volume_map": {},
        }


class ClusterVisualizer:
    """Visualizer for clustering analysis results."""

    def __init__(self):
        self.fig, self.axes = plt.subplots(2, 2, figsize=(16, 12))
        try:
            plt.style.use('seaborn-v0_8')
        except OSError:
            plt.style.use('default')

    def plot_clustering_results(self, analysis_results: Dict[str, Any], save_path: str = None) -> None:
        """Plot comprehensive clustering analysis results."""
        clustering_data = analysis_results['clustering_data']
        labels = analysis_results['labels']
        centers = analysis_results['centers']
        cluster_analysis = analysis_results['cluster_analysis']

        if len(clustering_data) == 0:
            logger.warning("No data to visualize")
            return

        # Color mapping
        unique_labels = np.unique(labels)
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))

        # 1. Clustering scatter plot
        self._plot_cluster_scatter(clustering_data, labels, centers, unique_labels, colors)

        # 2. Liquidity peaks bar chart
        self._plot_liquidity_peaks(analysis_results['liquidity_peaks'])

        # 3. Elbow method plot
        self._plot_elbow_method(analysis_results)

        # 4. Cluster statistics table
        self._plot_cluster_statistics_table(cluster_analysis)

        plt.tight_layout()

        if save_path:
            try:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Plot saved to {save_path}")
            except Exception as e:
                logger.error(f"Failed to save plot: {e}")

        try:
            plt.show()
        except Exception:
            # Handle cases where display is not available
            logger.debug("Display not available, plot saved only")

    def _plot_cluster_scatter(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        centers: np.ndarray,
        unique_labels: np.ndarray,
        colors: np.ndarray,
    ) -> None:
        """Plot clustering scatter plot."""
        ax = self.axes[0, 0]

        for i, label in enumerate(unique_labels):
            if label == -1:
                color = 'gray'
                label_name = 'Noise'
            else:
                color = colors[i]
                label_name = f'Cluster {label}'

            mask = labels == label
            cluster_points = data[mask]

            # Separate bid and ask points
            bids_mask = cluster_points[:, 2] == 0
            asks_mask = cluster_points[:, 2] == 1

            # Plot bid points (triangles)
            if np.any(bids_mask):
                ax.scatter(
                    cluster_points[bids_mask, 0],
                    cluster_points[bids_mask, 1],
                    c=[color],
                    marker='^',
                    s=50,
                    alpha=0.7,
                    label=f'{label_name} Bid'
                )

            # Plot ask points (inverted triangles)
            if np.any(asks_mask):
                ax.scatter(
                    cluster_points[asks_mask, 0],
                    cluster_points[asks_mask, 1],
                    c=[color],
                    marker='v',
                    s=50,
                    alpha=0.7,
                    label=f'{label_name} Ask'
                )

        # Plot cluster centers
        if len(centers) > 0:
            ax.scatter(
                centers[:, 0],
                centers[:, 1],
                c='red',
                marker='X',
                s=200,
                edgecolors='black',
                linewidth=2,
                label='Cluster Centers',
                zorder=10
            )

        ax.set_xlabel('Price (Standardized)')
        ax.set_ylabel('Volume (Standardized)')
        ax.set_title('Order Book Clustering Results')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)

    def _plot_liquidity_peaks(self, liquidity_peaks: List[Dict[str, Any]]) -> None:
        """Plot liquidity peaks bar chart."""
        ax = self.axes[0, 1]

        if not liquidity_peaks:
            ax.text(0.5, 0.5, 'No liquidity peaks found',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Liquidity Peaks by Cluster')
            return

        peak_prices = [peak['center_price'] for peak in liquidity_peaks]
        peak_volumes = [peak['total_volume'] for peak in liquidity_peaks]
        peak_colors = ['green' if peak['dominant_side'] == 'bid' else 'red'
                      for peak in liquidity_peaks]

        bars = ax.bar(range(len(peak_prices)), peak_volumes, color=peak_colors, alpha=0.7)
        ax.set_xlabel('Liquidity Peak Index')
        ax.set_ylabel('Total Volume')
        ax.set_title('Liquidity Peaks by Cluster')
        ax.set_xticks(range(len(peak_prices)))
        ax.set_xticklabels([f'Peak {i+1}' for i in range(len(peak_prices))])

        # Add value labels on bars
        for bar, volume in zip(bars, peak_volumes):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{volume:.1f}', ha='center', va='bottom')

        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='green', alpha=0.7, label='Bid Dominant'),
                         Patch(facecolor='red', alpha=0.7, label='Ask Dominant')]
        ax.legend(handles=legend_elements)

    def _plot_elbow_method(self, analysis_results: Dict[str, Any]) -> None:
        """Plot elbow method for optimal K selection."""
        ax = self.axes[1, 0]

        wcss = analysis_results.get('wcss', [])
        if not wcss:
            ax.text(0.5, 0.5, 'No WCSS data available',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Elbow Method for Optimal K')
            return

        k_range = range(2, 2 + len(wcss))
        ax.plot(k_range, wcss, 'bo-', markersize=8, linewidth=2)

        # Mark optimal K
        optimal_k = analysis_results.get('optimal_clusters', 2)
        if optimal_k in k_range:
            ax.axvline(x=optimal_k, color='red', linestyle='--',
                      label=f'Optimal K: {optimal_k}')
            # Add annotation
            idx = list(k_range).index(optimal_k)
            ax.annotate(f'K={optimal_k}',
                       xy=(optimal_k, wcss[idx]),
                       xytext=(optimal_k + 0.5, wcss[idx] + max(wcss) * 0.05),
                       arrowprops=dict(arrowstyle='->', color='red'),
                       fontsize=10, color='red')

        ax.set_xlabel('Number of Clusters')
        ax.set_ylabel('Within-Cluster Sum of Squares')
        ax.set_title('Elbow Method for Optimal K')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_cluster_statistics_table(
        self, cluster_analysis: Dict[int, Dict[str, Any]]
    ) -> None:
        """Plot cluster statistics as a table."""
        ax = self.axes[1, 1]

        if not cluster_analysis:
            ax.text(0.5, 0.5, 'No cluster statistics available',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Cluster Statistics')
            ax.axis('off')
            return

        # Prepare data for table
        cluster_stats = []
        for cluster_id, stats in cluster_analysis.items():
            cluster_stats.append({
                'Cluster': cluster_id,
                'Size': stats['size'],
                'Avg Price': f"{stats['avg_price']:.2f}",
                'Total Volume': f"{stats['total_volume']:.1f}",
                'Bid Ratio': f"{stats['bid_ratio']:.2f}",
                'Dominant': stats['dominant_side'].upper()
            })

        # Create DataFrame
        stats_df = pd.DataFrame(cluster_stats)

        # Create table
        table = ax.table(
            cellText=stats_df.values,
            colLabels=stats_df.columns,
            cellLoc='center',
            loc='center',
            bbox=[0, 0, 1, 1]
        )

        # Style table
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.8)

        # Color header row
        for i in range(len(stats_df.columns)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')

        # Color alternating rows
        for i in range(1, len(stats_df) + 1):
            for j in range(len(stats_df.columns)):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f0f0f0')

        ax.set_title('Cluster Statistics Summary')
        ax.axis('off')


def print_clustering_results(results: Dict[str, Any]) -> None:
    """Print clustering analysis results in the specified format."""
    print("聚类分析结果:")
    print(f"最优聚类数: {results['optimal_clusters']}")
    print(f"轮廓系数: {results['silhouette_score']:.3f}")

    print("\n流动性峰值区域:")
    for i, peak in enumerate(results['liquidity_peaks']):
        print(f"峰值 {i+1}: ${peak['center_price']:.2f}, "
              f"总量: {peak['total_volume']:.0f}, "
              f"方向: {peak['dominant_side']}, "
              f"纯度: {peak['purity']:.2f}")

    print("\n详细聚类统计:")
    for cluster_id, stats in results['cluster_analysis'].items():
        print(f"聚类 {cluster_id}: {stats['size']}个订单, "
              f"价格区间: ${stats['price_range'][0]:.2f}-${stats['price_range'][1]:.2f}, "
              f"总挂单量: {stats['total_volume']:.0f}")