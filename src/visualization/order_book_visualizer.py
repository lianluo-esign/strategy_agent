"""Order book depth distribution visualizer.

This module provides functionality to visualize order book depth data
by aggregating it at 1-dollar precision and creating distribution charts.
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from ..core.models import DepthSnapshot
from ..core.price_aggregator import aggregate_depth_by_one_dollar
from ..utils.config import VisualizationConfig

logger = logging.getLogger(__name__)


class OrderBookVisualizer:
    """Visualizes order book depth distribution with 1-dollar precision aggregation.

    This class handles the creation of order book distribution charts,
    including data aggregation, visualization generation, and file management.
    """

    def __init__(self, config: VisualizationConfig) -> None:
        """Initialize the visualizer with configuration.

        Args:
            config: Visualization configuration settings

        Raises:
            ValueError: If configuration settings are invalid
            OSError: If output directory cannot be created
        """
        self.config = config
        self._validate_config()
        self._setup_output_directory()
        self._setup_matplotlib_style()

    def _validate_config(self) -> None:
        """Validate visualization configuration.

        Raises:
            ValueError: If configuration values are invalid
        """
        if self.config.price_aggregation_precision <= 0:
            raise ValueError("Price aggregation precision must be positive")

        if self.config.max_price_levels <= 0:
            raise ValueError("Max price levels must be positive")

        if self.config.chart_width <= 0 or self.config.chart_height <= 0:
            raise ValueError("Chart dimensions must be positive")

        if self.config.chart_dpi <= 0:
            raise ValueError("Chart DPI must be positive")

        if self.config.retention_days <= 0:
            raise ValueError("Retention days must be positive")

    def _setup_output_directory(self) -> None:
        """Setup the output directory for saving visualization files.

        Raises:
            OSError: If directory cannot be created
        """
        self.output_path = Path(self.config.output_base_path)
        try:
            self.output_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Output directory ready: {self.output_path}")
        except OSError as e:
            logger.error(f"Failed to create output directory {self.output_path}: {e}")
            raise

    def _setup_matplotlib_style(self) -> None:
        """Setup matplotlib styling configuration."""
        try:
            # Set the style
            plt.style.use(self.config.chart_style)
        except OSError:
            # Fallback to default style if specified style not found
            logger.warning(f"Style '{self.config.chart_style}' not found, using default")
            plt.style.use('default')

        # Configure matplotlib for better quality
        plt.rcParams['figure.figsize'] = (
            self.config.chart_width / 100,
            self.config.chart_height / 100
        )
        plt.rcParams['figure.dpi'] = self.config.chart_dpi
        plt.rcParams['savefig.dpi'] = self.config.chart_dpi
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
        plt.rcParams['legend.fontsize'] = 11

    def create_order_book_distribution_chart(
        self,
        depth_snapshot: DepthSnapshot,
        symbol: str | None = None
    ) -> str:
        """Create and save order book depth distribution chart.

        Args:
            depth_snapshot: Order book depth snapshot data
            symbol: Trading symbol (defaults to snapshot symbol)

        Returns:
            Path to the saved image file

        Raises:
            ValueError: If depth snapshot data is invalid
            RuntimeError: If chart generation fails
        """
        if not depth_snapshot.bids and not depth_snapshot.asks:
            raise ValueError("Depth snapshot contains no bid or ask data")

        symbol = symbol or depth_snapshot.symbol

        try:
            # Aggregate data by 1-dollar precision
            aggregated_bids, aggregated_asks = self._aggregate_depth_data(
                depth_snapshot.bids, depth_snapshot.asks
            )

            if not aggregated_bids and not aggregated_asks:
                raise ValueError("No valid data after aggregation")

            # Sort prices from high to low as required
            prices_bids, volumes_bids = self._prepare_sorted_data(aggregated_bids, reverse=True)
            prices_asks, volumes_asks = self._prepare_sorted_data(aggregated_asks, reverse=True)

            # Create the chart
            fig = self._create_distribution_chart(
                prices_bids, volumes_bids,
                prices_asks, volumes_asks,
                symbol, depth_snapshot.timestamp
            )

            # Save the chart
            output_file = self._save_chart(fig, symbol, depth_snapshot.timestamp)

            logger.info(f"Order book distribution chart saved: {output_file}")
            return output_file

        except Exception as e:
            logger.error(f"Failed to create order book distribution chart: {e}")
            raise RuntimeError(f"Chart generation failed: {e}") from e

    def _aggregate_depth_data(
        self, bids: list[Any], asks: list[Any]
    ) -> tuple[dict[float, float], dict[float, float]]:
        """Aggregate depth data by configured precision.

        Args:
            bids: List of bid levels
            asks: List of ask levels

        Returns:
            Tuple of aggregated bids and asks dictionaries
        """
        try:
            aggregated_bids, aggregated_asks = aggregate_depth_by_one_dollar(bids, asks)

            # Convert Decimal keys to float for compatibility with matplotlib
            bids_float = {float(k): float(v) for k, v in aggregated_bids.items()}
            asks_float = {float(k): float(v) for k, v in aggregated_asks.items()}

            # Limit number of price levels if configured
            if self.config.max_price_levels > 0:
                # Sort by volume and keep top levels
                bids_items = sorted(bids_float.items(), key=lambda x: x[1], reverse=True)
                asks_items = sorted(asks_float.items(), key=lambda x: x[1], reverse=True)

                bids_float = dict(bids_items[:self.config.max_price_levels])
                asks_float = dict(asks_items[:self.config.max_price_levels])

            logger.debug(f"Aggregated {len(bids)} bids to {len(bids_float)} levels")
            logger.debug(f"Aggregated {len(asks)} asks to {len(asks_float)} levels")

            return bids_float, asks_float

        except Exception as e:
            logger.error(f"Failed to aggregate depth data: {e}")
            raise

    def _prepare_sorted_data(
        self, data: dict[float, float], reverse: bool = True
    ) -> tuple[list[float], list[float]]:
        """Prepare sorted price and volume data.

        Args:
            data: Dictionary of price to volume mappings
            reverse: Whether to sort in descending order (high to low)

        Returns:
            Tuple of (prices_list, volumes_list) sorted as specified
        """
        if not data:
            return [], []

        # Sort by price (high to low as required by specification)
        sorted_items = sorted(data.items(), key=lambda x: x[0], reverse=reverse)
        prices, volumes = zip(*sorted_items, strict=True) if sorted_items else ([], [])

        return list(prices), list(volumes)

    def _create_distribution_chart(
        self,
        prices_bids: list[float],
        volumes_bids: list[float],
        prices_asks: list[float],
        volumes_asks: list[float],
        symbol: str,
        timestamp: datetime
    ) -> Figure:
        """Create the order book distribution chart.

        Args:
            prices_bids: Sorted bid prices
            volumes_bids: Corresponding bid volumes
            prices_asks: Sorted ask prices
            volumes_asks: Corresponding ask volumes
            symbol: Trading symbol
            timestamp: Data timestamp

        Returns:
            matplotlib Figure object
        """
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(
            self.config.chart_width / 100,
            self.config.chart_height / 100
        ))
        fig.suptitle(
            f'{symbol} Order Book Depth Distribution\n{timestamp.strftime("%Y-%m-%d %H:%M:%S")}',
            fontsize=16, fontweight='bold'
        )

        # Plot 1: Side-by-side bars for bids and asks
        self._plot_side_by_side_distribution(ax1, prices_bids, volumes_bids, prices_asks, volumes_asks)

        # Plot 2: Overlay chart with different scaling
        self._plot_overlay_distribution(ax2, prices_bids, volumes_bids, prices_asks, volumes_asks)

        # Adjust layout
        plt.tight_layout()
        return fig

    def _plot_side_by_side_distribution(
        self,
        ax: Any,
        prices_bids: list[float],
        volumes_bids: list[float],
        prices_asks: list[float],
        volumes_asks: list[float]
    ) -> None:
        """Plot side-by-side bid/ask distribution.

        Args:
            ax: Matplotlib axis object
            prices_bids: Bid price levels
            volumes_bids: Bid volumes
            prices_asks: Ask price levels
            volumes_asks: Ask volumes
        """
        # Validate input data
        if not prices_bids and not prices_asks:
            return

        # Create price bins
        if prices_bids and prices_asks:
            all_prices = sorted(set(prices_bids + prices_asks))
        elif prices_bids:
            all_prices = sorted(prices_bids)
        else:
            all_prices = sorted(prices_asks)

        if not all_prices:
            return

        # Plot bids (green)
        if prices_bids and volumes_bids:
            # Align bid volumes with price bins
            bid_volumes_aligned = []
            for price in all_prices:
                idx = prices_bids.index(price) if price in prices_bids else -1
                bid_volumes_aligned.append(volumes_bids[idx] if idx >= 0 else 0)

            ax.bar(
                [p - 0.4 for p in all_prices],  # Offset to the left
                bid_volumes_aligned,
                width=0.8,
                alpha=0.7,
                color='green',
                label='Bids (Buy Orders)',
                edgecolor='darkgreen',
                linewidth=0.5
            )

        # Plot asks (red)
        if prices_asks and volumes_asks:
            # Align ask volumes with price bins
            ask_volumes_aligned = []
            for price in all_prices:
                idx = prices_asks.index(price) if price in prices_asks else -1
                ask_volumes_aligned.append(volumes_asks[idx] if idx >= 0 else 0)

            ax.bar(
                all_prices,  # Center position
                ask_volumes_aligned,
                width=0.8,
                alpha=0.7,
                color='red',
                label='Asks (Sell Orders)',
                edgecolor='darkred',
                linewidth=0.5
            )

        ax.set_xlabel('Price (USD)')
        ax.set_ylabel('Volume (BTC)')
        ax.set_title('Order Book Distribution by Price Level')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Format x-axis to show fewer labels if there are many price levels
        if len(all_prices) > 50:
            step = max(1, len(all_prices) // 20)
            ax.set_xticks(all_prices[::step])

    def _plot_overlay_distribution(
        self,
        ax: Any,
        prices_bids: list[float],
        volumes_bids: list[float],
        prices_asks: list[float],
        volumes_asks: list[float]
    ) -> None:
        """Plot overlay distribution with area charts.

        Args:
            ax: Matplotlib axis object
            prices_bids: Bid price levels
            volumes_bids: Bid volumes
            prices_asks: Ask price levels
            volumes_asks: Ask volumes
        """
        # Normalize volumes for better visualization
        max_volume = max(
            max(volumes_bids) if volumes_bids else 0,
            max(volumes_asks) if volumes_asks else 0
        )

        if max_volume == 0:
            return

        # Plot bids as area chart
        if prices_bids and volumes_bids:
            # Sort bids descending (high to low)
            sorted_bids = sorted(zip(prices_bids, volumes_bids, strict=True), reverse=True)
            bid_prices, bid_volumes = zip(*sorted_bids, strict=True) if sorted_bids else ([], [])

            normalized_bid_volumes = [v / max_volume * 100 for v in bid_volumes]
            ax.fill_between(
                bid_prices, 0, normalized_bid_volumes,
                alpha=0.5, color='green', label='Bids Volume %'
            )
            ax.plot(bid_prices, normalized_bid_volumes, 'g-', linewidth=2)

        # Plot asks as area chart
        if prices_asks and volumes_asks:
            # Sort asks descending (high to low)
            sorted_asks = sorted(zip(prices_asks, volumes_asks, strict=True), reverse=True)
            ask_prices, ask_volumes = zip(*sorted_asks, strict=True) if sorted_asks else ([], [])

            normalized_ask_volumes = [v / max_volume * 100 for v in ask_volumes]
            ax.fill_between(
                ask_prices, 0, normalized_ask_volumes,
                alpha=0.5, color='red', label='Asks Volume %'
            )
            ax.plot(ask_prices, normalized_ask_volumes, 'r-', linewidth=2)

        ax.set_xlabel('Price (USD) - High to Low')
        ax.set_ylabel('Normalized Volume (%)')
        ax.set_title('Normalized Volume Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _save_chart(self, fig: Figure, symbol: str, timestamp: datetime) -> str:
        """Save the chart to file.

        Args:
            fig: Matplotlib figure object
            symbol: Trading symbol
            timestamp: Data timestamp

        Returns:
            Path to the saved file
        """
        # Generate filename
        timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
        filename = f"order_book_distribution_{symbol}_{timestamp_str}.png"
        filepath = self.output_path / filename

        try:
            # Save the figure
            fig.savefig(
                filepath,
                dpi=self.config.chart_dpi,
                bbox_inches='tight',
                facecolor='white',
                edgecolor='none'
            )

            # Close the figure to free memory
            plt.close(fig)

            return str(filepath)

        except Exception as e:
            plt.close(fig)  # Ensure figure is closed even on error
            logger.error(f"Failed to save chart to {filepath}: {e}")
            raise

    def cleanup_old_files(self) -> int:
        """Remove old visualization files based on retention policy.

        Returns:
            Number of files removed
        """
        if not self.config.auto_cleanup:
            logger.debug("Auto cleanup is disabled")
            return 0

        cutoff_date = datetime.now() - timedelta(days=self.config.retention_days)
        removed_count = 0

        try:
            for file_path in self.output_path.glob("*.png"):
                if file_path.is_file():
                    file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                    if file_time < cutoff_date:
                        file_path.unlink()
                        removed_count += 1
                        logger.debug(f"Removed old file: {file_path}")

            if removed_count > 0:
                logger.info(f"Cleaned up {removed_count} old visualization files")

        except Exception as e:
            logger.error(f"Failed to cleanup old files: {e}")

        return removed_count

    def get_visualization_stats(self) -> dict[str, Any]:
        """Get statistics about visualizations.

        Returns:
            Dictionary containing visualization statistics
        """
        try:
            png_files = list(self.output_path.glob("*.png"))
            total_size = sum(f.stat().st_size for f in png_files if f.is_file())

            return {
                "output_directory": str(self.output_path),
                "total_files": len(png_files),
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "retention_days": self.config.retention_days,
                "auto_cleanup_enabled": self.config.auto_cleanup,
                "chart_dimensions": f"{self.config.chart_width}x{self.config.chart_height}",
                "chart_dpi": self.config.chart_dpi
            }
        except Exception as e:
            logger.error(f"Failed to get visualization stats: {e}")
            return {"error": str(e)}
