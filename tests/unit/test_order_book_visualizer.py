"""Unit tests for order book visualizer."""

import tempfile
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.core.models import DepthLevel, DepthSnapshot
from src.utils.config import VisualizationConfig
from src.visualization.order_book_visualizer import OrderBookVisualizer


class TestOrderBookVisualizer:
    """Test cases for OrderBookVisualizer class."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Create a temporary directory for test outputs
        self.temp_dir = tempfile.mkdtemp()

        # Create test configuration
        self.config = VisualizationConfig(
            enabled=True,
            price_aggregation_precision=1.0,
            max_price_levels=100,
            chart_width=800,
            chart_height=600,
            chart_dpi=150,
            chart_style="default",  # Use default style for testing
            output_base_path=self.temp_dir,
            retention_days=7,
            auto_cleanup=True,
        )

        # Create visualizer instance
        self.visualizer = OrderBookVisualizer(self.config)
        self.temp_path = Path(self.temp_dir)

    def teardown_method(self):
        """Clean up after each test method."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_initialization_with_valid_config(self):
        """Test visualizer initialization with valid configuration."""
        assert self.visualizer.config == self.config
        assert self.visualizer.output_path == Path(self.temp_dir)
        assert self.visualizer.output_path.exists()

    def test_initialization_with_invalid_config(self):
        """Test visualizer initialization with invalid configuration."""
        # Test with negative precision
        with pytest.raises(
            ValueError, match="Price aggregation precision must be positive"
        ):
            config = VisualizationConfig(price_aggregation_precision=-1.0)
            OrderBookVisualizer(config)

        # Test with zero chart dimensions
        with pytest.raises(ValueError, match="Chart dimensions must be positive"):
            config = VisualizationConfig(chart_width=0)
            OrderBookVisualizer(config)

        # Test with negative retention days
        with pytest.raises(ValueError, match="Retention days must be positive"):
            config = VisualizationConfig(retention_days=-1)
            OrderBookVisualizer(config)

    def test_aggregate_depth_data_with_valid_input(self):
        """Test depth data aggregation with valid input."""
        # Create test bid and ask levels (adjusted for 1-dollar rounding)
        bids = [
            DepthLevel(price=Decimal("50001.50"), quantity=Decimal("1.2")),
            DepthLevel(price=Decimal("50001.80"), quantity=Decimal("0.8")),
        ]

        asks = [
            DepthLevel(price=Decimal("50003.20"), quantity=Decimal("1.5")),
            DepthLevel(price=Decimal("50003.90"), quantity=Decimal("0.7")),
        ]

        # Test aggregation
        aggregated_bids, aggregated_asks = self.visualizer._aggregate_depth_data(
            bids, asks
        )

        # Verify bid aggregation (all should round down to 50001)
        assert 50001.0 in aggregated_bids
        assert aggregated_bids[50001.0] == 2.0  # 1.2 + 0.8

        # Verify ask aggregation (all should round down to 50003)
        assert 50003.0 in aggregated_asks
        assert aggregated_asks[50003.0] == 2.2  # 1.5 + 0.7

    def test_aggregate_depth_data_with_empty_input(self):
        """Test depth data aggregation with empty input."""
        aggregated_bids, aggregated_asks = self.visualizer._aggregate_depth_data([], [])

        assert aggregated_bids == {}
        assert aggregated_asks == {}

    def test_aggregate_depth_data_with_invalid_prices(self):
        """Test depth data aggregation with invalid prices."""
        # Create levels with zero or negative prices
        bids = [
            DepthLevel(price=Decimal("0"), quantity=Decimal("1.0")),
            DepthLevel(price=Decimal("-100"), quantity=Decimal("1.0")),
        ]

        aggregated_bids, aggregated_asks = self.visualizer._aggregate_depth_data(
            bids, []
        )

        # Should filter out invalid prices
        assert aggregated_bids == {}

    def test_prepare_sorted_data_descending(self):
        """Test data preparation with descending sort order."""
        test_data = {
            50005.0: 2.5,
            50001.0: 1.8,
            50003.0: 3.2,
            50002.0: 1.1,
        }

        prices, volumes = self.visualizer._prepare_sorted_data(test_data, reverse=True)

        # Should be sorted high to low
        assert prices == [50005.0, 50003.0, 50002.0, 50001.0]
        assert volumes == [2.5, 3.2, 1.1, 1.8]

    def test_prepare_sorted_data_ascending(self):
        """Test data preparation with ascending sort order."""
        test_data = {
            50005.0: 2.5,
            50001.0: 1.8,
            50003.0: 3.2,
        }

        prices, volumes = self.visualizer._prepare_sorted_data(test_data, reverse=False)

        # Should be sorted low to high
        assert prices == [50001.0, 50003.0, 50005.0]
        assert volumes == [1.8, 3.2, 2.5]

    def test_prepare_sorted_data_with_empty_input(self):
        """Test data preparation with empty input."""
        prices, volumes = self.visualizer._prepare_sorted_data({})

        assert prices == []
        assert volumes == []

    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.close")
    @patch("matplotlib.pyplot.subplots")
    def test_create_order_book_distribution_chart_success(
        self, mock_subplots, mock_close, mock_savefig
    ):
        """Test successful chart creation."""
        # Create mock figure and axes
        mock_fig = MagicMock()
        mock_ax1 = MagicMock()
        mock_ax2 = MagicMock()
        mock_subplots.return_value = (mock_fig, (mock_ax1, mock_ax2))

        # Create test depth snapshot
        snapshot = DepthSnapshot(
            symbol="BTCFDUSD",
            timestamp=datetime(2024, 1, 15, 10, 30, 0),
            bids=[
                DepthLevel(price=Decimal("50001.50"), quantity=Decimal("1.2")),
                DepthLevel(price=Decimal("50002.80"), quantity=Decimal("0.8")),
            ],
            asks=[
                DepthLevel(price=Decimal("50003.20"), quantity=Decimal("1.5")),
                DepthLevel(price=Decimal("50004.90"), quantity=Decimal("0.7")),
            ],
        )

        # Create chart
        output_file = self.visualizer.create_order_book_distribution_chart(snapshot)

        # Verify output file path format
        assert output_file.endswith(".png")
        assert "BTCFDUSD" in output_file
        assert "order_book_distribution" in output_file

        # Verify matplotlib functions were called
        mock_subplots.assert_called_once()
        mock_savefig.assert_called_once()
        mock_close.assert_called_once()

    def test_create_order_book_distribution_chart_with_empty_snapshot(self):
        """Test chart creation with empty depth snapshot."""
        snapshot = DepthSnapshot(
            symbol="BTCFDUSD",
            timestamp=datetime(2024, 1, 15, 10, 30, 0),
            bids=[],
            asks=[],
        )

        with pytest.raises(
            ValueError, match="Depth snapshot contains no bid or ask data"
        ):
            self.visualizer.create_order_book_distribution_chart(snapshot)

    def test_cleanup_old_files(self):
        """Test cleanup of old visualization files."""
        # Create some test files
        old_file = self.temp_path / "old_chart.png"
        recent_file = self.temp_path / "recent_chart.png"

        old_file.touch()
        recent_file.touch()

        # Mock file modification times
        import time

        old_time = time.time() - (8 * 24 * 60 * 60)  # 8 days ago
        recent_time = time.time() - (1 * 60 * 60)  # 1 hour ago

        import os

        os.utime(old_file, (old_time, old_time))
        os.utime(recent_file, (recent_time, recent_time))

        # Run cleanup
        removed_count = self.visualizer.cleanup_old_files()

        # Verify old file was removed, recent file remains
        assert removed_count == 1
        assert not old_file.exists()
        assert recent_file.exists()

    def test_cleanup_old_files_disabled(self):
        """Test cleanup when auto_cleanup is disabled."""
        self.visualizer.config.auto_cleanup = False

        removed_count = self.visualizer.cleanup_old_files()

        assert removed_count == 0

    def test_get_visualization_stats(self):
        """Test getting visualization statistics."""
        # Create a test file with some content
        test_file = self.visualizer.output_path / "test_chart.png"
        test_file.write_bytes(b"x" * 1024)  # 1KB of data

        stats = self.visualizer.get_visualization_stats()

        assert stats["output_directory"] == str(self.visualizer.output_path)
        assert stats["total_files"] == 1
        assert stats["total_size_mb"] > 0
        assert stats["retention_days"] == 7
        assert stats["auto_cleanup_enabled"] is True
        assert stats["chart_dimensions"] == "800x600"
        assert stats["chart_dpi"] == 150

    def test_get_visualization_stats_empty_directory(self):
        """Test getting visualization statistics from empty directory."""
        stats = self.visualizer.get_visualization_stats()

        assert stats["total_files"] == 0
        assert stats["total_size_mb"] == 0

    def test_max_price_levels_limit(self):
        """Test that maximum price levels is respected."""
        # Create many price levels
        bids = [
            DepthLevel(price=Decimal(f"{50000 + i}.50"), quantity=Decimal("1.0"))
            for i in range(200)  # More than max_price_levels (100)
        ]

        asks = [
            DepthLevel(price=Decimal(f"{50200 + i}.50"), quantity=Decimal("1.0"))
            for i in range(200)
        ]

        aggregated_bids, aggregated_asks = self.visualizer._aggregate_depth_data(
            bids, asks
        )

        # Should limit to max_price_levels
        assert len(aggregated_bids) <= 100
        assert len(aggregated_asks) <= 100

    @patch("matplotlib.pyplot.style.use")
    def test_matplotlib_style_fallback(self, mock_style_use):
        """Test fallback to default style when specified style is not found."""
        # Mock style.use to raise OSError for invalid style
        mock_style_use.side_effect = [OSError("Style not found"), None]

        config = VisualizationConfig(
            chart_style="nonexistent_style", output_base_path=self.temp_dir
        )

        # Should not raise exception, should fallback to default
        OrderBookVisualizer(config)

        # Verify both styles were attempted
        assert mock_style_use.call_count == 2
        mock_style_use.assert_any_call("nonexistent_style")
        mock_style_use.assert_any_call("default")

    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.close")
    @patch("matplotlib.pyplot.subplots")
    def test_chart_creation_with_custom_symbol(
        self, mock_subplots, mock_close, mock_savefig
    ):
        """Test chart creation with custom symbol override."""
        # Create mock figure and axes
        mock_fig = MagicMock()
        mock_ax1 = MagicMock()
        mock_ax2 = MagicMock()
        mock_subplots.return_value = (mock_fig, (mock_ax1, mock_ax2))

        snapshot = DepthSnapshot(
            symbol="ORIGINAL",
            timestamp=datetime(2024, 1, 15, 10, 30, 0),
            bids=[DepthLevel(price=Decimal("50001.50"), quantity=Decimal("1.2"))],
            asks=[DepthLevel(price=Decimal("50003.20"), quantity=Decimal("1.5"))],
        )

        # Create chart with custom symbol
        output_file = self.visualizer.create_order_book_distribution_chart(
            snapshot, symbol="CUSTOM"
        )

        # Verify savefig was called
        mock_savefig.assert_called_once()
        mock_close.assert_called_once()

        # Check that custom symbol appears in the filename
        assert "CUSTOM" in output_file
        assert "ORIGINAL" not in output_file
