"""Simplified liquidity peaks analyzer for order book analysis.

This module provides a lightweight approach to identify liquidity peaks
in order book data without complex machine learning clustering algorithms.
"""

import logging
from decimal import Decimal
from typing import Any

from .models import DepthSnapshot, SupportResistanceLevel

logger = logging.getLogger(__name__)

# Constants for algorithm parameters
PEAK_SCORE_NORMALIZATION_FACTOR = 3.0
MAX_PEAKS_RETURNED = 10
LOCAL_DENSITY_WINDOW_SIZE = 5


class LiquidityPeaksAnalyzer:
    """Simplified liquidity peaks analyzer for order book data.

    This analyzer identifies liquidity peaks by analyzing volume concentration
    at different price levels using a straightforward aggregation approach.
    """

    def __init__(
        self,
        min_volume_threshold: float = 10.0,
        peak_detection_window: int = LOCAL_DENSITY_WINDOW_SIZE,  # Price levels to consider for peak detection
        volume_weight: float = 2.0,
    ) -> None:
        """
        Initialize the liquidity peaks analyzer.

        Args:
            min_volume_threshold: Minimum volume to consider as a peak
            peak_detection_window: Number of price levels to analyze for peak detection
            volume_weight: Weight factor for volume in peak scoring
        """
        if min_volume_threshold <= 0:
            raise ValueError("min_volume_threshold must be positive")
        if peak_detection_window < 1:
            raise ValueError("peak_detection_window must be at least 1")
        if volume_weight <= 0:
            raise ValueError("volume_weight must be positive")

        self.min_volume_threshold = min_volume_threshold
        self.peak_detection_window = peak_detection_window
        self.volume_weight = volume_weight

        logger.info(
            f"Initialized LiquidityPeaksAnalyzer with min_volume={min_volume_threshold}, "
            f"peak_window={peak_detection_window}, volume_weight={volume_weight}"
        )

    def analyze_liquidity_peaks(self, snapshot: DepthSnapshot) -> dict[str, Any]:
        """
        Analyze order book data to identify liquidity peaks.

        Args:
            snapshot: Depth snapshot containing bids and asks

        Returns:
            Dictionary containing liquidity peak analysis results
        """
        # Input validation
        if not isinstance(snapshot, DepthSnapshot):
            raise TypeError("snapshot must be a DepthSnapshot object")
        if not snapshot.symbol:
            raise ValueError("snapshot must have a valid symbol")
        if not snapshot.bids and not snapshot.asks:
            logger.warning("Snapshot contains no bid or ask data")
            return self._create_empty_result()

        logger.info(f"Starting liquidity peaks analysis for {snapshot.symbol}")

        try:
            # Step 1: Aggregate order book data by 1-dollar precision
            aggregated_bids, aggregated_asks = self._aggregate_order_book_data(snapshot)

            if not aggregated_bids and not aggregated_asks:
                logger.warning("No order book data available for peak analysis")
                return self._create_empty_result()

            # Step 2: Identify liquidity peaks for bids and asks
            bid_peaks = self._identify_peaks_from_data(aggregated_bids, "bid")
            ask_peaks = self._identify_peaks_from_data(aggregated_asks, "ask")

            # Step 3: Convert peaks to SupportResistanceLevel objects
            liquidity_peaks = self._convert_peaks_to_support_resistance(
                bid_peaks, ask_peaks
            )

            # Step 4: Generate analysis summary
            analysis_summary = self._generate_analysis_summary(
                aggregated_bids, aggregated_asks, liquidity_peaks
            )

            results = {
                "liquidity_peaks": liquidity_peaks,
                "analysis_summary": analysis_summary,
                "bid_aggregation": aggregated_bids,
                "ask_aggregation": aggregated_asks,
                "total_bid_volume": sum(aggregated_bids.values()),
                "total_ask_volume": sum(aggregated_asks.values()),
                "peak_detection_stats": {
                    "bid_peaks_count": len(bid_peaks),
                    "ask_peaks_count": len(ask_peaks),
                    "total_peaks_count": len(liquidity_peaks),
                },
            }

            logger.info(
                f"Liquidity peaks analysis completed: "
                f"{len(liquidity_peaks)} peaks identified "
                f"(bid: {len(bid_peaks)}, ask: {len(ask_peaks)})"
            )

            return results

        except Exception as e:
            logger.error(f"Failed to analyze liquidity peaks: {e}")
            return self._create_empty_result()

    def _aggregate_order_book_data(
        self, snapshot: DepthSnapshot
    ) -> tuple[dict[Decimal, Decimal], dict[Decimal, Decimal]]:
        """Aggregate order book data by 1-dollar precision.

        Args:
            snapshot: Depth snapshot data

        Returns:
            Tuple of (aggregated_bids, aggregated_asks) dictionaries
        """
        # Use existing price aggregation utility
        from .price_aggregator import aggregate_depth_by_one_dollar

        return aggregate_depth_by_one_dollar(snapshot.bids, snapshot.asks)

    def _identify_peaks_from_data(
        self, data: dict[Decimal, Decimal], side: str
    ) -> list[dict[str, Any]]:
        """Identify liquidity peaks from aggregated order book data.

        Args:
            data: Aggregated price-volume data
            side: Either 'bid' or 'ask'

        Returns:
            List of peak data dictionaries
        """
        if not data:
            return []

        # Sort prices
        sorted_prices = sorted(data.keys(), reverse=(side == "bid"))

        peaks = []

        for i, price in enumerate(sorted_prices):
            volume = float(data[price])

            # Skip if volume below threshold
            if volume < self.min_volume_threshold:
                continue

            # Calculate peak score based on volume and local density
            peak_score = self._calculate_peak_score(
                price, volume, sorted_prices, data, side
            )

            # Consider as peak if score is significant
            if peak_score > 0:
                peak_data = {
                    "price": price,
                    "volume": volume,
                    "peak_score": peak_score,
                    "side": side,
                    "price_rank": i + 1,  # Position in sorted order
                }
                peaks.append(peak_data)

        # Sort peaks by peak score (descending)
        peaks.sort(key=lambda x: x["peak_score"], reverse=True)

        # Limit number of peaks returned
        return peaks[:MAX_PEAKS_RETURNED]

    def _calculate_peak_score(
        self,
        price: Decimal,
        volume: float,
        sorted_prices: list[Decimal],
        all_data: dict[Decimal, Decimal],
        side: str,
    ) -> float:
        """Calculate peak score for a given price level.

        Args:
            price: Price level to evaluate
            volume: Volume at this price level
            sorted_prices: All prices sorted by relevance
            all_data: Complete price-volume data
            side: Either 'bid' or 'ask'

        Returns:
            Peak score (higher is better)
        """
        if not all_data:
            return 0.0

        # Base score from volume (normalized)
        max_volume = float(max(all_data.values()))
        if max_volume > 0:
            volume_score = (volume / max_volume) * self.volume_weight
        else:
            volume_score = 0.0

        # Local density score - check nearby price levels
        local_density_score = self._calculate_local_density_score(
            price, sorted_prices, all_data
        )

        # Side-specific scoring
        side_preference = 1.0  # Can be adjusted for market conditions

        return volume_score + local_density_score + side_preference

    def _calculate_local_density_score(
        self,
        price: Decimal,
        sorted_prices: list[Decimal],
        all_data: dict[Decimal, Decimal],
    ) -> float:
        """Calculate local density score around a price level.

        Args:
            price: Price level to evaluate
            sorted_prices: All prices sorted
            all_data: Complete price-volume data

        Score Returns:
            Local density score (0.0 to 1.0)
        """
        if not all_data:
            return 0.0

        try:
            price_index = sorted_prices.index(price)
        except ValueError:
            return 0.0

        # Check neighboring price levels within window
        start_index = max(0, price_index - self.peak_detection_window // 2)
        end_index = min(
            len(sorted_prices), price_index + self.peak_detection_window // 2 + 1
        )

        neighboring_prices = sorted_prices[start_index:end_index]
        total_volume = sum(
            float(all_data.get(p, Decimal("0"))) for p in neighboring_prices
        )

        if total_volume > 0 and all_data:
            # Normalize by potential maximum volume in the window
            if len(all_data) > 0:
                max_possible_volume = (
                    len(neighboring_prices)
                    * float(max(all_data.values()))
                    / len(all_data)
                )
                if max_possible_volume > 0:
                    return min(1.0, total_volume / max_possible_volume)

        return 0.0

    def _convert_peaks_to_support_resistance(
        self, bid_peaks: list[dict[str, Any]], ask_peaks: list[dict[str, Any]]
    ) -> list[SupportResistanceLevel]:
        """Convert peaks to SupportResistanceLevel objects.

        Args:
            bid_peaks: List of bid peak data
            ask_peaks: List of ask peak data

        Returns:
            List of SupportResistanceLevel objects
        """
        liquidity_peaks = []

        # Convert bid peaks to support levels
        for peak_data in bid_peaks:
            support_level = SupportResistanceLevel(
                price=peak_data["price"],
                strength=min(
                    1.0, peak_data["peak_score"] / PEAK_SCORE_NORMALIZATION_FACTOR
                ),  # Normalize to 0-1 range
                level_type="support",
                volume_at_level=Decimal(str(peak_data["volume"])),
                confirmation_count=1,
                last_confirmed=None,
            )
            liquidity_peaks.append(support_level)

        # Convert ask peaks to resistance levels
        for peak_data in ask_peaks:
            resistance_level = SupportResistanceLevel(
                price=peak_data["price"],
                strength=min(
                    1.0, peak_data["peak_score"] / PEAK_SCORE_NORMALIZATION_FACTOR
                ),  # Normalize to 0-1 range
                level_type="resistance",
                volume_at_level=Decimal(str(peak_data["volume"])),
                confirmation_count=1,
                last_confirmed=None,
            )
            liquidity_peaks.append(resistance_level)

        return liquidity_peaks

    def _generate_analysis_summary(
        self,
        aggregated_bids: dict[Decimal, Decimal],
        aggregated_asks: dict[Decimal, Decimal],
        liquidity_peaks: list[SupportResistanceLevel],
    ) -> dict[str, Any]:
        """Generate analysis summary of the liquidity peaks.

        Args:
            aggregated_bids: Aggregated bid data
            aggregated_asks: Aggregated ask data
            liquidity_peaks: Identified liquidity peaks

        Returns:
            Analysis summary dictionary
        """
        total_bid_volume = float(sum(aggregated_bids.values()))
        total_ask_volume = float(sum(aggregated_asks.values()))
        total_volume = total_bid_volume + total_ask_volume

        bid_peaks = [p for p in liquidity_peaks if p.level_type == "support"]
        ask_peaks = [p for p in liquidity_peaks if p.level_type == "resistance"]

        # Calculate market balance
        if total_volume > 0:
            bid_ratio = total_bid_volume / total_volume
            ask_ratio = total_ask_volume / total_volume
        else:
            bid_ratio = ask_ratio = 0.5

        return {
            "total_volume": total_volume,
            "bid_ratio": bid_ratio,
            "ask_ratio": ask_ratio,
            "bid_peaks_count": len(bid_peaks),
            "ask_peaks_count": len(ask_peaks),
            "total_peaks_count": len(liquidity_peaks),
            "market_balance": "balanced"
            if abs(bid_ratio - ask_ratio) < 0.1
            else "bid_heavy"
            if bid_ratio > ask_ratio
            else "ask_heavy",
            "peak_density_score": sum(peak.strength for peak in liquidity_peaks)
            / len(liquidity_peaks)
            if liquidity_peaks
            else 0.0,
        }

    def _create_empty_result(self) -> dict[str, Any]:
        """Create empty result when analysis is not possible."""
        return {
            "liquidity_peaks": [],
            "analysis_summary": {},
            "bid_aggregation": {},
            "ask_aggregation": {},
            "total_bid_volume": 0.0,
            "total_ask_volume": 0.0,
            "peak_detection_stats": {
                "bid_peaks_count": 0,
                "ask_peaks_count": 0,
                "total_peaks_count": 0,
            },
        }


def print_liquidity_peaks_results(results: dict[str, Any]) -> None:
    """Print liquidity peaks analysis results in a clean, organized format.

    Args:
        results: Liquidity peaks analysis results
    """
    peaks = results.get("liquidity_peaks", [])
    summary = results.get("analysis_summary", {})

    if not peaks:
        print("未发现流动性峰值区域")
        return

    # Separate peaks by type
    ask_peaks = [p for p in peaks if p.level_type == "resistance"]
    bid_peaks = [p for p in peaks if p.level_type == "support"]

    # Sort by price (descending for asks, ascending for bids)
    ask_peaks.sort(key=lambda x: float(x.price), reverse=True)
    bid_peaks.sort(key=lambda x: float(x.price))

    print("\n=== 流动性峰值区域 ===")

    # Display ask peaks (resistance levels)
    if ask_peaks:
        print(f"\n🔻 卖盘阻力区域 (Ask Dominant):")
        for i, peak in enumerate(ask_peaks):
            print(
                f"  阻力 {i + 1}: ${peak.price:,.0f} | "
                f"挂单量: {peak.volume_at_level:,.0f} | "
                f"纯度: {peak.strength:.2f}"
            )

    # Display bid peaks (support levels)
    if bid_peaks:
        print(f"\n🟢 买盘支撑区域 (Bid Dominant):")
        for i, peak in enumerate(bid_peaks):
            print(
                f"  支撑 {i + 1}: ${peak.price:,.0f} | "
                f"挂单量: {peak.volume_at_level:,.0f} | "
                f"纯度: {peak.strength:.2f}"
            )
