"""Price aggregation utilities for configurable precision order book analysis.

This module provides functions and classes to aggregate depth snapshot data
by configurable precision to support more sophisticated wave peak analysis.
"""

import logging
from collections import defaultdict
from decimal import Decimal

from .models import DepthLevel

logger = logging.getLogger(__name__)


def aggregate_depth_by_one_dollar(
    bids: list[DepthLevel], asks: list[DepthLevel]
) -> tuple[dict[Decimal, Decimal], dict[Decimal, Decimal]]:
    """
    Aggregate depth snapshot data by 1-dollar precision.

    Each price level is rounded down to the nearest 1-dollar integer,
    and volumes are summed for levels within the same 1-dollar range.

    Args:
        bids: List of bid levels [(price, quantity), ...]
        asks: List of ask levels [(price, quantity), ...]

    Returns:
        Tuple containing:
        - Dict of aggregated bid prices to volumes {price: volume}
        - Dict of aggregated ask prices to volumes {price: volume}

    Raises:
        TypeError: If inputs are not in the expected format
        ValueError: If prices or quantities contain invalid values

    Example:
        Input:  [(50001.50, 1.2), (50002.30, 0.8), (50001.80, 2.1)]
        Output: {50001.00: 3.3}  # 50001.50 -> 50001, 50001.80 -> 50001, volumes summed
    """
    if not isinstance(bids, list):
        raise TypeError(f"bids must be a list, got {type(bids).__name__}")
    if not isinstance(asks, list):
        raise TypeError(f"asks must be a list, got {type(asks).__name__}")

    aggregated_bids: dict[Decimal, Decimal] = defaultdict(Decimal)
    aggregated_asks: dict[Decimal, Decimal] = defaultdict(Decimal)

    logger.debug(f"Aggregating {len(bids)} bid levels and {len(asks)} ask levels")

    # Validate and aggregate bids
    for i, bid_level in enumerate(bids):
        try:
            if not isinstance(bid_level.price, Decimal) or not isinstance(
                bid_level.quantity, Decimal
            ):
                logger.warning(f"Invalid bid level format at index {i}: {bid_level}")
                continue

            if bid_level.price <= 0:
                logger.warning(
                    f"Invalid bid price at index {i}: {bid_level.price} <= 0"
                )
                continue

            if bid_level.quantity <= 0:
                logger.debug(
                    f"Skipping zero/negative bid quantity at index {i}: {bid_level.quantity}"
                )
                continue

            # Round price down to nearest 1-dollar integer
            rounded_price = _round_down_to_dollar(bid_level.price)
            aggregated_bids[rounded_price] += bid_level.quantity

        except Exception as e:
            logger.error(f"Error processing bid level at index {i}: {e}")
            continue

    # Validate and aggregate asks
    for i, ask_level in enumerate(asks):
        try:
            if not isinstance(ask_level.price, Decimal) or not isinstance(
                ask_level.quantity, Decimal
            ):
                logger.warning(f"Invalid ask level format at index {i}: {ask_level}")
                continue

            if ask_level.price <= 0:
                logger.warning(
                    f"Invalid ask price at index {i}: {ask_level.price} <= 0"
                )
                continue

            if ask_level.quantity <= 0:
                logger.debug(
                    f"Skipping zero/negative ask quantity at index {i}: {ask_level.quantity}"
                )
                continue

            # Round price down to nearest 1-dollar integer
            rounded_price = _round_down_to_dollar(ask_level.price)
            aggregated_asks[rounded_price] += ask_level.quantity

        except Exception as e:
            logger.error(f"Error processing ask level at index {i}: {e}")
            continue

    logger.debug(f"Bids aggregated to {len(aggregated_bids)} price levels")
    logger.debug(f"Asks aggregated to {len(aggregated_asks)} price levels")

    # Convert defaultdicts to regular dicts for return
    return dict(aggregated_bids), dict(aggregated_asks)


def _round_down_to_dollar(price: Decimal) -> Decimal:
    """
    Round price down to the nearest 1-dollar integer.

    Examples:
        50001.50 -> 50001
        50000.99 -> 50000
        50123.45 -> 50123
    """
    if not isinstance(price, Decimal):
        raise TypeError(f"Expected Decimal, got {type(price)}")

    # Use integer division to round down to nearest dollar
    return Decimal(int(price))


def calculate_depth_statistics(
    aggregated_bids: dict[Decimal, Decimal], aggregated_asks: dict[Decimal, Decimal]
) -> dict[str, Decimal]:
    """
    Calculate depth statistics for aggregated order book data.

    Args:
        aggregated_bids: Aggregated bid data {price: volume}
        aggregated_asks: Aggregated ask data {price: volume}

    Returns:
        Dictionary containing depth statistics:
        - 'total_bid_volume': Total volume on bid side
        - 'total_ask_volume': Total volume on ask side
        - 'bid_price_levels': Number of bid price levels
        - 'ask_price_levels': Number of ask price levels
        - 'price_spread': Current best bid-ask spread
        - 'bid_ask_ratio': Bid to ask volume ratio
    """
    total_bid_volume = (
        sum(aggregated_bids.values()) if aggregated_bids else Decimal("0")
    )
    total_ask_volume = (
        sum(aggregated_asks.values()) if aggregated_asks else Decimal("0")
    )

    # Calculate price spread (best bid vs best ask)
    best_bid = max(aggregated_bids.keys()) if aggregated_bids else Decimal("0")
    best_ask = min(aggregated_asks.keys()) if aggregated_asks else Decimal("0")
    price_spread = best_ask - best_bid

    # Calculate bid/ask ratio
    bid_ask_ratio = total_bid_volume / (
        total_ask_volume + Decimal("0.01")
    )  # Avoid division by zero

    statistics = {
        "total_bid_volume": total_bid_volume,
        "total_ask_volume": total_ask_volume,
        "bid_price_levels": Decimal(str(len(aggregated_bids))),
        "ask_price_levels": Decimal(str(len(aggregated_asks))),
        "price_spread": price_spread,
        "bid_ask_ratio": bid_ask_ratio,
    }

    logger.debug(f"Depth statistics: {statistics}")
    return statistics


def identify_liquidity_clusters(
    price_volume_data: dict[Decimal, Decimal],
    min_cluster_volume: Decimal = Decimal("10.0"),
) -> list[dict[str, Decimal]]:
    """
    Identify high-liquidity clusters in aggregated depth data.

    Args:
        price_volume_data: Aggregated price-volume data
        min_cluster_volume: Minimum volume to qualify as a cluster

    Returns:
        List of cluster data dictionaries
    """
    clusters: list[dict[str, Decimal]] = []

    if not price_volume_data:
        return clusters

    # Sort by volume (descending) to find most significant areas first
    sorted_data = sorted(price_volume_data.items(), key=lambda x: x[1], reverse=True)

    # Find contiguous price ranges with significant volume
    i = 0
    while i < len(sorted_data):
        price, volume = sorted_data[i]

        if volume >= min_cluster_volume:
            cluster_start = i
            cluster_end = i
            cluster_volume = volume

            # Extend cluster while volume remains significant
            for j in range(i + 1, len(sorted_data)):
                next_price, next_volume = sorted_data[j]
                # Check if still in reasonable price range (within $5)
                if abs(next_price - price) <= Decimal(
                    "5"
                ) and next_volume >= min_cluster_volume / Decimal("2"):
                    cluster_end = j
                    cluster_volume += next_volume
                else:
                    break

            if cluster_end - cluster_start >= 1:  # At least 2 price levels
                cluster = {
                    "start_price": sorted_data[cluster_start][0],
                    "end_price": sorted_data[cluster_end][0],
                    "total_volume": cluster_volume,
                    "price_levels": Decimal(str(cluster_end - cluster_start + 1)),
                    "center_price": (
                        sorted_data[cluster_start][0] + sorted_data[cluster_end][0]
                    )
                    / Decimal("2"),
                }
                clusters.append(cluster)

            i = cluster_end + 1
        else:
            i += 1

    logger.debug(f"Identified {len(clusters)} liquidity clusters")
    return clusters


def convert_to_depth_levels(
    aggregated_bids: dict[Decimal, Decimal], aggregated_asks: dict[Decimal, Decimal]
) -> tuple[list[DepthLevel], list[DepthLevel]]:
    """
    Convert aggregated data back to DepthLevel format for compatibility.

    Args:
        aggregated_bids: Aggregated bid data {price: volume}
        aggregated_asks: Aggregated ask data {price: volume}

    Returns:
        Tuple of (bids, asks) as DepthLevel lists
    """
    bid_levels = [
        DepthLevel(price=price, quantity=volume)
        for price, volume in sorted(aggregated_bids.items(), reverse=True)
    ]

    ask_levels = [
        DepthLevel(price=price, quantity=volume)
        for price, volume in sorted(aggregated_asks.items())
    ]

    return bid_levels, ask_levels


def validate_aggregation_quality(
    original_bids: list[DepthLevel],
    original_asks: list[DepthLevel],
    aggregated_bids: dict[Decimal, Decimal],
    aggregated_asks: dict[Decimal, Decimal],
) -> dict[str, Decimal | int]:
    """
    Validate the quality of 1-dollar precision aggregation.

    Args:
        original_bids: Original bid levels before aggregation
        original_asks: Original ask levels before aggregation
        aggregated_bids: Aggregated bid data
        aggregated_asks: Aggregated ask data

    Returns:
        Dictionary with quality metrics:
        - 'volume_preservation': Percentage of volume preserved
        - 'price_levels_reduction': Reduction in price levels
        - 'aggregation_efficiency': Volume consolidation ratio
    """
    # Calculate original volumes
    original_bid_volume = sum(bid.quantity for bid in original_bids)
    original_ask_volume = sum(ask.quantity for ask in original_asks)
    total_original_volume = original_bid_volume + original_ask_volume

    # Calculate aggregated volumes
    aggregated_bid_volume = (
        sum(aggregated_bids.values()) if aggregated_bids else Decimal("0")
    )
    aggregated_ask_volume = (
        sum(aggregated_asks.values()) if aggregated_asks else Decimal("0")
    )
    total_aggregated_volume = aggregated_bid_volume + aggregated_ask_volume

    # Calculate quality metrics
    volume_preservation = (
        total_aggregated_volume / (total_original_volume + Decimal("0.01"))
    ) * Decimal("100")

    original_price_levels = len(original_bids) + len(original_asks)
    aggregated_price_levels = len(aggregated_bids) + len(aggregated_asks)
    price_levels_reduction = original_price_levels - aggregated_price_levels

    # Calculate aggregation efficiency (how much volume was consolidated)
    max_original_volume = max(
        max(bid.quantity for bid in original_bids) if original_bids else Decimal("0"),
        max(ask.quantity for ask in original_asks) if original_asks else Decimal("0"),
    )

    if max_original_volume > Decimal("0"):
        aggregation_efficiency = (
            total_original_volume
            / (original_price_levels * max_original_volume + Decimal("0.01"))
        ) * Decimal("100")
    else:
        aggregation_efficiency = Decimal("0")

    quality_metrics = {
        "volume_preservation": volume_preservation,
        "price_levels_reduction": Decimal(str(price_levels_reduction)),
        "aggregation_efficiency": aggregation_efficiency,
    }

    logger.info(f"Aggregation quality: {quality_metrics}")
    return quality_metrics


class PriceAggregator:
    """Configurable price aggregation for order book analysis.

    This class provides flexible price aggregation with configurable precision
    and validation, supporting various aggregation strategies.
    """

    def __init__(
        self, precision: float = 1.0, enabled: bool = True, max_price_levels: int = 5000
    ):
        """Initialize the price aggregator.

        Args:
            precision: Aggregation precision (e.g., 1.0 for $1, 0.1 for $0.1)
            enabled: Whether aggregation is enabled
            max_price_levels: Maximum number of price levels after aggregation
        """
        if precision <= 0:
            raise ValueError(f"Precision must be positive, got {precision}")
        if max_price_levels <= 0:
            raise ValueError(
                f"Max price levels must be positive, got {max_price_levels}"
            )

        self.precision = Decimal(str(precision))
        self.enabled = enabled
        self.max_price_levels = max_price_levels

        logger.info(
            f"Initialized PriceAggregator with precision={precision}, "
            f"enabled={enabled}, max_levels={max_price_levels}"
        )

    def aggregate_order_book_levels(
        self, bids: list[DepthLevel], asks: list[DepthLevel]
    ) -> tuple[dict[Decimal, Decimal], dict[Decimal, Decimal]]:
        """Aggregate order book levels using configured precision.

        Args:
            bids: List of bid levels
            asks: List of ask levels

        Returns:
            Tuple of (aggregated_bids, aggregated_asks) dictionaries
        """
        if not self.enabled:
            # Return original data as dict if aggregation is disabled
            logger.debug("Aggregation disabled, returning original data")
            return (
                {bid.price: bid.quantity for bid in bids},
                {ask.price: ask.quantity for ask in asks},
            )

        logger.debug(
            f"Aggregating {len(bids)} bid levels and {len(asks)} ask levels "
            f"with precision {self.precision}"
        )

        aggregated_bids: dict[Decimal, Decimal] = defaultdict(Decimal)
        aggregated_asks: dict[Decimal, Decimal] = defaultdict(Decimal)

        # Aggregate bids
        for bid_level in bids:
            if self._is_valid_level(bid_level):
                rounded_price = self._round_down_to_precision(bid_level.price)
                aggregated_bids[rounded_price] += bid_level.quantity

        # Aggregate asks
        for ask_level in asks:
            if self._is_valid_level(ask_level):
                rounded_price = self._round_down_to_precision(ask_level.price)
                aggregated_asks[rounded_price] += ask_level.quantity

        # Apply max price levels limit
        aggregated_bids = self._limit_price_levels(aggregated_bids, is_bids=True)
        aggregated_asks = self._limit_price_levels(aggregated_asks, is_bids=False)

        logger.debug(
            f"Aggregated to {len(aggregated_bids)} bid levels, "
            f"{len(aggregated_asks)} ask levels"
        )

        return dict(aggregated_bids), dict(aggregated_asks)

    def _is_valid_level(self, level: DepthLevel) -> bool:
        """Validate if a depth level is suitable for aggregation.

        Args:
            level: DepthLevel to validate

        Returns:
            True if valid, False otherwise
        """
        if not isinstance(level.price, Decimal) or not isinstance(
            level.quantity, Decimal
        ):
            return False

        if level.price <= 0 or level.quantity <= 0:
            return False

        return True

    def _round_down_to_precision(self, price: Decimal) -> Decimal:
        """Round price down to the configured precision.

        Args:
            price: Original price

        Returns:
            Price rounded down to configured precision
        """
        if not isinstance(price, Decimal):
            raise TypeError(f"Expected Decimal, got {type(price)}")

        # Round down to the nearest precision unit
        # For example: precision=1.0, price=50001.50 -> 50001.00
        #              precision=0.1, price=50001.53 -> 50001.50
        rounded = (price // self.precision) * self.precision
        return rounded

    def _limit_price_levels(
        self, price_data: dict[Decimal, Decimal], is_bids: bool
    ) -> dict[Decimal, Decimal]:
        """Limit the number of price levels based on configuration.

        Args:
            price_data: Price-volume data
            is_bids: Whether this is bid data (affects sorting)

        Returns:
            Limited price data dictionary
        """
        if len(price_data) <= self.max_price_levels:
            return price_data

        # Sort and limit
        if is_bids:
            # For bids, sort by price descending (keep highest prices)
            sorted_items = sorted(price_data.items(), key=lambda x: x[0], reverse=True)
        else:
            # For asks, sort by price ascending (keep lowest prices)
            sorted_items = sorted(price_data.items(), key=lambda x: x[0])

        limited_items = sorted_items[: self.max_price_levels]
        logger.debug(
            f"Limited {'bids' if is_bids else 'asks'} from {len(price_data)} "
            f"to {len(limited_items)} levels"
        )

        return dict(limited_items)

    def get_aggregation_stats(
        self,
        original_bids: list[DepthLevel],
        original_asks: list[DepthLevel],
        aggregated_bids: dict[Decimal, Decimal],
        aggregated_asks: dict[Decimal, Decimal],
    ) -> dict[str, Decimal | int]:
        """Get statistics about the aggregation process.

        Args:
            original_bids: Original bid levels
            original_asks: Original ask levels
            aggregated_bids: Aggregated bid data
            aggregated_asks: Aggregated ask data

        Returns:
            Dictionary with aggregation statistics
        """
        original_count = len(original_bids) + len(original_asks)
        aggregated_count = len(aggregated_bids) + len(aggregated_asks)

        # Calculate volume preservation
        original_volume = sum(bid.quantity for bid in original_bids) + sum(
            ask.quantity for ask in original_asks
        )
        aggregated_volume = sum(aggregated_bids.values()) + sum(
            aggregated_asks.values()
        )

        volume_preservation = (
            aggregated_volume / (original_volume + Decimal("0.01"))
        ) * 100

        return {
            "original_levels": original_count,
            "aggregated_levels": aggregated_count,
            "reduction_ratio": original_count / (aggregated_count + 1),
            "original_volume": float(original_volume),
            "aggregated_volume": float(aggregated_volume),
            "volume_preservation_percent": float(volume_preservation),
            "precision": float(self.precision),
            "enabled": self.enabled,
            "max_levels": self.max_price_levels,
        }
