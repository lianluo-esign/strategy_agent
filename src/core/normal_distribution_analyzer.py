"""Normal distribution peak analyzer for order book analysis.

This module provides statistical analysis methods to identify significant
price levels in order book data using normal distribution confidence intervals.
"""

import logging
import math
from collections import defaultdict
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


class OrderBookAggregator:
    """Aggregates order book data to specified price precision."""

    def __init__(self, price_precision: float = 1.0):
        """Initialize the aggregator with price precision.

        Args:
            price_precision: The precision level for price aggregation (default: 1.0)
        """
        self.price_precision = price_precision

    def aggregate_to_dollar_precision(
        self,
        order_book_data: Dict[str, List[Tuple[float, float]]]
    ) -> Tuple[Dict[float, float], Dict[float, float]]:
        """Aggregate order book data to 1-dollar precision (floor rounding).

        Args:
            order_book_data: Dictionary containing 'bids' and 'asks' lists

        Returns:
            Tuple of (aggregated_bids, aggregated_asks) dictionaries
        """
        aggregated_bids: Dict[float, float] = defaultdict(float)
        aggregated_asks: Dict[float, float] = defaultdict(float)

        # Process bids (buy orders)
        for price, quantity in order_book_data.get('bids', []):
            rounded_price = self._round_to_dollar(price)
            aggregated_bids[rounded_price] += quantity

        # Process asks (sell orders)
        for price, quantity in order_book_data.get('asks', []):
            rounded_price = self._round_to_dollar(price)
            aggregated_asks[rounded_price] += quantity

        return dict(aggregated_bids), dict(aggregated_asks)

    def _round_to_dollar(self, price: float) -> float:
        """Round price down to 1-dollar precision using floor operation.

        Args:
            price: The original price

        Returns:
            Price rounded down to 1-dollar precision
        """
        return math.floor(price / self.price_precision) * self.price_precision


class NormalDistributionAnalyzer:
    """Analyzes order book data using normal distribution statistical methods."""

    def __init__(self, confidence_level: float = 0.95):
        """Initialize the analyzer with confidence level.

        Args:
            confidence_level: Statistical confidence level (default: 0.95)
        """
        self.confidence_level = confidence_level
        self._z_score = self._calculate_z_score(confidence_level)

    def _calculate_z_score(self, confidence_level: float) -> float:
        """Calculate Z-score for the given confidence level.

        Args:
            confidence_level: Confidence level between 0 and 1

        Returns:
            Z-score for the confidence level
        """
        # For 95% confidence, Z-score is approximately 1.96
        # This is a simplified calculation - in production, consider using scipy.stats.norm.ppf
        if confidence_level == 0.95:
            return 1.96
        elif confidence_level == 0.90:
            return 1.645
        elif confidence_level == 0.99:
            return 2.576
        else:
            # Approximation for other levels
            return abs(4.0 * (confidence_level - 0.5))

    def find_peak_interval(
        self,
        price_quantities: Dict[float, float]
    ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """Find the peak interval using normal distribution analysis.

        Args:
            price_quantities: Dictionary of price to quantity mapping

        Returns:
            Tuple of (mean_price, lower_bound, upper_bound) or (None, None, None)
        """
        if not price_quantities:
            return None, None, None

        prices = list(price_quantities.keys())
        quantities = list(price_quantities.values())

        # Calculate weighted mean and standard deviation
        total_quantity = sum(quantities)
        if total_quantity == 0:
            return None, None, None

        mean_price = sum(price * quantity for price, quantity in zip(prices, quantities)) / total_quantity

        # Calculate weighted standard deviation
        variance = sum(
            quantity * ((price - mean_price) ** 2)
            for price, quantity in zip(prices, quantities)
        ) / total_quantity

        std_price = math.sqrt(variance)

        if std_price == 0:
            return mean_price, mean_price, mean_price

        # Calculate confidence interval
        margin_of_error = self._z_score * std_price
        lower_bound = mean_price - margin_of_error
        upper_bound = mean_price + margin_of_error

        return mean_price, lower_bound, upper_bound

    def analyze_distribution_peaks(
        self,
        aggregated_bids: Dict[float, float],
        aggregated_asks: Dict[float, float]
    ) -> Dict[str, Dict[str, Any]]:
        """Analyze distribution peaks for both bids and asks.

        Args:
            aggregated_bids: Aggregated bid data
            aggregated_asks: Aggregated ask data

        Returns:
            Dictionary containing peak analysis results
        """
        results: Dict[str, Dict[str, Any]] = {}

        # Analyze bid distribution
        if aggregated_bids:
            bid_mean, bid_lower, bid_upper = self.find_peak_interval(aggregated_bids)
            results['bids'] = {
                'mean_price': bid_mean,
                'peak_interval': (bid_lower, bid_upper) if bid_lower is not None else None,
                'total_volume': sum(aggregated_bids.values()),
                'peak_volume': self._get_volume_in_interval(
                    aggregated_bids, bid_lower, bid_upper
                ) if bid_lower is not None else 0,
                'z_score': self._z_score,
                'confidence_level': self.confidence_level
            }

        # Analyze ask distribution
        if aggregated_asks:
            ask_mean, ask_lower, ask_upper = self.find_peak_interval(aggregated_asks)
            results['asks'] = {
                'mean_price': ask_mean,
                'peak_interval': (ask_lower, ask_upper) if ask_lower is not None else None,
                'total_volume': sum(aggregated_asks.values()),
                'peak_volume': self._get_volume_in_interval(
                    aggregated_asks, ask_lower, ask_upper
                ) if ask_lower is not None else 0,
                'z_score': self._z_score,
                'confidence_level': self.confidence_level
            }

        return results

    def _get_volume_in_interval(
        self,
        price_quantities: Dict[float, float],
        lower: Optional[float],
        upper: Optional[float]
    ) -> float:
        """Get total volume within the specified price interval.

        Args:
            price_quantities: Dictionary of price to quantity
            lower: Lower bound of interval
            upper: Upper bound of interval

        Returns:
            Total volume within the interval
        """
        if lower is None or upper is None:
            return 0

        total_volume = 0.0
        for price, quantity in price_quantities.items():
            if lower <= price <= upper:
                total_volume += quantity
        return total_volume


class NormalDistributionPeakAnalyzer:
    """Main analyzer combining order book aggregation and normal distribution analysis."""

    def __init__(self, price_precision: float = 1.0, confidence_level: float = 0.95):
        """Initialize the peak analyzer.

        Args:
            price_precision: Price precision for aggregation (default: 1.0)
            confidence_level: Statistical confidence level (default: 0.95)
        """
        self.aggregator = OrderBookAggregator(price_precision)
        self.distribution_analyzer = NormalDistributionAnalyzer(confidence_level)

    def analyze_order_book(
        self,
        order_book_data: Dict[str, List[Tuple[float, float]]]
    ) -> Dict[str, Any]:
        """Perform complete order book analysis using normal distribution methods.

        Args:
            order_book_data: Raw order book data with bids and asks

        Returns:
            Dictionary containing complete analysis results
        """
        try:
            # Step 1: Aggregate to 1-dollar precision (floor rounding)
            aggregated_bids, aggregated_asks = self.aggregator.aggregate_to_dollar_precision(
                order_book_data
            )

            logger.debug(
                f"Aggregated {len(order_book_data.get('bids', []))} bids to "
                f"{len(aggregated_bids)} levels, {len(order_book_data.get('asks', []))} asks to "
                f"{len(aggregated_asks)} levels"
            )

            # Step 2: Analyze normal distribution peak intervals
            peak_analysis = self.distribution_analyzer.analyze_distribution_peaks(
                aggregated_bids, aggregated_asks
            )

            # Step 3: Extract key information
            analysis_result = {
                'aggregated_bids': aggregated_bids,
                'aggregated_asks': aggregated_asks,
                'peak_analysis': peak_analysis,
                'spread_analysis': self._analyze_spread(aggregated_bids, aggregated_asks),
                'market_metrics': self._calculate_market_metrics(aggregated_bids, aggregated_asks)
            }

            logger.info(
                f"Normal distribution analysis completed: "
                f"bid_peak={peak_analysis.get('bids', {}).get('mean_price')}, "
                f"ask_peak={peak_analysis.get('asks', {}).get('mean_price')}"
            )

            return analysis_result

        except Exception as e:
            logger.error(f"Error in normal distribution analysis: {e}")
            return {
                'aggregated_bids': {},
                'aggregated_asks': {},
                'peak_analysis': {},
                'spread_analysis': {},
                'market_metrics': {},
                'error': str(e)
            }

    def _analyze_spread(
        self,
        bids: Dict[float, float],
        asks: Dict[float, float]
    ) -> Dict[str, Any]:
        """Analyze bid-ask spread.

        Args:
            bids: Aggregated bid data
            asks: Aggregated ask data

        Returns:
            Dictionary containing spread analysis
        """
        if not bids or not asks:
            return {}

        best_bid = max(bids.keys())
        best_ask = min(asks.keys())
        spread = best_ask - best_bid

        return {
            'best_bid': best_bid,
            'best_ask': best_ask,
            'spread': spread,
            'mid_price': (best_bid + best_ask) / 2,
            'spread_percentage': (spread / best_ask) * 100 if best_ask > 0 else 0
        }

    def _calculate_market_metrics(
        self,
        bids: Dict[float, float],
        asks: Dict[float, float]
    ) -> Dict[str, Any]:
        """Calculate additional market metrics.

        Args:
            bids: Aggregated bid data
            asks: Aggregated ask data

        Returns:
            Dictionary containing market metrics
        """
        bid_volume = sum(bids.values())
        ask_volume = sum(asks.values())
        total_volume = bid_volume + ask_volume

        return {
            'total_bid_volume': bid_volume,
            'total_ask_volume': ask_volume,
            'total_volume': total_volume,
            'bid_ask_ratio': bid_volume / ask_volume if ask_volume > 0 else float('inf'),
            'volume_imbalance': (bid_volume - ask_volume) / total_volume if total_volume > 0 else 0,
            'price_levels_count': {
                'bid_levels': len(bids),
                'ask_levels': len(asks),
                'total_levels': len(bids) + len(asks)
            }
        }


def _convert_aggregated_data(aggregated_data: Dict[str, float]) -> Dict[Decimal, Decimal]:
    """Convert aggregated data to Decimal format.

    Args:
        aggregated_data: Dictionary with float values

    Returns:
        Dictionary with Decimal values
    """
    return {
        Decimal(str(price)): Decimal(str(quantity))
        for price, quantity in aggregated_data.items()
    }


def _convert_spread_analysis(spread_analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Convert spread analysis to Decimal format.

    Args:
        spread_analysis: Spread analysis data

    Returns:
        Dictionary with mixed Decimal and float values
    """
    if not spread_analysis:
        return {}

    return {
        'best_bid': Decimal(str(spread_analysis.get('best_bid', 0))),
        'best_ask': Decimal(str(spread_analysis.get('best_ask', 0))),
        'spread': Decimal(str(spread_analysis.get('spread', 0))),
        'mid_price': Decimal(str(spread_analysis.get('mid_price', 0))),
        'spread_percentage': float(spread_analysis.get('spread_percentage', 0))
    }


def _convert_peak_analysis(peak_analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Convert peak analysis to Decimal format.

    Args:
        peak_analysis: Peak analysis data

    Returns:
        Dictionary with Decimal values
    """
    if not peak_analysis:
        return {}

    decimal_peak_analysis = {}

    for side in ['bids', 'asks']:
        if side not in peak_analysis:
            continue

        side_data = peak_analysis[side].copy()

        # Convert mean price
        if side_data.get('mean_price') is not None:
            side_data['mean_price'] = Decimal(str(side_data['mean_price']))

        # Convert peak interval
        if side_data.get('peak_interval'):
            lower, upper = side_data['peak_interval']
            side_data['peak_interval'] = (
                Decimal(str(lower)) if lower is not None else None,
                Decimal(str(upper)) if upper is not None else None
            )

        # Convert volumes
        side_data['total_volume'] = Decimal(str(side_data.get('total_volume', 0)))
        side_data['peak_volume'] = Decimal(str(side_data.get('peak_volume', 0)))

        decimal_peak_analysis[side] = side_data

    return decimal_peak_analysis


def convert_to_decimal_format(analysis_result: Dict[str, Any]) -> Dict[str, Any]:
    """Convert analysis results to use Decimal objects for precision.

    Args:
        analysis_result: Analysis result with float values

    Returns:
        Analysis result with Decimal values for key fields
    """
    if 'error' in analysis_result:
        return analysis_result

    decimal_result = analysis_result.copy()

    # Ensure all required fields exist
    if 'aggregated_bids' not in decimal_result:
        decimal_result['aggregated_bids'] = {}
    if 'aggregated_asks' not in decimal_result:
        decimal_result['aggregated_asks'] = {}
    if 'spread_analysis' not in decimal_result:
        decimal_result['spread_analysis'] = {}
    if 'peak_analysis' not in decimal_result:
        decimal_result['peak_analysis'] = {}

    # Convert each component using helper functions
    decimal_result['aggregated_bids'] = _convert_aggregated_data(
        analysis_result.get('aggregated_bids', {})
    )

    decimal_result['aggregated_asks'] = _convert_aggregated_data(
        analysis_result.get('aggregated_asks', {})
    )

    decimal_result['spread_analysis'] = _convert_spread_analysis(
        analysis_result.get('spread_analysis', {})
    )

    decimal_result['peak_analysis'] = _convert_peak_analysis(
        analysis_result.get('peak_analysis', {})
    )

    return decimal_result