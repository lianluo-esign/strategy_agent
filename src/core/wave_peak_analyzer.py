"""Wave peak detection algorithms based on normal distribution analysis.

This module provides sophisticated wave peak detection using statistical
methods to identify significant price levels in order book data.
"""

import logging
import math
from collections import defaultdict
from decimal import Decimal
from typing import Dict, List, Optional, Tuple

from .models import DepthLevel

logger = logging.getLogger(__name__)


class WavePeak:
    """Represents a wave peak detected in order book analysis."""

    def __init__(
        self,
        center_price: Decimal,
        volume: Decimal,
        price_range_width: Decimal,
        z_score: float,
        confidence: float,
        bid_volume: Decimal = Decimal('0'),
        ask_volume: Decimal = Decimal('0'),
        peak_type: str = 'unknown'
    ):
        """Initialize wave peak information."""
        self.center_price = center_price
        self.volume = volume
        self.price_range_width = price_range_width
        self.z_score = z_score
        self.confidence = confidence
        self.bid_volume = bid_volume
        self.ask_volume = ask_volume
        self.peak_type = peak_type

    def __repr__(self) -> str:
        """String representation of wave peak."""
        return (
            f"WavePeak(center={self.center_price}, volume={self.volume}, "
            f"z_score={self.z_score:.2f}, confidence={self.confidence:.2f}, "
            f"type={self.peak_type})"
        )

    def to_dict(self) -> dict:
        """Convert wave peak to dictionary representation."""
        return {
            'center_price': float(self.center_price),
            'volume': float(self.volume),
            'price_range_width': float(self.price_range_width),
            'z_score': self.z_score,
            'confidence': self.confidence,
            'bid_volume': float(self.bid_volume),
            'ask_volume': float(self.ask_volume),
            'peak_type': self.peak_type,
            'lower_price': float(self.center_price - self.price_range_width / 2),
            'upper_price': float(self.center_price + self.price_range_width / 2),
        }


class PriceZone:
    """Represents a price zone with trading characteristics."""

    def __init__(
        self,
        lower_price: Decimal,
        upper_price: Decimal,
        zone_type: str,
        confidence: float,
        total_volume: Decimal,
        bid_ask_ratio: float = 1.0
    ):
        """Initialize price zone information."""
        self.lower_price = lower_price
        self.upper_price = upper_price
        self.zone_type = zone_type  # 'support' or 'resistance'
        self.confidence = confidence
        self.total_volume = total_volume
        self.bid_ask_ratio = bid_ask_ratio
        self.center_price = (lower_price + upper_price) / Decimal('2')
        self.width = upper_price - lower_price

    def __repr__(self) -> str:
        """String representation of price zone."""
        return (
            f"PriceZone({self.zone_type}: {self.lower_price}-{self.upper_price}, "
            f"volume={self.total_volume}, confidence={self.confidence:.2f})"
        )


def detect_normal_distribution_peaks(
    price_volume_data: Dict[Decimal, Decimal],
    min_peak_volume: Decimal = Decimal('5.0'),
    z_score_threshold: float = 1.5,
    min_peak_confidence: float = 0.3
) -> List[WavePeak]:
    """
    Detect wave peaks using normal distribution analysis.

    This function identifies significant price levels where order volume
    concentrates beyond random distribution expectations.

    Args:
        price_volume_data: Dictionary of price to volume (1-dollar precision aggregated)
        min_peak_volume: Minimum volume to qualify as a peak
        z_score_threshold: Z-score threshold for peak significance (default 1.5σ)
        min_peak_confidence: Minimum confidence score for valid peaks

    Returns:
        List of WavePeak objects ordered by volume (descending)
    """
    if not price_volume_data:
        logger.warning("Empty price volume data provided to peak detection")
        return []

    logger.info(f"Detecting peaks in {len(price_volume_data)} price levels")

    # Extract prices and volumes for analysis
    prices = []
    volumes = []
    for price, volume in price_volume_data.items():
        if volume >= min_peak_volume:
            prices.append(float(price))
            volumes.append(float(volume))

    if len(prices) < 3:
        logger.debug("Insufficient data points for statistical peak detection")
        return []

    try:
        # Calculate weighted statistics
        mean_price, std_price = _calculate_weighted_statistics(prices, volumes)

        if std_price == 0:
            logger.debug("Zero standard deviation - no variance in price distribution")
            return []

        logger.debug(f"Price distribution: mean={mean_price:.2f}, std={std_price:.2f}")

        peaks = []

        # Detect peaks using local maxima and statistical significance
        for i in range(1, len(prices) - 1):
            current_price = Decimal(str(prices[i]))
            current_volume = Decimal(str(volumes[i]))

            prev_price = Decimal(str(prices[i-1]))
            prev_volume = Decimal(str(volumes[i-1]))
            next_price = Decimal(str(prices[i+1]))
            next_volume = Decimal(str(volumes[i+1]))

            # Check if current point is a local maximum
            is_local_peak = current_volume > prev_volume and current_volume > next_volume

            # Calculate Z-score for statistical significance
            z_score = abs(float(current_price - Decimal(str(mean_price))) / std_price) if std_price > 0 else 0

            # Calculate confidence based on Z-score
            confidence = min(z_score / z_score_threshold, 1.0) if z_score_threshold > 0 else 1.0
            confidence = max(confidence, min_peak_confidence)

            if is_local_peak and confidence >= min_peak_confidence:
                # Calculate price range width based on volume distribution
                price_range_width = Decimal(str(2 * std_price))  # ±1σ range

                peak = WavePeak(
                    center_price=current_price,
                    volume=current_volume,
                    price_range_width=price_range_width,
                    z_score=z_score,
                    confidence=confidence,
                    peak_type='statistical_peak'
                )
                peaks.append(peak)
                logger.debug(f"Peak detected: {peak}")

        # Sort peaks by volume (descending)
        peaks.sort(key=lambda x: x.volume, reverse=True)

        logger.info(f"Detected {len(peaks)} statistical peaks")
        return peaks

    except Exception as e:
        logger.error(f"Error in peak detection: {e}")
        return []


def detect_volume_based_peaks(
    price_volume_data: Dict[Decimal, Decimal],
    min_relative_volume: float = 2.0,
    min_absolume: Decimal = Decimal('10.0')
) -> List[WavePeak]:
    """
    Detect peaks based on volume concentration analysis.

    This is a complementary method that identifies peaks where volume
    is significantly higher than surrounding price levels.

    Args:
        price_volume_data: Dictionary of price to volume
        min_relative_volume: Minimum volume ratio relative to neighbors
        min_absolume: Minimum absolute volume to qualify

    Returns:
        List of WavePeak objects ordered by volume
    """
    if not price_volume_data:
        return []

    peaks = []
    sorted_items = sorted(price_volume_data.items(), key=lambda x: x[0])

    for i in range(1, len(sorted_items) - 1):
        current_price, current_volume = sorted_items[i]
        prev_price, prev_volume = sorted_items[i-1]
        next_price, next_volume = sorted_items[i+1]

        # Check volume significance
        avg_neighbor_volume = (prev_volume + next_volume) / 2
        relative_volume_ratio = float(current_volume / avg_neighbor_volume) if avg_neighbor_volume > 0 else 0

        if current_volume >= min_absolume and relative_volume_ratio >= min_relative_volume:
            # Estimate price range width based on local density
            price_spread = abs(next_price - prev_price) / 2 if i < len(sorted_items) - 1 else abs(current_price - prev_price)
            price_range_width = Decimal(str(price_spread * 4))  # Estimate range width

            # Calculate confidence based on volume significance
            confidence = min(relative_volume_ratio / 5.0, 1.0)  # Scale to 0-1 range

            peak = WavePeak(
                center_price=current_price,
                volume=current_volume,
                price_range_width=price_range_width,
                z_score=0.0,  # Not calculated for volume-based detection
                confidence=confidence,
                peak_type='volume_concentration'
            )
            peaks.append(peak)

    peaks.sort(key=lambda x: x.volume, reverse=True)
    logger.info(f"Detected {len(peaks)} volume-based peaks")
    return peaks


def analyze_wave_formation(
    peaks: List[WavePeak],
    min_peak_distance: Decimal = Decimal('5.0'),
    max_price_range: Decimal = Decimal('50.0')
) -> List[PriceZone]:
    """
    Analyze wave peaks to identify price zones and wave formation.

    Args:
        peaks: List of detected wave peaks
        min_peak_distance: Minimum price distance between significant peaks
        max_price_range: Maximum price range for a single zone

    Returns:
        List of PriceZone objects representing significant price areas
    """
    if len(peaks) < 2:
        logger.debug("Insufficient peaks for wave formation analysis")
        return []

    zones = []
    sorted_peaks = sorted(peaks, key=lambda x: x.center_price)

    # Group nearby peaks into zones
    i = 0
    while i < len(sorted_peaks):
        current_peak = sorted_peaks[i]
        zone_peaks = [current_peak]
        zone_volume = current_peak.volume

        # Find peaks within reasonable distance for zone formation
        for j in range(i + 1, len(sorted_peaks)):
            next_peak = sorted_peaks[j]

            # Check if peaks are close enough to form a zone
            distance = abs(next_peak.center_price - current_peak.center_price)

            if distance <= min_peak_distance and distance <= max_price_range:
                zone_peaks.append(next_peak)
                zone_volume += next_peak.volume
                i = j  # Skip to next peak after grouping
            else:
                break

        # Create price zone from grouped peaks
        if len(zone_peaks) >= 2:
            min_price = min(peak.center_price for peak in zone_peaks)
            max_price = max(peak.center_price for peak in zone_peaks)
            avg_confidence = sum(peak.confidence for peak in zone_peaks) / len(zone_peaks)
            avg_volume = zone_volume / len(zone_peaks)

            # Determine zone type based on volume distribution
            bid_volume = sum(peak.bid_volume for peak in zone_peaks)
            ask_volume = sum(peak.ask_volume for peak in zone_peaks)
            bid_ask_ratio = float(bid_volume / (ask_volume + Decimal('0.01'))) if ask_volume > 0 else 1.0

            zone_type = 'resistance' if ask_volume > bid_volume else 'support'

            zone = PriceZone(
                lower_price=min_price,
                upper_price=max_price,
                zone_type=zone_type,
                confidence=avg_confidence,
                total_volume=avg_volume,
                bid_ask_ratio=bid_ask_ratio
            )
            zones.append(zone)

            logger.debug(f"Created {zone_type} zone: {zone.lower_price}-{zone.upper_price}")

        i += 1

    # Sort zones by confidence (descending)
    zones.sort(key=lambda x: x.confidence, reverse=True)

    logger.info(f"Identified {len(zones)} price zones from wave formation")
    return zones


def _calculate_weighted_statistics(prices: List[float], volumes: List[float]) -> Tuple[float, float]:
    """
    Calculate weighted mean and standard deviation.

    Args:
        prices: List of price values
        volumes: List of corresponding volumes (weights)

    Returns:
        Tuple of (weighted_mean, weighted_std)
    """
    if not prices or not volumes or len(prices) != len(volumes):
        logger.warning("Invalid inputs for weighted statistics calculation")
        return 0.0, 0.0

    total_volume = sum(volumes)
    if total_volume == 0:
        return 0.0, 0.0

    # Calculate weighted mean
    weighted_mean = sum(p * v for p, v in zip(prices, volumes)) / total_volume

    # Calculate weighted variance and standard deviation
    weighted_variance = sum(v * ((p - weighted_mean) ** 2) for p, v in zip(prices, volumes)) / total_volume
    weighted_std = math.sqrt(weighted_variance)

    return weighted_mean, weighted_std


def detect_combined_peaks(
    price_volume_data: Dict[Decimal, Decimal],
    statistical_params: Optional[Dict] = None,
    volume_params: Optional[Dict] = None
) -> List[WavePeak]:
    """
    Detect peaks using combined statistical and volume-based methods.

    Args:
        price_volume_data: Dictionary of price to volume
        statistical_params: Parameters for statistical peak detection
        volume_params: Parameters for volume-based peak detection

    Returns:
        List of WavePeak objects from combined detection
    """
    # Use default parameters if not provided
    if statistical_params is None:
        statistical_params = {
            'min_peak_volume': 5.0,
            'z_score_threshold': 1.5,
            'min_peak_confidence': 0.3
        }

    if volume_params is None:
        volume_params = {
            'min_relative_volume': 2.0,
            'min_absolume': 10.0
        }

    # Detect peaks using both methods
    statistical_peaks = detect_normal_distribution_peaks(price_volume_data, **statistical_params)
    volume_peaks = detect_volume_based_peaks(price_volume_data, **volume_params)

    # Combine and deduplicate peaks (within 2% price range)
    all_peaks = []
    used_prices = set()

    for peak in statistical_peaks + volume_peaks:
        peak_price_key = f"{peak.center_price:.2f}"
        if peak_price_key not in used_prices:
            all_peaks.append(peak)
            used_prices.add(peak_price_key)

    # Sort by volume and confidence
    all_peaks.sort(key=lambda x: (x.volume, x.confidence), reverse=True)

    logger.info(f"Combined peak detection: {len(statistical_peaks)} statistical, {len(volume_peaks)} volume-based, {len(all_peaks)} unique peaks")

    return all_peaks


def validate_peak_detection_quality(
    original_levels: int,
    detected_peaks: List[WavePeak],
    price_volume_data: Dict[Decimal, Decimal]
) -> Dict[str, float]:
    """
    Validate the quality of peak detection results.

    Args:
        original_levels: Number of original price levels
        detected_peaks: List of detected peaks
        price_volume_data: Original price-volume data

    Returns:
        Dictionary with quality metrics
    """
    total_original_volume = sum(price_volume_data.values())
    total_peak_volume = sum(peak.volume for peak in detected_peaks)

    # Calculate preservation rate
    volume_preservation_rate = float(total_peak_volume / (total_original_volume + Decimal('0.01'))) if total_original_volume > 0 else 0

    # Calculate compression ratio
    compression_ratio = original_levels / len(detected_peaks) if detected_peaks else 0

    # Calculate average confidence
    avg_confidence = sum(peak.confidence for peak in detected_peaks) / len(detected_peaks) if detected_peaks else 0

    # Calculate coverage (how many significant levels were captured)
    significant_volume_levels = len([v for v in price_volume_data.values() if v >= 5.0])
    coverage_rate = len(detected_peaks) / significant_volume_levels if significant_volume_levels > 0 else 0

    quality_metrics = {
        'volume_preservation_rate': volume_preservation_rate,
        'compression_ratio': compression_ratio,
        'avg_confidence': avg_confidence,
        'coverage_rate': coverage_rate,
        'peak_count': len(detected_peaks),
        'significant_levels_count': significant_volume_levels,
    }

    logger.info(f"Peak detection quality: {quality_metrics}")
    return quality_metrics