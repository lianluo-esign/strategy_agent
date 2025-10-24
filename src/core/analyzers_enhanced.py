"""Enhanced market analyzers with wave peak detection and 1-dollar precision aggregation.

This module provides sophisticated market analysis using statistical methods
to identify significant price levels and wave patterns.
"""

import logging
from collections import defaultdict
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)

from .models import (
    DepthSnapshot,
    EnhancedMarketAnalysisResult,
    MarketAnalysisResult,
    MinuteTradeData,
    SupportResistanceLevel,
)

# Import new modules
try:
    from .price_aggregator import (
        aggregate_depth_by_one_dollar,
        calculate_depth_statistics,
        identify_liquidity_clusters,
        convert_to_depth_levels,
        validate_aggregation_quality,
    )
    from .wave_peak_analyzer import (
        detect_combined_peaks,
        analyze_wave_formation,
        validate_peak_detection_quality,
        WavePeak,
        PriceZone,
    )
except ImportError as e:
    logger.warning(f"Could not import enhanced analyzer modules: {e}")
    # Fallback imports
    from .price_aggregator import (
        aggregate_depth_by_one_dollar,
        calculate_depth_statistics,
        identify_liquidity_clusters,
        convert_to_depth_levels,
        validate_aggregation_quality,
    )
    from .wave_peak_analyzer import (
        detect_combined_peaks,
        analyze_wave_formation,
        validate_peak_detection_quality,
        WavePeak,
        PriceZone,
    )


class EnhancedMarketAnalyzer:
    """
    Enhanced market analyzer with 1-dollar precision aggregation and wave peak detection.

    This analyzer combines traditional support/resistance analysis with sophisticated
    statistical methods to identify wave patterns and significant price levels.
    """

    def __init__(self, min_volume_threshold: Decimal = Decimal('1.0'), analysis_window_minutes: int = 180):
        """
        Initialize the enhanced market analyzer.

        Args:
            min_volume_threshold: Minimum volume threshold for analysis
            analysis_window_minutes: Analysis window in minutes
        """
        self.min_volume_threshold = min_volume_threshold
        self.analysis_window_minutes = analysis_window_minutes
        logger.info(f"Initialized EnhancedMarketAnalyzer with min_volume={min_volume_threshold}, window={analysis_window_minutes}min")

    def analyze_market(
        self,
        snapshot: DepthSnapshot,
        trade_data_list: List[MinuteTradeData],
        symbol: str,
        enhanced_mode: bool = True
    ) -> MarketAnalysisResult | EnhancedMarketAnalysisResult:
        """
        Perform comprehensive market analysis.

        Args:
            snapshot: Current depth snapshot
            trade_data_list: List of trade data
            symbol: Trading symbol
            enhanced_mode: Whether to return enhanced results

        Returns:
            MarketAnalysisResult or EnhancedMarketAnalysisResult based on mode
        """
        try:
            if enhanced_mode:
                return self._analyze_enhanced(snapshot, trade_data_list, symbol)
            else:
                return self._analyze_legacy(snapshot, trade_data_list, symbol)
        except Exception as e:
            logger.error(f"Market analysis failed: {e}")
            # Return empty result on error
            if enhanced_mode:
                return EnhancedMarketAnalysisResult(
                    timestamp=datetime.now(),
                    symbol=symbol,
                    aggregated_bids={},
                    aggregated_asks={},
                    wave_peaks=[],
                    support_zones=[],
                    resistance_zones=[],
                )
            else:
                return MarketAnalysisResult(
                    timestamp=datetime.now(),
                    symbol=symbol,
                )

    def _analyze_enhanced(
        self,
        snapshot: DepthSnapshot,
        trade_data_list: List[MinuteTradeData],
        symbol: str
    ) -> EnhancedMarketAnalysisResult:
        """
        Perform enhanced analysis with wave peak detection.

        Args:
            snapshot: Depth snapshot
            trade_data_list: Trade data list
            symbol: Trading symbol

        Returns:
            EnhancedMarketAnalysisResult with comprehensive analysis
        """
        logger.info(f"Starting enhanced market analysis for {symbol}")

        # Step 1: Aggregate depth data by 1-dollar precision
        aggregated_bids, aggregated_asks = self._aggregate_depth_snapshot(snapshot)

        # Step 2: Aggregate trade data
        aggregated_trades = self._aggregate_trade_data(trade_data_list)

        # Step 3: Detect wave peaks using combined methods
        wave_peaks = self._detect_wave_peaks(aggregated_trades)

        # Step 4: Analyze price formation
        support_zones, resistance_zones = self._analyze_price_formation(wave_peaks)

        # Step 5: Generate traditional support/resistance levels for backward compatibility
        support_levels, resistance_levels = self._generate_support_resistance_levels(
            wave_peaks, support_zones, resistance_zones
        )

        # Step 6: Calculate additional analysis metrics
        poc_levels = self._identify_poc_levels(aggregated_trades)
        liquidity_vacuum_zones = self._identify_liquidity_vacuum_zones(aggregated_bids, aggregated_asks)
        resonance_zones = self._identify_resonance_zones(support_levels, resistance_levels, aggregated_trades)

        # Step 7: Calculate quality metrics
        depth_statistics = self._calculate_depth_statistics(
            snapshot.bids, snapshot.asks, aggregated_bids, aggregated_asks
        )
        peak_quality = self._calculate_peak_detection_quality(wave_peaks, aggregated_trades)

        result = EnhancedMarketAnalysisResult(
            timestamp=datetime.now(),
            symbol=symbol,
            aggregated_bids=aggregated_bids,
            aggregated_asks=aggregated_asks,
            wave_peaks=wave_peaks,
            support_zones=support_zones,
            resistance_zones=resistance_zones,
            support_levels=support_levels,
            resistance_levels=resistance_levels,
            poc_levels=poc_levels,
            liquidity_vacuum_zones=liquidity_vacuum_zones,
            resonance_zones=resonance_zones,
            depth_statistics=depth_statistics,
            peak_detection_quality=peak_quality,
        )

        logger.info(f"Enhanced analysis completed: {len(wave_peaks)} peaks, {len(support_zones)} support zones, {len(resistance_zones)} resistance zones")
        return result

    def _analyze_legacy(
        self,
        snapshot: DepthSnapshot,
        trade_data_list: List[MinuteTradeData],
        symbol: str
    ) -> MarketAnalysisResult:
        """
        Perform legacy analysis for backward compatibility.

        Args:
            snapshot: Depth snapshot
            trade_data_list: Trade data list
            symbol: Trading symbol

        Returns:
            MarketAnalysisResult with traditional analysis
        """
        # For now, delegate to enhanced mode and convert to legacy format
        enhanced_result = self._analyze_enhanced(snapshot, trade_data_list, symbol)

        return MarketAnalysisResult(
            timestamp=enhanced_result.timestamp,
            symbol=enhanced_result.symbol,
            support_levels=enhanced_result.support_levels,
            resistance_levels=enhanced_result.resistance_levels,
            poc_levels=enhanced_result.poc_levels,
            liquidity_vacuum_zones=enhanced_result.liquidity_vacuum_zones,
            resonance_zones=enhanced_result.resonance_zones,
        )

    def _aggregate_depth_snapshot(self, snapshot: DepthSnapshot) -> Tuple[Dict[Decimal, Decimal], Dict[Decimal, Decimal]]:
        """Aggregate depth snapshot by 1-dollar precision."""
        return aggregate_depth_by_one_dollar(snapshot.bids, snapshot.asks)

    def _aggregate_trade_data(self, trade_data_list: List[MinuteTradeData]) -> Dict[Decimal, Decimal]:
        """Aggregate trade data by 1-dollar precision."""
        aggregated_trades = defaultdict(Decimal)

        for minute_data in trade_data_list:
            for price_level_data in minute_data.price_levels.values():
                price_key = price_level_data.price_level.quantize(Decimal('1'))
                aggregated_trades[price_key] += price_level_data.total_volume

        return dict(aggregated_trades)

    def _detect_wave_peaks(self, aggregated_trades: Dict[Decimal, Decimal]) -> List[WavePeak]:
        """Detect wave peaks using combined statistical methods."""
        return detect_combined_peaks(aggregated_trades)

    def _analyze_price_formation(self, wave_peaks: List[WavePeak]) -> Tuple[List[PriceZone], List[PriceZone]]:
        """Analyze price formation from wave peaks."""
        zones = analyze_wave_formation(wave_peaks)

        support_zones = [zone for zone in zones if zone.zone_type == 'support']
        resistance_zones = [zone for zone in zones if zone.zone_type == 'resistance']

        return support_zones, resistance_zones

    def _generate_support_resistance_levels(
        self,
        wave_peaks: List[WavePeak],
        support_zones: List[PriceZone],
        resistance_zones: List[PriceZone]
    ) -> Tuple[List[SupportResistanceLevel], List[SupportResistanceLevel]]:
        """Generate traditional support/resistance levels from enhanced analysis."""
        support_levels = []
        resistance_levels = []

        # Convert support zones to traditional levels
        for zone in support_zones:
            level = SupportResistanceLevel(
                price=zone.center_price,
                strength=zone.confidence,
                level_type='support',
                volume_at_level=zone.total_volume,
                confirmation_count=1,
                last_confirmed=datetime.now()
            )
            support_levels.append(level)

        # Convert resistance zones to traditional levels
        for zone in resistance_zones:
            level = SupportResistanceLevel(
                price=zone.center_price,
                strength=zone.confidence,
                level_type='resistance',
                volume_at_level=zone.total_volume,
                confirmation_count=1,
                last_confirmed=datetime.now()
            )
            resistance_levels.append(level)

        # Add standalone wave peaks
        for peak in wave_peaks:
            if peak.peak_type == 'statistical_peak' and peak.confidence > 0.7:
                level_type = 'support' if peak.bid_volume > peak.ask_volume else 'resistance'
                level = SupportResistanceLevel(
                    price=peak.center_price,
                    strength=peak.confidence,
                    level_type=level_type,
                    volume_at_level=peak.volume,
                    confirmation_count=1,
                    last_confirmed=datetime.now()
                )

                if level_type == 'support':
                    support_levels.append(level)
                else:
                    resistance_levels.append(level)

        return support_levels, resistance_levels

    def _identify_poc_levels(self, aggregated_trades: Dict[Decimal, Decimal]) -> List[Decimal]:
        """Identify Point of Control (POC) levels."""
        if not aggregated_trades:
            return []

        max_volume = max(aggregated_trades.values())
        poc_levels = [
            price for price, volume in aggregated_trades.items()
            if volume >= max_volume * Decimal('0.8')  # Top 20% volume levels
        ]

        return poc_levels

    def _identify_liquidity_vacuum_zones(
        self,
        aggregated_bids: Dict[Decimal, Decimal],
        aggregated_asks: Dict[Decimal, Decimal]
    ) -> List[Decimal]:
        """Identify liquidity vacuum zones (price ranges with low volume)."""
        all_prices = sorted(set(aggregated_bids.keys()) | set(aggregated_asks.keys()))

        if len(all_prices) < 3:
            return []

        # Calculate average volume
        all_volumes = list(aggregated_bids.values()) + list(aggregated_asks.values())
        avg_volume = sum(all_volumes) / len(all_volumes) if all_volumes else Decimal('0')

        vacuum_zones = []
        for i in range(1, len(all_prices)):
            prev_price = all_prices[i-1]
            curr_price = all_prices[i]

            prev_volume = aggregated_bids.get(prev_price, aggregated_asks.get(prev_price, Decimal('0')))
            curr_volume = aggregated_bids.get(curr_price, aggregated_asks.get(curr_price, Decimal('0')))

            # If both adjacent price levels have low volume, mark as vacuum
            if prev_volume < avg_volume * Decimal('0.2') and curr_volume < avg_volume * Decimal('0.2'):
                vacuum_zones.append((prev_price + curr_price) / Decimal('2'))

        return vacuum_zones

    def _identify_resonance_zones(
        self,
        support_levels: List[SupportResistanceLevel],
        resistance_levels: List[SupportResistanceLevel],
        aggregated_trades: Dict[Decimal, Decimal]
    ) -> List[Decimal]:
        """Identify resonance zones where support and resistance converge."""
        resonance_zones = []

        for support in support_levels:
            for resistance in resistance_levels:
                # If support and resistance are close (within $2), mark as resonance zone
                if abs(support.price - resistance.price) <= Decimal('2'):
                    center_price = (support.price + resistance.price) / Decimal('2')

                    # Check if there's significant volume at this price
                    volume_at_center = sum(
                        volume for price, volume in aggregated_trades.items()
                        if abs(price - center_price) <= Decimal('1')
                    )

                    if volume_at_center > self.min_volume_threshold * Decimal('10'):
                        resonance_zones.append(center_price)

        return resonance_zones

    def _calculate_depth_statistics(
        self,
        original_bids: List,
        original_asks: List,
        aggregated_bids: Dict[Decimal, Decimal],
        aggregated_asks: Dict[Decimal, Decimal]
    ) -> Dict[str, Decimal]:
        """Calculate depth statistics for quality assessment."""
        original_bid_volume = sum(bid.quantity for bid in original_bids)
        original_ask_volume = sum(ask.quantity for ask in original_asks)
        aggregated_bid_volume = sum(aggregated_bids.values())
        aggregated_ask_volume = sum(aggregated_asks.values())

        # Calculate compression ratios
        bid_compression = Decimal(str(len(original_bids))) / Decimal(str(max(len(aggregated_bids), 1)))
        ask_compression = Decimal(str(len(original_asks))) / Decimal(str(max(len(aggregated_asks), 1)))

        # Calculate volume preservation
        total_original = original_bid_volume + original_ask_volume
        total_aggregated = aggregated_bid_volume + aggregated_ask_volume
        volume_preservation = total_aggregated / (total_original + Decimal('0.01')) if total_original > 0 else Decimal('0')

        return {
            'bid_compression_ratio': bid_compression,
            'ask_compression_ratio': ask_compression,
            'volume_preservation_rate': volume_preservation,
            'original_levels': Decimal(str(len(original_bids) + len(original_asks))),
            'aggregated_levels': Decimal(str(len(aggregated_bids) + len(aggregated_asks))),
        }

    def _calculate_peak_detection_quality(
        self,
        wave_peaks: List[WavePeak],
        aggregated_trades: Dict[Decimal, Decimal]
    ) -> Dict[str, float]:
        """Calculate quality metrics for peak detection."""
        if not wave_peaks:
            return {
                'peak_count': 0,
                'avg_confidence': 0.0,
                'coverage_rate': 0.0,
            }

        total_peaks = len(wave_peaks)
        avg_confidence = sum(peak.confidence for peak in wave_peaks) / total_peaks

        # Calculate coverage (percentage of significant volume captured by peaks)
        total_volume = sum(aggregated_trades.values())
        peak_volume = sum(peak.volume for peak in wave_peaks)
        coverage_rate = float(peak_volume / (total_volume + 0.01)) if total_volume > 0 else 0.0

        return {
            'peak_count': total_peaks,
            'avg_confidence': avg_confidence,
            'coverage_rate': coverage_rate,
        }