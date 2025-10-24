"""Enhanced market analyzers with normal distribution peak detection.

This module provides sophisticated market analysis using normal distribution
methods to identify significant price levels and confidence intervals.
"""

import logging
from collections import defaultdict
from datetime import datetime
from decimal import Decimal
from typing import Any

from .models import (
    DepthSnapshot,
    EnhancedMarketAnalysisResult,
    MarketAnalysisResult,
    MinuteTradeData,
    SupportResistanceLevel,
)
from .normal_distribution_analyzer import (
    NormalDistributionPeakAnalyzer,
    convert_to_decimal_format,
)
from .sklearn_cluster_analyzer import SklearnClusterAnalyzer

logger = logging.getLogger(__name__)


class NormalDistributionMarketAnalyzer:
    """
    Enhanced market analyzer with normal distribution peak detection.

    This analyzer combines traditional support/resistance analysis with
    sophisticated normal distribution methods to identify confidence intervals
    and significant price levels.
    """

    def __init__(
        self,
        min_volume_threshold: Decimal = Decimal("1.0"),
        analysis_window_minutes: int = 180,
        confidence_level: float = 0.95,
    ):
        """
        Initialize the enhanced market analyzer.

        Args:
            min_volume_threshold: Minimum volume threshold for analysis
            analysis_window_minutes: Analysis window in minutes
            confidence_level: Statistical confidence level for peak detection
        """
        self.min_volume_threshold = min_volume_threshold
        self.analysis_window_minutes = analysis_window_minutes
        self.confidence_level = confidence_level

        # Initialize normal distribution analyzer
        self.peak_analyzer = NormalDistributionPeakAnalyzer(
            price_precision=1.0,
            confidence_level=confidence_level
        )

        # Initialize sklearn cluster analyzer
        self.cluster_analyzer = SklearnClusterAnalyzer(
            min_samples=3,
            eps_multiplier=0.02,
            max_clusters=8,
            volume_weight=2.0
        )

        logger.info(
            f"Initialized NormalDistributionMarketAnalyzer with confidence_level={confidence_level}, "
            f"window={analysis_window_minutes}min, sklearn clustering enabled"
        )

    def analyze_market(
        self,
        snapshot: DepthSnapshot,
        trade_data_list: list[MinuteTradeData],
        symbol: str,
        enhanced_mode: bool = True,
    ) -> MarketAnalysisResult | EnhancedMarketAnalysisResult:
        """
        Perform comprehensive market analysis using normal distribution methods.

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
                    normal_distribution_peaks={},
                    confidence_intervals={},
                )
            else:
                return MarketAnalysisResult(
                    timestamp=datetime.now(),
                    symbol=symbol,
                )

    def _analyze_enhanced(
        self,
        snapshot: DepthSnapshot,
        trade_data_list: list[MinuteTradeData],
        symbol: str,
    ) -> EnhancedMarketAnalysisResult:
        """
        Perform enhanced analysis with normal distribution peak detection.

        Args:
            snapshot: Depth snapshot
            trade_data_list: Trade data list
            symbol: Trading symbol

        Returns:
            EnhancedMarketAnalysisResult with comprehensive analysis
        """
        logger.info(f"Starting enhanced normal distribution analysis for {symbol}")

        # Step 1: Prepare order book data for normal distribution analysis
        order_book_data = self._prepare_order_book_data(snapshot)

        # Step 2: Perform normal distribution peak analysis
        nd_analysis = self.peak_analyzer.analyze_order_book(order_book_data)

        # Step 2.5: Perform sklearn clustering analysis
        clustering_results = self.cluster_analyzer.analyze_order_book_clustering(snapshot)

        # Step 3: Convert to decimal format for consistency
        nd_analysis_decimal = convert_to_decimal_format(nd_analysis)

        # Step 4: Aggregate trade data
        aggregated_trades = self._aggregate_trade_data(trade_data_list)

        # Step 5: Generate traditional support/resistance levels from normal distribution results
        support_levels, resistance_levels = self._generate_support_resistance_from_nd(
            nd_analysis_decimal, symbol
        )

        # Step 6: Calculate additional analysis metrics
        poc_levels = self._identify_poc_levels(aggregated_trades)
        liquidity_vacuum_zones = self._identify_liquidity_vacuum_zones(
            nd_analysis_decimal.get('aggregated_bids', {}),
            nd_analysis_decimal.get('aggregated_asks', {})
        )
        resonance_zones = self._identify_resonance_zones(
            support_levels, resistance_levels, aggregated_trades
        )

        # Step 7: Calculate quality metrics
        depth_statistics = self._calculate_depth_statistics(
            snapshot.bids, snapshot.asks,
            nd_analysis_decimal.get('aggregated_bids', {}),
            nd_analysis_decimal.get('aggregated_asks', {})
        )
        peak_quality = self._calculate_nd_peak_quality(nd_analysis_decimal, aggregated_trades)

        result = EnhancedMarketAnalysisResult(
            timestamp=datetime.now(),
            symbol=symbol,
            aggregated_bids=nd_analysis_decimal.get('aggregated_bids', {}),
            aggregated_asks=nd_analysis_decimal.get('aggregated_asks', {}),
            wave_peaks=[],  # Empty for now, could be populated later
            support_zones=[],  # Convert from confidence intervals
            resistance_zones=[],
            support_levels=support_levels,
            resistance_levels=resistance_levels,
            poc_levels=poc_levels,
            liquidity_vacuum_zones=liquidity_vacuum_zones,
            resonance_zones=resonance_zones,
            depth_statistics=depth_statistics,
            peak_detection_quality=peak_quality,
            # New fields for normal distribution analysis
            normal_distribution_peaks=nd_analysis_decimal.get('peak_analysis', {}),
            confidence_intervals=self._extract_confidence_intervals(nd_analysis_decimal),
            market_metrics=nd_analysis_decimal.get('market_metrics', {}),
            spread_analysis=nd_analysis_decimal.get('spread_analysis', {}),
            # New fields for sklearn clustering analysis
            clustering_results=clustering_results,
            optimal_clusters=clustering_results.get('optimal_clusters', 0),
            silhouette_score=clustering_results.get('silhouette_score', 0.0),
            liquidity_peaks=self._convert_clustering_peaks_to_support_resistance(
                clustering_results.get('liquidity_peaks', [])
            ),
        )

        logger.info(
            f"Normal distribution analysis completed: "
            f"bid_peak={nd_analysis_decimal.get('peak_analysis', {}).get('bids', {}).get('mean_price')}, "
            f"ask_peak={nd_analysis_decimal.get('peak_analysis', {}).get('asks', {}).get('mean_price')}, "
            f"confidence_level={self.confidence_level}"
        )

        # Log sklearn clustering results
        logger.info(
            f"Sklearn clustering analysis completed: "
            f"optimal_clusters={clustering_results.get('optimal_clusters', 0)}, "
            f"silhouette_score={clustering_results.get('silhouette_score', 0.0):.3f}, "
            f"liquidity_peaks={len(clustering_results.get('liquidity_peaks', []))}"
        )

        # Print clustering results in the specified format
        if clustering_results.get('optimal_clusters', 0) > 0:
            print(f"\n=== SKLEARN聚类分析结果 ===")
            print(f"最优聚类数: {clustering_results['optimal_clusters']}")
            print(f"轮廓系数: {clustering_results['silhouette_score']:.3f}")

            print(f"\n流动性峰值区域:")
            for i, peak in enumerate(clustering_results.get('liquidity_peaks', [])):
                print(f"峰值 {i+1}: ${peak['center_price']:.2f}, "
                      f"总量: {peak['total_volume']:.0f}, "
                      f"方向: {peak['dominant_side']}, "
                      f"纯度: {peak['purity']:.2f}")

            print(f"\n详细聚类统计:")
            cluster_analysis = clustering_results.get('cluster_analysis', {})
            for cluster_id, stats in cluster_analysis.items():
                print(f"聚类 {cluster_id}: {stats['size']}个订单, "
                      f"价格区间: ${stats['price_range'][0]:.2f}-${stats['price_range'][1]:.2f}, "
                      f"总挂单量: {stats['total_volume']:.0f}")
            print("=" * 40)

        return result

    def _analyze_legacy(
        self,
        snapshot: DepthSnapshot,
        trade_data_list: list[MinuteTradeData],
        symbol: str,
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

    def _prepare_order_book_data(self, snapshot: DepthSnapshot) -> dict[str, list]:
        """Prepare order book data for normal distribution analysis.

        Args:
            snapshot: Depth snapshot

        Returns:
            Order book data in the expected format
        """
        order_book_data = {
            'bids': [
                (float(level.price), float(level.quantity))
                for level in snapshot.bids
            ],
            'asks': [
                (float(level.price), float(level.quantity))
                for level in snapshot.asks
            ]
        }
        return order_book_data

    def _aggregate_trade_data(
        self, trade_data_list: list[MinuteTradeData]
    ) -> dict[Decimal, Decimal]:
        """Aggregate trade data by 1-dollar precision."""
        aggregated_trades = defaultdict(Decimal)

        for minute_data in trade_data_list:
            for price_level_data in minute_data.price_levels.values():
                # Handle both PriceLevelData objects and dictionaries (from Redis)
                if isinstance(price_level_data, dict):
                    # Data is stored as dictionary from Redis JSON deserialization
                    price_key = Decimal(str(price_level_data["price_level"])).quantize(Decimal("1"))
                    total_volume = Decimal(str(price_level_data["total_volume"]))
                else:
                    # Data is a PriceLevelData object
                    price_key = price_level_data.price_level.quantize(Decimal("1"))
                    total_volume = price_level_data.total_volume

                aggregated_trades[price_key] += total_volume

        return dict(aggregated_trades)

    def _generate_support_resistance_from_nd(
        self,
        nd_analysis: dict[str, Any],
        symbol: str,
    ) -> tuple[list[SupportResistanceLevel], list[SupportResistanceLevel]]:
        """Generate support/resistance levels from normal distribution analysis.

        Args:
            nd_analysis: Normal distribution analysis results
            symbol: Trading symbol

        Returns:
            Tuple of (support_levels, resistance_levels)
        """
        support_levels = []
        resistance_levels = []

        peak_analysis = nd_analysis.get('peak_analysis', {})

        # Generate support levels from bid peak interval
        if 'bids' in peak_analysis:
            bid_data = peak_analysis['bids']
            if bid_data.get('peak_interval') and bid_data.get('mean_price'):
                lower_price, upper_price = bid_data['peak_interval']
                if lower_price is not None and upper_price is not None:
                    # Use the lower bound as support level
                    support_level = SupportResistanceLevel(
                        price=lower_price,
                        strength=bid_data.get('z_score', 1.96) / 3.0,  # Normalize to 0-1 range
                        level_type="support",
                        volume_at_level=bid_data.get('peak_volume', Decimal("0")),
                        confirmation_count=1,
                        last_confirmed=datetime.now(),
                    )
                    support_levels.append(support_level)

        # Generate resistance levels from ask peak interval
        if 'asks' in peak_analysis:
            ask_data = peak_analysis['asks']
            if ask_data.get('peak_interval') and ask_data.get('mean_price'):
                lower_price, upper_price = ask_data['peak_interval']
                if lower_price is not None and upper_price is not None:
                    # Use the upper bound as resistance level
                    resistance_level = SupportResistanceLevel(
                        price=upper_price,
                        strength=ask_data.get('z_score', 1.96) / 3.0,  # Normalize to 0-1 range
                        level_type="resistance",
                        volume_at_level=ask_data.get('peak_volume', Decimal("0")),
                        confirmation_count=1,
                        last_confirmed=datetime.now(),
                    )
                    resistance_levels.append(resistance_level)

        return support_levels, resistance_levels

    def _extract_confidence_intervals(self, nd_analysis: dict[str, Any]) -> dict[str, Any]:
        """Extract confidence intervals from analysis results.

        Args:
            nd_analysis: Normal distribution analysis results

        Returns:
            Dictionary containing confidence intervals
        """
        peak_analysis = nd_analysis.get('peak_analysis', {})
        confidence_intervals = {}

        if 'bids' in peak_analysis:
            bid_data = peak_analysis['bids']
            confidence_intervals['bid'] = {
                'interval': bid_data.get('peak_interval'),
                'mean_price': bid_data.get('mean_price'),
                'confidence_level': bid_data.get('confidence_level'),
                'z_score': bid_data.get('z_score'),
            }

        if 'asks' in peak_analysis:
            ask_data = peak_analysis['asks']
            confidence_intervals['ask'] = {
                'interval': ask_data.get('peak_interval'),
                'mean_price': ask_data.get('mean_price'),
                'confidence_level': ask_data.get('confidence_level'),
                'z_score': ask_data.get('z_score'),
            }

        return confidence_intervals

    def _identify_poc_levels(
        self, aggregated_trades: dict[Decimal, Decimal]
    ) -> list[Decimal]:
        """Identify Point of Control (POC) levels."""
        if not aggregated_trades:
            return []

        max_volume = max(aggregated_trades.values())
        poc_levels = [
            price
            for price, volume in aggregated_trades.items()
            if volume >= max_volume * Decimal("0.8")  # Top 20% volume levels
        ]

        return poc_levels

    def _identify_liquidity_vacuum_zones(
        self,
        aggregated_bids: dict[Decimal, Decimal],
        aggregated_asks: dict[Decimal, Decimal],
    ) -> list[Decimal]:
        """Identify liquidity vacuum zones (price ranges with low volume)."""
        all_prices = sorted(set(aggregated_bids.keys()) | set(aggregated_asks.keys()))

        if len(all_prices) < 3:
            return []

        # Calculate average volume
        all_volumes = list(aggregated_bids.values()) + list(aggregated_asks.values())
        avg_volume = (
            sum(all_volumes) / len(all_volumes) if all_volumes else Decimal("0")
        )

        vacuum_zones = []
        for i in range(1, len(all_prices)):
            prev_price = all_prices[i - 1]
            curr_price = all_prices[i]

            prev_volume = aggregated_bids.get(
                prev_price, aggregated_asks.get(prev_price, Decimal("0"))
            )
            curr_volume = aggregated_bids.get(
                curr_price, aggregated_asks.get(curr_price, Decimal("0"))
            )

            # If both adjacent price levels have low volume, mark as vacuum
            if prev_volume < avg_volume * Decimal("0.2") and curr_volume < avg_volume * Decimal("0.2"):
                vacuum_zones.append((prev_price + curr_price) / Decimal("2"))

        return vacuum_zones

    def _identify_resonance_zones(
        self,
        support_levels: list[SupportResistanceLevel],
        resistance_levels: list[SupportResistanceLevel],
        aggregated_trades: dict[Decimal, Decimal],
    ) -> list[Decimal]:
        """Identify resonance zones where support and resistance converge."""
        resonance_zones = []

        for support in support_levels:
            for resistance in resistance_levels:
                # If support and resistance are close (within $2), mark as resonance zone
                if abs(support.price - resistance.price) <= Decimal("2"):
                    center_price = (support.price + resistance.price) / Decimal("2")

                    # Check if there's significant volume at this price
                    volume_at_center = sum(
                        volume
                        for price, volume in aggregated_trades.items()
                        if abs(price - center_price) <= Decimal("1")
                    )

                    if volume_at_center > self.min_volume_threshold * Decimal("10"):
                        resonance_zones.append(center_price)

        return resonance_zones

    def _calculate_depth_statistics(
        self,
        original_bids: list,
        original_asks: list,
        aggregated_bids: dict[Decimal, Decimal],
        aggregated_asks: dict[Decimal, Decimal],
    ) -> dict[str, Decimal]:
        """Calculate depth statistics for quality assessment."""
        original_bid_volume = sum(bid.quantity for bid in original_bids)
        original_ask_volume = sum(ask.quantity for ask in original_asks)
        aggregated_bid_volume = sum(aggregated_bids.values())
        aggregated_ask_volume = sum(aggregated_asks.values())

        # Calculate compression ratios
        bid_compression = Decimal(str(len(original_bids))) / Decimal(
            str(max(len(aggregated_bids), 1))
        )
        ask_compression = Decimal(str(len(original_asks))) / Decimal(
            str(max(len(aggregated_asks), 1))
        )

        # Calculate volume preservation
        total_original = original_bid_volume + original_ask_volume
        total_aggregated = aggregated_bid_volume + aggregated_ask_volume
        volume_preservation = (
            total_aggregated / (total_original + Decimal("0.01"))
            if total_original > 0
            else Decimal("0")
        )

        return {
            "bid_compression_ratio": bid_compression,
            "ask_compression_ratio": ask_compression,
            "volume_preservation_rate": volume_preservation,
            "original_levels": Decimal(str(len(original_bids) + len(original_asks))),
            "aggregated_levels": Decimal(
                str(len(aggregated_bids) + len(aggregated_asks))
            ),
        }

    def _calculate_nd_peak_quality(
        self, nd_analysis: dict[str, Any], aggregated_trades: dict[Decimal, Decimal]
    ) -> dict[str, float]:
        """Calculate quality metrics for normal distribution peak detection."""
        if not nd_analysis or 'peak_analysis' not in nd_analysis:
            return {
                "peak_count": 0,
                "avg_confidence": 0.0,
                "coverage_rate": 0.0,
                "confidence_level": self.confidence_level,
            }

        peak_analysis = nd_analysis['peak_analysis']
        total_peaks = len([side for side in ['bids', 'asks'] if side in peak_analysis])

        # Average confidence based on z-scores
        z_scores = []
        for side in ['bids', 'asks']:
            if side in peak_analysis:
                z_scores.append(peak_analysis[side].get('z_score', 0))

        avg_confidence = sum(z_scores) / len(z_scores) if z_scores else 0.0

        # Calculate coverage (percentage of significant volume captured by peaks)
        total_volume = sum(aggregated_trades.values())
        peak_volume = 0.0
        for side in ['bids', 'asks']:
            if side in peak_analysis:
                peak_volume += float(peak_analysis[side].get('peak_volume', 0))

        coverage_rate = (
            peak_volume / (float(total_volume) + 0.01) if total_volume > 0 else 0.0
        )

        return {
            "peak_count": total_peaks,
            "avg_confidence": avg_confidence,
            "coverage_rate": coverage_rate,
            "confidence_level": self.confidence_level,
        }

    def _convert_clustering_peaks_to_support_resistance(
        self, liquidity_peaks: list[dict[str, Any]]
    ) -> list[SupportResistanceLevel]:
        """Convert sklearn clustering peaks to SupportResistanceLevel objects."""
        support_resistance_levels = []

        for peak in liquidity_peaks:
            level_type = "support" if peak["dominant_side"] == "bid" else "resistance"

            support_resistance_level = SupportResistanceLevel(
                price=Decimal(str(peak["center_price"])),
                strength=min(peak["purity"], 1.0),  # Ensure strength doesn't exceed 1.0
                level_type=level_type,
                volume_at_level=Decimal(str(abs(peak["total_volume"]))),
                confirmation_count=1,
                last_confirmed=datetime.now(),
            )

            support_resistance_levels.append(support_resistance_level)

        return support_resistance_levels


# Keep the existing MarketAnalyzer as fallback
class MarketAnalyzer:
    """Main market analyzer that supports both normal distribution and traditional methods."""

    def __init__(
        self,
        min_volume_threshold: float = 0.1,
        analysis_window_minutes: int = 180,
        enhanced_mode: bool = False,
        use_normal_distribution: bool = True,
        confidence_level: float = 0.95,
    ):
        """Initialize the market analyzer."""
        self.min_volume_threshold = min_volume_threshold
        self.analysis_window_minutes = analysis_window_minutes
        self.enhanced_mode = enhanced_mode
        self.use_normal_distribution = use_normal_distribution

        if use_normal_distribution:
            # Use normal distribution analyzer
            self.nd_analyzer = NormalDistributionMarketAnalyzer(
                min_volume_threshold=Decimal(str(min_volume_threshold)),
                analysis_window_minutes=analysis_window_minutes,
                confidence_level=confidence_level,
            )
            logger.info("Using normal distribution market analyzer")
        else:
            # Fall back to importing traditional analyzer
            try:
                from . import analyzers
                self.traditional_analyzer = analyzers.MarketAnalyzer(
                    min_volume_threshold=min_volume_threshold,
                    analysis_window_minutes=analysis_window_minutes,
                    enhanced_mode=enhanced_mode,
                )
                logger.info("Using traditional market analyzer")
            except ImportError:
                logger.error("Failed to import traditional analyzer")
                self.nd_analyzer = NormalDistributionMarketAnalyzer(
                    min_volume_threshold=Decimal(str(min_volume_threshold)),
                    analysis_window_minutes=analysis_window_minutes,
                    confidence_level=confidence_level,
                )

    def analyze_market(
        self,
        snapshot,
        trade_data_list,
        symbol: str = "BTCFDUSD",
        enhanced_mode: bool = True,
    ):
        """Perform market analysis using the selected method."""
        if self.use_normal_distribution and hasattr(self, 'nd_analyzer'):
            return self.nd_analyzer.analyze_market(
                snapshot, trade_data_list, symbol, enhanced_mode
            )
        elif hasattr(self, 'traditional_analyzer'):
            return self.traditional_analyzer.analyze_market(
                snapshot, trade_data_list, symbol, enhanced_mode
            )
        else:
            # Fallback to normal distribution
            return self.nd_analyzer.analyze_market(
                snapshot, trade_data_list, symbol, enhanced_mode
            )
