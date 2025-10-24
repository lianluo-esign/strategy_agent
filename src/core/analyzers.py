"""Market data analyzers for support/resistance and order flow analysis."""

import logging
from collections import defaultdict
from datetime import datetime
from decimal import Decimal

from .models import (
    DepthSnapshot,
    MarketAnalysisResult,
    MinuteTradeData,
    SupportResistanceLevel,
)

logger = logging.getLogger(__name__)


def _to_decimal(value: int | float | str | Decimal) -> Decimal:
    """Convert a value to Decimal safely.

    Args:
        value: Value to convert to Decimal

    Returns:
        Decimal representation of the input value

    Raises:
        TypeError: If value cannot be converted to Decimal
    """
    if isinstance(value, Decimal):
        return value
    elif isinstance(value, int):
        return Decimal(value)  # Faster for integers
    elif isinstance(value, (float, str)):
        return Decimal(str(value))  # Safe for floats and strings
    else:
        raise TypeError(f"Cannot convert {type(value)} to Decimal")


def _safe_decimal_division(
    numerator: Decimal,
    denominator: Decimal | int | float
) -> Decimal:
    """Safely divide Decimal by a numeric denominator.

    Args:
        numerator: Decimal numerator
        denominator: Numeric denominator

    Returns:
        Result of division as Decimal

    Raises:
        ValueError: If denominator is zero
        TypeError: If denominator type is invalid
    """
    # Convert denominator to Decimal for consistent calculation
    decimal_denominator = _to_decimal(denominator)

    if decimal_denominator == 0:
        raise ValueError("Denominator cannot be zero")

    return numerator / decimal_denominator


class DepthSnapshotAnalyzer:
    """Analyzes depth snapshots to identify support and resistance levels."""

    def __init__(self, min_volume_threshold: float = 0.1, price_zone_size: float = 0.50):
        """Initialize the depth analyzer."""
        self.min_volume_threshold = min_volume_threshold
        self.price_zone_size = Decimal(str(price_zone_size))

    def analyze_support_resistance(
        self,
        snapshot: DepthSnapshot,
        lookback_levels: int = 100
    ) -> tuple[list[SupportResistanceLevel], list[SupportResistanceLevel]]:
        """Analyze depth snapshot to identify support and resistance levels."""
        support_levels = []
        resistance_levels = []

        # Analyze bid side (support)
        if len(snapshot.bids) > 0:
            bid_levels = snapshot.bids[:lookback_levels]  # Top N bid levels
            support_levels = self._find_significant_levels(
                bid_levels, 'support', snapshot.symbol
            )

        # Analyze ask side (resistance)
        if len(snapshot.asks) > 0:
            ask_levels = snapshot.asks[:lookback_levels]  # Top N ask levels
            resistance_levels = self._find_significant_levels(
                ask_levels, 'resistance', snapshot.symbol
            )

        return support_levels, resistance_levels

    def _find_significant_levels(
        self,
        levels: list,
        level_type: str,
        symbol: str
    ) -> list[SupportResistanceLevel]:
        """Find significant support/resistance levels from order book data."""
        significant_levels = []

        if not levels:
            return significant_levels

        # Group by price zones (using configurable zone size)
        price_zones = self._group_by_price_zones(levels, zone_size=self.price_zone_size)

        # Calculate total volume for all levels
        total_volume = sum(level.quantity for level in levels)

        # Identify significant zones
        for zone_price, zone_levels in price_zones.items():
            zone_volume = sum(level.quantity for level in zone_levels)

            if zone_volume < Decimal(str(self.min_volume_threshold)):
                continue

            # Calculate strength based on volume concentration
            strength = min(float(zone_volume / total_volume) * 10, 1.0)  # Normalize to 0-1

            if strength > 0.1:  # Only include levels with meaningful strength
                level = SupportResistanceLevel(
                    price=zone_price,
                    strength=float(strength),
                    level_type=level_type,
                    volume_at_level=zone_volume,
                    confirmation_count=1,  # Initial confirmation
                    last_confirmed=datetime.now()
                )
                significant_levels.append(level)

        return sorted(significant_levels, key=lambda x: x.strength, reverse=True)

    def _group_by_price_zones(
        self,
        levels: list,
        zone_size: Decimal
    ) -> dict[Decimal, list]:
        """Group order book levels by price zones."""
        zones = defaultdict(list)

        for level in levels:
            # Find the zone center price
            zone_price = (level.price // zone_size) * zone_size
            zones[zone_price].append(level)

        return zones

    def identify_liquidity_vacuum_zones(
        self,
        snapshot: DepthSnapshot,
        price_range: tuple[Decimal, Decimal] | None = None
    ) -> list[Decimal]:
        """Identify price ranges with low liquidity (vacuum zones)."""
        if price_range is None:
            # Use the visible order book range
            min_price = min(level.price for level in snapshot.asks)
            max_price = max(level.price for level in snapshot.bids)
        else:
            min_price, max_price = price_range

        vacuum_zones = []

        # Calculate average volume per price level
        all_levels = snapshot.bids + snapshot.asks
        if not all_levels:
            return vacuum_zones

        avg_volume = sum(level.quantity for level in all_levels) / Decimal(str(len(all_levels)))
        low_volume_threshold = avg_volume * Decimal('0.1')  # 10% of average volume

        # Scan for low volume areas
        # This is a simplified implementation - in practice, you'd want more sophisticated detection
        for i in range(len(all_levels) - 1):
            current_level = all_levels[i]
            next_level = all_levels[i + 1]

            # Check price gap and low volume
            price_gap = abs(next_level.price - current_level.price)
            if price_gap > Decimal('1.0'):  # $1+ gap
                avg_gap_volume = (current_level.quantity + next_level.quantity) / Decimal('2')
                if avg_gap_volume < low_volume_threshold:
                    vacuum_zones.append((current_level.price + next_level.price) / Decimal('2'))

        return vacuum_zones


class OrderFlowAnalyzer:
    """Analyzes order flow data to confirm support/resistance levels."""

    def __init__(self, analysis_window_minutes: int = 180):  # 3 hours default
        """Initialize the order flow analyzer."""
        self.analysis_window_minutes = analysis_window_minutes

    def analyze_order_flow(
        self,
        trade_data_list: list[MinuteTradeData],
        support_levels: list[SupportResistanceLevel],
        resistance_levels: list[SupportResistanceLevel]
    ) -> tuple[list[SupportResistanceLevel], list[SupportResistanceLevel], list[Decimal]]:
        """Analyze order flow to confirm levels and find POCs."""
        if not trade_data_list:
            return support_levels, resistance_levels, []

        # Calculate Point of Control (POC) levels
        poc_levels = self._find_poc_levels(trade_data_list)

        # Confirm support/resistance levels with order flow
        confirmed_support = self._confirm_levels_with_order_flow(
            support_levels, trade_data_list, 'support'
        )
        confirmed_resistance = self._confirm_levels_with_order_flow(
            resistance_levels, trade_data_list, 'resistance'
        )

        return confirmed_support, confirmed_resistance, poc_levels

    def _find_poc_levels(self, trade_data_list: list[MinuteTradeData]) -> list[Decimal]:
        """Find Point of Control levels - prices with highest volume concentration."""
        # Aggregate all volume by price level
        price_volume_map = defaultdict(Decimal)

        try:
            for minute_data in trade_data_list:
                if not hasattr(minute_data, 'price_levels') or not minute_data.price_levels:
                    continue

                for price_level, level_data in minute_data.price_levels.items():
                    if not hasattr(level_data, 'total_volume'):
                        continue
                    # Ensure both price and volume are Decimal
                    price_decimal = _to_decimal(price_level)
                    volume_decimal = _to_decimal(level_data.total_volume)
                    price_volume_map[price_decimal] += volume_decimal

            if not price_volume_map:
                logger.debug("No price volume data found for POC calculation")
                return []

            # Find top volume levels (top 10% by volume)
            sorted_levels = sorted(
                price_volume_map.items(),
                key=lambda x: float(x[1]),  # Convert to float for comparison
                reverse=True
            )

            top_count = max(1, len(sorted_levels) // 10)  # Top 10%
            poc_levels = [price for price, _ in sorted_levels[:top_count]]

            return poc_levels

        except Exception as e:
            logger.error(f"Error calculating POC levels: {e}")
            return []

    def _confirm_levels_with_order_flow(
        self,
        levels: list[SupportResistanceLevel],
        trade_data_list: list[MinuteTradeData],
        level_type: str
    ) -> list[SupportResistanceLevel]:
        """Confirm support/resistance levels using order flow data."""
        confirmed_levels = []

        for level in levels:
            confirmation_score = self._calculate_level_confirmation(
                level, trade_data_list, level_type
            )

            if confirmation_score > 0.3:  # Minimum confirmation threshold
                confirmed_level = SupportResistanceLevel(
                    price=level.price,
                    strength=level.strength * confirmation_score,  # Adjust strength
                    level_type=level.level_type,
                    volume_at_level=level.volume_at_level,
                    confirmation_count=level.confirmation_count + 1,
                    last_confirmed=datetime.now()
                )
                confirmed_levels.append(confirmed_level)

        return confirmed_levels

    def _calculate_level_confirmation(
        self,
        level: SupportResistanceLevel,
        trade_data_list: list[MinuteTradeData],
        level_type: str
    ) -> float:
        """Calculate how well order flow confirms a support/resistance level."""
        confirmation_score = 0.0
        relevant_data_count = 0

        # Define price tolerance around the level
        tolerance = Decimal('2.0')  # $2 tolerance

        for minute_data in trade_data_list:
            # Check if there's trading activity near this level
            nearby_prices = [
                price for price in minute_data.price_levels.keys()
                if abs(price - level.price) <= tolerance
            ]

            if not nearby_prices:
                continue

            relevant_data_count += 1
            level_confirmation = 0.0

            for price in nearby_prices:
                price_data = minute_data.price_levels[price]

                # Handle different data structures (dict vs object)
                try:
                    if hasattr(price_data, 'delta'):
                        delta = price_data.delta
                        total_volume = price_data.total_volume
                    elif isinstance(price_data, dict):
                        delta = _to_decimal(price_data.get('delta', 0))
                        total_volume = _to_decimal(price_data.get('total_volume', 0.01))
                    else:
                        logger.warning(f"Unexpected price_data type: {type(price_data)}")
                        continue

                    if total_volume <= 0:
                        continue

                    # Calculate confirmation contribution (convert to float)
                    confirmation_value = float(abs(delta) / total_volume)

                    if level_type == 'support':
                        # Support should show buying pressure when price approaches
                        if delta > 0:  # More buying than selling
                            level_confirmation += confirmation_value
                        else:
                            level_confirmation -= confirmation_value

                    elif level_type == 'resistance':
                        # Resistance should show selling pressure when price approaches
                        if delta < 0:  # More selling than buying
                            level_confirmation += confirmation_value
                        else:
                            level_confirmation -= confirmation_value

                except Exception as e:
                    logger.warning(f"Error processing price data for {price}: {e}")
                    continue

            # Average confirmation for this minute
            if nearby_prices:
                confirmation_score += level_confirmation / len(nearby_prices)

        # Normalize confirmation score
        if relevant_data_count > 0:
            confirmation_score = max(0, confirmation_score / relevant_data_count)
            return min(confirmation_score, 1.0)  # Cap at 1.0

        return 0.0


class MarketAnalyzer:
    """Main market analyzer that combines depth and order flow analysis."""

    def __init__(
        self,
        min_volume_threshold: float = 0.1,
        analysis_window_minutes: int = 180
    ):
        """Initialize the market analyzer."""
        self.depth_analyzer = DepthSnapshotAnalyzer(min_volume_threshold)
        self.order_flow_analyzer = OrderFlowAnalyzer(analysis_window_minutes)

    def analyze_market(
        self,
        snapshot: DepthSnapshot | None,
        trade_data_list: list[MinuteTradeData],
        symbol: str = "BTCFDUSD"
    ) -> MarketAnalysisResult:
        """Perform comprehensive market analysis."""
        if not snapshot and not trade_data_list:
            return MarketAnalysisResult(
                timestamp=datetime.now(),
                symbol=symbol
            )

        try:
            # Initialize result
            result = MarketAnalysisResult(
                timestamp=datetime.now(),
                symbol=symbol
            )

            # Step 1: Depth snapshot analysis (static support/resistance)
            if snapshot:
                result.support_levels, result.resistance_levels = self.depth_analyzer.analyze_support_resistance(snapshot)
                result.liquidity_vacuum_zones = self.depth_analyzer.identify_liquidity_vacuum_zones(snapshot)

            # Step 2: Order flow analysis (dynamic confirmation)
            if trade_data_list and (result.support_levels or result.resistance_levels):
                confirmed_support, confirmed_resistance, poc_levels = self.order_flow_analyzer.analyze_order_flow(
                    trade_data_list, result.support_levels, result.resistance_levels
                )

                # Use confirmed levels if available, otherwise fall back to original
                result.support_levels = confirmed_support if confirmed_support else result.support_levels
                result.resistance_levels = confirmed_resistance if confirmed_resistance else result.resistance_levels
                result.poc_levels = poc_levels

            # Step 3: Find resonance zones (where multiple signals align)
            result.resonance_zones = self._find_resonance_zones(result)

            logger.info(
                f"Market analysis completed: "
                f"{len(result.support_levels)} supports, "
                f"{len(result.resistance_levels)} resistances, "
                f"{len(result.resonance_zones)} resonance zones"
            )

            return result

        except Exception as e:
            logger.error(f"Error during market analysis: {e}")
            # Return a minimal valid result instead of failing
            return MarketAnalysisResult(
                timestamp=datetime.now(),
                symbol=symbol,
                support_levels=[],
                resistance_levels=[],
                resonance_zones=[],
                liquidity_vacuum_zones=[],
                poc_levels=[]
            )

    def _find_resonance_zones(self, result: MarketAnalysisResult) -> list[Decimal]:
        """Find resonance zones where multiple signals align."""
        resonance_zones = []

        # Create price level signal map
        price_signals = defaultdict(int)

        # Add support levels
        for level in result.support_levels:
            price_signals[level.price] += 2 if level.strength > 0.5 else 1

        # Add resistance levels
        for level in result.resistance_levels:
            price_signals[level.price] += 2 if level.strength > 0.5 else 1

        # Add POC levels
        for poc_price in result.poc_levels:
            price_signals[poc_price] += 1

        # Find zones with multiple signals
        for price, signal_count in price_signals.items():
            if signal_count >= 3:  # Minimum 3 signals to qualify as resonance
                resonance_zones.append(price)

        return sorted(resonance_zones)
