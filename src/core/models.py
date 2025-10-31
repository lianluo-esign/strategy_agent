"""Data models for the Strategy Agent system."""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from decimal import ROUND_HALF_UP, Decimal
from typing import TYPE_CHECKING, Any

logger = logging.getLogger(__name__)

# Import classes from other modules for type hints
if TYPE_CHECKING:
    from .wave_peak_analyzer import PriceZone, WavePeak


@dataclass
class DepthLevel:
    """Represents a single level in the order book depth."""

    price: Decimal
    quantity: Decimal

    def __post_init__(self) -> None:
        """Convert to Decimal if needed."""
        if not isinstance(self.price, Decimal):
            self.price = Decimal(str(self.price))
        if not isinstance(self.quantity, Decimal):
            self.quantity = Decimal(str(self.quantity))


@dataclass
class DepthSnapshot:
    """Represents a complete order book depth snapshot."""

    symbol: str
    timestamp: datetime
    bids: list[DepthLevel] = field(default_factory=list)
    asks: list[DepthLevel] = field(default_factory=list)

    def get_bid_price_levels(self) -> list[Decimal]:
        """Get all bid price levels."""
        return [level.price for level in self.bids]

    def get_ask_price_levels(self) -> list[Decimal]:
        """Get all ask price levels."""
        return [level.price for level in self.asks]

    def get_best_bid(self) -> Decimal | None:
        """Get the best bid price."""
        return max(self.get_bid_price_levels()) if self.bids else None

    def get_best_ask(self) -> Decimal | None:
        """Get the best ask price."""
        return min(self.get_ask_price_levels()) if self.asks else None


@dataclass
class Trade:
    """Represents a single trade event."""

    symbol: str
    price: Decimal
    quantity: Decimal
    is_buyer_maker: bool
    timestamp: datetime
    trade_id: str

    def __post_init__(self) -> None:
        """Convert to Decimal if needed."""
        if not isinstance(self.price, Decimal):
            self.price = Decimal(str(self.price))
        if not isinstance(self.quantity, Decimal):
            self.quantity = Decimal(str(self.quantity))


@dataclass
class PriceLevelData:
    """Aggregated trade data for a specific price level."""

    price_level: Decimal
    buy_volume: Decimal = Decimal("0")
    sell_volume: Decimal = Decimal("0")
    total_volume: Decimal = Decimal("0")
    delta: Decimal = Decimal("0")  # buy_volume - sell_volume
    trade_count: int = 0

    def add_trade(self, trade: Trade) -> None:
        """Add a trade to this price level."""
        self.total_volume += trade.quantity
        self.trade_count += 1

        if trade.is_buyer_maker:
            # If buyer is maker, it's a sell trade (aggressive seller)
            self.sell_volume += trade.quantity
        else:
            # If seller is maker, it's a buy trade (aggressive buyer)
            self.buy_volume += trade.quantity

        self.delta = self.buy_volume - self.sell_volume

    def to_dict(self) -> dict:
        """Convert to dictionary for Redis storage."""
        return {
            "price_level": float(self.price_level),
            "buy_volume": float(self.buy_volume),
            "sell_volume": float(self.sell_volume),
            "total_volume": float(self.total_volume),
            "delta": float(self.delta),
            "trade_count": self.trade_count,
        }


@dataclass
class MinuteTradeData:
    """Aggregated trade data for a one-minute interval with thread-safe operations."""

    timestamp: datetime
    price_levels: dict[Decimal, PriceLevelData] = field(default_factory=dict)
    max_price_levels: int = 1000  # Memory limit for price levels
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)  # Async lock for thread safety

    async def add_trade(self, trade: Trade) -> None:
        """Add a trade to the appropriate price level in a thread-safe manner."""
        async with self._lock:
            await self._add_trade_unsafe(trade)

    def add_trade_sync(self, trade: Trade) -> None:
        """Add a trade synchronously (for backward compatibility)."""
        # Note: This method is not thread-safe and should be used carefully
        if len(self.price_levels) >= self.max_price_levels:
            logger.warning(
                f"Maximum price levels ({self.max_price_levels}) reached, skipping trade"
            )
            return

        # Round price to $1 precision
        price_level_key = trade.price.quantize(Decimal("1"), rounding=ROUND_HALF_UP)

        if price_level_key not in self.price_levels:
            self.price_levels[price_level_key] = PriceLevelData(
                price_level=price_level_key
            )

        self.price_levels[price_level_key].add_trade(trade)

    async def _add_trade_unsafe(self, trade: Trade) -> None:
        """Internal unsafe trade addition (assumes lock is held)."""
        # Check memory limit
        if len(self.price_levels) >= self.max_price_levels:
            logger.warning(
                f"Maximum price levels ({self.max_price_levels}) reached, skipping trade"
            )
            return

        # Round price to $1 precision
        price_level_key = trade.price.quantize(Decimal("1"), rounding=ROUND_HALF_UP)

        if price_level_key not in self.price_levels:
            self.price_levels[price_level_key] = PriceLevelData(
                price_level=price_level_key
            )

        self.price_levels[price_level_key].add_trade(trade)

    async def cleanup_low_volume_levels(
        self, min_volume_threshold: Decimal = Decimal("0.001")
    ) -> int:
        """Remove price levels with very low volume to save memory (thread-safe)."""
        async with self._lock:
            return self._cleanup_low_volume_levels_unsafe(min_volume_threshold)

    def _cleanup_low_volume_levels_unsafe(self, min_volume_threshold: Decimal) -> int:
        """Internal unsafe cleanup (assumes lock is held)."""
        to_remove = [
            price
            for price, data in self.price_levels.items()
            if data.total_volume < min_volume_threshold
        ]

        for price in to_remove:
            del self.price_levels[price]

        if to_remove:
            logger.debug(f"Cleaned up {len(to_remove)} low-volume price levels")

        return len(to_remove)

    def cleanup_low_volume_levels_sync(
        self, min_volume_threshold: Decimal = Decimal("0.001")
    ) -> int:
        """Remove price levels with very low volume to save memory (deprecated sync version)."""
        logger.warning("Using deprecated sync cleanup method - consider migrating to async version")
        # Fallback to unsafe synchronous operation
        return self._cleanup_low_volume_levels_unsafe(min_volume_threshold)

    async def get_price_levels_copy(self) -> dict[Decimal, PriceLevelData]:
        """Get a thread-safe copy of price levels."""
        async with self._lock:
            return dict(self.price_levels)

    async def get_statistics(self) -> dict[str, Any]:
        """Get current statistics in a thread-safe manner."""
        async with self._lock:
            total_trades = sum(
                level.trade_count for level in self.price_levels.values()
            )
            total_volume = sum(
                level.total_volume for level in self.price_levels.values()
            )
            return {
                "price_levels_count": len(self.price_levels),
                "total_trades": total_trades,
                "total_volume": float(total_volume),
                "max_price_levels": self.max_price_levels
            }

    async def to_dict_async(self) -> dict:
        """Convert to dictionary for Redis storage in a thread-safe manner."""
        async with self._lock:
            return {
                "timestamp": self.timestamp.isoformat(),
                "price_levels": {str(k): v.to_dict() for k, v in self.price_levels.items()},
            }

    def to_dict(self) -> dict:
        """Convert to dictionary for Redis storage (synchronous - use with caution)."""
        # Note: This method is not thread-safe and should only be used when you know
        # there are no concurrent modifications happening
        return {
            "timestamp": self.timestamp.isoformat(),
            "price_levels": {str(k): v.to_dict() for k, v in self.price_levels.items()},
        }


@dataclass
class SupportResistanceLevel:
    """Represents a support or resistance level."""

    price: Decimal
    strength: float  # 0.0 to 1.0
    level_type: str  # 'support' or 'resistance'
    volume_at_level: Decimal
    confirmation_count: int = 0
    last_confirmed: datetime | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "price": float(self.price),
            "strength": self.strength,
            "level_type": self.level_type,
            "volume_at_level": float(self.volume_at_level),
            "confirmation_count": self.confirmation_count,
            "last_confirmed": self.last_confirmed.isoformat()
            if self.last_confirmed
            else None,
        }


@dataclass
class MarketAnalysisResult:
    """Results from market analysis."""

    timestamp: datetime
    symbol: str
    support_levels: list[SupportResistanceLevel] = field(default_factory=list)
    resistance_levels: list[SupportResistanceLevel] = field(default_factory=list)
    poc_levels: list[Decimal] = field(default_factory=list)  # Point of Control levels
    liquidity_vacuum_zones: list[Decimal] = field(default_factory=list)
    resonance_zones: list[Decimal] = field(
        default_factory=list
    )  # High-probability zones

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "support_levels": [level.to_dict() for level in self.support_levels],
            "resistance_levels": [level.to_dict() for level in self.resistance_levels],
            "poc_levels": [float(poc) for poc in self.poc_levels],
            "liquidity_vacuum_zones": [
                float(zone) for zone in self.liquidity_vacuum_zones
            ],
            "resonance_zones": [float(zone) for zone in self.resonance_zones],
        }


@dataclass
class TradingRecommendation:
    """Trading recommendation from AI analysis."""

    timestamp: datetime
    symbol: str
    action: str  # 'buy', 'sell', 'hold'
    price_range: tuple[Decimal, Decimal]  # Recommended price range
    confidence: float  # 0.0 to 1.0
    reasoning: str
    risk_level: str  # 'low', 'medium', 'high'

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "action": self.action,
            "price_range": [float(self.price_range[0]), float(self.price_range[1])],
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "risk_level": self.risk_level,
        }


@dataclass
class EnhancedMarketAnalysisResult:
    """Enhanced market analysis result with wave peak detection."""

    timestamp: datetime
    symbol: str

    # 1-dollar precision aggregated data
    aggregated_bids: dict[Decimal, Decimal] = field(default_factory=dict)
    aggregated_asks: dict[Decimal, Decimal] = field(default_factory=dict)

    # Wave peak analysis
    wave_peaks: list["WavePeak"] = field(default_factory=list)
    support_zones: list["PriceZone"] = field(default_factory=list)
    resistance_zones: list["PriceZone"] = field(default_factory=list)

    # Traditional analysis (backward compatibility)
    support_levels: list[SupportResistanceLevel] = field(default_factory=list)
    resistance_levels: list[SupportResistanceLevel] = field(default_factory=list)
    poc_levels: list[Decimal] = field(default_factory=list)
    liquidity_vacuum_zones: list[Decimal] = field(default_factory=list)
    resonance_zones: list[Decimal] = field(default_factory=list)

    # Statistics and quality metrics
    depth_statistics: dict[str, Decimal] = field(default_factory=dict)
    peak_detection_quality: dict[str, float] = field(default_factory=dict)

    # Normal distribution analysis results
    normal_distribution_peaks: dict[str, Any] = field(default_factory=dict)
    confidence_intervals: dict[str, Any] = field(default_factory=dict)
    market_metrics: dict[str, Any] = field(default_factory=dict)
    spread_analysis: dict[str, Any] = field(default_factory=dict)

    # Liquidity peak identification (simplified from detailed clustering)
    liquidity_peaks: list[SupportResistanceLevel] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "aggregated_bids": {
                str(k): float(v) for k, v in self.aggregated_bids.items()
            },
            "aggregated_asks": {
                str(k): float(v) for k, v in self.aggregated_asks.items()
            },
            "wave_peaks": [peak.to_dict() for peak in self.wave_peaks],
            "support_zones": [zone.to_dict() for zone in self.support_zones],
            "resistance_zones": [zone.to_dict() for zone in self.resistance_zones],
            "support_levels": [level.to_dict() for level in self.support_levels],
            "resistance_levels": [level.to_dict() for level in self.resistance_levels],
            "poc_levels": [float(poc) for poc in self.poc_levels],
            "liquidity_vacuum_zones": [
                float(zone) for zone in self.liquidity_vacuum_zones
            ],
            "resonance_zones": [float(zone) for zone in self.resonance_zones],
            "depth_statistics": {k: float(v) for k, v in self.depth_statistics.items()},
            "peak_detection_quality": self.peak_detection_quality,
            # Normal distribution analysis
            "normal_distribution_peaks": self.normal_distribution_peaks,
            "confidence_intervals": self.confidence_intervals,
            "market_metrics": self.market_metrics,
            "spread_analysis": self.spread_analysis,
            # Liquidity peak identification
            "liquidity_peaks": [peak.to_dict() for peak in self.liquidity_peaks],
        }
