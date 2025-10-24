"""Redis client for caching market data."""

import asyncio
import json
import logging
from datetime import datetime
from decimal import Decimal
from pathlib import Path

import aiofiles
import redis
from redis.asyncio import Redis as AsyncRedis

from .constants import (
    REDIS_ANALYSIS_RESULTS_KEY,
    REDIS_DEPTH_SNAPSHOT_KEY,
    REDIS_TRADES_WINDOW_KEY,
    TRADES_WINDOW_SIZE_MINUTES,
)
from .models import DepthLevel, DepthSnapshot, MarketAnalysisResult, MinuteTradeData

logger = logging.getLogger(__name__)


class RedisDataStore:
    """Redis client for storing and retrieving market data."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        storage_dir: str = "storage",
    ):
        """Initialize Redis connection."""
        self.redis = redis.Redis(
            host=host,
            port=port,
            db=db,
            decode_responses=True,
            socket_timeout=5,
            socket_connect_timeout=5,
        )
        self.async_redis = AsyncRedis(
            host=host, port=port, db=db, decode_responses=True
        )
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)

    def test_connection(self) -> bool:
        """Test Redis connection."""
        try:
            self.redis.ping()
            logger.info("Redis connection successful")
            return True
        except Exception as e:
            logger.error(f"Redis connection failed: {e}")
            return False

    async def store_depth_snapshot(self, snapshot: DepthSnapshot) -> None:
        """Store a depth snapshot in Redis (overwrite mode)."""
        try:
            data = {
                "symbol": snapshot.symbol,
                "timestamp": snapshot.timestamp.isoformat(),
                "bids": [
                    [float(level.price), float(level.quantity)]
                    for level in snapshot.bids
                ],
                "asks": [
                    [float(level.price), float(level.quantity)]
                    for level in snapshot.asks
                ],
            }

            # Store/overwrite the snapshot as a single value
            self.redis.set(REDIS_DEPTH_SNAPSHOT_KEY, json.dumps(data))

            logger.debug(
                f"Stored depth snapshot for {snapshot.symbol} at {snapshot.timestamp}"
            )

        except Exception as e:
            logger.error(f"Failed to store depth snapshot: {e}")
            raise

    def get_latest_depth_snapshot(self) -> DepthSnapshot | None:
        """Get the depth snapshot."""
        try:
            data_str = self.redis.get(REDIS_DEPTH_SNAPSHOT_KEY)
            if not data_str:
                return None

            data = json.loads(data_str)
            return DepthSnapshot(
                symbol=data["symbol"],
                timestamp=datetime.fromisoformat(data["timestamp"]),
                bids=[
                    DepthLevel(price=Decimal(str(price)), quantity=Decimal(str(qty)))
                    for price, qty in data["bids"]
                ],
                asks=[
                    DepthLevel(price=Decimal(str(price)), quantity=Decimal(str(qty)))
                    for price, qty in data["asks"]
                ],
            )

        except Exception as e:
            logger.error(f"Failed to get depth snapshot: {e}")
            return None

    async def store_minute_trade_data(self, trade_data: MinuteTradeData) -> None:
        """Store minute trade data in the sliding window."""
        try:
            data = trade_data.to_dict()

            # Store the data
            self.redis.lpush(REDIS_TRADES_WINDOW_KEY, json.dumps(data))

            # Check if we need to remove expired data and serialize to disk
            await self._handle_expired_trade_data()

            logger.debug(f"Stored minute trade data for {trade_data.timestamp}")

        except Exception as e:
            logger.error(f"Failed to store minute trade data: {e}")
            raise

    async def _handle_expired_trade_data(self) -> None:
        """Handle expired trade data by serializing to disk asynchronously."""
        try:
            # Get current number of items in the trades window
            current_count = self.redis.llen(REDIS_TRADES_WINDOW_KEY)

            if current_count > TRADES_WINDOW_SIZE_MINUTES:
                # Calculate how many items are expired
                expired_count = current_count - TRADES_WINDOW_SIZE_MINUTES

                # Get the expired items (they are at the end of the list)
                expired_items = self.redis.lrange(
                    REDIS_TRADES_WINDOW_KEY, TRADES_WINDOW_SIZE_MINUTES, -1
                )

                # Serialize expired items to disk asynchronously
                await self._serialize_trade_data_to_files(expired_items)

                # Trim the list to keep only the recent data
                self.redis.ltrim(
                    REDIS_TRADES_WINDOW_KEY, 0, TRADES_WINDOW_SIZE_MINUTES - 1
                )

                logger.info(
                    f"Serialized {expired_count} expired trade data items to disk"
                )

        except Exception as e:
            logger.error(f"Failed to handle expired trade data: {e}")
            # Don't raise here to avoid interrupting main flow

    async def _serialize_trade_data_to_files(self, expired_items: list[str]) -> None:
        """Serialize expired trade data to JSON files asynchronously."""
        try:
            # Create concurrent tasks for file writing
            tasks = [self._write_trade_data_file(item) for item in expired_items]

            # Execute all file writes concurrently
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

        except Exception as e:
            logger.error(f"Failed to serialize trade data to files: {e}")

    async def _write_trade_data_file(self, data_str: str) -> None:
        """Write a single trade data item to a JSON file asynchronously."""
        try:
            data = json.loads(data_str)
            timestamp = datetime.fromisoformat(data["timestamp"])

            # Create filename based on timestamp (one file per minute)
            filename = f"trades_{timestamp.strftime('%Y%m%d_%H%M')}.json"
            filepath = self.storage_dir / filename

            # Write data to file asynchronously
            async with aiofiles.open(filepath, "w") as f:
                await f.write(json.dumps(data, indent=2, ensure_ascii=False))

            logger.debug(f"Serialized trade data to {filepath}")

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON data for file serialization: {e}")
        except OSError as e:
            logger.error(f"File system error writing trade data: {e}")
        except Exception as e:
            logger.error(f"Failed to write trade data file: {e}")

    def get_recent_trade_data(self, minutes: int = 60) -> list[MinuteTradeData]:
        """Get recent trade data for analysis."""
        try:
            data_list = self.redis.lrange(REDIS_TRADES_WINDOW_KEY, 0, minutes - 1)
            trade_data_list = []

            for data_str in data_list:
                try:
                    data = json.loads(data_str)
                    trade_data = MinuteTradeData(
                        timestamp=datetime.fromisoformat(data["timestamp"])
                    )

                    # Reconstruct price level data
                    for price_str, price_data in data["price_levels"].items():
                        price_level = Decimal(price_str)
                        trade_data.price_levels[price_level] = price_data

                    trade_data_list.append(trade_data)

                except Exception as e:
                    logger.warning(f"Failed to parse trade data: {e}")
                    continue

            return trade_data_list

        except Exception as e:
            logger.error(f"Failed to get recent trade data: {e}")
            return []

    async def store_analysis_result(self, result: MarketAnalysisResult) -> None:
        """Store market analysis results."""
        try:
            key = f"{REDIS_ANALYSIS_RESULTS_KEY}:{result.timestamp.timestamp()}"
            data = result.to_dict()

            # Store the result
            self.redis.setex(key, 3600, json.dumps(data))  # Expire after 1 hour

            logger.debug(
                f"Stored analysis result for {result.symbol} at {result.timestamp}"
            )

        except Exception as e:
            logger.error(f"Failed to store analysis result: {e}")
            raise

    def get_latest_analysis_result(self) -> MarketAnalysisResult | None:
        """Get the most recent analysis result."""
        try:
            # Get all analysis result keys and find the most recent
            keys = self.redis.keys(f"{REDIS_ANALYSIS_RESULTS_KEY}:*")
            if not keys:
                return None

            # Sort keys by timestamp (extracted from key name)
            latest_key = max(keys, key=lambda k: float(k.split(":")[-1]))
            data_str = self.redis.get(latest_key)

            if not data_str:
                return None

            data = json.loads(data_str)
            # Reconstruct MarketAnalysisResult from data
            result = MarketAnalysisResult(
                timestamp=datetime.fromisoformat(data["timestamp"]),
                symbol=data["symbol"],
            )

            # Add other fields as needed
            return result

        except Exception as e:
            logger.error(f"Failed to get latest analysis result: {e}")
            return None

    def depth_snapshot_exists(self) -> bool:
        """Check if a depth snapshot exists."""
        return self.redis.exists(REDIS_DEPTH_SNAPSHOT_KEY) > 0

    def get_trade_window_count(self) -> int:
        """Get the number of stored trade data points."""
        return self.redis.llen(REDIS_TRADES_WINDOW_KEY)

    def clear_all_data(self) -> None:
        """Clear all stored data (for testing)."""
        try:
            self.redis.delete(REDIS_DEPTH_SNAPSHOT_KEY)
            self.redis.delete(REDIS_TRADES_WINDOW_KEY)

            # Delete all analysis results
            keys = self.redis.keys(f"{REDIS_ANALYSIS_RESULTS_KEY}:*")
            if keys:
                self.redis.delete(*keys)

            logger.info("Cleared all data from Redis")

        except Exception as e:
            logger.error(f"Failed to clear data: {e}")
            raise

    async def close(self) -> None:
        """Close Redis connections."""
        try:
            if hasattr(self.async_redis, "aclose"):
                await self.async_redis.aclose()
        except Exception as e:
            logger.warning(f"Error closing async Redis connection: {e}")

        try:
            if hasattr(self.redis, "close"):
                self.redis.close()
        except Exception as e:
            logger.warning(f"Error closing Redis connection: {e}")
