"""Data collector agent for market data acquisition."""

import asyncio
import logging
import signal
import sys
from datetime import datetime, timedelta
from typing import Optional

from ..core.redis_client import RedisDataStore
from ..core.models import MinuteTradeData, Trade
from ..utils.binance_client import BinanceAPIClient, BinanceWebSocketClient
from ..utils.config import Settings

logger = logging.getLogger(__name__)


class DataCollectorAgent:
    """Agent responsible for collecting and storing market data."""

    def __init__(self, settings: Settings):
        """Initialize the data collector agent."""
        self.settings = settings
        self.redis_store = RedisDataStore(
            host=settings.redis.host,
            port=settings.redis.port,
            db=settings.redis.db
        )
        self.api_client = BinanceAPIClient(
            base_url=settings.binance.rest_api_base,
            timeout=settings.binance.timeout
        )
        self.websocket_client = BinanceWebSocketClient(symbol=settings.binance.symbol)

        # Trade aggregation state
        self.current_minute_data = MinuteTradeData(timestamp=datetime.now())
        self.last_aggregation_time = datetime.now()

        # Control flags
        self.is_running = False
        self.shutdown_event = asyncio.Event()

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame) -> None:
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, initiating shutdown...")
        self.is_running = False
        self.shutdown_event.set()

    async def start(self) -> None:
        """Start the data collection process."""
        logger.info("Starting Data Collector Agent")

        # Test Redis connection
        if not self.redis_store.test_connection():
            logger.error("Failed to connect to Redis. Exiting...")
            return

        # Initialize with depth snapshot
        await self._initialize_depth_snapshot()

        # Start concurrent tasks
        tasks = [
            self._depth_snapshot_collector(),
            self._websocket_trade_collector(),
            self._trade_aggregator()
        ]

        try:
            self.is_running = True
            await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"Data collector error: {e}")
        finally:
            await self._shutdown()

    async def _initialize_depth_snapshot(self) -> None:
        """Initialize with a depth snapshot."""
        logger.info("Fetching initial depth snapshot")
        snapshot = await self.api_client.get_depth_snapshot(
            symbol=self.settings.binance.symbol,
            limit=self.settings.data_collector.depth_snapshot.limit
        )

        if snapshot:
            await self.redis_store.store_depth_snapshot(snapshot)
            logger.info(f"Initial depth snapshot stored for {snapshot.symbol}")
        else:
            logger.error("Failed to fetch initial depth snapshot")

    async def _depth_snapshot_collector(self) -> None:
        """Periodically collect depth snapshots."""
        interval = self.settings.data_collector.depth_snapshot.update_interval_seconds

        while self.is_running:
            try:
                logger.debug("Collecting depth snapshot")
                snapshot = await self.api_client.get_depth_snapshot(
                    symbol=self.settings.binance.symbol,
                    limit=self.settings.data_collector.depth_snapshot.limit
                )

                if snapshot:
                    await self.redis_store.store_depth_snapshot(snapshot)
                    logger.debug(f"Depth snapshot stored for {snapshot.symbol}")
                else:
                    logger.warning("Failed to fetch depth snapshot")

                # Wait for next collection
                await asyncio.sleep(interval)

            except Exception as e:
                logger.error(f"Depth snapshot collector error: {e}")
                await asyncio.sleep(5)  # Short delay before retry

    async def _websocket_trade_collector(self) -> None:
        """Collect real-time trade data via WebSocket."""
        max_retries = 5
        retry_count = 0

        while self.is_running and retry_count < max_retries:
            try:
                logger.info("Connecting to trade WebSocket")
                connected = await self.websocket_client.connect()

                if not connected:
                    retry_count += 1
                    wait_time = min(2 ** retry_count, 30)  # Exponential backoff
                    logger.warning(f"WebSocket connection failed, retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                    continue

                retry_count = 0  # Reset retry count on successful connection

                # Listen for trades
                await self.websocket_client.listen_trades(self._handle_trade)

            except Exception as e:
                logger.error(f"WebSocket collector error: {e}")
                retry_count += 1
                await asyncio.sleep(5)

            finally:
                await self.websocket_client.disconnect()

        if retry_count >= max_retries:
            logger.error("Max WebSocket retries exceeded, stopping trade collection")

    async def _handle_trade(self, trade: Trade) -> None:
        """Handle incoming trade data."""
        try:
            # Add trade to current minute aggregation
            self.current_minute_data.add_trade(trade)
            logger.debug(f"Processed trade: {trade.price}@{trade.quantity}")

        except Exception as e:
            logger.error(f"Error handling trade: {e}")

    async def _trade_aggregator(self) -> None:
        """Aggregate trades by minute and store to Redis."""
        interval = self.settings.data_collector.order_flow.aggregation_interval_seconds

        while self.is_running:
            try:
                current_time = datetime.now()

                # Check if we should aggregate current minute's data
                if current_time >= self.last_aggregation_time + timedelta(seconds=interval):
                    await self._aggregate_and_store_minute_data()

                    # Reset for next minute
                    self.current_minute_data = MinuteTradeData(timestamp=current_time)
                    self.last_aggregation_time = current_time

                await asyncio.sleep(1)  # Check every second

            except Exception as e:
                logger.error(f"Trade aggregator error: {e}")
                await asyncio.sleep(5)

    async def _aggregate_and_store_minute_data(self) -> None:
        """Store the aggregated minute data to Redis."""
        if not self.current_minute_data.price_levels:
            logger.debug("No trades to aggregate for current minute")
            return

        try:
            await self.redis_store.store_minute_trade_data(self.current_minute_data)

            total_trades = sum(
                level.trade_count for level in self.current_minute_data.price_levels.values()
            )
            total_volume = sum(
                level.total_volume for level in self.current_minute_data.price_levels.values()
            )

            logger.debug(
                f"Stored minute data: {total_trades} trades, "
                f"{total_volume:.4f} volume across {len(self.current_minute_data.price_levels)} price levels"
            )

        except Exception as e:
            logger.error(f"Failed to store minute data: {e}")

    async def _shutdown(self) -> None:
        """Cleanup and shutdown the agent."""
        logger.info("Shutting down Data Collector Agent")

        self.is_running = False

        # Store any remaining aggregated data
        if self.current_minute_data.price_levels:
            await self._aggregate_and_store_minute_data()

        # Close connections
        await self.websocket_client.disconnect()
        await self.redis_store.close()
        await self.api_client.close_async_session()

        logger.info("Data Collector Agent shutdown complete")

    def get_status(self) -> dict:
        """Get current agent status."""
        return {
            'is_running': self.is_running,
            'websocket_connected': self.websocket_client.is_connected,
            'current_minute_trades': len(self.current_minute_data.price_levels),
            'depth_snapshots_count': self.redis_store.get_depth_snapshot_count(),
            'trade_window_count': self.redis_store.get_trade_window_count(),
            'last_update': datetime.now().isoformat()
        }


async def main() -> None:
    """Main entry point for the data collector agent."""
    import argparse

    parser = argparse.ArgumentParser(description="Strategy Agent Data Collector")
    parser.add_argument(
        "--config",
        default="config/development.yaml",
        help="Configuration file path"
    )
    args = parser.parse_args()

    # Load settings
    settings = Settings.load_from_file(args.config)
    settings.setup_logging()

    # Create and start agent
    agent = DataCollectorAgent(settings)

    try:
        await agent.start()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())