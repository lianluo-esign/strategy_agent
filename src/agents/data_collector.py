"""Data collector agent for market data acquisition."""

import asyncio
import logging
import signal
import sys
from datetime import datetime, timedelta
from typing import Any, Optional

from ..core.models import MinuteTradeData, Trade
from ..core.redis_client import RedisDataStore
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
            db=settings.redis.db,
            storage_dir=settings.redis.storage_dir
        )
        self.api_client = BinanceAPIClient(
            base_url=settings.binance.rest_api_base,
            timeout=settings.binance.timeout
        )
        self.websocket_client = BinanceWebSocketClient(symbol=settings.binance.symbol)

        # Trade aggregation state
        self.current_minute_data = MinuteTradeData(timestamp=datetime.now())
        self.last_aggregation_time = datetime.now()

        # Control flags and shutdown management
        self.is_running = False
        self.shutdown_event = asyncio.Event()
        self.tasks: list[asyncio.Task] = []
        self.loop: asyncio.AbstractEventLoop | None = None

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum: int, frame: Any) -> None:
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.is_running = False

        # Trigger shutdown event
        if not self.shutdown_event.is_set():
            self.shutdown_event.set()

        # If we have an event loop, schedule immediate shutdown
        if self.loop and not self.loop.is_closed():
            self.loop.call_soon_threadsafe(self._schedule_immediate_shutdown)

    def _schedule_immediate_shutdown(self) -> None:
        """Schedule immediate shutdown from the event loop."""
        logger.info("Scheduling immediate task cancellation...")
        # Cancel all running tasks
        for task in self.tasks:
            if not task.done():
                task.cancel()

    async def start(self) -> None:
        """Start the data collection process."""
        logger.info("Starting Data Collector Agent")

        # Store current event loop for signal handling
        self.loop = asyncio.get_running_loop()

        # Test Redis connection
        if not self.redis_store.test_connection():
            logger.error("Failed to connect to Redis. Exiting...")
            return

        # Initialize with depth snapshot
        await self._initialize_depth_snapshot()

        # Start concurrent tasks with proper cancellation handling
        try:
            self.is_running = True

            # Create tasks and store references for cancellation
            task1 = asyncio.create_task(self._depth_snapshot_collector())
            task2 = asyncio.create_task(self._websocket_trade_collector())
            task3 = asyncio.create_task(self._trade_aggregator())

            self.tasks = [task1, task2, task3]

            # Wait for tasks with proper exception handling
            await asyncio.gather(*self.tasks, return_exceptions=True)

        except asyncio.CancelledError:
            logger.info("Tasks cancelled, shutting down...")
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

                # Wait for next collection with cancellation support
                try:
                    await asyncio.sleep(interval)  # asyncio.sleep is already cancellable
                except asyncio.CancelledError:
                    logger.info("Depth snapshot collector cancelled during sleep")
                    break

            except asyncio.CancelledError:
                logger.info("Depth snapshot collector cancelled")
                break
            except Exception as e:
                logger.error(f"Depth snapshot collector error: {e}")
                try:
                    await asyncio.sleep(5)
                except asyncio.CancelledError:
                    break

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

                    # Use asyncio.sleep with cancellation support
                    try:
                        await asyncio.sleep(wait_time)  # asyncio.sleep is already cancellable
                    except asyncio.CancelledError:
                        logger.info("WebSocket retry cancelled during sleep")
                        break

                    continue

                retry_count = 0  # Reset retry count on successful connection

                # Listen for trades with cancellation support
                await self._listen_trades_with_cancellation()

            except asyncio.CancelledError:
                logger.info("WebSocket collector cancelled")
                break
            except Exception as e:
                logger.error(f"WebSocket collector error: {e}")
                retry_count += 1
                if self.is_running:
                    try:
                        await asyncio.sleep(5)  # asyncio.sleep is already cancellable
                    except asyncio.CancelledError:
                        break

            finally:
                await self.websocket_client.disconnect()

        if retry_count >= max_retries:
            logger.error("Max WebSocket retries exceeded, stopping trade collection")

    async def _listen_trades_with_cancellation(self) -> None:
        """Listen for trades with proper cancellation support."""
        try:
            # Create a cancellation-aware task for WebSocket listening
            listen_task = asyncio.create_task(self.websocket_client.listen_trades(self._handle_trade))
            shutdown_task = asyncio.create_task(self.shutdown_event.wait())

            # Wait for either the task to complete or shutdown to be requested
            done, pending = await asyncio.wait(
                [listen_task, shutdown_task],
                return_when=asyncio.FIRST_COMPLETED
            )

            # Cancel pending tasks
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

            # Handle completed tasks
            for task in done:
                if task == shutdown_task:  # shutdown_event completed
                    logger.info("Shutdown requested, stopping WebSocket listener")
                    if not listen_task.done():
                        listen_task.cancel()
                        try:
                            await listen_task
                        except asyncio.CancelledError:
                            pass
                elif task.exception():
                    # Re-raise exceptions from the listen task
                    task.result()  # This will raise the exception

        except asyncio.CancelledError:
            logger.info("WebSocket listening cancelled")
            raise

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

                # Check every second with cancellation support
                try:
                    await asyncio.sleep(1)  # asyncio.sleep is already cancellable
                except asyncio.CancelledError:
                    logger.info("Trade aggregator cancelled during sleep")
                    break

            except asyncio.CancelledError:
                logger.info("Trade aggregator cancelled")
                break
            except Exception as e:
                logger.error(f"Trade aggregator error: {e}", exc_info=True)
                try:
                    await asyncio.sleep(5)
                except asyncio.CancelledError:
                    break

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
        """Cleanup and shutdown the agent with timeout protection."""
        logger.info("Shutting down Data Collector Agent")

        self.is_running = False

        # Store any remaining aggregated data
        if self.current_minute_data.price_levels:
            try:
                await asyncio.wait_for(self._aggregate_and_store_minute_data(), timeout=5)
            except TimeoutError:
                logger.warning("Timeout storing remaining aggregated data")
            except Exception as e:
                logger.error(f"Error storing remaining data: {e}")

        # Cancel any remaining tasks
        await self._cancel_remaining_tasks()

        # Close connections with timeout
        await self._close_connections_with_timeout()

        logger.info("Data Collector Agent shutdown complete")

    async def _cancel_remaining_tasks(self) -> None:
        """Cancel any remaining tasks with timeout."""
        if not self.tasks:
            return

        logger.info(f"Cancelling {len(self.tasks)} remaining tasks...")

        # Cancel all tasks
        for task in self.tasks:
            if not task.done():
                task.cancel()

        # Wait for tasks to complete with timeout
        try:
            await asyncio.wait_for(
                asyncio.gather(*self.tasks, return_exceptions=True),
                timeout=10
            )
        except TimeoutError:
            logger.warning("Timeout waiting for tasks to cancel")
        except Exception as e:
            logger.error(f"Error cancelling tasks: {e}")

        self.tasks.clear()

    async def _close_connections_with_timeout(self) -> None:
        """Close connections with timeout protection."""
        logger.info("Closing connections...")

        # Close WebSocket connection
        try:
            await asyncio.wait_for(self.websocket_client.disconnect(), timeout=5)
        except TimeoutError:
            logger.warning("Timeout closing WebSocket connection")
        except Exception as e:
            logger.error(f"Error closing WebSocket: {e}")

        # Close Redis connection
        try:
            await asyncio.wait_for(self.redis_store.close(), timeout=5)
        except TimeoutError:
            logger.warning("Timeout closing Redis connection")
        except Exception as e:
            logger.error(f"Error closing Redis: {e}")

        # Close API session
        try:
            await asyncio.wait_for(self.api_client.close_async_session(), timeout=5)
        except TimeoutError:
            logger.warning("Timeout closing API session")
        except Exception as e:
            logger.error(f"Error closing API session: {e}")

    def get_status(self) -> dict:
        """Get current agent status."""
        return {
            'is_running': self.is_running,
            'websocket_connected': self.websocket_client.is_connected,
            'current_minute_trades': len(self.current_minute_data.price_levels),
            'depth_snapshot_available': self.redis_store.depth_snapshot_exists(),
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
        # Start with timeout protection for graceful shutdown
        await asyncio.wait_for(agent.start(), timeout=None)  # No timeout for normal operation
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        # The signal handler will trigger graceful shutdown
    except TimeoutError:
        logger.error("Operation timed out")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        # This handles the case where Ctrl+C is pressed before asyncio.run completes
        print("\nReceived interrupt signal, exiting...")
        sys.exit(0)
