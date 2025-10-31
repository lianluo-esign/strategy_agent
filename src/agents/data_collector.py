"""Data collector agent for market data acquisition."""

import asyncio
import logging
import signal
import sys
from datetime import datetime, timedelta
from typing import Any

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
            storage_dir=settings.redis.storage_dir,
            max_storage_files=settings.redis.max_storage_files,
        )
        self.api_client = BinanceAPIClient(
            base_url=settings.binance.rest_api_base, timeout=settings.binance.timeout
        )
        self.websocket_client = BinanceWebSocketClient(symbol=settings.binance.symbol)

        # Trade aggregation state
        self.current_minute_data = MinuteTradeData(timestamp=datetime.now())
        self.last_aggregation_time = datetime.now()

        # Monitoring and metrics
        self.start_time = datetime.now()
        self.total_trades_processed = 0
        self.total_connection_failures = 0
        self.last_health_check = None
        self.memory_usage_stats = {
            "peak_price_levels": 0,
            "total_cleanups": 0,
            "last_cleanup_time": None
        }

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
            task4 = asyncio.create_task(self._monitoring_task())  # New monitoring task

            self.tasks = [task1, task2, task3, task4]

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
        logger.debug(
            f"Depth snapshot config: limit={self.settings.data_collector.depth_snapshot.limit}, "
            f"update_interval={self.settings.data_collector.depth_snapshot.update_interval_seconds}s"
        )
        snapshot = await self.api_client.get_depth_snapshot(
            symbol=self.settings.binance.symbol,
            limit=self.settings.data_collector.depth_snapshot.limit,
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
                    limit=self.settings.data_collector.depth_snapshot.limit,
                )

                if snapshot:
                    await self.redis_store.store_depth_snapshot(snapshot)
                    logger.debug(f"Depth snapshot stored for {snapshot.symbol}")
                else:
                    logger.warning("Failed to fetch depth snapshot")

                # Wait for next collection with cancellation support
                try:
                    await asyncio.sleep(
                        interval
                    )  # asyncio.sleep is already cancellable
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
        """Collect real-time trade data via WebSocket with enhanced reconnection logic."""
        max_retries = 10
        retry_count = 0
        consecutive_failures = 0
        max_consecutive_failures = 3

        while self.is_running and retry_count < max_retries:
            try:
                logger.info(f"Connecting to trade WebSocket (attempt {retry_count + 1}/{max_retries})")

                # Reset connection state
                await self.websocket_client.disconnect()
                await asyncio.sleep(1)  # Brief delay before reconnection

                connected = await self.websocket_client.connect()

                if not connected:
                    retry_count += 1
                    consecutive_failures += 1

                    # Exponential backoff with jitter
                    base_wait = min(2 ** retry_count, 60)
                    jitter = base_wait * 0.1 * (hash(str(retry_count)) % 10)
                    wait_time = base_wait + jitter

                    logger.warning(
                        f"WebSocket connection failed (consecutive failures: {consecutive_failures}), "
                        f"retrying in {wait_time:.1f}s..."
                    )

                    # Check if we should abort due to too many consecutive failures
                    if consecutive_failures >= max_consecutive_failures:
                        logger.error(f"Too many consecutive failures ({max_consecutive_failures}), "
                                   "extending backoff period")
                        wait_time *= 2  # Double the wait time

                    # Use asyncio.sleep with cancellation support
                    try:
                        await asyncio.sleep(wait_time)
                    except asyncio.CancelledError:
                        logger.info("WebSocket retry cancelled during sleep")
                        break

                    continue

                # Connection successful - reset counters
                retry_count = 0
                consecutive_failures = 0
                logger.info("WebSocket connection established successfully")

                # Listen for trades with enhanced cancellation and error handling
                await self._listen_trades_with_enhanced_cancellation()

            except asyncio.CancelledError:
                logger.info("WebSocket collector cancelled")
                break
            except Exception as e:
                logger.error(f"WebSocket collector error: {e}", exc_info=True)
                retry_count += 1
                consecutive_failures += 1

                if self.is_running:
                    # Progressive backoff for errors
                    error_wait = min(5 * retry_count, 30)
                    logger.info(f"Error occurred, waiting {error_wait}s before retry...")

                    try:
                        await asyncio.sleep(error_wait)
                    except asyncio.CancelledError:
                        break

            finally:
                try:
                    await self.websocket_client.disconnect()
                except Exception as e:
                    logger.warning(f"Error during WebSocket disconnect: {e}")

        if retry_count >= max_retries:
            logger.error(f"Max WebSocket retries ({max_retries}) exceeded, stopping trade collection")

            # Send notification about extended downtime
            try:
                logger.error("CRITICAL: WebSocket data collection has stopped. Manual intervention may be required.")
            except Exception:
                pass  # Don't let notification errors crash the system

    async def _listen_trades_with_cancellation(self) -> None:
        """Listen for trades with proper cancellation support (legacy method)."""
        return await self._listen_trades_with_enhanced_cancellation()

    async def _listen_trades_with_enhanced_cancellation(self) -> None:
        """Listen for trades with enhanced cancellation, error handling, and health monitoring."""
        listen_task = None
        health_monitor_task = None
        shutdown_task = None

        try:
            logger.info("Starting enhanced WebSocket listening with health monitoring")

            # Create main listening task
            listen_task = asyncio.create_task(
                self.websocket_client.listen_trades(self._handle_trade)
            )

            # Create health monitoring task
            health_monitor_task = asyncio.create_task(
                self._websocket_health_monitor()
            )

            # Create shutdown monitoring task
            shutdown_task = asyncio.create_task(self.shutdown_event.wait())

            # Wait for any task to complete
            done, pending = await asyncio.wait(
                [listen_task, health_monitor_task, shutdown_task],
                return_when=asyncio.FIRST_COMPLETED
            )

            # Analyze which task completed
            completed_task = None
            for task in done:
                completed_task = task
                if task == shutdown_task:
                    logger.info("Shutdown requested, stopping WebSocket listener")
                elif task == health_monitor_task:
                    logger.warning("Health monitor completed - potential issue detected")
                elif task == listen_task:
                    if task.exception():
                        logger.error("WebSocket listening task failed with exception")
                    else:
                        logger.info("WebSocket listening task completed normally")

            # Cancel all pending tasks
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    logger.warning(f"Error cancelling pending task: {e}")

            # Re-raise any exception from the completed task
            if completed_task and completed_task.exception():
                raise completed_task.exception()

        except asyncio.CancelledError:
            logger.info("Enhanced WebSocket listening cancelled")
            raise
        except Exception as e:
            logger.error(f"Enhanced WebSocket listening failed: {e}", exc_info=True)
            raise
        finally:
            # Cleanup tasks
            for task in [listen_task, health_monitor_task, shutdown_task]:
                if task and not task.done():
                    task.cancel()
                    try:
                        await task
                    except (asyncio.CancelledError, Exception):
                        pass

    async def _websocket_health_monitor(self) -> None:
        """Monitor WebSocket health during active listening."""
        logger.debug("Starting WebSocket health monitor")
        last_health_check = datetime.now()
        health_check_interval = 60  # Check health every minute

        while not self.shutdown_event.is_set() and self.websocket_client.is_connected:
            try:
                # Wait for health check interval or shutdown
                try:
                    await asyncio.wait_for(
                        self.shutdown_event.wait(),
                        timeout=health_check_interval
                    )
                    # Shutdown was requested
                    break
                except asyncio.TimeoutError:
                    # Normal timeout, perform health check
                    pass

                # Perform health check
                health_status = await self.websocket_client.health_check()
                if health_status["status"] != "healthy":
                    logger.warning(f"WebSocket health check failed: {health_status}")
                    # Health issues detected - trigger reconnection
                    break

                last_health_check = datetime.now()
                logger.debug("WebSocket health check passed")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"WebSocket health monitor error: {e}")
                await asyncio.sleep(5)  # Brief delay before retrying

        logger.debug("WebSocket health monitor stopped")

    async def _handle_trade(self, trade: Trade) -> None:
        """Handle incoming trade data with thread-safe operations and comprehensive error handling."""
        try:
            # Validate trade data
            if not self._validate_trade(trade):
                logger.warning(f"Invalid trade data received: {trade}")
                return

            # Add trade to current minute aggregation in a thread-safe manner
            await self.current_minute_data.add_trade(trade)
            self.total_trades_processed += 1
            logger.debug(f"Processed trade: {trade.price}@{trade.quantity}")

            # Update monitoring metrics
            stats = await self.current_minute_data.get_statistics()
            if stats["price_levels_count"] > self.memory_usage_stats["peak_price_levels"]:
                self.memory_usage_stats["peak_price_levels"] = stats["price_levels_count"]

            # Periodically check for memory cleanup
            if stats["price_levels_count"] > stats["max_price_levels"] * 0.8:  # 80% threshold
                cleaned = await self.current_minute_data.cleanup_low_volume_levels()
                if cleaned > 0:
                    logger.debug(f"Memory cleanup: removed {cleaned} low-volume price levels")
                    self.memory_usage_stats["total_cleanups"] += 1
                    self.memory_usage_stats["last_cleanup_time"] = datetime.now().isoformat()

        except Exception as e:
            logger.error(f"Error handling trade: {e}", exc_info=True)
            # Consider this as a non-fatal error and continue processing

    def _validate_trade(self, trade: Trade) -> bool:
        """Validate trade data before processing."""
        try:
            # Check for None values
            if not trade or trade.price is None or trade.quantity is None:
                return False

            # Check for valid values
            if trade.price <= 0 or trade.quantity <= 0:
                return False

            # Check timestamp (should be recent)
            now = datetime.now()
            if trade.timestamp > now + timedelta(minutes=1):  # Allow 1 minute clock skew
                logger.warning(f"Trade timestamp too far in future: {trade.timestamp}")
                return False

            if trade.timestamp < now - timedelta(hours=1):  # Reject trades older than 1 hour
                logger.warning(f"Trade timestamp too old: {trade.timestamp}")
                return False

            return True

        except Exception as e:
            logger.error(f"Trade validation error: {e}")
            return False

    async def _trade_aggregator(self) -> None:
        """Aggregate trades by minute and store to Redis with memory management."""
        interval = self.settings.data_collector.order_flow.aggregation_interval_seconds
        memory_cleanup_interval = 10  # Clean memory every 10 aggregations
        cleanup_counter = 0

        while self.is_running:
            try:
                current_time = datetime.now()

                # Check if we should aggregate current minute's data
                if current_time >= self.last_aggregation_time + timedelta(
                    seconds=interval
                ):
                    await self._aggregate_and_store_minute_data()

                    cleanup_counter += 1

                    # Periodic memory cleanup
                    if cleanup_counter >= memory_cleanup_interval:
                        await self._perform_memory_cleanup()
                        cleanup_counter = 0

                    # Reset for next minute with minimal memory overhead
                    old_data = self.current_minute_data
                    self.current_minute_data = MinuteTradeData(timestamp=current_time)
                    self.last_aggregation_time = current_time

                    # Explicitly clear old data
                    await self._clear_minute_data(old_data)

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

    async def _perform_memory_cleanup(self) -> None:
        """Perform comprehensive memory cleanup."""
        try:
            # Clean current minute data
            cleaned_count = await self.current_minute_data.cleanup_low_volume_levels()
            if cleaned_count > 0:
                logger.info(f"Memory cleanup: removed {cleaned_count} low-volume price levels")

            # Force garbage collection hint
            import gc
            gc.collect()

            logger.debug("Memory cleanup completed")

        except Exception as e:
            logger.error(f"Memory cleanup error: {e}")

    async def _clear_minute_data(self, minute_data: MinuteTradeData) -> None:
        """Clear minute data to free memory."""
        try:
            # Clear price levels to free memory
            async with minute_data._lock:
                minute_data.price_levels.clear()
            logger.debug("Cleared minute data memory")
        except Exception as e:
            logger.warning(f"Error clearing minute data: {e}")

    async def _aggregate_and_store_minute_data(self) -> None:
        """Store the aggregated minute data to Redis with thread-safe operations."""
        # Get statistics in a thread-safe manner
        stats = await self.current_minute_data.get_statistics()

        if stats["price_levels_count"] == 0:
            logger.debug("No trades to aggregate for current minute")
            return

        try:
            # Get thread-safe copy and store
            minute_data_dict = await self.current_minute_data.to_dict_async()

            # Create a temporary MinuteTradeData object with the async data
            temp_data = MinuteTradeData(
                timestamp=self.current_minute_data.timestamp,
                price_levels=dict(self.current_minute_data.price_levels)  # Copy current state
            )

            await self.redis_store.store_minute_trade_data(temp_data)

            logger.debug(
                f"Stored minute data: {stats['total_trades']} trades, "
                f"{stats['total_volume']:.4f} volume across {stats['price_levels_count']} price levels"
            )

            # Perform cleanup after storage
            cleaned_count = await self.current_minute_data.cleanup_low_volume_levels()
            if cleaned_count > 0:
                logger.debug(f"Cleaned up {cleaned_count} low-volume price levels")

        except Exception as e:
            logger.error(f"Failed to store minute data: {e}")

    async def _shutdown(self) -> None:
        """Cleanup and shutdown the agent with timeout protection."""
        logger.info("Shutting down Data Collector Agent")

        self.is_running = False

        # Store any remaining aggregated data
        if self.current_minute_data.price_levels:
            try:
                await asyncio.wait_for(
                    self._aggregate_and_store_minute_data(), timeout=5
                )
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
                asyncio.gather(*self.tasks, return_exceptions=True), timeout=10
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

    async def get_status_async(self) -> dict:
        """Get current agent status with detailed information (async version)."""
        # Get current minute data statistics
        minute_stats = await self.current_minute_data.get_statistics()

        # Get WebSocket status
        websocket_status = self.websocket_client.get_connection_status()

        # Get Redis status
        redis_connected = self.redis_store.test_connection()
        redis_stats = {
            "connected": redis_connected,
            "depth_snapshot_available": self.redis_store.depth_snapshot_exists(),
            "trade_window_count": self.redis_store.get_trade_window_count()
        }

        return {
            "is_running": self.is_running,
            "timestamp": datetime.now().isoformat(),
            "websocket": websocket_status,
            "redis": redis_stats,
            "current_minute_data": minute_stats,
            "tasks": {
                "total_tasks": len(self.tasks),
                "active_tasks": len([t for t in self.tasks if not t.done()]),
                "task_names": [type(t._coro).__name__ if t._coro else "unknown" for t in self.tasks if not t.done()]
            }
        }

    def get_status(self) -> dict:
        """Get current agent status (synchronous version for backward compatibility)."""
        return {
            "is_running": self.is_running,
            "websocket_connected": self.websocket_client.is_connected,
            "current_minute_trades": len(self.current_minute_data.price_levels),
            "depth_snapshot_available": self.redis_store.depth_snapshot_exists(),
            "trade_window_count": self.redis_store.get_trade_window_count(),
            "last_update": datetime.now().isoformat(),
        }

    async def health_check(self) -> dict:
        """Perform comprehensive health check of the data collector."""
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "checks": {}
        }

        # Check if agent is running
        health_status["checks"]["agent_running"] = {
            "status": "pass" if self.is_running else "fail",
            "message": "Agent is running" if self.is_running else "Agent is not running"
        }

        # Check WebSocket connection
        if self.websocket_client.is_connected:
            websocket_health = await self.websocket_client.health_check()
            health_status["checks"]["websocket"] = websocket_health
            if websocket_health["status"] != "healthy":
                health_status["status"] = "unhealthy"
        else:
            health_status["checks"]["websocket"] = {
                "status": "fail",
                "message": "WebSocket not connected"
            }
            health_status["status"] = "unhealthy"

        # Check Redis connection
        try:
            redis_connected = self.redis_store.test_connection()
            health_status["checks"]["redis"] = {
                "status": "pass" if redis_connected else "fail",
                "message": "Redis connection working" if redis_connected else "Redis connection failed"
            }
            if not redis_connected and health_status["status"] == "healthy":
                health_status["status"] = "degraded"
        except Exception as e:
            health_status["checks"]["redis"] = {
                "status": "fail",
                "message": f"Redis health check error: {e}"
            }
            health_status["status"] = "unhealthy"

        # Check task status
        active_tasks = len([t for t in self.tasks if not t.done()])
        health_status["checks"]["tasks"] = {
            "status": "pass" if active_tasks > 0 else "fail",
            "message": f"{active_tasks} active tasks",
            "active_tasks": active_tasks,
            "total_tasks": len(self.tasks)
        }

        # Check data collection rate
        minute_stats = await self.current_minute_data.get_statistics()
        if minute_stats["total_trades"] == 0:
            health_status["checks"]["data_collection"] = {
                "status": "warning",
                "message": "No trades collected in current minute"
            }
            if health_status["status"] == "healthy":
                health_status["status"] = "degraded"
        else:
            health_status["checks"]["data_collection"] = {
                "status": "pass",
                "message": f"Collected {minute_stats['total_trades']} trades"
            }

        return health_status

    async def _monitoring_task(self) -> None:
        """Background monitoring task for health checks and metrics collection."""
        logger.info("Starting monitoring task")
        monitoring_interval = 60  # Check every minute
        last_stats_log = datetime.now()

        while self.is_running:
            try:
                # Wait for monitoring interval or shutdown
                try:
                    await asyncio.wait_for(
                        self.shutdown_event.wait(),
                        timeout=monitoring_interval
                    )
                    # Shutdown was requested
                    break
                except asyncio.TimeoutError:
                    # Normal timeout, perform monitoring
                    pass

                # Perform health check
                health_status = await self.health_check()
                self.last_health_check = health_status

                # Log health status if there are issues
                if health_status["status"] != "healthy":
                    logger.warning(f"Health check status: {health_status['status']}")
                    for check_name, check_result in health_status["checks"].items():
                        if check_result["status"] != "pass":
                            logger.warning(f"  {check_name}: {check_result['message']}")

                # Log periodic statistics
                current_time = datetime.now()
                if (current_time - last_stats_log).total_seconds() >= 300:  # Every 5 minutes
                    await self._log_performance_stats()
                    last_stats_log = current_time

                # Update connection failure counter
                if not self.websocket_client.is_connected:
                    self.total_connection_failures += 1

            except asyncio.CancelledError:
                logger.info("Monitoring task cancelled")
                break
            except Exception as e:
                logger.error(f"Monitoring task error: {e}", exc_info=True)
                await asyncio.sleep(10)  # Brief delay before retrying

        logger.info("Monitoring task stopped")

    async def _log_performance_stats(self) -> None:
        """Log comprehensive performance statistics."""
        try:
            uptime = datetime.now() - self.start_time
            current_stats = await self.current_minute_data.get_statistics()

            stats = {
                "uptime_hours": uptime.total_seconds() / 3600,
                "total_trades_processed": self.total_trades_processed,
                "trades_per_minute": self.total_trades_processed / max(uptime.total_seconds() / 60, 1),
                "current_minute_price_levels": current_stats["price_levels_count"],
                "peak_price_levels": self.memory_usage_stats["peak_price_levels"],
                "total_memory_cleanups": self.memory_usage_stats["total_cleanups"],
                "total_connection_failures": self.total_connection_failures,
                "websocket_connected": self.websocket_client.is_connected,
                "redis_connected": self.redis_store.test_connection(),
                "trade_window_count": self.redis_store.get_trade_window_count()
            }

            logger.info(f"Performance Statistics: {stats}")

        except Exception as e:
            logger.error(f"Error logging performance stats: {e}")

    def get_performance_metrics(self) -> dict:
        """Get comprehensive performance metrics."""
        uptime = datetime.now() - self.start_time

        return {
            "uptime": {
                "seconds": uptime.total_seconds(),
                "minutes": uptime.total_seconds() / 60,
                "hours": uptime.total_seconds() / 3600
            },
            "trading": {
                "total_processed": self.total_trades_processed,
                "average_per_minute": self.total_trades_processed / max(uptime.total_seconds() / 60, 1),
                "current_minute_levels": len(self.current_minute_data.price_levels)
            },
            "memory": self.memory_usage_stats,
            "connections": {
                "websocket": {
                    "connected": self.websocket_client.is_connected,
                    "total_failures": self.total_connection_failures
                },
                "redis": {
                    "connected": self.redis_store.test_connection(),
                    "trade_window_count": self.redis_store.get_trade_window_count()
                }
            },
            "system": {
                "is_running": self.is_running,
                "active_tasks": len([t for t in self.tasks if not t.done()]),
                "total_tasks": len(self.tasks)
            }
        }


async def main() -> None:
    """Main entry point for the data collector agent."""
    import argparse

    parser = argparse.ArgumentParser(description="Strategy Agent Data Collector")
    parser.add_argument(
        "--config", default="config/development.yaml", help="Configuration file path"
    )
    args = parser.parse_args()

    # Load settings
    settings = Settings.load_from_file(args.config)
    settings.setup_logging()

    # Create and start agent
    agent = DataCollectorAgent(settings)

    try:
        # Start with timeout protection for graceful shutdown
        await asyncio.wait_for(
            agent.start(), timeout=None
        )  # No timeout for normal operation
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
