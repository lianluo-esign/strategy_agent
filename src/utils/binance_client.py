"""Binance API client for market data collection."""

import asyncio
import json
import logging
from datetime import datetime
from decimal import Decimal

import aiohttp
import requests
import websockets
from websockets.exceptions import ConnectionClosed

from ..core.constants import (
    BINANCE_REST_API_BASE,
    BINANCE_WEBSOCKET_BASE,
    BTC_FDUSD_SYMBOL,
    DEPTH_SNAPSHOT_LIMIT,
    ERROR_WEBSOCKET_CONNECTION,
    WEBSOCKET_TRADE_STREAM,
)
from ..core.models import DepthLevel, DepthSnapshot, Trade

logger = logging.getLogger(__name__)


class BinanceAPIClient:
    """Binance REST API client."""

    def __init__(self, base_url: str = BINANCE_REST_API_BASE, timeout: int = 30):
        """Initialize API client."""
        self.base_url = base_url
        self.timeout = timeout
        self.session = requests.Session()
        self._async_session = None

    async def get_async_session(self) -> aiohttp.ClientSession:
        """Get or create async session for connection reuse."""
        if self._async_session is None or self._async_session.closed:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            connector = aiohttp.TCPConnector(
                limit=100,  # Connection pool limit
                limit_per_host=30,
                keepalive_timeout=30,
                enable_cleanup_closed=True,
            )
            self._async_session = aiohttp.ClientSession(
                timeout=timeout, connector=connector
            )
        return self._async_session

    async def close_async_session(self) -> None:
        """Close async session."""
        if self._async_session and not self._async_session.closed:
            await self._async_session.close()

    async def get_depth_snapshot(
        self, symbol: str = BTC_FDUSD_SYMBOL, limit: int = DEPTH_SNAPSHOT_LIMIT
    ) -> DepthSnapshot | None:
        """Get order book depth snapshot."""
        try:
            url = f"{self.base_url}/api/v3/depth"
            params = {"symbol": symbol, "limit": limit}

            session = await self.get_async_session()
            async with session.get(url, params=params) as response:
                if response.status != 200:
                    logger.error(
                        f"Binance API error: {response.status} - {await response.text()}"
                    )
                    return None

                data = await response.json()
                return self._parse_depth_snapshot(data, symbol)

        except aiohttp.ClientError as e:
            logger.error(f"HTTP client error getting depth snapshot: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to get depth snapshot: {e}")
            return None

    def _parse_depth_snapshot(self, data: dict, symbol: str) -> DepthSnapshot:
        """Parse depth snapshot from API response."""
        timestamp = (
            datetime.now()
        )  # Use current time since Binance depth API doesn't provide timestamp

        bids = [
            DepthLevel(price=Decimal(str(price)), quantity=Decimal(str(qty)))
            for price, qty in data["bids"]
        ]

        asks = [
            DepthLevel(price=Decimal(str(price)), quantity=Decimal(str(qty)))
            for price, qty in data["asks"]
        ]

        return DepthSnapshot(symbol=symbol, timestamp=timestamp, bids=bids, asks=asks)


class BinanceWebSocketClient:
    """Binance WebSocket client for real-time data with proper heartbeat and reconnection."""

    def __init__(self, symbol: str = BTC_FDUSD_SYMBOL):
        """Initialize WebSocket client."""
        self.symbol = symbol
        self.websocket_url = f"{BINANCE_WEBSOCKET_BASE}/{WEBSOCKET_TRADE_STREAM}"
        self.websocket = None
        self.is_connected = False
        self._ping_task: asyncio.Task | None = None
        self._connection_task: asyncio.Task | None = None
        self._shutdown_event = asyncio.Event()
        self._last_ping_time = datetime.now()
        self._connection_retries = 0
        self._max_connection_retries = 5
        self._ping_interval = 30  # seconds between pings
        self._ping_timeout = 10   # seconds to wait for pong response

    async def connect(self) -> bool:
        """Connect to WebSocket with proper error handling and reconnection logic."""
        try:
            if self._connection_retries >= self._max_connection_retries:
                logger.error(f"Max connection retries ({self._max_connection_retries}) exceeded")
                return False

            # Reset shutdown event for new connection
            self._shutdown_event.clear()

            self.websocket = await websockets.connect(
                self.websocket_url,
                ping_interval=None,  # We'll handle ping/pong manually
                ping_timeout=None,
                close_timeout=10,
                max_size=2**20,  # 1MB max message size
                compression=None,  # Disable compression for lower latency
            )

            self.is_connected = True
            self._connection_retries = 0
            logger.info(f"Connected to Binance WebSocket for {self.symbol} (attempt {self._connection_retries + 1})")

            # Start the heartbeat monitor task
            self._ping_task = asyncio.create_task(self._heartbeat_monitor())

            return True

        except Exception as e:
            self._connection_retries += 1
            wait_time = min(2 ** self._connection_retries, 30)  # Exponential backoff
            logger.error(f"Failed to connect to WebSocket (attempt {self._connection_retries}): {e}")
            logger.info(f"Retrying in {wait_time} seconds...")
            self.is_connected = False
            return False

    async def disconnect(self) -> None:
        """Disconnect from WebSocket with proper cleanup."""
        logger.info("Disconnecting from Binance WebSocket")

        # Signal shutdown to all tasks
        self._shutdown_event.set()

        # Cancel heartbeat monitor task
        if self._ping_task and not self._ping_task.done():
            self._ping_task.cancel()
            try:
                await self._ping_task
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.warning(f"Error cancelling ping task: {e}")

        # Cancel connection monitoring task
        if self._connection_task and not self._connection_task.done():
            self._connection_task.cancel()
            try:
                await self._connection_task
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.warning(f"Error cancelling connection task: {e}")

        # Close WebSocket connection
        if self.websocket:
            try:
                await asyncio.wait_for(self.websocket.close(), timeout=5)
                logger.info("WebSocket connection closed successfully")
            except asyncio.TimeoutError:
                logger.warning("WebSocket close operation timed out")
            except Exception as e:
                logger.warning(f"Error closing WebSocket: {e}")
            finally:
                self.websocket = None

        self.is_connected = False
        logger.info("WebSocket disconnection complete")

    async def listen_trades(self, callback):
        """Listen for trade events and call callback for each trade."""
        if not self.is_connected or not self.websocket:
            raise ConnectionError(ERROR_WEBSOCKET_CONNECTION)

        try:
            async for message in self.websocket:
                try:
                    data = json.loads(message)
                    trade = self._parse_trade_message(data)
                    if trade:
                        await callback(trade)

                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse WebSocket message: {e}")
                except Exception as e:
                    logger.error(f"Error processing trade message: {e}")

        except ConnectionClosed:
            logger.warning("WebSocket connection closed")
            self.is_connected = False
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            self.is_connected = False

    def _parse_trade_message(self, data: dict) -> Trade | None:
        """Parse trade message from WebSocket."""
        try:
            # Validate required fields
            required_fields = ["p", "q", "m", "T", "a"]
            if not all(field in data for field in required_fields):
                logger.warning(f"Missing required fields in trade message: {data}")
                return None

            # Validate data types
            if not all(
                isinstance(data[field], (int, float, str))
                for field in ["p", "q", "T", "a"]
            ):
                logger.warning(f"Invalid data types in trade message: {data}")
                return None

            if not isinstance(data["m"], bool):
                logger.warning(f"Invalid maker flag in trade message: {data}")
                return None

            # Validate values
            price = Decimal(str(data["p"]))
            quantity = Decimal(str(data["q"]))

            if price <= 0 or quantity <= 0:
                logger.warning(f"Invalid price or quantity in trade message: {data}")
                return None

            # Binance aggregated trade format
            return Trade(
                symbol=self.symbol,
                price=price,
                quantity=quantity,
                is_buyer_maker=data["m"],  # True if buyer is the maker
                timestamp=datetime.fromtimestamp(data["T"] / 1000),
                trade_id=str(data["a"]),
            )

        except (ValueError, TypeError, KeyError) as e:
            logger.error(f"Failed to parse trade message: {e}, data: {data}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error parsing trade message: {e}, data: {data}")
            return None

    async def _heartbeat_monitor(self) -> None:
        """Monitor WebSocket connection and send periodic pings."""
        logger.info("Starting WebSocket heartbeat monitor")

        while not self._shutdown_event.is_set() and self.is_connected:
            try:
                # Wait for ping interval or shutdown
                try:
                    await asyncio.wait_for(
                        self._shutdown_event.wait(),
                        timeout=self._ping_interval
                    )
                    # Shutdown event was triggered
                    break
                except asyncio.TimeoutError:
                    # Normal timeout, proceed with ping
                    pass

                # Check if we're still connected before pinging
                if not self.is_connected or not self.websocket:
                    logger.warning("WebSocket disconnected during heartbeat monitoring")
                    break

                # Send ping and wait for response
                ping_success = await self._send_ping_with_timeout()
                if ping_success:
                    self._last_ping_time = datetime.now()
                    logger.debug("WebSocket ping successful")
                else:
                    logger.error("WebSocket ping failed - connection may be lost")
                    self.is_connected = False
                    break

            except asyncio.CancelledError:
                logger.info("Heartbeat monitor cancelled")
                break
            except Exception as e:
                logger.error(f"Heartbeat monitor error: {e}")
                await asyncio.sleep(1)  # Brief delay before retrying

        logger.info("Heartbeat monitor stopped")

    async def _send_ping_with_timeout(self) -> bool:
        """Send ping with timeout and handle response."""
        if not self.websocket:
            return False

        try:
            # Send ping
            await asyncio.wait_for(
                self.websocket.ping(),
                timeout=self._ping_timeout
            )
            return True

        except asyncio.TimeoutError:
            logger.warning("WebSocket ping timeout")
            return False
        except Exception as e:
            logger.warning(f"WebSocket ping error: {e}")
            return False

    async def ping(self) -> bool:
        """Ping the WebSocket to check connection (legacy method)."""
        if not self.is_connected or not self.websocket:
            return False

        try:
            await self.websocket.ping()
            return True
        except Exception:
            self.is_connected = False
            return False

    def get_connection_status(self) -> dict:
        """Get detailed connection status."""
        return {
            "is_connected": self.is_connected,
            "connection_retries": self._connection_retries,
            "last_ping_time": self._last_ping_time.isoformat() if self._last_ping_time else None,
            "ping_interval": self._ping_interval,
            "ping_timeout": self._ping_timeout,
            "websocket_url": self.websocket_url,
            "symbol": self.symbol
        }

    async def health_check(self) -> dict:
        """Perform comprehensive health check."""
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "checks": {}
        }

        # Connection check
        if self.is_connected and self.websocket:
            try:
                # Quick ping test
                ping_result = await asyncio.wait_for(
                    self.websocket.ping(),
                    timeout=3
                )
                health_status["checks"]["connection"] = {
                    "status": "pass",
                    "message": "WebSocket connection responsive"
                }
            except Exception as e:
                health_status["checks"]["connection"] = {
                    "status": "fail",
                    "message": f"WebSocket connection test failed: {e}"
                }
                health_status["status"] = "unhealthy"
        else:
            health_status["checks"]["connection"] = {
                "status": "fail",
                "message": "WebSocket not connected"
            }
            health_status["status"] = "unhealthy"

        # Heartbeat check
        if self._ping_task and not self._ping_task.done():
            health_status["checks"]["heartbeat"] = {
                "status": "pass",
                "message": "Heartbeat monitor running"
            }
        else:
            health_status["checks"]["heartbeat"] = {
                "status": "fail",
                "message": "Heartbeat monitor not running"
            }
            if health_status["status"] == "healthy":
                health_status["status"] = "degraded"

        return health_status
