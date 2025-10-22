"""Binance API client for market data collection."""

import asyncio
import json
import logging
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional

import aiohttp
import requests
import websockets
from websockets.exceptions import ConnectionClosed

from ..core.constants import (
    BINANCE_REST_API_BASE,
    BINANCE_WEBSOCKET_BASE,
    BTC_FDUSD_SYMBOL,
    DEPTH_SNAPSHOT_LIMIT,
    WEBSOCKET_TRADE_STREAM,
    ERROR_BINANCE_API,
    ERROR_WEBSOCKET_CONNECTION
)
from ..core.models import DepthSnapshot, DepthLevel, Trade

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
                enable_cleanup_closed=True
            )
            self._async_session = aiohttp.ClientSession(
                timeout=timeout,
                connector=connector
            )
        return self._async_session

    async def close_async_session(self) -> None:
        """Close async session."""
        if self._async_session and not self._async_session.closed:
            await self._async_session.close()

    async def get_depth_snapshot(self, symbol: str = BTC_FDUSD_SYMBOL, limit: int = DEPTH_SNAPSHOT_LIMIT) -> Optional[DepthSnapshot]:
        """Get order book depth snapshot."""
        try:
            url = f"{self.base_url}/api/v3/depth"
            params = {
                "symbol": symbol,
                "limit": limit
            }

            session = await self.get_async_session()
            async with session.get(url, params=params) as response:
                if response.status != 200:
                    logger.error(f"Binance API error: {response.status} - {await response.text()}")
                    return None

                data = await response.json()
                return self._parse_depth_snapshot(data, symbol)

        except aiohttp.ClientError as e:
            logger.error(f"HTTP client error getting depth snapshot: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to get depth snapshot: {e}")
            return None

    def _parse_depth_snapshot(self, data: Dict, symbol: str) -> DepthSnapshot:
        """Parse depth snapshot from API response."""
        timestamp = datetime.fromtimestamp(data['lastUpdateId'] / 1000)

        bids = [DepthLevel(price=Decimal(str(price)), quantity=Decimal(str(qty)))
                for price, qty in data['bids']]

        asks = [DepthLevel(price=Decimal(str(price)), quantity=Decimal(str(qty)))
                for price, qty in data['asks']]

        return DepthSnapshot(
            symbol=symbol,
            timestamp=timestamp,
            bids=bids,
            asks=asks
        )


class BinanceWebSocketClient:
    """Binance WebSocket client for real-time data."""

    def __init__(self, symbol: str = BTC_FDUSD_SYMBOL):
        """Initialize WebSocket client."""
        self.symbol = symbol
        self.websocket_url = f"{BINANCE_WEBSOCKET_BASE}/{WEBSOCKET_TRADE_STREAM}"
        self.websocket = None
        self.is_connected = False

    async def connect(self) -> bool:
        """Connect to WebSocket."""
        try:
            self.websocket = await websockets.connect(
                self.websocket_url,
                ping_interval=20,
                ping_timeout=10,
                close_timeout=10
            )
            self.is_connected = True
            logger.info(f"Connected to Binance WebSocket for {self.symbol}")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to WebSocket: {e}")
            self.is_connected = False
            return False

    async def disconnect(self) -> None:
        """Disconnect from WebSocket."""
        if self.websocket:
            try:
                await self.websocket.close()
                logger.info("Disconnected from Binance WebSocket")
            except Exception as e:
                logger.warning(f"Error closing WebSocket: {e}")
            finally:
                self.websocket = None
                self.is_connected = False

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

    def _parse_trade_message(self, data: Dict) -> Optional[Trade]:
        """Parse trade message from WebSocket."""
        try:
            # Validate required fields
            required_fields = ['p', 'q', 'm', 'T', 'a']
            if not all(field in data for field in required_fields):
                logger.warning(f"Missing required fields in trade message: {data}")
                return None

            # Validate data types
            if not all(isinstance(data[field], (int, float, str)) for field in ['p', 'q', 'T', 'a']):
                logger.warning(f"Invalid data types in trade message: {data}")
                return None

            if not isinstance(data['m'], bool):
                logger.warning(f"Invalid maker flag in trade message: {data}")
                return None

            # Validate values
            price = Decimal(str(data['p']))
            quantity = Decimal(str(data['q']))

            if price <= 0 or quantity <= 0:
                logger.warning(f"Invalid price or quantity in trade message: {data}")
                return None

            # Binance aggregated trade format
            return Trade(
                symbol=self.symbol,
                price=price,
                quantity=quantity,
                is_buyer_maker=data['m'],  # True if buyer is the maker
                timestamp=datetime.fromtimestamp(data['T'] / 1000),
                trade_id=str(data['a'])
            )

        except (ValueError, TypeError, KeyError) as e:
            logger.error(f"Failed to parse trade message: {e}, data: {data}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error parsing trade message: {e}, data: {data}")
            return None

    async def ping(self) -> bool:
        """Ping the WebSocket to check connection."""
        if not self.is_connected or not self.websocket:
            return False

        try:
            await self.websocket.ping()
            return True
        except Exception:
            self.is_connected = False
            return False