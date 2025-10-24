"""Constants for the Strategy Agent system."""

# Redis Keys
REDIS_DEPTH_SNAPSHOT_KEY = "depth_snapshot_5000"
REDIS_TRADES_WINDOW_KEY = "trades_window"
REDIS_ANALYSIS_RESULTS_KEY = "analysis_results"

# Binance API
BINANCE_REST_API_BASE = "https://api.binance.com"
BINANCE_WEBSOCKET_BASE = "wss://stream.binance.com:9443"
BTC_FDUSD_SYMBOL = "BTCFDUSD"

# Data Collection Constants
DEPTH_SNAPSHOT_LIMIT = 5000
TRADES_WINDOW_SIZE_MINUTES = 48 * 60  # 48 hours
PRICE_PRECISION = 1.0  # $1 precision for aggregation
AGGREGATION_INTERVAL_SECONDS = 60

# Analysis Constants
ANALYSIS_INTERVAL_SECONDS = 60
MIN_ORDER_VOLUME_THRESHOLD = 0.01
SUPPORT_RESISTANCE_THRESHOLD = 0.1

# WebSocket Messages
WEBSOCKET_TRADE_STREAM = f"ws/{BTC_FDUSD_SYMBOL.lower()}@aggTrade"

# Error Messages
ERROR_REDIS_CONNECTION = "Failed to connect to Redis"
ERROR_BINANCE_API = "Binance API request failed"
ERROR_WEBSOCKET_CONNECTION = "WebSocket connection failed"
