"""Configuration management for the Strategy Agent."""

import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class RedisConfig(BaseModel):
    """Redis configuration."""
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    decode_responses: bool = True
    socket_timeout: int = 5
    socket_connect_timeout: int = 5
    storage_dir: str = "storage"


class BinanceConfig(BaseModel):
    """Binance API configuration."""
    rest_api_base: str = "https://api.binance.com"
    websocket_base: str = "wss://stream.binance.com:9443"
    symbol: str = "BTCFDUSD"
    rate_limit_requests_per_minute: int = 1200
    timeout: int = 30


class DepthSnapshotConfig(BaseModel):
    """Depth snapshot collection configuration."""
    limit: int = 5000
    update_interval_seconds: int = 60
    window_size: int = 60


class OrderFlowConfig(BaseModel):
    """Order flow data collection configuration."""
    websocket_url: str = "wss://stream.binance.com:9443/ws/btcfdusd@aggTrade"
    window_size_minutes: int = 48 * 60  # 48 hours
    price_precision: float = 1.0  # $1 precision
    aggregation_interval_seconds: int = 60


class DataCollectorConfig(BaseModel):
    """Data collector configuration."""
    depth_snapshot: DepthSnapshotConfig = Field(default_factory=DepthSnapshotConfig)
    order_flow: OrderFlowConfig = Field(default_factory=OrderFlowConfig)


class DeepSeekConfig(BaseModel):
    """DeepSeek AI configuration."""
    api_key: str
    base_url: str = "https://api.deepseek.com/v1"
    model: str = "deepseek-chat"
    max_tokens: int = 4000
    temperature: float = 0.1

    def __init__(self, **data: Any) -> None:
        """Initialize with environment variable support."""
        # Support environment variable for API key
        if 'api_key' not in data:
            data['api_key'] = os.getenv('DEEPSEEK_API_KEY', '')
        super().__init__(**data)


class AnalysisConfig(BaseModel):
    """Analysis configuration."""
    interval_seconds: int = 60
    min_order_volume_threshold: float = 0.01
    support_resistance_threshold: float = 0.1


class AnalyzerConfig(BaseModel):
    """Analyzer configuration."""
    deepseek: DeepSeekConfig
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)


class LoggingConfig(BaseModel):
    """Logging configuration."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: str = "logs/strategy_agent.log"
    max_file_size_mb: int = 100
    backup_count: int = 5


class AppConfig(BaseModel):
    """Application configuration."""
    name: str = "strategy-agent"
    environment: str = "development"
    log_level: str = "DEBUG"


class Settings(BaseSettings):
    """Main settings class."""
    app: AppConfig = Field(default_factory=AppConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    binance: BinanceConfig = Field(default_factory=BinanceConfig)
    data_collector: DataCollectorConfig = Field(default_factory=DataCollectorConfig)
    analyzer: AnalyzerConfig
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore"
    }

    @classmethod
    def load_from_file(cls, config_path: str) -> "Settings":
        """Load settings from YAML file."""
        import yaml

        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_file) as f:
            config_data = yaml.safe_load(f)

        return cls(**config_data)

    def setup_logging(self) -> None:
        """Setup logging based on configuration."""
        import logging
        import logging.handlers
        from pathlib import Path

        # Validate logging configuration
        if self.logging.max_file_size_mb <= 0:
            raise ValueError("Log file size must be positive")
        if self.logging.backup_count < 0:
            raise ValueError("Log backup count cannot be negative")

        # Create logs directory if it doesn't exist
        log_file = Path(self.logging.file_path)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        # Configure logging
        log_level = getattr(logging, self.logging.level.upper())
        formatter = logging.Formatter(self.logging.format)

        # File handler with rotation
        file_handler = logging.handlers.RotatingFileHandler(
            self.logging.file_path,
            maxBytes=self.logging.max_file_size_mb * 1024 * 1024,
            backupCount=self.logging.backup_count
        )
        file_handler.setFormatter(formatter)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        # Root logger configuration
        root_logger = logging.getLogger()
        root_logger.setLevel(log_level)
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)

        # Set specific logger levels
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("websockets").setLevel(logging.WARNING)
