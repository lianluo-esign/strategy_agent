"""Configuration management for the Strategy Agent."""

import os
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


def _expand_env_vars(config_data: Any) -> Any:
    """
    Recursively expand environment variables in configuration data.

    Supports ${VAR_NAME} pattern in strings, dictionaries, and lists.
    Missing environment variables are replaced with empty string.

    Args:
        config_data: Configuration data structure to process

    Returns:
        Configuration data with expanded environment variables

    Raises:
        ValueError: If required environment variables are missing
    """
    if isinstance(config_data, dict):
        return {k: _expand_env_vars(v) for k, v in config_data.items()}
    elif isinstance(config_data, list):
        return [_expand_env_vars(item) for item in config_data]
    elif isinstance(config_data, str):
        # Expand ${VAR_NAME} patterns with pre-compiled regex
        pattern = r'\$\{([^}]+)\}'
        def replace_var(match: re.Match[str]) -> str:
            var_name = match.group(1)
            value = os.getenv(var_name)
            if value is None:
                # Log warning but don't fail - let validation handle required vars
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"Environment variable {var_name} not found, using empty string")
                return ""
            return value

        return re.sub(pattern, replace_var, config_data)
    else:
        return config_data


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
        """Load settings from YAML file with environment variable expansion."""
        import yaml

        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_file) as f:
            config_data = yaml.safe_load(f)

        # Expand environment variables
        config_data = _expand_env_vars(config_data)

        return cls(**config_data)

    def validate_required_env_vars(self) -> None:
        """
        Validate that required environment variables are set.

        Raises:
            ValueError: If required environment variables are missing
        """
        required_vars = {
            'DEEPSEEK_API_KEY': 'DeepSeek API key is required but not set'
        }

        missing_vars = []
        for var_name, error_msg in required_vars.items():
            if not os.getenv(var_name):
                missing_vars.append(f"  - {var_name}: {error_msg}")

        if missing_vars:
            raise ValueError(
                "Missing required environment variables:\n" + "\n".join(missing_vars)
            )

    def validate_config_values(self) -> None:
        """
        Validate configuration value constraints.

        Raises:
            ValueError: If configuration values are invalid
        """
        import logging

        # Validate Redis configuration
        if not (1 <= self.redis.port <= 65535):
            raise ValueError(f"Redis port must be 1-65535, got {self.redis.port}")

        # Validate Binance configuration
        if self.binance.timeout <= 0:
            raise ValueError(f"Binance timeout must be positive, got {self.binance.timeout}")

        # Validate symbol format
        if not re.match(r'^[A-Z]+[A-Z0-9]*$', self.binance.symbol):
            raise ValueError(f"Invalid symbol format: {self.binance.symbol}")

        # Validate data collector configuration
        if self.data_collector.depth_snapshot.limit <= 0:
            raise ValueError("Depth snapshot limit must be positive")

        if self.data_collector.depth_snapshot.update_interval_seconds <= 0:
            raise ValueError("Update interval must be positive")

        # Validate analyzer configuration
        if not self.analyzer.deepseek.api_key or self.analyzer.deepseek.api_key.strip() == "":
            raise ValueError("DeepSeek API key cannot be empty")

        if self.analyzer.deepseek.max_tokens <= 0:
            raise ValueError("DeepSeek max tokens must be positive")

        # Validate logging configuration
        if self.logging.max_file_size_mb <= 0:
            raise ValueError("Log file size must be positive")

        if self.logging.backup_count < 0:
            raise ValueError("Log backup count cannot be negative")

        logging.getLogger(__name__).info("Configuration validation passed")

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
