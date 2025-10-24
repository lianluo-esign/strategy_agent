"""Unit tests for configuration management system."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from utils.config import (
    Settings,
    _expand_env_vars,
    RedisConfig,
    BinanceConfig,
    DeepSeekConfig,
    AppConfig,
    LoggingConfig,
)


class TestExpandEnvVars:
    """Test environment variable expansion functionality."""

    def test_expand_simple_env_var(self):
        """Test expanding a simple environment variable."""
        os.environ["TEST_VAR"] = "test_value"
        config_data = {"key": "${TEST_VAR}"}
        result = _expand_env_vars(config_data)
        assert result["key"] == "test_value"
        del os.environ["TEST_VAR"]

    def test_expand_missing_env_var_with_warning(self):
        """Test expanding missing environment variable logs warning."""
        config_data = {"key": "${MISSING_VAR}"}
        with patch("logging.getLogger") as mock_get_logger:
            mock_logger = mock_get_logger.return_value
            result = _expand_env_vars(config_data)
            assert result["key"] == ""
            mock_logger.warning.assert_called_once_with(
                "Environment variable MISSING_VAR not found, using empty string"
            )

    def test_expand_nested_dict(self):
        """Test expanding environment variables in nested dictionary."""
        os.environ["NESTED_VAR"] = "nested_value"
        config_data = {
            "level1": {
                "level2": "${NESTED_VAR}"
            }
        }
        result = _expand_env_vars(config_data)
        assert result["level1"]["level2"] == "nested_value"
        del os.environ["NESTED_VAR"]

    def test_expand_list(self):
        """Test expanding environment variables in list."""
        os.environ["LIST_VAR"] = "list_value"
        config_data = {"items": ["item1", "${LIST_VAR}", "item3"]}
        result = _expand_env_vars(config_data)
        assert result["items"] == ["item1", "list_value", "item3"]
        del os.environ["LIST_VAR"]

    def test_expand_mixed_types(self):
        """Test expanding with mixed data types."""
        os.environ["STRING_VAR"] = "string"
        config_data = {
            "string": "${STRING_VAR}",
            "number": 42,
            "boolean": True,
            "nested": {
                "value": "${STRING_VAR}"
            }
        }
        result = _expand_env_vars(config_data)
        assert result["string"] == "string"
        assert result["number"] == 42
        assert result["boolean"] is True
        assert result["nested"]["value"] == "string"
        del os.environ["STRING_VAR"]

    def test_expand_multiple_vars_in_string(self):
        """Test expanding multiple environment variables in one string."""
        os.environ["VAR1"] = "hello"
        os.environ["VAR2"] = "world"
        config_data = {"message": "${VAR1} ${VAR2}!"}
        result = _expand_env_vars(config_data)
        assert result["message"] == "hello world!"
        del os.environ["VAR1"]
        del os.environ["VAR2"]


class TestSettings:
    """Test Settings class functionality."""

    def test_default_initialization(self):
        """Test settings initialization with default values."""
        settings = Settings(
            analyzer={
                "deepseek": {
                    "api_key": "test_key"
                }
            }
        )

        assert settings.app.name == "strategy-agent"
        assert settings.redis.host == "localhost"
        assert settings.redis.port == 6379
        assert settings.binance.symbol == "BTCFDUSD"
        assert settings.analyzer.deepseek.api_key == "test_key"

    def test_load_from_file_with_env_vars(self):
        """Test loading settings from YAML file with environment variables."""
        os.environ["DEEPSEEK_API_KEY"] = "loaded_key"

        config_data = {
            "app": {
                "name": "test-app",
                "environment": "test"
            },
            "analyzer": {
                "deepseek": {
                    "api_key": "${DEEPSEEK_API_KEY}",
                    "model": "deepseek-chat"
                }
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name

        try:
            settings = Settings.load_from_file(config_path)
            assert settings.app.name == "test-app"
            assert settings.app.environment == "test"
            assert settings.analyzer.deepseek.api_key == "loaded_key"
            assert settings.analyzer.deepseek.model == "deepseek-chat"
        finally:
            os.unlink(config_path)
            del os.environ["DEEPSEEK_API_KEY"]

    def test_load_from_nonexistent_file(self):
        """Test loading from nonexistent file raises error."""
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            Settings.load_from_file("/nonexistent/config.yaml")

    def test_validate_required_env_vars_success(self):
        """Test successful validation of required environment variables."""
        os.environ["DEEPSEEK_API_KEY"] = "valid_key"

        settings = Settings(
            analyzer={
                "deepseek": {
                    "api_key": "valid_key"
                }
            }
        )

        # Should not raise any exception
        settings.validate_required_env_vars()
        del os.environ["DEEPSEEK_API_KEY"]

    def test_validate_required_env_vars_missing(self):
        """Test validation failure with missing environment variables."""
        # Ensure DEEPSEEK_API_KEY is not set
        if "DEEPSEEK_API_KEY" in os.environ:
            del os.environ["DEEPSEEK_API_KEY"]

        settings = Settings(
            analyzer={
                "deepseek": {
                    "api_key": ""
                }
            }
        )

        with pytest.raises(ValueError, match="Missing required environment variables"):
            settings.validate_required_env_vars()

    def test_validate_config_values_success(self):
        """Test successful configuration value validation."""
        settings = Settings(
            analyzer={
                "deepseek": {
                    "api_key": "valid_key",
                    "max_tokens": 1000
                }
            }
        )

        # Should not raise any exception
        settings.validate_config_values()

    def test_validate_invalid_redis_port(self):
        """Test validation failure with invalid Redis port."""
        settings = Settings(
            redis={"port": 70000},  # Invalid port
            analyzer={
                "deepseek": {
                    "api_key": "valid_key"
                }
            }
        )

        with pytest.raises(ValueError, match="Redis port must be 1-65535"):
            settings.validate_config_values()

    def test_validate_invalid_binance_timeout(self):
        """Test validation failure with invalid Binance timeout."""
        settings = Settings(
            binance={"timeout": -1},  # Invalid timeout
            analyzer={
                "deepseek": {
                    "api_key": "valid_key"
                }
            }
        )

        with pytest.raises(ValueError, match="Binance timeout must be positive"):
            settings.validate_config_values()

    def test_validate_invalid_symbol_format(self):
        """Test validation failure with invalid symbol format."""
        settings = Settings(
            binance={"symbol": "invalid-symbol"},
            analyzer={
                "deepseek": {
                    "api_key": "valid_key"
                }
            }
        )

        with pytest.raises(ValueError, match="Invalid symbol format"):
            settings.validate_config_values()

    def test_validate_empty_deepseek_api_key(self):
        """Test validation failure with empty DeepSeek API key."""
        settings = Settings(
            analyzer={
                "deepseek": {
                    "api_key": ""
                }
            }
        )

        with pytest.raises(ValueError, match="DeepSeek API key cannot be empty"):
            settings.validate_config_values()

    def test_validate_invalid_depth_limit(self):
        """Test validation failure with invalid depth snapshot limit."""
        settings = Settings(
            data_collector={
                "depth_snapshot": {
                    "limit": 0
                }
            },
            analyzer={
                "deepseek": {
                    "api_key": "valid_key"
                }
            }
        )

        with pytest.raises(ValueError, match="Depth snapshot limit must be positive"):
            settings.validate_config_values()

    def test_validate_invalid_log_config(self):
        """Test validation failure with invalid logging configuration."""
        settings = Settings(
            logging={
                "max_file_size_mb": -1,
                "backup_count": -1
            },
            analyzer={
                "deepseek": {
                    "api_key": "valid_key"
                }
            }
        )

        # Test each validation separately since the method only raises on first error
        with pytest.raises(ValueError, match="Log file size must be positive"):
            settings.validate_config_values()

        # Test backup count validation separately
        settings_valid_size = Settings(
            logging={
                "max_file_size_mb": 10,  # Valid size
                "backup_count": -1  # Invalid backup count
            },
            analyzer={
                "deepseek": {
                    "api_key": "valid_key"
                }
            }
        )

        with pytest.raises(ValueError, match="Log backup count cannot be negative"):
            settings_valid_size.validate_config_values()

    def test_setup_logging(self):
        """Test logging setup functionality."""
        settings = Settings(
            logging={
                "level": "DEBUG",
                "format": "%(levelname)s - %(message)s",
                "file_path": "test.log",
                "max_file_size_mb": 1,
                "backup_count": 1
            },
            analyzer={
                "deepseek": {
                    "api_key": "valid_key"
                }
            }
        )

        # Should not raise any exception and create log file directory
        settings.setup_logging()

        # Check that log file parent directory was created
        log_path = Path(settings.logging.file_path)
        assert log_path.parent.exists()

    def test_deepseek_env_var_fallback(self):
        """Test DeepSeekConfig fallback to environment variable."""
        os.environ["DEEPSEEK_API_KEY"] = "env_key"

        # Initialize without api_key in data
        config = DeepSeekConfig()

        assert config.api_key == "env_key"
        del os.environ["DEEPSEEK_API_KEY"]


class TestConfigModels:
    """Test individual configuration model classes."""

    def test_redis_config_defaults(self):
        """Test RedisConfig default values."""
        config = RedisConfig()
        assert config.host == "localhost"
        assert config.port == 6379
        assert config.db == 0
        assert config.decode_responses is True
        assert config.socket_timeout == 5
        assert config.socket_connect_timeout == 5

    def test_binance_config_defaults(self):
        """Test BinanceConfig default values."""
        config = BinanceConfig()
        assert config.rest_api_base == "https://api.binance.com"
        assert config.websocket_base == "wss://stream.binance.com:9443"
        assert config.symbol == "BTCFDUSD"
        assert config.rate_limit_requests_per_minute == 1200
        assert config.timeout == 30

    def test_app_config_defaults(self):
        """Test AppConfig default values."""
        config = AppConfig()
        assert config.name == "strategy-agent"
        assert config.environment == "development"
        assert config.log_level == "DEBUG"

    def test_logging_config_defaults(self):
        """Test LoggingConfig default values."""
        config = LoggingConfig()
        assert config.level == "INFO"
        assert "%(asctime)s" in config.format
        assert config.file_path == "logs/strategy_agent.log"
        assert config.max_file_size_mb == 100
        assert config.backup_count == 5

    def test_deepseek_config_custom_values(self):
        """Test DeepSeekConfig with custom values."""
        config = DeepSeekConfig(
            api_key="custom_key",
            base_url="https://custom.api.com",
            model="custom-model",
            max_tokens=2000,
            temperature=0.5
        )

        assert config.api_key == "custom_key"
        assert config.base_url == "https://custom.api.com"
        assert config.model == "custom-model"
        assert config.max_tokens == 2000
        assert config.temperature == 0.5