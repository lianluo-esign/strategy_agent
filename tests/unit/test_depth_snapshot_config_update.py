"""Unit tests for depth snapshot configuration update.

Tests the removal of window_size configuration and validation of
the new single-key overwrite mechanism configuration.
"""

from unittest.mock import patch

from src.utils.config import DataCollectorConfig, DepthSnapshotConfig, Settings


class TestDepthSnapshotConfigUpdate:
    """Test depth snapshot configuration updates."""

    def test_depth_snapshot_config_default_values(self):
        """Test that DepthSnapshotConfig has correct default values without window_size."""
        config = DepthSnapshotConfig()

        assert config.limit == 5000
        assert config.update_interval_seconds == 60

        # Ensure window_size is not present
        assert not hasattr(config, 'window_size')

    def test_depth_snapshot_config_custom_values(self):
        """Test DepthSnapshotConfig with custom values."""
        config = DepthSnapshotConfig(
            limit=1000,
            update_interval_seconds=30
        )

        assert config.limit == 1000
        assert config.update_interval_seconds == 30

    def test_depth_snapshot_config_json_schema(self):
        """Test that window_size is not in the JSON schema."""
        schema = DepthSnapshotConfig.model_json_schema()

        assert 'properties' in schema
        assert 'limit' in schema['properties']
        assert 'update_interval_seconds' in schema['properties']
        assert 'window_size' not in schema['properties']

    def test_data_collector_config_composition(self):
        """Test DataCollectorConfig composition with updated DepthSnapshotConfig."""
        config = DataCollectorConfig()

        assert config.depth_snapshot.limit == 5000
        assert config.depth_snapshot.update_interval_seconds == 60
        assert not hasattr(config.depth_snapshot, 'window_size')

    def test_config_serialization(self):
        """Test that configuration can be serialized without window_size."""
        config = DepthSnapshotConfig(
            limit=1500,
            update_interval_seconds=90
        )

        # Test model_dump (Pydantic v2 method)
        data = config.model_dump()
        assert 'limit' in data
        assert 'update_interval_seconds' in data
        assert 'window_size' not in data

        # Test JSON serialization
        json_str = config.model_dump_json()
        assert 'limit' in json_str
        assert 'update_interval_seconds' in json_str
        assert 'window_size' not in json_str

    def test_config_deserialization_from_dict(self):
        """Test creating config from dictionary without window_size."""
        data = {
            'limit': 2500,
            'update_interval_seconds': 75
        }

        config = DepthSnapshotConfig(**data)
        assert config.limit == 2500
        assert config.update_interval_seconds == 75

    def test_config_deserialization_ignores_window_size(self):
        """Test that deserialization ignores window_size if present."""
        data = {
            'limit': 2500,
            'update_interval_seconds': 75,
            'window_size': 50  # Should be ignored
        }

        config = DepthSnapshotConfig(**data)
        assert config.limit == 2500
        assert config.update_interval_seconds == 75
        assert not hasattr(config, 'window_size')

    def test_config_copy_with_modification(self):
        """Test copying configuration and modifying values."""
        original = DepthSnapshotConfig(
            limit=4000,
            update_interval_seconds=120
        )

        # Copy with modification
        modified = original.model_copy(update={'limit': 6000})

        assert modified.limit == 6000
        assert modified.update_interval_seconds == 120
        assert not hasattr(modified, 'window_size')

    def test_config_equality(self):
        """Test configuration equality without window_size."""
        config1 = DepthSnapshotConfig(limit=1000, update_interval_seconds=30)
        config2 = DepthSnapshotConfig(limit=1000, update_interval_seconds=30)
        config3 = DepthSnapshotConfig(limit=2000, update_interval_seconds=30)

        assert config1 == config2
        assert config1 != config3

    def test_config_repr(self):
        """Test string representation of configuration."""
        config = DepthSnapshotConfig(limit=1500, update_interval_seconds=45)

        repr_str = repr(config)
        assert 'limit=1500' in repr_str
        assert 'update_interval_seconds=45' in repr_str
        assert 'window_size' not in repr_str.lower()

    def test_config_documentation_string(self):
        """Test that configuration class has updated documentation."""
        # Check that docstring mentions removal of window_size
        docstring = DepthSnapshotConfig.__doc__
        assert 'window_size' in docstring.lower()
        assert 'removed' in docstring.lower()

    @patch('pathlib.Path.exists')
    @patch('yaml.safe_load')
    def test_settings_load_from_updated_config(self, mock_yaml_load, mock_path_exists):
        """Test Settings.load_from_file with updated configuration."""
        mock_path_exists.return_value = True
        mock_yaml_load.return_value = {
            'data_collector': {
                'depth_snapshot': {
                    'limit': 3000,
                    'update_interval_seconds': 45
                }
            },
            'redis': {
                'host': 'localhost',
                'port': 6379,
                'db': 0
            },
            'binance': {
                'symbol': 'BTCFDUSD',
                'timeout': 30
            },
            'analyzer': {
                'deepseek': {
                    'api_key': 'test-key'
                }
            }
        }

        settings = Settings.load_from_file('test_config.yaml')

        assert settings.data_collector.depth_snapshot.limit == 3000
        assert settings.data_collector.depth_snapshot.update_interval_seconds == 45
        assert not hasattr(settings.data_collector.depth_snapshot, 'window_size')

    @patch('pathlib.Path.exists')
    @patch('yaml.safe_load')
    def test_settings_load_ignores_window_size(self, mock_yaml_load, mock_path_exists):
        """Test that Settings.load_from_file ignores window_size if present in YAML."""
        mock_path_exists.return_value = True
        mock_yaml_load.return_value = {
            'data_collector': {
                'depth_snapshot': {
                    'limit': 3000,
                    'update_interval_seconds': 45,
                    'window_size': 100  # This should be ignored
                }
            },
            'redis': {
                'host': 'localhost',
                'port': 6379,
                'db': 0
            },
            'binance': {
                'symbol': 'BTCFDUSD',
                'timeout': 30
            },
            'analyzer': {
                'deepseek': {
                    'api_key': 'test-key'
                }
            }
        }

        settings = Settings.load_from_file('test_config.yaml')

        assert settings.data_collector.depth_snapshot.limit == 3000
        assert settings.data_collector.depth_snapshot.update_interval_seconds == 45
        assert not hasattr(settings.data_collector.depth_snapshot, 'window_size')


class TestBackwardCompatibility:
    """Test backward compatibility for configuration loading."""

    @patch('pathlib.Path.exists')
    @patch('yaml.safe_load')
    def test_old_config_file_loading(self, mock_yaml_load, mock_path_exists):
        """Test loading old config files that still have window_size."""
        mock_path_exists.return_value = True
        mock_yaml_load.return_value = {
            'data_collector': {
                'depth_snapshot': {
                    'limit': 5000,
                    'update_interval_seconds': 60,
                    'window_size': 60
                }
            },
            'redis': {
                'host': 'localhost',
                'port': 6379,
                'db': 0
            },
            'binance': {
                'symbol': 'BTCFDUSD',
                'timeout': 30
            },
            'analyzer': {
                'deepseek': {
                    'api_key': 'test-key'
                }
            }
        }

        # Should not raise an error
        settings = Settings.load_from_file('old_config.yaml')

        # Should ignore window_size
        assert settings.data_collector.depth_snapshot.limit == 5000
        assert settings.data_collector.depth_snapshot.update_interval_seconds == 60
        assert not hasattr(settings.data_collector.depth_snapshot, 'window_size')

    def test_config_extra_fields_ignored(self):
        """Test that extra fields in configuration are ignored."""
        # Pydantic should ignore extra fields by default
        config = DepthSnapshotConfig(
            limit=1000,
            update_interval_seconds=30,
            extra_field='should_be_ignored',
            window_size='should_also_be_ignored'
        )

        assert config.limit == 1000
        assert config.update_interval_seconds == 30
        assert not hasattr(config, 'extra_field')
        assert not hasattr(config, 'window_size')
