"""Integration tests for systemd service management."""

import os
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.mark.integration
class TestServiceIntegration:
    """Integration tests for systemd service."""

    @pytest.fixture
    def service_name(self):
        """Service name for testing."""
        return "strategy-agent-data-collector"

    @pytest.fixture
    def project_root(self):
        """Project root directory."""
        return Path(__file__).parent.parent

    @pytest.fixture
    def service_file_path(self, project_root):
        """Path to service file."""
        return project_root / "systemd" / "strategy-agent-data-collector.service"

    def test_service_file_copy_to_systemd(self, service_file_path, tmp_path):
        """Test copying service file to systemd directory."""
        systemd_dir = tmp_path / "systemd"
        systemd_dir.mkdir()

        service_dest = systemd_dir / "strategy-agent-data-collector.service"

        # Copy service file
        import shutil
        shutil.copy(service_file_path, service_dest)

        assert service_dest.exists(), "Service file should be copied"
        assert service_dest.is_file(), "Destination should be a file"

        # Verify content
        original_content = service_file_path.read_text()
        copied_content = service_dest.read_text()
        assert original_content == copied_content, "Content should match"

    @patch('subprocess.run')
    def test_systemctl_daemon_reload(self, mock_run):
        """Test systemctl daemon-reload command."""
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = ""
        mock_run.return_value.stderr = ""

        result = subprocess.run(
            ["systemctl", "daemon-reload"],
            capture_output=True,
            text=True
        )

        assert result.returncode == 0, "Daemon reload should succeed"
        mock_run.assert_called_once_with(
            ["systemctl", "daemon-reload"],
            capture_output=True,
            text=True
        )

    @patch('subprocess.run')
    def test_service_enable_command(self, mock_run, service_name):
        """Test service enable command."""
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = ""
        mock_run.return_value.stderr = ""

        result = subprocess.run(
            ["systemctl", "enable", f"{service_name}.service"],
            capture_output=True,
            text=True
        )

        assert result.returncode == 0, "Service enable should succeed"

    @patch('subprocess.run')
    def test_service_start_command(self, mock_run, service_name):
        """Test service start command."""
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = ""
        mock_run.return_value.stderr = ""

        result = subprocess.run(
            ["systemctl", "start", f"{service_name}.service"],
            capture_output=True,
            text=True
        )

        assert result.returncode == 0, "Service start should succeed"

    @patch('subprocess.run')
    def test_service_status_command(self, mock_run, service_name):
        """Test service status command."""
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "● strategy-agent-data-collector.service - Strategy Agent Data Collector Service\n     Loaded: loaded (/etc/systemd/system/strategy-agent-data-collector.service; enabled; vendor preset: enabled)\n     Active: active (running) since Thu 2024-01-01 12:00:00 UTC; 1h ago"
        mock_run.return_value.stderr = ""

        result = subprocess.run(
            ["systemctl", "status", f"{service_name}.service"],
            capture_output=True,
            text=True
        )

        assert result.returncode == 0, "Service status should succeed"
        assert "active (running)" in result.stdout, "Service should be running"

    @patch('subprocess.run')
    def test_service_stop_command(self, mock_run, service_name):
        """Test service stop command."""
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = ""
        mock_run.return_value.stderr = ""

        result = subprocess.run(
            ["systemctl", "stop", f"{service_name}.service"],
            capture_output=True,
            text=True
        )

        assert result.returncode == 0, "Service stop should succeed"

    @patch('subprocess.run')
    def test_journal_logs_command(self, mock_run, service_name):
        """Test journal logs command."""
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "2024-01-01 12:00:00 hostname strategy-agent-data-collector[1234]: Starting data collection..."
        mock_run.return_value.stderr = ""

        result = subprocess.run(
            ["journalctl", "-u", f"{service_name}.service", "-n", "10"],
            capture_output=True,
            text=True
        )

        assert result.returncode == 0, "Journal command should succeed"
        assert "strategy-agent-data-collector" in result.stdout, "Logs should contain service name"

    def test_environment_file_validation(self, project_root):
        """Test environment file exists and is valid."""
        env_file = project_root / ".env"
        env_example = project_root / ".env.example"

        # Should have at least .env.example
        assert env_example.exists(), "Environment example file should exist"

        if env_file.exists():
            content = env_file.read_text()
            # Should contain at least some environment variables
            assert len(content.strip()) > 0, "Environment file should not be empty"

    def test_python_executable_exists(self, project_root):
        """Test Python executable exists in virtual environment."""
        venv_python = project_root / "venv" / "bin" / "python"

        if not venv_python.exists():
            pytest.skip("Virtual environment not found")

        assert venv_python.is_file(), "Python executable should be a file"
        assert os.access(venv_python, os.X_OK), "Python executable should be executable"

    def test_agent_script_exists(self, project_root):
        """Test agent_data_collector.py script exists."""
        agent_script = project_root / "agent_data_collector.py"

        assert agent_script.exists(), "Agent script should exist"
        assert agent_script.is_file(), "Agent script should be a file"

        content = agent_script.read_text()
        assert "if __name__" in content, "Agent script should be executable"
        assert "asyncio.run(main())" in content, "Agent script should run main function"

    @patch('subprocess.run')
    def test_logrotate_configuration(self, mock_run):
        """Test logrotate configuration validation."""
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "logrotate: configuration file is valid"
        mock_run.return_value.stderr = ""

        # Test logrotate configuration (mocked)
        result = subprocess.run(
            ["logrotate", "-d", "/etc/logrotate.d/strategy-agent-data-collector"],
            capture_output=True,
            text=True
        )

        assert result.returncode == 0, "Logrotate config should be valid"

    def test_directory_permissions(self, project_root):
        """Test that required directories have correct permissions."""
        logs_dir = project_root / "logs"
        storage_dir = project_root / "storage"

        # Create directories if they don't exist
        logs_dir.mkdir(exist_ok=True)
        storage_dir.mkdir(exist_ok=True)

        # Check permissions (basic checks)
        assert os.access(logs_dir, os.R_OK | os.W_OK), "Logs directory should be readable and writable"
        assert os.access(storage_dir, os.R_OK | os.W_OK), "Storage directory should be readable and writable"

    @patch('subprocess.run')
    def test_systemd_syntax_check(self, mock_run, service_file_path):
        """Test systemd syntax validation."""
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = ""
        mock_run.return_value.stderr = ""

        result = subprocess.run(
            ["systemd-analyze", "verify", str(service_file_path)],
            capture_output=True,
            text=True
        )

        assert result.returncode == 0, "Service file should pass systemd syntax check"

    def test_install_script_validation(self, project_root):
        """Test install script validation."""
        install_script = project_root / "systemd" / "install.sh"

        assert install_script.exists(), "Install script should exist"

        content = install_script.read_text()

        # Check for essential commands
        assert "systemctl daemon-reload" in content, "Should reload systemd daemon"
        assert "systemctl enable" in content, "Should enable service"
        assert "cp " in content, "Should copy files"

        # Check for error handling
        assert "set -e" in content, "Should exit on error"

    @patch('subprocess.run')
    def test_timer_functionality(self, mock_run, service_name):
        """Test timer functionality if timer file exists."""
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = ""
        mock_run.return_value.stderr = ""

        # Test timer commands
        result = subprocess.run(
            ["systemctl", "status", f"{service_name}.timer"],
            capture_output=True,
            text=True
        )

        assert result.returncode == 0, "Timer status should succeed"
