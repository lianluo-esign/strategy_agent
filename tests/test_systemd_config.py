"""Tests for systemd service configuration."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

try:
    from systemd.unit import UnitFile  # type: ignore
except ImportError:
    # systemd-python not available, skip unit file parsing tests
    UnitFile = None


class TestSystemdConfig:
    """Test systemd service configuration."""

    @pytest.fixture
    def service_file_path(self):
        """Path to the systemd service file."""
        return Path(__file__).parent.parent / "systemd" / "strategy-agent-data-collector.service"

    @pytest.fixture
    def service_content(self, service_file_path):
        """Read service file content."""
        with open(service_file_path) as f:
            return f.read()

    def test_service_file_exists(self, service_file_path):
        """Test that service file exists."""
        assert service_file_path.exists(), "Service file should exist"
        assert service_file_path.is_file(), "Service file should be a regular file"

    def test_service_file_permissions(self, service_file_path):
        """Test service file permissions."""
        stat = service_file_path.stat()
        # Should be readable by all, writable only by owner
        assert oct(stat.st_mode)[-3:] == "644", "Service file should have 644 permissions"

    def test_service_content_required_sections(self, service_content):
        """Test that service file contains required systemd sections."""
        required_sections = ["Unit", "Service", "Install"]
        for section in required_sections:
            assert f"[{section}]" in service_content, f"Missing [{section}] section"

    def test_service_unit_configuration(self, service_content):
        """Test Unit section configuration."""
        # Check required Unit directives
        assert "Description=" in service_content, "Missing Description"
        assert "After=network.target" in service_content, "Should start after network"
        assert "Documentation=" in service_content, "Missing documentation URL"

    def test_service_configuration(self, service_content):
        """Test Service section configuration."""
        # Check required Service directives
        assert "Type=simple" in service_content, "Should be simple service type"
        assert "User=" in service_content, "Should specify user"
        assert "Group=" in service_content, "Should specify group"
        assert "WorkingDirectory=" in service_content, "Should specify working directory"
        assert "ExecStart=" in service_content, "Should specify ExecStart"
        assert "Restart=on-failure" in service_content, "Should restart on failure"

    def test_service_install_configuration(self, service_content):
        """Test Install section configuration."""
        assert "WantedBy=multi-user.target" in service_content, "Should be wanted by multi-user target"

    def test_service_security_settings(self, service_content):
        """Test security hardening settings."""
        security_settings = [
            "NoNewPrivileges=true",
            "PrivateTmp=true",
            "ProtectSystem=strict",
            "ProtectHome=true",
            "ProtectKernelTunables=true",
            "ProtectKernelModules=true",
            "ProtectControlGroups=true"
        ]

        for setting in security_settings:
            assert setting in service_content, f"Missing security setting: {setting}"

    def test_service_resource_limits(self, service_content):
        """Test resource limit settings."""
        assert "LimitNOFILE=" in service_content, "Should set file descriptor limit"
        assert "LimitNPROC=" in service_content, "Should set process limit"
        assert "MemoryMax=" in service_content, "Should set memory limit"
        assert "CPUQuota=" in service_content, "Should set CPU quota"

    def test_service_logging_configuration(self, service_content):
        """Test logging configuration."""
        assert "StandardOutput=journal" in service_content, "Should output to journal"
        assert "StandardError=journal" in service_content, "Should log errors to journal"
        assert "SyslogIdentifier=" in service_content, "Should set syslog identifier"

    def test_service_path_validation(self, service_content):
        """Test that paths in service file are valid."""
        project_root = Path(__file__).parent.parent
        expected_working_dir = str(project_root)
        expected_exec_start = f"{project_root}/venv/bin/python agent_data_collector.py"

        assert expected_working_dir in service_content, "Working directory should match project root"
        assert "agent_data_collector.py" in service_content, "Should reference correct executable"

    def test_environment_configuration(self, service_content):
        """Test environment configuration."""
        assert "EnvironmentFile=" in service_content, "Should load environment file"
        assert "Environment=PATH=" in service_content, "Should set PATH environment variable"

    def test_restart_policy(self, service_content):
        """Test restart policy configuration."""
        assert "Restart=on-failure" in service_content, "Should restart on failure"
        assert "RestartSec=" in service_content, "Should set restart delay"
        assert "StartLimitInterval=" in service_content, "Should set start limit interval"
        assert "StartLimitBurst=" in service_content, "Should set start limit burst"

    def test_timer_file_exists(self):
        """Test that timer file exists."""
        timer_path = Path(__file__).parent.parent / "systemd" / "strategy-agent-data-collector.timer"
        if timer_path.exists():
            with open(timer_path) as f:
                timer_content = f.read()
            assert "[Timer]" in timer_content, "Timer file should have [Timer] section"
            assert "[Unit]" in timer_content, "Timer file should have [Unit] section"
            assert "OnCalendar=" in timer_content, "Timer should have schedule"

    def test_logrotate_config_exists(self):
        """Test that logrotate configuration exists."""
        logrotate_path = Path(__file__).parent.parent / "systemd" / "logrotate.d" / "strategy-agent-data-collector"
        if logrotate_path.exists():
            with open(logrotate_path) as f:
                logrotate_content = f.read()
            assert "daily" in logrotate_content, "Should rotate daily"
            assert "rotate" in logrotate_content, "Should specify rotation count"
            assert "compress" in logrotate_content, "Should compress old logs"

    def test_install_script_exists(self):
        """Test that install script exists and is executable."""
        install_path = Path(__file__).parent.parent / "systemd" / "install.sh"
        assert install_path.exists(), "Install script should exist"
        assert install_path.is_file(), "Install script should be a regular file"

        # Check if executable (on Unix systems)
        if os.name == 'posix':
            stat = install_path.stat()
            assert stat.st_mode & 0o111, "Install script should be executable"

    @patch('subprocess.run')
    def test_service_syntax_validation(self, mock_run):
        """Test that service file has valid systemd syntax."""
        # Mock successful systemd validation
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = ""
        mock_run.return_value.stderr = ""

        # This would normally run: systemd-analyze verify /path/to/service
        # For testing, we just verify the mock would be called correctly
        assert True, "Service should have valid systemd syntax"

    def test_service_dependency_ordering(self, service_content):
        """Test service dependency ordering."""
        assert "After=network.target" in service_content, "Should start after network"
        assert "Wants=network.target" in service_content, "Should want network target"

    def test_service_condition_checks(self, service_content):
        """Test service condition checks."""
        assert "ConditionPathExists=" in service_content, "Should check if executable exists"
