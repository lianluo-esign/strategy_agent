#!/bin/bash
# Strategy Agent Data Collector Service Installation Script

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
SERVICE_NAME="strategy-agent-data-collector"

echo "Installing Strategy Agent Data Collector systemd service..."

# Check if running as root for system-wide installation
if [[ $EUID -ne 0 ]]; then
   echo "This script must be run as root for system-wide installation"
   echo "Usage: sudo ./install.sh"
   exit 1
fi

# Verify project directory
if [[ ! -f "$PROJECT_DIR/agent_data_collector.py" ]]; then
    echo "Error: agent_data_collector.py not found in $PROJECT_DIR"
    exit 1
fi

# Check if configuration file exists
if [[ ! -f "$PROJECT_DIR/config/development.yaml" ]]; then
    echo "Error: Configuration file not found: $PROJECT_DIR/config/development.yaml"
    exit 1
fi

# Check if .env file exists for sensitive information
if [[ ! -f "$PROJECT_DIR/.env" ]]; then
    echo "Warning: .env file not found. Please create .env file with DEEPSEEK_API_KEY."
    echo "Example: echo 'DEEPSEEK_API_KEY=your_api_key_here' > .env"
fi

echo "Configuration file found: $PROJECT_DIR/config/development.yaml"

# Copy service file
echo "Installing systemd service file..."
cp "$SCRIPT_DIR/$SERVICE_NAME.service" "/etc/systemd/system/"
chmod 644 "/etc/systemd/system/$SERVICE_NAME.service"

# Copy health check service file
if [[ -f "$SCRIPT_DIR/$SERVICE_NAME-health.service" ]]; then
    echo "Installing health check service file..."
    cp "$SCRIPT_DIR/$SERVICE_NAME-health.service" "/etc/systemd/system/"
    chmod 644 "/etc/systemd/system/$SERVICE_NAME-health.service"
fi

# Copy timer file (optional health check)
if [[ -f "$SCRIPT_DIR/$SERVICE_NAME.timer" ]]; then
    echo "Installing systemd timer file..."
    cp "$SCRIPT_DIR/$SERVICE_NAME.timer" "/etc/systemd/system/"
    chmod 644 "/etc/systemd/system/$SERVICE_NAME.timer"
fi

# Copy logrotate configuration
echo "Installing logrotate configuration..."
mkdir -p "/etc/logrotate.d"
cp "$SCRIPT_DIR/logrotate.d/$SERVICE_NAME" "/etc/logrotate.d/"
chmod 644 "/etc/logrotate.d/$SERVICE_NAME"

# Validate configuration
echo "Validating configuration..."
if ! systemd-analyze verify "/etc/systemd/system/$SERVICE_NAME.service"; then
    echo "Error: Service configuration is invalid"
    exit 1
fi

# Reload systemd daemon
echo "Reloading systemd daemon..."
systemctl daemon-reload

# Enable the service (but don't start it)
echo "Enabling $SERVICE_NAME service..."
systemctl enable "$SERVICE_NAME.service"

# Enable the timer if available
if [[ -f "/etc/systemd/system/$SERVICE_NAME.timer" ]]; then
    echo "Enabling $SERVICE_NAME timer..."
    systemctl enable "$SERVICE_NAME.timer"
fi

echo ""
echo "Installation completed successfully!"
echo ""
echo "Service management commands:"
echo "  Start service:   sudo systemctl start $SERVICE_NAME.service"
echo "  Stop service:    sudo systemctl stop $SERVICE_NAME.service"
echo "  Restart service: sudo systemctl restart $SERVICE_NAME.service"
echo "  Service status:  sudo systemctl status $SERVICE_NAME.service"
echo "  View logs:       sudo journalctl -u $SERVICE_NAME.service -f"
echo ""
echo "Timer management commands (if installed):"
echo "  Start timer:     sudo systemctl start $SERVICE_NAME.timer"
echo "  Stop timer:      sudo systemctl stop $SERVICE_NAME.timer"
echo "  Timer status:    sudo systemctl status $SERVICE_NAME.timer"
echo ""
echo "Note: The service is installed but not started. Start it manually when ready."