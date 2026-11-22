#!/bin/sh
# Install the studio systemd service on the Jetson Nano (run on the host with sudo).
set -e
# Folder of this script
HERE=$(cd "$(dirname "$0")" && pwd)
# Helper script location used by the service
mkdir -p /opt/studio
# Copy the container wait helper
install -m 755 "$HERE/wait_container.sh" /opt/studio/wait_container.sh
# Copy the unit file
install -m 644 "$HERE/studio.service" /etc/systemd/system/studio.service
# Reload systemd and start at every boot
systemctl daemon-reload
# Enable and start now
systemctl enable --now studio.service
# Show the result
systemctl --no-pager status studio.service || true
