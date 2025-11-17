#!/bin/bash
# Setup SSH key for password-less login to Jetson
# Usage: ./setup_ssh_key.sh

JETSON_IP="${1:-100.111.167.60}"
JETSON_USER="geoffrey"

echo "==================================="
echo "SSH Key Setup for Jetson"
echo "==================================="
echo "Target: ${JETSON_USER}@${JETSON_IP}"
echo ""
echo "You will be prompted for the Jetson password: 926494"
echo ""

# Copy SSH public key to Jetson
ssh-copy-id -i ~/.ssh/id_rsa.pub ${JETSON_USER}@${JETSON_IP}

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ SSH key copied successfully!"
    echo "You can now SSH without password: ssh ${JETSON_USER}@${JETSON_IP}"
else
    echo ""
    echo "✗ Failed to copy SSH key"
    exit 1
fi
