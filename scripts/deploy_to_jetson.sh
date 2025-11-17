#!/bin/bash

# Deploy HookAnalyzer to Jetson Orin Nano
# Usage: ./deploy_to_jetson.sh [jetson_ip]

set -e

# Configuration
JETSON_IP="${1:-100.111.167.60}"
JETSON_USER="geoffrey"
JETSON_DIR="/home/geoffrey/HookAnalyzer"
LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "==================================="
echo "HookAnalyzer Jetson Deployment"
echo "==================================="
echo "Jetson IP: $JETSON_IP"
echo "Local dir: $LOCAL_DIR"
echo "Remote dir: $JETSON_DIR"
echo ""

# Function to run commands on Jetson
jetson_ssh() {
    ssh -o StrictHostKeyChecking=no ${JETSON_USER}@${JETSON_IP} "$@"
}

# Function to copy files to Jetson
jetson_scp() {
    scp -o StrictHostKeyChecking=no -r "$1" ${JETSON_USER}@${JETSON_IP}:"$2"
}

# Test connection
echo "[1/6] Testing connection to Jetson..."
if ! ping -c 1 ${JETSON_IP} > /dev/null 2>&1; then
    echo "Error: Cannot reach Jetson at ${JETSON_IP}"
    exit 1
fi
echo "✓ Connection OK"

# Create directory on Jetson
echo "[2/6] Creating directory on Jetson..."
jetson_ssh "mkdir -p ${JETSON_DIR}"
echo "✓ Directory created"

# Sync source code
echo "[3/6] Syncing source code..."
rsync -avz --progress \
    --exclude 'build/' \
    --exclude '.git/' \
    --exclude '__pycache__/' \
    --exclude '*.pyc' \
    --exclude 'models/*.trt' \
    ${LOCAL_DIR}/ ${JETSON_USER}@${JETSON_IP}:${JETSON_DIR}/
echo "✓ Source code synced"

# Build on Jetson
echo "[4/6] Building on Jetson..."
jetson_ssh "cd ${JETSON_DIR} && \
    mkdir -p build && \
    cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release \
          -DENABLE_TENSORRT=ON \
          -DENABLE_PROFILING=ON \
          -DBUILD_EXAMPLES=ON \
          .. && \
    make -j\$(nproc)"
echo "✓ Build completed"

# Run tests (optional)
echo "[5/6] Running tests..."
jetson_ssh "cd ${JETSON_DIR}/build && ctest --output-on-failure" || true
echo "✓ Tests completed"

# Create systemd service (optional)
echo "[6/6] Setting up systemd service..."
jetson_ssh "sudo tee /etc/systemd/system/hookanalyzer.service > /dev/null" << 'EOF'
[Unit]
Description=HookAnalyzer Inference Service
After=network.target

[Service]
Type=simple
User=geoffrey
WorkingDirectory=/home/geoffrey/HookAnalyzer
ExecStart=/usr/bin/python3 /home/geoffrey/HookAnalyzer/api/server/main.py
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

jetson_ssh "sudo systemctl daemon-reload"
echo "✓ Service configured (use 'sudo systemctl start hookanalyzer' to start)"

echo ""
echo "==================================="
echo "Deployment completed successfully!"
echo "==================================="
echo ""
echo "Next steps:"
echo "1. SSH to Jetson: ssh ${JETSON_USER}@${JETSON_IP}"
echo "2. Run demo: cd ${JETSON_DIR}/build && ./examples/simple_demo"
echo "3. Start service: sudo systemctl start hookanalyzer"
echo ""
