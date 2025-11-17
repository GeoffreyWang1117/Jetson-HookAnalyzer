#!/bin/bash

# Run HookAnalyzer in Docker development environment
# Usage: ./run_docker_dev.sh

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "==================================="
echo "Starting HookAnalyzer Docker Dev"
echo "==================================="
echo ""

# Build Docker image
echo "Building Docker image..."
docker build -t hookanalyzer:dev \
    -f "${PROJECT_DIR}/docker/Dockerfile.local" \
    "${PROJECT_DIR}"

# Run container
echo "Starting container..."
docker run --rm -it \
    --gpus all \
    --name hookanalyzer-dev \
    -v "${PROJECT_DIR}:/workspace" \
    -v "${PROJECT_DIR}/models:/workspace/models" \
    -v "${PROJECT_DIR}/data:/workspace/data" \
    -p 8000:8000 \
    -p 8888:8888 \
    --shm-size=4g \
    hookanalyzer:dev

echo ""
echo "Container exited."
