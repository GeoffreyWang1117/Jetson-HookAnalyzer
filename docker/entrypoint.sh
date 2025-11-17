#!/bin/bash
set -e

# Display GPU information
echo "=== GPU Information ==="
nvidia-smi || echo "nvidia-smi not available"
nvcc --version || echo "nvcc not available"

# Check CUDA availability
if command -v python3 &> /dev/null; then
    python3 -c "import torch; print(f'PyTorch CUDA available: {torch.cuda.is_available()}')" 2>/dev/null || true
fi

echo "=== HookAnalyzer Development Environment ==="
echo "Workspace: /workspace"
echo "Models: /workspace/models"
echo "=================================="

# Execute the command
exec "$@"
