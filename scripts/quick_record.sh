#!/bin/bash
# Quick 3-minute recording script
# Optimized for asciinema

echo "Installing asciinema if needed..."
which asciinema || sudo apt-get install -y asciinema

echo ""
echo "========================================"
echo "  Quick Recording Script"
echo "========================================"
echo ""
echo "This will:"
echo "  1. Record a 3-minute demo"
echo "  2. Run all tests automatically"
echo "  3. Generate shareable link"
echo ""
read -p "Press ENTER to start recording..."

cd ~/HookAnalyzer

# Start recording
asciinema rec -t "HookAnalyzer - CUDA Performance Framework" hookanalyzer-demo.cast --overwrite <<'DEMO'
# Set environment
export PATH=/usr/local/cuda-12.6/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.6
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH

# Run demo
bash scripts/demo_video.sh

# Exit recording
exit
DEMO

echo ""
echo "Recording saved to: hookanalyzer-demo.cast"
echo ""
echo "To upload and share:"
echo "  asciinema upload hookanalyzer-demo.cast"
echo ""
echo "To play locally:"
echo "  asciinema play hookanalyzer-demo.cast"
echo ""
