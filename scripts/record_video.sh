#!/bin/bash

# Video Recording Script for HookAnalyzer Demo
# Uses asciinema + agg to create MP4/GIF video

set -e

echo "==================================="
echo "HookAnalyzer Video Recording Script"
echo "==================================="
echo

# Change to project directory
cd ~/HookAnalyzer

# Check if agg is installed (for converting asciinema to gif/video)
if ! command -v agg &> /dev/null; then
    echo "Installing agg (asciinema gif generator)..."

    # Download agg binary for ARM64
    AGG_VERSION="1.4.3"
    wget -q https://github.com/asciinema/agg/releases/download/v${AGG_VERSION}/agg-aarch64-unknown-linux-gnu -O /tmp/agg
    chmod +x /tmp/agg
    sudo mv /tmp/agg /usr/local/bin/agg

    echo "✓ agg installed"
fi

# Set terminal size for better video quality
export COLUMNS=120
export LINES=30

# Record the demo
echo "Starting asciinema recording..."
echo "Terminal size: ${COLUMNS}x${LINES}"
echo

# Create timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
CAST_FILE="demo_${TIMESTAMP}.cast"
GIF_FILE="demo_${TIMESTAMP}.gif"
MP4_FILE="demo_${TIMESTAMP}.mp4"

# Record with asciinema
asciinema rec \
    --cols ${COLUMNS} \
    --rows ${LINES} \
    --title "HookAnalyzer - CUDA Performance Framework on Jetson Orin Nano" \
    --command "bash scripts/demo_video.sh" \
    --overwrite \
    "${CAST_FILE}"

echo
echo "✓ Recording complete: ${CAST_FILE}"

# Convert to GIF using agg
echo
echo "Converting to GIF..."
agg \
    --font-size 14 \
    --theme monokai \
    "${CAST_FILE}" \
    "${GIF_FILE}"

echo "✓ GIF created: ${GIF_FILE}"

# Convert GIF to MP4 using ffmpeg
echo
echo "Converting to MP4..."
ffmpeg -i "${GIF_FILE}" \
    -movflags faststart \
    -pix_fmt yuv420p \
    -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" \
    "${MP4_FILE}" \
    -y -loglevel error

echo "✓ MP4 created: ${MP4_FILE}"

# Show file sizes
echo
echo "==================================="
echo "Recording Complete!"
echo "==================================="
echo
ls -lh "${CAST_FILE}" "${GIF_FILE}" "${MP4_FILE}"
echo
echo "Files created:"
echo "  - ${CAST_FILE} (asciinema format)"
echo "  - ${GIF_FILE} (animated GIF)"
echo "  - ${MP4_FILE} (video file)"
echo
echo "To upload asciinema:"
echo "  asciinema upload ${CAST_FILE}"
echo
echo "To transfer to local machine:"
echo "  scp geoffrey@100.111.167.60:~/HookAnalyzer/${MP4_FILE} ."
echo
