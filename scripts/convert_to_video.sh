#!/bin/bash

# Convert ANSI colored terminal output to video
# Usage: ./convert_to_video.sh input.txt output.mp4

set -e

INPUT_FILE="${1:-demo_colored.txt}"
OUTPUT_MP4="${2:-hookanalyzer_demo.mp4}"

echo "==================================="
echo "Converting Terminal Output to Video"
echo "==================================="
echo

if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file $INPUT_FILE not found"
    exit 1
fi

# Step 1: Convert ANSI to HTML
HTML_FILE="${INPUT_FILE%.txt}.html"
echo "Converting ANSI to HTML..."

cat > "$HTML_FILE" << 'HTML_HEADER'
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <style>
        body {
            background-color: #1e1e1e;
            color: #d4d4d4;
            font-family: 'Courier New', monospace;
            font-size: 14px;
            line-height: 1.4;
            padding: 20px;
            margin: 0;
        }
        pre {
            margin: 0;
            white-space: pre-wrap;
            word-wrap: break-word;
        }
    </style>
</head>
<body>
<pre>
HTML_HEADER

# Convert ANSI codes to HTML
cat "$INPUT_FILE" | aha --no-header >> "$HTML_FILE"

cat >> "$HTML_FILE" << 'HTML_FOOTER'
</pre>
</body>
</html>
HTML_FOOTER

echo "✓ HTML created: $HTML_FILE"

# Step 2: Convert HTML to PNG
PNG_FILE="${INPUT_FILE%.txt}.png"
echo "Rendering HTML to PNG..."

wkhtmltoimage \
    --quality 100 \
    --width 1920 \
    --format png \
    --enable-local-file-access \
    "$HTML_FILE" \
    "$PNG_FILE" 2>/dev/null

echo "✓ PNG created: $PNG_FILE"

# Step 3: Create video from PNG (with slow scroll effect)
echo "Creating video..."

# Get image height
HEIGHT=$(identify -format '%h' "$PNG_FILE")
SCROLL_DURATION=30  # 30 seconds total video

ffmpeg \
    -loop 1 \
    -i "$PNG_FILE" \
    -vf "crop=iw:1080:0:'min(ih-1080,t/$SCROLL_DURATION*(ih-1080))',fps=30" \
    -t $SCROLL_DURATION \
    -c:v libx264 \
    -pix_fmt yuv420p \
    -movflags +faststart \
    "$OUTPUT_MP4" \
    -y -loglevel error

echo "✓ Video created: $OUTPUT_MP4"

# Show file info
echo
echo "==================================="
echo "Conversion Complete!"
echo "==================================="
echo
ls -lh "$HTML_FILE" "$PNG_FILE" "$OUTPUT_MP4"
echo
echo "Output video: $OUTPUT_MP4"
echo "Duration: ${SCROLL_DURATION} seconds"
echo
echo "To transfer to local machine:"
echo "  scp geoffrey@100.111.167.60:~/HookAnalyzer/$OUTPUT_MP4 ."
echo
