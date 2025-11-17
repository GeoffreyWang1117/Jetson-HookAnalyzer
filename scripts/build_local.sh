#!/bin/bash

# Build HookAnalyzer locally (for development)
# Usage: ./build_local.sh [debug|release]

set -e

BUILD_TYPE="${1:-Release}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${PROJECT_DIR}/build"

echo "==================================="
echo "Building HookAnalyzer"
echo "==================================="
echo "Build type: $BUILD_TYPE"
echo "Project dir: $PROJECT_DIR"
echo "Build dir: $BUILD_DIR"
echo ""

# Create build directory
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

# Run CMake
echo "[1/3] Running CMake..."
cmake -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
      -DBUILD_TESTS=ON \
      -DBUILD_BENCHMARKS=ON \
      -DBUILD_EXAMPLES=ON \
      -DENABLE_PROFILING=ON \
      ..

# Build
echo "[2/3] Building..."
make -j$(nproc)

# Run tests
echo "[3/3] Running tests..."
ctest --output-on-failure || true

echo ""
echo "==================================="
echo "Build completed!"
echo "==================================="
echo "Executables:"
echo "  - ${BUILD_DIR}/examples/simple_demo"
echo ""
echo "Libraries:"
echo "  - ${BUILD_DIR}/libhook_analyzer.so"
echo "  - ${BUILD_DIR}/core/cuda_hook/libcuda_hook.so"
echo "  - ${BUILD_DIR}/core/scheduler/libscheduler.so"
echo "  - ${BUILD_DIR}/core/profiler/libprofiler.so"
echo ""
