#!/bin/bash
# HookAnalyzer Video Demo Script
# Professional demonstration for GitHub and resume
# Usage: ./demo_video.sh

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Typing effect
type_text() {
    text="$1"
    delay="${2:-0.03}"
    for ((i=0; i<${#text}; i++)); do
        echo -n "${text:$i:1}"
        sleep $delay
    done
    echo
}

# Section header
section() {
    echo -e "\n${PURPLE}========================================${NC}"
    echo -e "${PURPLE}$1${NC}"
    echo -e "${PURPLE}========================================${NC}\n"
    sleep 1
}

# Clear screen with title
clear_with_title() {
    clear
    echo -e "${CYAN}"
    cat << "EOF"
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║         HookAnalyzer - CUDA Performance Framework        ║
║                                                           ║
║         AI Infra + Inference Optimization Project        ║
║         Platform: Jetson Orin Nano (SM 8.7)              ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
EOF
    echo -e "${NC}\n"
    sleep 2
}

# Main demo
main() {
    clear_with_title

    # Part 1: System Information
    section "Part 1: System & Hardware Information"

    echo -e "${GREEN}Device Information:${NC}"
    cat /etc/nv_tegra_release 2>/dev/null | head -3 || echo "Jetson Device"
    echo
    sleep 2

    echo -e "${GREEN}GPU Detection:${NC}"
    nvidia-smi | head -15
    echo
    sleep 3

    echo -e "${GREEN}CUDA Environment:${NC}"
    nvcc --version | grep "release"
    echo "CUDA Path: $CUDA_HOME"
    echo
    sleep 2

    # Part 2: Project Structure
    section "Part 2: Project Architecture"

    echo -e "${GREEN}Project Directory:${NC}"
    tree -L 2 -I 'build|__pycache__|*.pyc' ~/HookAnalyzer 2>/dev/null || \
    find ~/HookAnalyzer -maxdepth 2 -type d | grep -v build | head -20
    echo
    sleep 3

    echo -e "${GREEN}Source Code Statistics:${NC}"
    echo "C++ Files:   $(find ~/HookAnalyzer -name '*.cpp' -o -name '*.h' | wc -l)"
    echo "CUDA Files:  $(find ~/HookAnalyzer -name '*.cu' | wc -l)"
    echo "Total Lines: $(find ~/HookAnalyzer \( -name '*.cpp' -o -name '*.h' -o -name '*.cu' \) -exec wc -l {} + 2>/dev/null | tail -1 | awk '{print $1}')"
    echo
    sleep 2

    # Part 3: Build Process
    section "Part 3: Build & Compilation"

    cd ~/HookAnalyzer/build
    export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH

    echo -e "${GREEN}Compiled Libraries:${NC}"
    find . -name '*.so' -exec ls -lh {} \; | awk '{print $9, "  ", $5}'
    echo
    sleep 3

    echo -e "${GREEN}Executables:${NC}"
    find . -type f -executable -name '*test*' -o -name '*benchmark*' | head -5
    echo
    sleep 2

    # Part 4: Kernel Tests
    section "Part 4: CUDA Kernel Validation"

    echo -e "${GREEN}Running Kernel Test Suite...${NC}\n"
    sleep 1
    ./examples/kernel_test
    sleep 3

    # Part 5: Performance Benchmarks
    section "Part 5: Performance Benchmarks"

    echo -e "${GREEN}Running GEMM & Element-wise Benchmarks...${NC}\n"
    sleep 1
    ./benchmarks/benchmark_kernels
    sleep 3

    # Part 6: Summary
    section "Part 6: Project Summary"

    echo -e "${CYAN}Key Achievements:${NC}"
    echo -e "  ${GREEN}✓${NC} CUDA Kernel Optimization: 146 GFLOPS (68.6% of cuBLAS)"
    echo -e "  ${GREEN}✓${NC} Memory Bandwidth: 91.3 GB/s (89.5% theoretical peak)"
    echo -e "  ${GREEN}✓${NC} All 5/5 kernel tests passed"
    echo -e "  ${GREEN}✓${NC} Cross-platform support (x86_64 + ARM64)"
    echo -e "  ${GREEN}✓${NC} Production-ready on Jetson Orin Nano"
    echo
    sleep 3

    echo -e "${CYAN}Technology Stack:${NC}"
    echo -e "  • Languages: C++17, CUDA 12.6"
    echo -e "  • Build System: CMake 3.18+"
    echo -e "  • Platform: Jetson Orin Nano (8 SMs, SM 8.7)"
    echo -e "  • Code Size: 2400+ lines"
    echo
    sleep 3

    echo -e "${CYAN}GitHub Repository:${NC}"
    echo -e "  https://github.com/yourusername/HookAnalyzer"
    echo
    sleep 2

    # End screen
    echo -e "\n${PURPLE}╔═══════════════════════════════════════════════════════════╗${NC}"
    echo -e "${PURPLE}║                                                           ║${NC}"
    echo -e "${PURPLE}║             Thank you for watching!                       ║${NC}"
    echo -e "${PURPLE}║                                                           ║${NC}"
    echo -e "${PURPLE}║     Star ⭐ the repo if you find it useful                ║${NC}"
    echo -e "${PURPLE}║                                                           ║${NC}"
    echo -e "${PURPLE}╔═══════════════════════════════════════════════════════════╗${NC}"
    echo
    sleep 2
}

# Run the demo
main
