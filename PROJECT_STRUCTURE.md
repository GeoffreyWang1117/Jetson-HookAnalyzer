# Project Structure

## Overview

This document describes the organization of the HookAnalyzer project.

## Directory Layout

```
HookAnalyzer/
├── README.md                    # English documentation (main)
├── README_CN.md                 # Chinese documentation
├── LICENSE                      # MIT License
├── CMakeLists.txt              # Root CMake configuration
├── requirements.txt            # Python dependencies
├── CONTRIBUTING.md             # Contribution guidelines
├── DEPLOYMENT_GUIDE.md         # Deployment instructions
├── PROJECT_STRUCTURE.md        # This file
│
├── core/                       # Core framework components
│   ├── cuda_hook/             # CUDA API interception layer
│   │   ├── cuda_hook.h
│   │   ├── cuda_hook.cpp
│   │   └── CMakeLists.txt
│   ├── scheduler/             # Multi-model inference scheduler
│   │   ├── scheduler.h
│   │   ├── scheduler.cpp
│   │   └── CMakeLists.txt
│   └── profiler/              # Performance profiling utilities
│       ├── profiler.h
│       ├── profiler.cpp
│       └── CMakeLists.txt
│
├── engines/                    # Inference engine adapters
│   ├── tensorrt_adapter/      # TensorRT C++ wrapper
│   │   ├── tensorrt_engine.h
│   │   ├── tensorrt_engine.cpp (~350 LOC)
│   │   └── CMakeLists.txt
│   └── onnx_adapter/          # ONNX Runtime wrapper (planned)
│       └── CMakeLists.txt
│
├── kernels/                    # Custom CUDA kernels
│   └── optimized/             # Optimized kernel implementations
│       ├── kernels.h
│       ├── kernels.cu         # GEMM, element-wise ops, activations
│       ├── gemm_optimized_v2.cu  # Advanced GEMM variants
│       └── CMakeLists.txt
│
├── benchmarks/                 # Performance benchmarking tools
│   ├── benchmark_kernels.cpp
│   ├── gemm_analysis.cpp      # GEMM performance analysis
│   ├── gemm_compare.cpp       # GEMM variant comparison
│   └── CMakeLists.txt
│
├── examples/                   # Usage examples and tests
│   ├── simple_demo_minimal.cpp
│   ├── kernel_test.cpp        # Kernel validation suite
│   ├── test_tensorrt.cpp      # TensorRT inference test
│   └── CMakeLists.txt
│
├── scripts/                    # Build and deployment scripts
│   ├── setup_yolov8.py        # YOLOv8 model setup
│   ├── test_yolov8_simple.py
│   ├── test_yolov8_inference.py
│   ├── record_video.sh        # Demo recording
│   └── convert_to_video.sh
│
├── api/                        # RESTful API service
│   └── server/
│       ├── main.py            # FastAPI application
│       ├── config.py
│       ├── models.py
│       └── routes/
│
├── monitoring/                 # Metrics and monitoring
│   ├── metrics/               # Prometheus exporter
│   └── dashboard/             # Grafana configurations
│
├── tests/                      # Unit and integration tests
│   ├── test_scheduler.cpp
│   ├── test_profiler.cpp
│   └── CMakeLists.txt
│
├── docker/                     # Docker configurations
│   ├── Dockerfile
│   └── docker-compose.yml
│
├── docs/                       # Documentation
│   ├── experiments/           # Experimental reports
│   │   ├── EXPERIMENT1_REPORT.md       # GEMM optimization analysis
│   │   ├── EXPERIMENT3_PROGRESS.md     # Experiment 3 progress log
│   │   ├── EXPERIMENT3_RESULTS.md      # TensorRT integration results
│   │   ├── EXPERIMENT_ROADMAP.md       # Future experiments
│   │   ├── VERIFICATION_REPORT.md      # Initial validation
│   │   ├── FINAL_SUMMARY.md           # Project completion summary
│   │   └── VIDEO_RECORDING_GUIDE.md   # Video demo guide
│   ├── media/                 # Media files
│   │   ├── hookanalyzer_demo.mp4      # Demo video
│   │   ├── demo_colored.png           # Terminal screenshot
│   │   └── demo_output.txt            # Sample output
│   ├── PROJECT_OVERVIEW.md    # High-level overview
│   ├── DEMO_SAMPLES.md        # Demo samples
│   ├── quick_start.md         # Quick start guide
│   └── README_WITH_VIDEO.md   # Documentation with video
│
├── models/                     # Model files (not in repo)
├── logs/                       # Runtime logs
└── data/                       # Sample data

```

## Key Components

### Core Framework (core/)

**CUDA Hook Layer** (`cuda_hook/`)
- Intercepts CUDA API calls (cudaMalloc, cudaFree, etc.)
- Tracks GPU memory allocations and deallocations
- Provides performance profiling hooks

**Scheduler** (`scheduler/`)
- Priority-based multi-model task scheduling
- Dynamic batching support
- CUDA stream management
- Resource allocation and contention handling

**Profiler** (`profiler/`)
- Real-time performance metrics collection
- GPU utilization tracking
- Latency and throughput measurement

### Inference Engines (engines/)

**TensorRT Adapter** (`tensorrt_adapter/`)
- Complete C++ TensorRT wrapper (~350 LOC)
- Engine loading and serialization
- Synchronous and asynchronous inference
- FP16 precision optimization
- Comprehensive benchmarking capabilities
- **Status:** ✅ Production-ready

**ONNX Adapter** (`onnx_adapter/`)
- **Status:** 📋 Planned for future implementation

### Custom Kernels (kernels/optimized/)

**Implemented Kernels:**
- GEMM (General Matrix Multiply)
  - 16×16 tiled implementation (100% occupancy on Jetson)
  - 32×32, 64×64 variants for comparison
  - Double-buffered variant
- Element-wise operations (add, multiply, etc.)
- Activation functions (ReLU, Sigmoid)
- **Performance:** 146 GFLOPS (68.6% of cuBLAS on 512×512 matrices)

### Benchmarking Tools (benchmarks/)

- `benchmark_kernels.cpp` - Comprehensive kernel performance suite
- `gemm_analysis.cpp` - GEMM optimization analysis
- `gemm_compare.cpp` - Compare different GEMM variants

### Documentation (docs/)

**Experimental Reports** (`experiments/`)
- Detailed reports for completed experiments
- Performance data and analysis
- Optimization insights and lessons learned

**Media Files** (`media/`)
- Demo videos
- Screenshots
- Sample outputs

## Build Artifacts

After building with CMake, the `build/` directory contains:

```
build/
├── core/
│   ├── libhook_analyzer.so
│   ├── libcuda_hook.so
│   └── ...
├── kernels/
│   └── liboptimized_kernels.so
├── engines/
│   └── libtensorrt_adapter.so
├── examples/
│   ├── kernel_test
│   ├── simple_demo
│   └── test_tensorrt
└── benchmarks/
    ├── benchmark_kernels
    ├── gemm_analysis
    └── gemm_compare
```

## File Naming Conventions

- **Headers:** `*.h` (C++ headers)
- **Source:** `*.cpp` (C++ source), `*.cu` (CUDA source)
- **Documentation:** `*.md` (Markdown)
- **Scripts:** `*.py` (Python), `*.sh` (Shell)
- **Configs:** `*.txt`, `*.yml`, `*.json`

## Important Files

### Configuration
- `CMakeLists.txt` - Build system configuration
- `requirements.txt` - Python dependencies

### Documentation
- `README.md` / `README_CN.md` - Main documentation (English/Chinese)
- `docs/experiments/EXPERIMENT3_RESULTS.md` - YOLOv8 TensorRT results
- `docs/experiments/EXPERIMENT1_REPORT.md` - GEMM optimization analysis

### Media
- `docs/media/hookanalyzer_demo.mp4` - Project demonstration video

## Git Ignored Files

The following are excluded from version control (see `.gitignore`):

- `build/` - CMake build artifacts
- `models/*.engine` - TensorRT engine files
- `models/*.onnx` - ONNX model files
- `logs/*.log` - Runtime logs
- `*.pyc`, `__pycache__/` - Python bytecode
- `.vscode/`, `.idea/` - IDE configurations

## Development Workflow

1. **Core Development:** Modify files in `core/`, `engines/`, `kernels/`
2. **Build:** Use CMake in `build/` directory
3. **Test:** Run executables from `build/examples/` or `build/benchmarks/`
4. **Document:** Update relevant `.md` files in `docs/`
5. **Commit:** Follow conventional commit messages

## Notes

- All C++ source files use C++17 standard
- CUDA files compiled with nvcc (CUDA 12.6)
- TensorRT version: 10.3.0
- Platform: Jetson Orin Nano (Ampere SM 8.7)

---

**Last Updated:** 2025-11-17
**Maintainer:** Geoffrey
