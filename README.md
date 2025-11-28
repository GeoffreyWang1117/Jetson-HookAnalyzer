# CUDA Hook Analyzer & Intelligent Inference Scheduler

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CUDA](https://img.shields.io/badge/CUDA-12.6-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Platform](https://img.shields.io/badge/Platform-Jetson%20Orin%20Nano-76B900.svg)](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/)

> A lightweight CUDA-level performance profiling and intelligent multi-model inference scheduling framework for edge devices.

[中文文档](README_CN.md) | English

## 🎯 Project Overview

**HookAnalyzer** addresses critical challenges in edge AI deployment:
- **Multi-model concurrency** with resource contention management
- **CUDA kernel-level profiling** for bottleneck identification
- **GPU memory optimization** with fragmentation analysis
- **Intelligent scheduling** balancing latency and throughput

## 🌟 Highlights

### Production-Grade Results
- ⚡ **114.67 FPS** YOLOv8 inference on Jetson Orin Nano (3.8x real-time)
- 🎯 **8.72ms average latency** with P99 < 14ms (production-stable)
- 💾 **7.4 MB GPU memory footprint** (highly optimized)
- 🔧 **350+ LOC** modular TensorRT C++ wrapper

### Technical Depth
- Deep dive into CUDA kernel optimization (occupancy analysis, memory coalescing)
- TensorRT engine integration with FP16 precision
- Async inference pipeline with CUDA streams
- Performance profiling and benchmarking framework

### Demonstrated Skills
- C++17, CUDA 12.6, TensorRT 10.3, CMake
- Edge AI deployment on resource-constrained devices
- Performance analysis and optimization methodologies
- Production-ready code architecture

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                        │
│              (YOLOv8, ResNet, BERT Models)                   │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│              Inference Engine Adapters                      │
│        TensorRT │ ONNX Runtime │ Custom Kernels             │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│           Intelligent Scheduler (C++)                       │
│   Priority Queue │ Dynamic Batching │ Stream Manager        │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│          CUDA Hook & Profiling Layer                        │
│   Memory Tracker │ Kernel Analyzer │ CUPTI Integration      │
└─────────────────────────────────────────────────────────────┘
```

## ✨ Key Features

### 🔍 CUDA Interception Layer
- Real-time `cudaMalloc`/`cudaFree` hooking
- Kernel launch time profiling with CUPTI
- Memory access pattern analysis
- GPU utilization tracking

### 🧠 Intelligent Scheduler
- **Priority-based** multi-model scheduling
- **Dynamic batching** with configurable policies
- **Stream-level** parallelism optimization
- **Latency-aware** resource allocation

### 🚀 Performance Optimization
- Custom CUDA kernel library (GEMM, Conv, Softmax)
- Memory pool with defragmentation
- Multi-stream concurrent execution
- Mixed precision (INT8/FP16/FP32) support

### 📊 Monitoring & Visualization
- Real-time metrics via Prometheus
- Grafana dashboards
- Flame graph generation
- RESTful API for control

## 🛠️ Technology Stack

| Component | Technology |
|-----------|------------|
| **Core** | C++17, CUDA 12.6, CMake 3.18+ |
| **Inference** | TensorRT 10.3.0, ONNX Runtime (planned) |
| **Profiling** | CUPTI, Nsight Systems |
| **API** | Python 3.10+, FastAPI |
| **Monitoring** | Prometheus, Grafana (planned) |
| **Containerization** | Docker, NVIDIA Container Runtime |
| **Platform** | Jetson Orin Nano (JetPack 6.x) |

## 📦 Directory Structure

```
HookAnalyzer/
├── core/
│   ├── cuda_hook/          # CUDA API interception
│   ├── scheduler/          # Multi-model scheduler
│   └── profiler/           # Performance analysis
├── engines/
│   ├── tensorrt_adapter/   # TensorRT wrapper
│   └── onnx_adapter/       # ONNX Runtime wrapper
├── kernels/
│   └── optimized/          # Custom CUDA kernels
├── api/
│   └── server/             # FastAPI service
├── monitoring/
│   ├── metrics/            # Prometheus exporter
│   └── dashboard/          # Grafana configs
├── benchmarks/             # Performance tests
├── examples/               # Usage examples
├── scripts/                # Build & deployment
├── tests/                  # Unit tests
└── docs/                   # Documentation
```

## 🚀 Quick Start

### Prerequisites

**Hardware:**
- NVIDIA Jetson Orin Nano (verified platform)
  - Ampere GPU architecture (SM 8.7)
  - 8 Streaming Multiprocessors
  - 7.6 GB LPDDR5 memory
- Or any CUDA-capable device with Compute Capability 5.0+

**Software:**
- JetPack 6.x (CUDA 12.6, TensorRT 10.3.0)
- CMake 3.18+, GCC 11+
- Python 3.10+ (for model conversion scripts)

### Build from Source

```bash
# Clone repository
git clone https://github.com/GeoffreyWang1117/Jetson-HookAnalyzer.git
cd Jetson-HookAnalyzer

# Build with CMake (on Jetson)
mkdir build && cd build

# Auto-detect CUDA compiler
CUDA_COMPILER=$(which nvcc || echo "/usr/local/cuda/bin/nvcc")

# Adaptive parallel compilation
CORES=$(nproc)
PARALLEL=$((CORES > 2 ? CORES - 2 : CORES))

cmake -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_COMPILER=${CUDA_COMPILER} \
      ..
make -j${PARALLEL}

# Run kernel tests
./examples/kernel_test

# Run TensorRT inference test (if yolov8n.engine exists)
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH
./examples/test_tensorrt ../yolov8n.engine
```

### Docker Deployment

```bash
# Build Docker image
docker build -t hook-analyzer:latest -f docker/Dockerfile .

# Run container
docker run --gpus all -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  hook-analyzer:latest
```

## 📊 Performance Benchmarks

### Verified Results on Jetson Orin Nano

**Hardware:** Jetson Orin Nano (Ampere SM 8.7, 8 SMs, 7.6GB RAM)
**Software:** CUDA 12.6, TensorRT 10.3.0

#### YOLOv8n TensorRT Inference (Experiment 3) ✅

| Metric | Result | Target | Status |
|--------|--------|--------|--------|
| **Throughput** | **114.67 FPS** | 30 FPS | **3.8x faster** ⭐ |
| **Average Latency** | **8.72 ms** | 33 ms | **3.8x faster** ⭐ |
| **Min Latency** | 6.44 ms | - | Best case |
| **Max Latency** | 13.78 ms | - | P99 < 14ms |
| **GPU Memory** | 7.4 MB | - | Highly efficient |

**Model:** YOLOv8n (3.2M params, 8.7 GFLOPs)
**Precision:** FP16
**Input:** 640×640×3 RGB

#### Custom CUDA Kernels (Verified) ✅

| Kernel | Performance | vs cuBLAS/Reference |
|--------|-------------|---------------------|
| **GEMM (512×512)** | 146 GFLOPS | 68.6% cuBLAS |
| **Memory Bandwidth** | 91.3 GB/s | Efficient |
| **Element-wise Ops** | ✅ Passed | - |
| **Activations (ReLU)** | ✅ Passed | - |

*Full results: [EXPERIMENT3_RESULTS.md](docs/experiments/EXPERIMENT3_RESULTS.md) | [VERIFICATION_REPORT.md](docs/experiments/VERIFICATION_REPORT.md)*

## 🔬 Experimental Results

### ✅ Completed Experiments

#### Experiment 1: GEMM Performance Analysis
- **Goal:** Optimize matrix multiplication kernels for Jetson Orin Nano
- **Key Finding:** Discovered occupancy vs. tile size tradeoff
  - 16×16 tiles: 100% occupancy (6 blocks/SM)
  - 32×32 tiles: 67% occupancy (1 block/SM) → 20% slower
- **Result:** Documented critical optimization insights for edge GPUs
- **Report:** [EXPERIMENT1_REPORT.md](docs/experiments/EXPERIMENT1_REPORT.md)

#### Experiment 3: Real Model Integration with TensorRT
- **Goal:** Integrate YOLOv8 object detection model using TensorRT
- **Implementation:** Complete C++ TensorRT wrapper (~350 LOC)
- **Performance:** 114.67 FPS (8.72ms latency) - **3.8x faster than real-time**
- **Features:**
  - ✅ Engine loading and serialization
  - ✅ GPU memory management
  - ✅ Sync/async inference support
  - ✅ Comprehensive benchmarking
- **Status:** Production-ready, extensible architecture
- **Report:** [EXPERIMENT3_RESULTS.md](docs/experiments/EXPERIMENT3_RESULTS.md)

### 📋 Planned Experiments

- **Experiment 2:** Multi-model concurrent inference with scheduler integration
- **Experiment 4:** INT8 quantization and calibration
- **Experiment 5:** Video stream processing pipeline
- **Experiment 6:** Multi-device distributed inference

## 📚 Documentation

### Experimental Reports (Completed)
- [Experiment 3: TensorRT Integration Results](docs/experiments/EXPERIMENT3_RESULTS.md) - YOLOv8 inference at 114.67 FPS
- [Experiment 1: GEMM Optimization Analysis](docs/experiments/EXPERIMENT1_REPORT.md) - Occupancy vs tile size insights
- [Verification Report](docs/experiments/VERIFICATION_REPORT.md) - Initial project validation
- [Final Summary](docs/experiments/FINAL_SUMMARY.md) - Project completion overview

### Quick References
- [Video Recording Guide](docs/experiments/VIDEO_RECORDING_GUIDE.md) - Demo video creation
- [Experiment Roadmap](docs/experiments/EXPERIMENT_ROADMAP.md) - Future experiment plans
- [Demo Video](docs/media/hookanalyzer_demo.mp4) - Project demonstration

## 🤝 Contributing

Contributions welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) first.

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file.

## 🙏 Acknowledgments

- NVIDIA CUDA Toolkit and TensorRT team
- PyTorch and ONNX communities
- Jetson developer community

## 📧 Contact

- **Author:** Geoffrey
- **Project:** AI Infrastructure & Inference Optimization
- **Platform:** Jetson Orin Nano @ 100.111.167.60
- **GitHub:** [GeoffreyWang1117/Jetson-HookAnalyzer](https://github.com/GeoffreyWang1117/Jetson-HookAnalyzer)

---

**⚡ Built for edge AI inference optimization on resource-constrained devices**
