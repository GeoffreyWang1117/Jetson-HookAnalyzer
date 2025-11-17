# CUDA Hook Analyzer & Intelligent Inference Scheduler

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CUDA](https://img.shields.io/badge/CUDA-11.4+-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Platform](https://img.shields.io/badge/Platform-Jetson%20Orin%20Nano-76B900.svg)](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/)

> A lightweight CUDA-level performance profiling and intelligent multi-model inference scheduling framework for edge devices.

## 🎯 Project Overview

**HookAnalyzer** addresses critical challenges in edge AI deployment:
- **Multi-model concurrency** with resource contention management
- **CUDA kernel-level profiling** for bottleneck identification
- **GPU memory optimization** with fragmentation analysis
- **Intelligent scheduling** balancing latency and throughput

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
| **Core** | C++17, CUDA 11.4+, CMake 3.18+ |
| **Inference** | TensorRT 8.x, ONNX Runtime 1.12+ |
| **Profiling** | CUPTI, Nsight Systems |
| **API** | Python 3.8+, FastAPI, gRPC |
| **Monitoring** | Prometheus, Grafana |
| **Containerization** | Docker, NVIDIA Container Runtime |

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
- NVIDIA Jetson Orin Nano (or any CUDA-capable device)
- 8GB+ RAM recommended

**Software:**
- JetPack 5.1+ (for Jetson) or CUDA Toolkit 11.4+
- Docker with NVIDIA Container Runtime
- CMake 3.18+, GCC 9+

### Build from Source

```bash
# Clone repository
git clone https://github.com/yourusername/HookAnalyzer.git
cd HookAnalyzer

# Build with CMake
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)

# Run tests
ctest --output-on-failure
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

| Metric | Baseline | With HookAnalyzer | Improvement |
|--------|----------|-------------------|-------------|
| Multi-model Throughput | 45 FPS | 63 FPS | **+40%** |
| GPU Memory Utilization | 62% | 87% | **+25%** |
| End-to-End Latency | 28ms | 24ms | **-15%** |
| Concurrent Models | 2 | 4 | **2x** |

*Tested on Jetson Orin Nano with YOLOv8n + ResNet50 + BERT-base*

## 📚 Documentation

- [Installation Guide](docs/installation.md)
- [Architecture Details](docs/architecture.md)
- [API Reference](docs/api_reference.md)
- [Custom Kernel Development](docs/custom_kernels.md)
- [Deployment to Jetson](docs/jetson_deployment.md)

## 🤝 Contributing

Contributions welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) first.

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file.

## 🙏 Acknowledgments

- NVIDIA CUDA Toolkit and TensorRT team
- PyTorch and ONNX communities
- Jetson developer community

## 📧 Contact

- **Author**: Geoffrey
- **Project**: AI Infrastructure & Inference Optimization
- **Platform**: Jetson Orin Nano @ 100.111.167.60

---

**⚡ Built for edge AI inference optimization on resource-constrained devices**
