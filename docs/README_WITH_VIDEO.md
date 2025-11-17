# HookAnalyzer - README示例（含视频）

这是一个包含视频演示的README示例，可以直接复制到你的项目README.md中。

---

# CUDA Hook Analyzer & Intelligent Inference Scheduler

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CUDA](https://img.shields.io/badge/CUDA-12.6+-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Platform](https://img.shields.io/badge/Platform-Jetson%20Orin%20Nano-76B900.svg)](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/)
[![Performance](https://img.shields.io/badge/GEMM-146%20GFLOPS-blue.svg)]()
[![Bandwidth](https://img.shields.io/badge/Bandwidth-91.3%20GB%2Fs-orange.svg)]()

> A lightweight CUDA-level performance profiling and intelligent multi-model inference scheduling framework for edge devices.

## 🎥 Demo Video

**Watch HookAnalyzer in action on Jetson Orin Nano** (3 minutes)

<!-- 替换为你的实际视频链接 -->
[![Demo Video](https://img.shields.io/badge/▶-Watch%20Demo%20on%20YouTube-red?style=for-the-badge&logo=youtube)](YOUR_YOUTUBE_LINK)

**Or try the interactive terminal recording:**

<!-- 替换为你的asciinema链接 -->
[![asciicast](https://asciinema.org/a/YOUR_CAST_ID.svg)](https://asciinema.org/a/YOUR_CAST_ID)

<details>
<summary>📸 Click to see screenshots</summary>

### GPU Detection & System Info
![GPU Info](docs/screenshots/gpu_detection.png)

### Kernel Tests - All Passing ✅
![Kernel Tests](docs/screenshots/kernel_tests.png)

### Performance Benchmarks
![Benchmarks](docs/screenshots/benchmarks.png)

</details>

---

## 🚀 Quick Start

**The fastest way to see it working:**

```bash
# Clone
git clone https://github.com/yourusername/HookAnalyzer.git
cd HookAnalyzer

# Build on Jetson
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j6

# Run demo
./examples/kernel_test

# Run benchmarks
./benchmarks/benchmark_kernels
```

---

## 📊 Performance Highlights

Benchmarked on **Jetson Orin Nano** (SM 8.7, 8 SMs, 7.6GB RAM):

| Metric | Result | vs Baseline | Status |
|--------|--------|-------------|--------|
| **GEMM (512x512)** | 146 GFLOPS | 68.6% of cuBLAS | ⭐⭐⭐⭐ |
| **Memory Bandwidth** | 91.3 GB/s | 89.5% of theoretical | ⭐⭐⭐⭐⭐ |
| **Add Operation** | 87.1 GB/s | Memory-bound | ⭐⭐⭐⭐ |
| **ReLU Activation** | 74.7 GB/s | Optimized | ⭐⭐⭐⭐ |

**All 5/5 kernel tests passing** ✅

---

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
- Shared memory tiling optimization
- Memory coalescing for bandwidth efficiency
- Mixed precision (INT8/FP16/FP32) support

### 📊 Monitoring & Visualization
- Real-time metrics via Prometheus
- CUDA event profiling
- Chrome trace export for flame graphs
- RESTful API for control

---

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

---

## 🛠️ Technology Stack

| Component | Technology |
|-----------|------------|
| **Core** | C++17, CUDA 12.6, CMake 3.18+ |
| **Inference** | TensorRT 8.x, ONNX Runtime 1.12+ |
| **Profiling** | CUPTI, Nsight Systems |
| **API** | Python 3.8+, FastAPI |
| **Deployment** | Docker, NVIDIA Container Runtime |

---

## 📚 Documentation

- [Quick Start Guide](docs/quick_start.md)
- [Architecture Details](docs/PROJECT_OVERVIEW.md)
- [Video Recording Guide](VIDEO_RECORDING_GUIDE.md) - How to create your own demo
- [Deployment to Jetson](DEPLOYMENT_GUIDE.md)
- [Verification Report](VERIFICATION_REPORT.md) - Performance test results

---

## 🧪 Reproducing the Results

**Want to verify the benchmarks yourself?**

1. **Get the hardware** - Jetson Orin Nano (or any CUDA-capable device)
2. **Clone and build** - Follow Quick Start above
3. **Run tests** - `./examples/kernel_test`
4. **Run benchmarks** - `./benchmarks/benchmark_kernels`
5. **Compare results** - Check against VERIFICATION_REPORT.md

All code is open source - feel free to audit and improve!

---

## 🤝 Contributing

Contributions welcome! Areas we're looking to improve:

- [ ] Large matrix GEMM optimization (currently 15% of cuBLAS)
- [ ] TensorCore support for Ampere GPUs
- [ ] TensorRT engine integration
- [ ] Multi-GPU / distributed inference
- [ ] More activation functions

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file.

---

## 🙏 Acknowledgments

- NVIDIA CUDA Toolkit and TensorRT team
- Jetson developer community
- All contributors and testers

---

## 📧 Contact

**Author**: Geoffrey
**Project**: AI Infrastructure & Inference Optimization
**Platform**: Jetson Orin Nano @ 100.111.167.60

Questions? Open an issue or reach out!

---

<div align="center">

**⚡ Built for edge AI inference optimization on resource-constrained devices**

[⭐ Star this repo](https://github.com/yourusername/HookAnalyzer) if you find it useful!

</div>
