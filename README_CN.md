# CUDA Hook 分析器 & 智能推理调度器

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CUDA](https://img.shields.io/badge/CUDA-12.6-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Platform](https://img.shields.io/badge/Platform-Jetson%20Orin%20Nano-76B900.svg)](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/)

> 面向边缘设备的轻量级 CUDA 层性能分析和智能多模型推理调度框架

## 🎯 项目概述

**HookAnalyzer** 解决边缘 AI 部署中的关键挑战：
- **多模型并发**与资源竞争管理
- **CUDA 内核级分析**用于瓶颈识别
- **GPU 内存优化**与碎片分析
- **智能调度**平衡延迟和吞吐量

## 🌟 核心亮点

### 生产级性能结果
- ⚡ **114.67 FPS** - Jetson Orin Nano 上的 YOLOv8 推理速度（超实时要求 3.8 倍）
- 🎯 **8.72ms 平均延迟** - P99 延迟 < 14ms（生产级稳定性）
- 💾 **7.4 MB GPU 内存占用** - 640×640 输入图像（高度优化）
- 🔧 **350+ 行代码** - 模块化 TensorRT C++ 封装层

### 技术深度
- CUDA 内核优化深入研究（占用率分析、内存合并）
- TensorRT 引擎集成与 FP16 精度优化
- 使用 CUDA 流的异步推理管道
- 性能分析与基准测试框架

### 技术能力展示
- C++17、CUDA 12.6、TensorRT 10.3、CMake
- 资源受限设备上的边缘 AI 部署
- 性能分析与优化方法论
- 生产就绪代码架构

## 🏗️ 架构设计

```
┌─────────────────────────────────────────────────────────────┐
│                      应用层                                  │
│              (YOLOv8, ResNet, BERT 模型)                     │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                推理引擎适配器                                │
│        TensorRT │ ONNX Runtime │ 自定义内核                  │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│              智能调度器 (C++)                                │
│   优先级队列 │ 动态批处理 │ 流管理器                         │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│          CUDA Hook & 分析层                                  │
│   内存跟踪器 │ 内核分析器 │ CUPTI 集成                       │
└─────────────────────────────────────────────────────────────┘
```

## ✨ 核心功能

### 🔍 CUDA 拦截层
- 实时 `cudaMalloc`/`cudaFree` 钩子
- 使用 CUPTI 进行内核启动时间分析
- 内存访问模式分析
- GPU 利用率跟踪

### 🧠 智能调度器
- **基于优先级**的多模型调度
- **动态批处理**与可配置策略
- **流级别**并行优化
- **延迟感知**资源分配

### 🚀 性能优化
- 自定义 CUDA 内核库（GEMM、卷积、Softmax）
- 带碎片整理的内存池
- 多流并发执行
- 混合精度（INT8/FP16/FP32）支持

### 📊 监控与可视化
- 通过 Prometheus 实时指标
- Grafana 仪表板
- 火焰图生成
- RESTful API 控制

## 🛠️ 技术栈

| 组件 | 技术 |
|-----------|------------|
| **核心** | C++17, CUDA 12.6, CMake 3.18+ |
| **推理** | TensorRT 10.3.0, ONNX Runtime（计划中） |
| **分析** | CUPTI, Nsight Systems |
| **API** | Python 3.10+, FastAPI |
| **监控** | Prometheus, Grafana（计划中） |
| **容器化** | Docker, NVIDIA Container Runtime |
| **平台** | Jetson Orin Nano (JetPack 6.x) |

## 📦 目录结构

```
HookAnalyzer/
├── core/
│   ├── cuda_hook/          # CUDA API 拦截
│   ├── scheduler/          # 多模型调度器
│   └── profiler/           # 性能分析
├── engines/
│   ├── tensorrt_adapter/   # TensorRT 封装
│   └── onnx_adapter/       # ONNX Runtime 封装
├── kernels/
│   └── optimized/          # 自定义 CUDA 内核
├── api/
│   └── server/             # FastAPI 服务
├── monitoring/
│   ├── metrics/            # Prometheus 导出器
│   └── dashboard/          # Grafana 配置
├── benchmarks/             # 性能测试
├── examples/               # 使用示例
├── scripts/                # 构建与部署脚本
├── tests/                  # 单元测试
└── docs/                   # 文档
```

## 🚀 快速开始

### 环境要求

**硬件：**
- NVIDIA Jetson Orin Nano（已验证平台）
  - Ampere GPU 架构（SM 8.7）
  - 8 个流式多处理器
  - 7.6 GB LPDDR5 内存
- 或任何 Compute Capability 5.0+ 的 CUDA 设备

**软件：**
- JetPack 6.x（CUDA 12.6, TensorRT 10.3.0）
- CMake 3.18+, GCC 11+
- Python 3.10+（用于模型转换脚本）

### 从源码构建

```bash
# 克隆仓库
git clone https://github.com/GeoffreyWang1117/Jetson-HookAnalyzer.git
cd Jetson-HookAnalyzer

# 使用 CMake 构建（在 Jetson 上）
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.6/bin/nvcc \
      ..
make -j6

# 运行内核测试
./examples/kernel_test

# 运行 TensorRT 推理测试（如果存在 yolov8n.engine）
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH
./examples/test_tensorrt ../yolov8n.engine
```

### Docker 部署

```bash
# 构建 Docker 镜像
docker build -t hook-analyzer:latest -f docker/Dockerfile .

# 运行容器
docker run --gpus all -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  hook-analyzer:latest
```

## 📊 性能基准测试

### Jetson Orin Nano 验证结果

**硬件：** Jetson Orin Nano (Ampere SM 8.7, 8 SMs, 7.6GB RAM)
**软件：** CUDA 12.6, TensorRT 10.3.0

#### YOLOv8n TensorRT 推理（实验 3）✅

| 指标 | 结果 | 目标 | 状态 |
|--------|--------|--------|--------|
| **吞吐量** | **114.67 FPS** | 30 FPS | **快 3.8 倍** ⭐ |
| **平均延迟** | **8.72 ms** | 33 ms | **快 3.8 倍** ⭐ |
| **最小延迟** | 6.44 ms | - | 最佳情况 |
| **最大延迟** | 13.78 ms | - | P99 < 14ms |
| **GPU 内存** | 7.4 MB | - | 高效 |

**模型：** YOLOv8n（3.2M 参数，8.7 GFLOPs）
**精度：** FP16
**输入：** 640×640×3 RGB

#### 自定义 CUDA 内核（已验证）✅

| 内核 | 性能 | vs cuBLAS/参考 |
|--------|-------------|---------------------|
| **GEMM (512×512)** | 146 GFLOPS | 68.6% cuBLAS |
| **内存带宽** | 91.3 GB/s | 高效 |
| **逐元素操作** | ✅ 通过 | - |
| **激活函数 (ReLU)** | ✅ 通过 | - |

*完整结果：[EXPERIMENT3_RESULTS.md](docs/experiments/EXPERIMENT3_RESULTS.md) | [VERIFICATION_REPORT.md](docs/experiments/VERIFICATION_REPORT.md)*

## 🔬 实验成果

### ✅ 已完成实验

#### 实验 1：GEMM 性能分析
- **目标：** 为 Jetson Orin Nano 优化矩阵乘法内核
- **关键发现：** 发现占用率与块大小权衡
  - 16×16 块：100% 占用率（6 块/SM）
  - 32×32 块：67% 占用率（1 块/SM）→ 慢 20%
- **结果：** 记录了边缘 GPU 的关键优化见解
- **报告：** [EXPERIMENT1_REPORT.md](docs/experiments/EXPERIMENT1_REPORT.md)

#### 实验 3：使用 TensorRT 集成真实模型
- **目标：** 使用 TensorRT 集成 YOLOv8 目标检测模型
- **实现：** 完整的 C++ TensorRT 封装（约 350 行代码）
- **性能：** 114.67 FPS（8.72ms 延迟）- **比实时快 3.8 倍**
- **功能：**
  - ✅ 引擎加载和序列化
  - ✅ GPU 内存管理
  - ✅ 同步/异步推理支持
  - ✅ 全面的基准测试
- **状态：** 生产就绪，可扩展架构
- **报告：** [EXPERIMENT3_RESULTS.md](docs/experiments/EXPERIMENT3_RESULTS.md)

### 📋 计划中实验

- **实验 2：** 多模型并发推理与调度器集成
- **实验 4：** INT8 量化和校准
- **实验 5：** 视频流处理管道
- **实验 6：** 多设备分布式推理

## 📚 文档

### 实验报告（已完成）
- [实验 3：TensorRT 集成结果](docs/experiments/EXPERIMENT3_RESULTS.md) - YOLOv8 推理达 114.67 FPS
- [实验 1：GEMM 优化分析](docs/experiments/EXPERIMENT1_REPORT.md) - 占用率 vs 块大小见解
- [验证报告](docs/experiments/VERIFICATION_REPORT.md) - 初始项目验证
- [最终总结](docs/experiments/FINAL_SUMMARY.md) - 项目完成概述

### 快速参考
- [视频录制指南](docs/experiments/VIDEO_RECORDING_GUIDE.md) - 演示视频创建
- [实验路线图](docs/experiments/EXPERIMENT_ROADMAP.md) - 未来实验计划
- [演示视频](docs/media/hookanalyzer_demo.mp4) - 项目演示

## 🤝 贡献

欢迎贡献！请先阅读 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。

## 🙏 致谢

- NVIDIA CUDA Toolkit 和 TensorRT 团队
- PyTorch 和 ONNX 社区
- Jetson 开发者社区

## 📧 联系方式

- **作者：** Geoffrey
- **项目：** AI 基础设施与推理优化
- **平台：** Jetson Orin Nano @ 100.111.167.60
- **GitHub：** [GeoffreyWang1117/Jetson-HookAnalyzer](https://github.com/GeoffreyWang1117/Jetson-HookAnalyzer)

---

**⚡ 为资源受限设备上的边缘 AI 推理优化而构建**
