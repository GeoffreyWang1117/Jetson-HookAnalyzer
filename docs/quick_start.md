# HookAnalyzer Quick Start Guide

## Prerequisites

### Hardware Requirements
- **Jetson Orin Nano** (or any CUDA-capable device)
- 8GB+ RAM recommended
- 16GB+ storage

### Software Requirements
- **JetPack 5.1+** (for Jetson) or **CUDA Toolkit 11.4+**
- CMake 3.18+
- GCC 9+
- Python 3.8+
- Docker (optional, for containerized deployment)

## Installation

### Option 1: Build from Source (Recommended)

```bash
# Clone the repository
git clone https://github.com/GeoffreyWang1117/Jetson-HookAnalyzer.git
cd Jetson-HookAnalyzer

# Build
./scripts/build_local.sh release

# Run tests
cd build
ctest --output-on-failure

# Run demo
./examples/simple_demo
```

### Option 2: Docker Development

```bash
# Build and run Docker container
./scripts/run_docker_dev.sh

# Inside container
cd /workspace
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
./examples/simple_demo
```

### Option 3: Deploy to Jetson

```bash
# From your development machine
./scripts/deploy_to_jetson.sh 100.111.167.60

# SSH to Jetson
ssh geoffrey@100.111.167.60

# Run demo
cd HookAnalyzer/build
./examples/simple_demo
```

## Basic Usage

### 1. Memory Tracking

```cpp
#include "cuda_hook/cuda_hook.h"

auto& hook_manager = CudaHookManager::getInstance();
hook_manager.initialize();

// Allocate GPU memory
float* d_data;
cudaMalloc(&d_data, 1024 * sizeof(float));

// Get memory statistics
auto stats = hook_manager.getMemoryStats();
std::cout << "Current allocated: " << stats.current_allocated << " bytes" << std::endl;
std::cout << "Peak allocated: " << stats.peak_allocated << " bytes" << std::endl;

cudaFree(d_data);
```

### 2. Inference Scheduling

```cpp
#include "scheduler/scheduler.h"

SchedulerConfig config;
config.num_worker_threads = 4;
config.num_cuda_streams = 4;

InferenceScheduler scheduler(config);
scheduler.start();

// Submit inference task
InferenceTask task;
task.model_id = "yolov8";
task.priority = 1;
task.callback = []() {
    // Your inference code here
};

uint64_t task_id = scheduler.submitTask(task);
scheduler.waitForTask(task_id);

auto stats = scheduler.getStats();
std::cout << "Throughput: " << stats.throughput_tasks_per_sec << " tasks/sec" << std::endl;
```

### 3. Performance Profiling

```cpp
#include "profiler/profiler.h"

Profiler profiler;
profiler.initialize();

{
    PROFILE_SCOPE(profiler, "my_kernel");
    // Your CUDA kernel launch
    myKernel<<<grid, block>>>();
    cudaDeviceSynchronize();
}

profiler.exportChromeTrace("trace.json");
```

### 4. Custom CUDA Kernels

```cpp
#include "kernels/optimized/kernels.h"

const int N = 1024;
float *d_a, *d_b, *d_c;

// Allocate and initialize...

// Element-wise addition
kernels::addFloat(d_a, d_b, d_c, N);

// Matrix multiplication
int M = 512, N = 512, K = 512;
kernels::gemmFloatOptimized(d_A, d_B, d_C, M, N, K);
```

## REST API Usage

Start the API server:

```bash
python3 api/server/main.py
```

Submit inference request:

```bash
curl -X POST http://localhost:8000/inference \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "yolov8",
    "input_data": [1.0, 2.0, 3.0],
    "priority": 1
  }'
```

Get system stats:

```bash
curl http://localhost:8000/system/stats
```

View metrics:

```bash
curl http://localhost:8000/metrics
```

## Monitoring with Grafana

```bash
# Start monitoring stack
docker-compose -f docker/docker-compose.yml up -d prometheus grafana

# Access Grafana at http://localhost:3000
# Default credentials: admin/admin
```

## Troubleshooting

### CUDA not found
```bash
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

### TensorRT not found
```bash
# On Jetson
sudo apt-get install tensorrt

# Set environment
export LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu:$LD_LIBRARY_PATH
```

### Permission denied for GPU
```bash
sudo usermod -aG video $USER
# Log out and back in
```

## Next Steps

- [Project Structure](../PROJECT_STRUCTURE.md) - Complete project organization
- [Experiment 3 Results](experiments/EXPERIMENT3_RESULTS.md) - YOLOv8 TensorRT integration (114.67 FPS)
- [Experiment 1 Report](experiments/EXPERIMENT1_REPORT.md) - GEMM optimization analysis
- [Improvement Roadmap](../IMPROVEMENT_ROADMAP.md) - Future development plans
- [Demo Video](media/hookanalyzer_demo.mp4) - Project demonstration

## Support

- GitHub Issues: https://github.com/GeoffreyWang1117/Jetson-HookAnalyzer/issues
- GitHub Repository: https://github.com/GeoffreyWang1117/Jetson-HookAnalyzer
