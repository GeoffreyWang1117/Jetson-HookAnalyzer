# HookAnalyzer 改进路线图

基于项目全面分析，按优先级排列的改进建议。

## 🔴 高优先级（立即修复）

### 1. 添加 CUDA 错误处理

**受影响文件：**
- `engines/tensorrt_adapter/tensorrt_engine.cpp:107`
- `core/cuda_hook/cuda_hook.cpp:57,237,240`
- `core/scheduler/scheduler.cpp:27-33`

**问题：**
```cpp
// ❌ 错误示例
cudaMalloc(&device_buffers_[i], bytes);  // 无错误检查

// ✅ 正确做法
cudaError_t err = cudaMalloc(&device_buffers_[i], bytes);
if (err != cudaSuccess) {
    std::cerr << "CUDA allocation failed: " << cudaGetErrorString(err) << std::endl;
    throw std::runtime_error("GPU memory allocation failed");
}
```

**预估工作量：** 2-4 小时
**收益：** 提高生产稳定性，易于调试

---

### 2. 实现 Python-C++ 绑定

**受影响文件：**
- `api/server/main.py` (4 处 TODO)

**当前状态：**
```python
# TODO: Integrate with actual C++ scheduler
output_data = [x * 2.0 for x in request.input_data[:10]]  # 假数据
```

**实现方案：**

**选项 A：pybind11（推荐）**
```cpp
// bindings/python_bindings.cpp
#include <pybind11/pybind11.h>
#include "tensorrt_engine.h"

namespace py = pybind11;

PYBIND11_MODULE(hookanalyzer, m) {
    py::class_<TensorRTEngine>(m, "TensorRTEngine")
        .def(py::init<const std::string&>())
        .def("infer", &TensorRTEngine::infer)
        .def("benchmark", &TensorRTEngine::benchmark);
}
```

**选项 B：ctypes（快速方案）**
```python
import ctypes
lib = ctypes.CDLL('./build/libtensorrt_adapter.so')
```

**预估工作量：** 1-2 天
**收益：** API 可实际使用，不再是演示

---

### 3. 集成测试框架（Google Test）

**新增文件：**
- `tests/test_tensorrt.cpp`
- `tests/test_scheduler.cpp`
- `tests/test_profiler.cpp`

**CMakeLists.txt 更新：**
```cmake
# tests/CMakeLists.txt
include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG release-1.12.1
)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(test_tensorrt test_tensorrt.cpp)
target_link_libraries(test_tensorrt GTest::gtest_main tensorrt_adapter)
gtest_discover_tests(test_tensorrt)
```

**测试示例：**
```cpp
#include <gtest/gtest.h>
#include "tensorrt_engine.h"

TEST(TensorRTEngineTest, LoadInvalidEngine) {
    EXPECT_THROW(TensorRTEngine("/nonexistent.engine"), std::runtime_error);
}

TEST(TensorRTEngineTest, InferencePerformance) {
    TensorRTEngine engine("yolov8n.engine");
    auto stats = engine.benchmark(10, 100);
    EXPECT_GT(stats.avg_latency_ms, 0.0f);
    EXPECT_LT(stats.avg_latency_ms, 20.0f);  // 性能回归检测
}
```

**预估工作量：** 1 周
**收益：** 防止回归，提高代码质量

---

### 4. 创建缺失的文档

**需要创建的文档：**

#### 4.1 `docs/api_reference.md`
```markdown
# API Reference

## REST API Endpoints

### POST /api/inference
**Description:** Submit inference task

**Request:**
```json
{
  "model": "yolov8",
  "input_data": [1.0, 2.0, ...],
  "priority": "HIGH"
}
```

**Response:**
```json
{
  "task_id": "abc123",
  "status": "queued",
  "estimated_latency_ms": 8.72
}
```
```

#### 4.2 `docs/architecture.md`
- 系统架构图详解
- 各模块职责
- 数据流图

#### 4.3 `docs/custom_kernels.md`
- CUDA 内核开发指南
- 性能优化技巧
- Tile 大小选择策略

#### 4.4 `docs/troubleshooting.md`（新增）
- 常见编译错误
- 运行时问题诊断
- 性能问题排查

**预估工作量：** 2-3 天
**收益：** 降低使用门槛，减少支持负担

---

## 🟡 中优先级（短期改进）

### 5. 实现真实的 Profiler 指标

**受影响文件：**
- `core/profiler/profiler.cpp:111-132`

**集成 NVML：**
```cpp
#include <nvml.h>

void Profiler::collectMetrics() {
    nvmlInit();

    nvmlDevice_t device;
    nvmlDeviceGetHandleByIndex(0, &device);

    // SM 利用率
    nvmlUtilization_t utilization;
    nvmlDeviceGetUtilizationRates(device, &utilization);
    metrics.sm_utilization = utilization.gpu / 100.0f;

    // 温度
    unsigned int temp;
    nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &temp);
    metrics.temperature_celsius = static_cast<float>(temp);

    // 功耗
    unsigned int power;
    nvmlDeviceGetPowerUsage(device, &power);
    metrics.power_usage_watts = power / 1000.0f;

    nvmlShutdown();
}
```

**CMakeLists.txt 更新：**
```cmake
find_library(NVML_LIBRARY nvidia-ml HINTS /usr/lib/aarch64-linux-gnu)
target_link_libraries(profiler ${NVML_LIBRARY})
```

**预估工作量：** 4-6 小时
**收益：** 真实性能监控数据

---

### 6. 实现动态批处理

**受影响文件：**
- `core/scheduler/scheduler.cpp:272-298`

**当前问题：** `tryBatchTasks()` 函数存在但从未被调用

**修复方案：**
```cpp
// scheduler.cpp:100 附近
void InferenceScheduler::workerThread(int worker_id) {
    while (running_) {
        std::vector<InferenceTask> batch;

        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            queue_cv_.wait(lock, [this] {
                return !task_queue_.empty() || !running_;
            });

            if (!running_) break;

            // 尝试批处理
            InferenceTask first = task_queue_.top();
            batch = tryBatchTasks(first);  // ✅ 实际调用

            // 从队列移除批处理的任务
            for (const auto& task : batch) {
                task_queue_.pop();
            }
        }

        // 执行批量推理
        executeBatch(batch, worker_id);
    }
}
```

**预估工作量：** 1-2 天
**收益：** 提升吞吐量 20-40%

---

### 7. 消除硬编码值

**受影响文件：**
- `CMakeLists.txt:30`
- `README.md:149-154`
- `kernels/optimized/kernels.cu:27`

**修复方案：**

**7.1 CUDA 架构**
```cmake
# CMakeLists.txt
option(CUDA_ARCHITECTURES "Target CUDA architectures" "87;75;72;61")
set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCHITECTURES})
```

**7.2 编译并行数**
```cmake
# 添加到 README.md
CORES=$(nproc)
PARALLEL=$((CORES > 2 ? CORES - 2 : CORES))
make -j${PARALLEL}
```

**7.3 Tile 大小可配置**
```cpp
// kernels.h
cudaError_t gemmFloatOptimized(
    const float* A, const float* B, float* C,
    int M, int N, int K,
    int tile_size = 16,  // 新增参数
    cudaStream_t stream = 0
);
```

**预估工作量：** 4 小时
**收益：** 提高可移植性和灵活性

---

## 🟢 低优先级（长期改进）

### 8. 实现内存池

**新增文件：**
- `core/memory_pool/memory_pool.h`
- `core/memory_pool/memory_pool.cpp`

**接口设计：**
```cpp
class GPUMemoryPool {
public:
    void* allocate(size_t bytes);
    void deallocate(void* ptr);
    void defragment();

private:
    std::map<size_t, std::vector<void*>> free_blocks_;
    std::unordered_map<void*, size_t> allocated_blocks_;
};
```

**预估工作量：** 1 周
**收益：** 减少内存碎片，提升分配性能

---

### 9. 实现 ONNX Runtime 适配器

**新增文件：**
- `engines/onnx_adapter/onnx_engine.h`
- `engines/onnx_adapter/onnx_engine.cpp`

**依赖：**
```bash
# 安装 ONNX Runtime
wget https://github.com/microsoft/onnxruntime/releases/download/v1.16.3/onnxruntime-linux-aarch64-1.16.3.tgz
tar -xzf onnxruntime-linux-aarch64-1.16.3.tgz
```

**预估工作量：** 3-5 天
**收益：** 支持更多模型格式

---

### 10. 性能优化

**10.1 使用 CUDA Events 进行精确计时**
```cpp
// tensorrt_engine.cpp:257
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

for (int i = 0; i < iterations; i++) {
    cudaEventRecord(start);
    context_->enqueueV3(stream);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    latencies.push_back(ms);
}
```

**10.2 优化统计收集**
```cpp
// scheduler.cpp:254
// 使用 circular buffer 替代 vector
#include <boost/circular_buffer.hpp>
boost::circular_buffer<double> queue_wait_times_(1000);
```

**预估工作量：** 1-2 天
**收益：** 更准确的基准测试

---

## 📊 改进优先级矩阵

| 改进项 | 优先级 | 工作量 | 收益 | 建议时间 |
|--------|--------|--------|------|----------|
| CUDA 错误处理 | 🔴 高 | 2-4h | 高 | 立即 |
| Python 绑定 | 🔴 高 | 1-2d | 极高 | 本周 |
| 测试框架 | 🔴 高 | 1w | 高 | 2 周内 |
| 创建文档 | 🔴 高 | 2-3d | 中 | 2 周内 |
| NVML 集成 | 🟡 中 | 4-6h | 中 | 1 个月内 |
| 动态批处理 | 🟡 中 | 1-2d | 高 | 1 个月内 |
| 消除硬编码 | 🟡 中 | 4h | 中 | 1 个月内 |
| 内存池 | 🟢 低 | 1w | 中 | 3 个月内 |
| ONNX 适配器 | 🟢 低 | 3-5d | 低 | 按需 |
| 性能优化 | 🟢 低 | 1-2d | 低 | 按需 |

---

## 🎯 建议的实施顺序

### 第 1 周：生产稳定性
1. ✅ 添加所有 CUDA 错误处理
2. ✅ 修复内存泄漏风险
3. ✅ 改进错误消息

### 第 2-3 周：API 可用性
1. ✅ 实现 Python 绑定（pybind11）
2. ✅ 测试 API 与 C++ 后端集成
3. ✅ 添加 API 文档

### 第 4-5 周：测试与文档
1. ✅ 集成 Google Test
2. ✅ 编写 20+ 单元测试
3. ✅ 创建缺失文档

### 第 6-8 周：功能完善
1. ✅ 实现动态批处理
2. ✅ 集成 NVML
3. ✅ 消除硬编码值

### 未来（可选）
- 内存池实现
- ONNX 适配器
- 多 GPU 支持
- 分布式推理

---

## 📝 快速行动清单（本周可完成）

**今天（2 小时）：**
- [ ] 为 `tensorrt_engine.cpp:107` 添加错误检查
- [ ] 为 `cuda_hook.cpp:57` 添加错误检查
- [ ] 更新 `quick_start.md` 移除损坏链接

**明天（4 小时）：**
- [ ] 创建 `docs/api_reference.md`
- [ ] 创建 `docs/troubleshooting.md`
- [ ] 添加 `.clang-format` 代码格式化配置

**本周末（8 小时）：**
- [ ] 集成 pybind11
- [ ] 创建 Python 绑定示例
- [ ] 测试 API 实际推理

---

## 🔧 工具和资源

**推荐工具：**
- `clang-tidy` - 静态代码分析
- `valgrind` - 内存泄漏检测
- `nsight-systems` - 性能分析
- `cppcheck` - 代码质量检查

**参考文档：**
- pybind11: https://pybind11.readthedocs.io/
- Google Test: https://google.github.io/googletest/
- NVML API: https://docs.nvidia.com/deploy/nvml-api/
- TensorRT Best Practices: https://docs.nvidia.com/deeplearning/tensorrt/

---

## 📈 预期成果

完成所有高优先级改进后：

**代码质量：**
- ✅ 生产级错误处理
- ✅ 80%+ 测试覆盖率
- ✅ 完整文档

**功能完整性：**
- ✅ API 实际可用（非演示）
- ✅ 真实性能监控
- ✅ 动态批处理工作

**用户体验：**
- ✅ 清晰的错误消息
- ✅ 完整的使用文档
- ✅ 易于部署

**简历价值：**
- ✅ 生产级代码质量
- ✅ 完整的测试框架
- ✅ 端到端可用系统

---

**创建时间：** 2025-11-17
**维护者：** Geoffrey
**状态：** 活跃开发中
