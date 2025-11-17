#pragma once

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime.h>
#include <string>
#include <vector>
#include <memory>
#include <fstream>
#include <iostream>

namespace hook_analyzer {
namespace tensorrt {

class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << msg << std::endl;
        }
    }
};

class TensorRTEngine {
public:
    TensorRTEngine(const std::string& engine_path);
    ~TensorRTEngine();

    // Inference
    bool infer(const std::vector<void*>& inputs, std::vector<void*>& outputs);

    // Async inference
    bool inferAsync(const std::vector<void*>& inputs,
                    std::vector<void*>& outputs,
                    cudaStream_t stream);

    // Get input/output info
    int getNumInputs() const;
    int getNumOutputs() const;
    std::vector<int> getInputShape(int index) const;
    std::vector<int> getOutputShape(int index) const;

    // Performance
    struct InferenceStats {
        float avg_latency_ms;
        float min_latency_ms;
        float max_latency_ms;
        int iterations;
    };

    InferenceStats benchmark(int warmup_iterations = 10,
                            int benchmark_iterations = 100);

private:
    bool loadEngine(const std::string& engine_path);
    void allocateBuffers();

    Logger logger_;
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;

    std::vector<void*> device_buffers_;
    std::vector<size_t> buffer_sizes_;

    int num_bindings_;
    int batch_size_;
};

} // namespace tensorrt
} // namespace hook_analyzer
