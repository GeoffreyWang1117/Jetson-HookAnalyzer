#include "tensorrt_engine.h"
#include <iostream>
#include <chrono>
#include <algorithm>

namespace hook_analyzer {
namespace tensorrt {

TensorRTEngine::TensorRTEngine(const std::string& engine_path)
    : num_bindings_(0), batch_size_(1) {

    if (!loadEngine(engine_path)) {
        throw std::runtime_error("Failed to load TensorRT engine");
    }

    allocateBuffers();
}

TensorRTEngine::~TensorRTEngine() {
    // Free device buffers
    for (auto buffer : device_buffers_) {
        if (buffer) {
            cudaFree(buffer);
        }
    }
    device_buffers_.clear();
}

bool TensorRTEngine::loadEngine(const std::string& engine_path) {
    std::cout << "Loading TensorRT engine from: " << engine_path << std::endl;

    // Read engine file
    std::ifstream file(engine_path, std::ios::binary);
    if (!file.good()) {
        std::cerr << "Error: Cannot open engine file: " << engine_path << std::endl;
        return false;
    }

    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> engine_data(size);
    file.read(engine_data.data(), size);
    file.close();

    // Create runtime and deserialize engine
    runtime_.reset(nvinfer1::createInferRuntime(logger_));
    if (!runtime_) {
        std::cerr << "Error: Failed to create TensorRT runtime" << std::endl;
        return false;
    }

    engine_.reset(runtime_->deserializeCudaEngine(engine_data.data(), size));
    if (!engine_) {
        std::cerr << "Error: Failed to deserialize CUDA engine" << std::endl;
        return false;
    }

    // Create execution context
    context_.reset(engine_->createExecutionContext());
    if (!context_) {
        std::cerr << "Error: Failed to create execution context" << std::endl;
        return false;
    }

    num_bindings_ = engine_->getNbIOTensors();

    std::cout << "✓ Engine loaded successfully" << std::endl;
    std::cout << "  Number of bindings: " << num_bindings_ << std::endl;

    // Print input/output info
    for (int i = 0; i < num_bindings_; i++) {
        auto name = engine_->getIOTensorName(i);
        auto dims = engine_->getTensorShape(name);
        auto dtype = engine_->getTensorDataType(name);
        auto mode = engine_->getTensorIOMode(name);

        std::cout << "  " << (mode == nvinfer1::TensorIOMode::kINPUT ? "Input " : "Output")
                  << i << ": " << name << " [";
        for (int j = 0; j < dims.nbDims; j++) {
            std::cout << dims.d[j];
            if (j < dims.nbDims - 1) std::cout << "x";
        }
        std::cout << "]" << std::endl;
    }

    return true;
}

void TensorRTEngine::allocateBuffers() {
    device_buffers_.resize(num_bindings_);
    buffer_sizes_.resize(num_bindings_);

    for (int i = 0; i < num_bindings_; i++) {
        auto name = engine_->getIOTensorName(i);
        auto dims = context_->getTensorShape(name);

        // Calculate buffer size
        size_t size = 1;
        for (int j = 0; j < dims.nbDims; j++) {
            size *= dims.d[j];
        }

        // Allocate device memory
        size_t bytes = size * sizeof(float);  // Assuming FP32
        cudaMalloc(&device_buffers_[i], bytes);
        buffer_sizes_[i] = size;

        std::cout << "  Allocated buffer " << i << ": "
                  << bytes / (1024.0 * 1024.0) << " MB" << std::endl;
    }
}

bool TensorRTEngine::infer(const std::vector<void*>& inputs,
                           std::vector<void*>& outputs) {
    return inferAsync(inputs, outputs, 0);
}

bool TensorRTEngine::inferAsync(const std::vector<void*>& inputs,
                               std::vector<void*>& outputs,
                               cudaStream_t stream) {
    // Copy input data to device
    int input_idx = 0;
    for (int i = 0; i < num_bindings_; i++) {
        auto name = engine_->getIOTensorName(i);
        if (engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
            if (input_idx >= inputs.size()) {
                std::cerr << "Error: Not enough input buffers provided" << std::endl;
                return false;
            }

            size_t bytes = buffer_sizes_[i] * sizeof(float);
            cudaMemcpyAsync(device_buffers_[i], inputs[input_idx],
                           bytes, cudaMemcpyHostToDevice, stream);

            context_->setTensorAddress(name, device_buffers_[i]);
            input_idx++;
        } else {
            context_->setTensorAddress(name, device_buffers_[i]);
        }
    }

    // Execute inference
    bool status = context_->enqueueV3(stream);
    if (!status) {
        std::cerr << "Error: Inference execution failed" << std::endl;
        return false;
    }

    // Copy output data from device
    int output_idx = 0;
    for (int i = 0; i < num_bindings_; i++) {
        auto name = engine_->getIOTensorName(i);
        if (engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kOUTPUT) {
            if (output_idx >= outputs.size()) {
                std::cerr << "Error: Not enough output buffers provided" << std::endl;
                return false;
            }

            size_t bytes = buffer_sizes_[i] * sizeof(float);
            cudaMemcpyAsync(outputs[output_idx], device_buffers_[i],
                           bytes, cudaMemcpyDeviceToHost, stream);
            output_idx++;
        }
    }

    // Synchronize stream
    cudaStreamSynchronize(stream);

    return true;
}

int TensorRTEngine::getNumInputs() const {
    int count = 0;
    for (int i = 0; i < num_bindings_; i++) {
        auto name = engine_->getIOTensorName(i);
        if (engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
            count++;
        }
    }
    return count;
}

int TensorRTEngine::getNumOutputs() const {
    int count = 0;
    for (int i = 0; i < num_bindings_; i++) {
        auto name = engine_->getIOTensorName(i);
        if (engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kOUTPUT) {
            count++;
        }
    }
    return count;
}

std::vector<int> TensorRTEngine::getInputShape(int index) const {
    int current_input = 0;
    for (int i = 0; i < num_bindings_; i++) {
        auto name = engine_->getIOTensorName(i);
        if (engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
            if (current_input == index) {
                auto dims = engine_->getTensorShape(name);
                std::vector<int> shape;
                for (int j = 0; j < dims.nbDims; j++) {
                    shape.push_back(dims.d[j]);
                }
                return shape;
            }
            current_input++;
        }
    }
    return {};
}

std::vector<int> TensorRTEngine::getOutputShape(int index) const {
    int current_output = 0;
    for (int i = 0; i < num_bindings_; i++) {
        auto name = engine_->getIOTensorName(i);
        if (engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kOUTPUT) {
            if (current_output == index) {
                auto dims = engine_->getTensorShape(name);
                std::vector<int> shape;
                for (int j = 0; j < dims.nbDims; j++) {
                    shape.push_back(dims.d[j]);
                }
                return shape;
            }
            current_output++;
        }
    }
    return {};
}

TensorRTEngine::InferenceStats TensorRTEngine::benchmark(int warmup_iterations,
                                                         int benchmark_iterations) {
    std::cout << "\n=== Benchmarking TensorRT Inference ===" << std::endl;
    std::cout << "Warmup iterations: " << warmup_iterations << std::endl;
    std::cout << "Benchmark iterations: " << benchmark_iterations << std::endl;

    // Get input shape
    auto input_shape = getInputShape(0);
    size_t input_size = 1;
    for (auto dim : input_shape) {
        input_size *= dim;
    }

    // Allocate host memory
    std::vector<float> input_data(input_size, 1.0f);

    auto output_shape = getOutputShape(0);
    size_t output_size = 1;
    for (auto dim : output_shape) {
        output_size *= dim;
    }
    std::vector<float> output_data(output_size);

    std::vector<void*> inputs = {input_data.data()};
    std::vector<void*> outputs = {output_data.data()};

    // Warmup
    std::cout << "Warming up..." << std::endl;
    for (int i = 0; i < warmup_iterations; i++) {
        infer(inputs, outputs);
    }

    // Benchmark
    std::cout << "Running benchmark..." << std::endl;
    std::vector<float> latencies;
    latencies.reserve(benchmark_iterations);

    for (int i = 0; i < benchmark_iterations; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        infer(inputs, outputs);
        auto end = std::chrono::high_resolution_clock::now();

        float latency = std::chrono::duration<float, std::milli>(end - start).count();
        latencies.push_back(latency);
    }

    // Calculate statistics
    float sum = 0.0f;
    float min_latency = latencies[0];
    float max_latency = latencies[0];

    for (float lat : latencies) {
        sum += lat;
        min_latency = std::min(min_latency, lat);
        max_latency = std::max(max_latency, lat);
    }

    float avg_latency = sum / benchmark_iterations;

    InferenceStats stats;
    stats.avg_latency_ms = avg_latency;
    stats.min_latency_ms = min_latency;
    stats.max_latency_ms = max_latency;
    stats.iterations = benchmark_iterations;

    std::cout << "\n=== Benchmark Results ===" << std::endl;
    std::cout << "Average latency: " << avg_latency << " ms" << std::endl;
    std::cout << "Min latency: " << min_latency << " ms" << std::endl;
    std::cout << "Max latency: " << max_latency << " ms" << std::endl;
    std::cout << "Throughput: " << (1000.0f / avg_latency) << " FPS" << std::endl;

    return stats;
}

} // namespace tensorrt
} // namespace hook_analyzer
