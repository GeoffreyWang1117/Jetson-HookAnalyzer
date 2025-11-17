#include <iostream>
#include <vector>
#include <cuda_runtime.h>

#include "cuda_hook/cuda_hook.h"
#include "scheduler/scheduler.h"
#include "profiler/profiler.h"
#include "kernels/optimized/kernels.h"

using namespace hook_analyzer;

void printGPUInfo() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    std::cout << "=== GPU Information ===" << std::endl;
    std::cout << "Number of GPUs: " << deviceCount << std::endl;

    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);

        std::cout << "\nGPU " << i << ": " << prop.name << std::endl;
        std::cout << "  Compute Capability: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "  Total Memory: " << prop.totalGlobalMem / (1024*1024) << " MB" << std::endl;
        std::cout << "  SM Count: " << prop.multiProcessorCount << std::endl;
        std::cout << "  Max Threads per Block: " << prop.maxThreadsPerBlock << std::endl;
    }
    std::cout << "=======================" << std::endl << std::endl;
}

void demoMemoryTracking() {
    std::cout << "=== Memory Tracking Demo ===" << std::endl;

    auto& hook_manager = CudaHookManager::getInstance();
    hook_manager.initialize();

    // Allocate some GPU memory
    float* d_array1;
    float* d_array2;
    float* d_result;

    size_t size = 1024 * 1024; // 1M floats
    size_t bytes = size * sizeof(float);

    cudaMalloc(&d_array1, bytes);
    cudaMalloc(&d_array2, bytes);
    cudaMalloc(&d_result, bytes);

    // Get memory statistics
    auto stats = hook_manager.getMemoryStats();
    std::cout << "Current allocated: " << stats.current_allocated / (1024*1024) << " MB" << std::endl;
    std::cout << "Peak allocated: " << stats.peak_allocated / (1024*1024) << " MB" << std::endl;
    std::cout << "Active allocations: " << stats.num_active_allocations << std::endl;
    std::cout << "Fragmentation ratio: " << stats.fragmentation_ratio << std::endl;

    // Free memory
    cudaFree(d_array1);
    cudaFree(d_array2);
    cudaFree(d_result);

    stats = hook_manager.getMemoryStats();
    std::cout << "After free - Current allocated: "
              << stats.current_allocated / (1024*1024) << " MB" << std::endl;

    std::cout << "===========================" << std::endl << std::endl;
}

void demoCustomKernels() {
    std::cout << "=== Custom Kernels Demo ===" << std::endl;

    Profiler profiler;
    profiler.initialize();

    const int N = 1024;
    size_t bytes = N * sizeof(float);

    // Host arrays
    std::vector<float> h_a(N, 2.0f);
    std::vector<float> h_b(N, 3.0f);
    std::vector<float> h_c(N, 0.0f);

    // Device arrays
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice);

    // Test element-wise addition
    {
        PROFILE_SCOPE(profiler, "add_kernel");
        kernels::addFloat(d_a, d_b, d_c, N);
        cudaDeviceSynchronize();
    }

    cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost);
    std::cout << "Add result: " << h_c[0] << " (expected: 5.0)" << std::endl;

    // Test ReLU
    std::fill(h_a.begin(), h_a.end(), -1.0f);
    cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice);

    {
        PROFILE_SCOPE(profiler, "relu_kernel");
        kernels::reluFloat(d_a, d_c, N);
        cudaDeviceSynchronize();
    }

    cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost);
    std::cout << "ReLU result: " << h_c[0] << " (expected: 0.0)" << std::endl;

    // Print profiling results
    auto events = profiler.getEvents();
    std::cout << "\nProfiling Results:" << std::endl;
    for (const auto& event : events) {
        std::cout << "  " << event.name << ": " << event.duration_ms << " ms" << std::endl;
    }

    // Cleanup
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    std::cout << "===========================" << std::endl << std::endl;
}

void demoScheduler() {
    std::cout << "=== Scheduler Demo ===" << std::endl;

    SchedulerConfig config;
    config.num_worker_threads = 2;
    config.num_cuda_streams = 4;
    config.enable_priority_scheduling = true;

    InferenceScheduler scheduler(config);
    scheduler.start();

    // Submit some dummy tasks
    for (int i = 0; i < 10; i++) {
        InferenceTask task;
        task.model_id = "test_model";
        task.priority = i % 3;  // Varying priorities

        task.callback = [i]() {
            // Simulate some work
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            std::cout << "Task " << i << " completed" << std::endl;
        };

        uint64_t task_id = scheduler.submitTask(task);
        std::cout << "Submitted task " << task_id << " with priority " << task.priority << std::endl;
    }

    // Wait for completion
    std::this_thread::sleep_for(std::chrono::seconds(1));

    auto stats = scheduler.getStats();
    std::cout << "\nScheduler Statistics:" << std::endl;
    std::cout << "  Total submitted: " << stats.total_tasks_submitted << std::endl;
    std::cout << "  Total completed: " << stats.total_tasks_completed << std::endl;
    std::cout << "  Total failed: " << stats.total_tasks_failed << std::endl;
    std::cout << "  Avg queue wait: " << stats.avg_queue_wait_time_ms << " ms" << std::endl;
    std::cout << "  Avg execution: " << stats.avg_execution_time_ms << " ms" << std::endl;
    std::cout << "  Throughput: " << stats.throughput_tasks_per_sec << " tasks/sec" << std::endl;

    scheduler.stop();

    std::cout << "===========================" << std::endl << std::endl;
}

int main(int argc, char** argv) {
    std::cout << "==================================" << std::endl;
    std::cout << "HookAnalyzer Simple Demo" << std::endl;
    std::cout << "==================================" << std::endl << std::endl;

    printGPUInfo();
    demoMemoryTracking();
    demoCustomKernels();
    demoScheduler();

    std::cout << "Demo completed successfully!" << std::endl;

    return 0;
}
