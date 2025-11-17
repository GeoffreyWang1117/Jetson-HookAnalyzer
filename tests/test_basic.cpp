#include <iostream>
#include <cassert>
#include "cuda_hook/cuda_hook.h"
#include "scheduler/scheduler.h"
#include "profiler/profiler.h"

using namespace hook_analyzer;

void test_cuda_hook() {
    std::cout << "Testing CUDA Hook Manager..." << std::endl;

    auto& hook_manager = CudaHookManager::getInstance();
    assert(hook_manager.initialize());

    // Test memory allocation tracking
    float* d_data;
    cudaMalloc(&d_data, 1024 * sizeof(float));

    auto stats = hook_manager.getMemoryStats();
    assert(stats.current_allocated >= 1024 * sizeof(float));

    cudaFree(d_data);

    std::cout << "✓ CUDA Hook tests passed" << std::endl;
}

void test_scheduler() {
    std::cout << "Testing Scheduler..." << std::endl;

    SchedulerConfig config;
    config.num_worker_threads = 2;

    InferenceScheduler scheduler(config);
    assert(scheduler.start());

    // Submit a task
    InferenceTask task;
    task.model_id = "test";
    task.priority = 1;
    task.callback = []() {
        // Empty task
    };

    uint64_t task_id = scheduler.submitTask(task);
    assert(task_id > 0);

    scheduler.stop();

    std::cout << "✓ Scheduler tests passed" << std::endl;
}

void test_profiler() {
    std::cout << "Testing Profiler..." << std::endl;

    Profiler profiler;
    assert(profiler.initialize());

    profiler.startEvent("test_event");
    profiler.endEvent("test_event");

    auto events = profiler.getEvents();
    assert(!events.empty());

    std::cout << "✓ Profiler tests passed" << std::endl;
}

int main() {
    std::cout << "==================================" << std::endl;
    std::cout << "Running HookAnalyzer Tests" << std::endl;
    std::cout << "==================================" << std::endl;

    try {
        test_cuda_hook();
        test_scheduler();
        test_profiler();

        std::cout << "\n==================================" << std::endl;
        std::cout << "All tests passed!" << std::endl;
        std::cout << "==================================" << std::endl;

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed: " << e.what() << std::endl;
        return 1;
    }
}
