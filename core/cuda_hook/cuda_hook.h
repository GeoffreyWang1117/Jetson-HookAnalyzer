#pragma once

#include <cuda_runtime.h>
#include <cuda.h>
#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <mutex>
#include <memory>
#include <chrono>
#include <vector>

namespace hook_analyzer {

// Memory allocation record
struct MemoryAllocation {
    void* ptr;
    size_t size;
    std::chrono::time_point<std::chrono::high_resolution_clock> timestamp;
    std::string tag;
    int device_id;
};

// Kernel launch record
struct KernelLaunch {
    std::string name;
    dim3 grid_dim;
    dim3 block_dim;
    size_t shared_mem;
    cudaStream_t stream;
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time;
    std::chrono::time_point<std::chrono::high_resolution_clock> end_time;
    float duration_ms;
};

// CUDA Hook Manager
class CudaHookManager {
public:
    static CudaHookManager& getInstance();

    // Initialize hooks
    bool initialize();
    void shutdown();

    // Memory tracking
    void trackAllocation(void* ptr, size_t size, const std::string& tag = "");
    void trackDeallocation(void* ptr);
    size_t getTotalAllocated() const;
    size_t getPeakMemoryUsage() const;
    float getFragmentationRatio() const;

    // Kernel tracking
    void trackKernelLaunch(const std::string& name, dim3 grid, dim3 block,
                          size_t shared_mem, cudaStream_t stream);
    void trackKernelEnd(const std::string& name);

    // Statistics
    struct MemoryStats {
        size_t current_allocated;
        size_t peak_allocated;
        size_t total_allocations;
        size_t total_deallocations;
        float fragmentation_ratio;
        size_t num_active_allocations;
    };

    struct KernelStats {
        std::string name;
        uint64_t call_count;
        float total_time_ms;
        float avg_time_ms;
        float min_time_ms;
        float max_time_ms;
    };

    MemoryStats getMemoryStats() const;
    std::vector<KernelStats> getKernelStats() const;

    // Reset statistics
    void resetStats();

    // Enable/disable tracking
    void setTrackingEnabled(bool enabled) { tracking_enabled_ = enabled; }
    bool isTrackingEnabled() const { return tracking_enabled_; }

private:
    CudaHookManager() = default;
    ~CudaHookManager() = default;
    CudaHookManager(const CudaHookManager&) = delete;
    CudaHookManager& operator=(const CudaHookManager&) = delete;

    mutable std::mutex mutex_;
    bool initialized_ = false;
    bool tracking_enabled_ = true;

    // Memory tracking data
    std::unordered_map<void*, MemoryAllocation> allocations_;
    size_t current_allocated_ = 0;
    size_t peak_allocated_ = 0;
    size_t total_allocations_ = 0;
    size_t total_deallocations_ = 0;

    // Kernel tracking data
    std::unordered_map<std::string, std::vector<KernelLaunch>> kernel_history_;
    std::unordered_map<std::string, KernelLaunch> active_kernels_;
};

// Hooked CUDA functions (will be implemented with LD_PRELOAD or CUPTI)
extern "C" {
    cudaError_t cudaMalloc(void** devPtr, size_t size);
    cudaError_t cudaFree(void* devPtr);
    cudaError_t cudaMallocManaged(void** devPtr, size_t size, unsigned int flags);
    cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind);
}

// RAII helper for tracking scopes
class ScopedCudaTracker {
public:
    explicit ScopedCudaTracker(const std::string& scope_name);
    ~ScopedCudaTracker();

private:
    std::string scope_name_;
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time_;
};

// Macro for easy scope tracking
#define CUDA_TRACK_SCOPE(name) hook_analyzer::ScopedCudaTracker _tracker(name)

} // namespace hook_analyzer
