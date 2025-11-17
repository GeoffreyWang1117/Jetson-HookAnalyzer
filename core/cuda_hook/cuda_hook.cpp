#include "cuda_hook.h"
#include <dlfcn.h>
#include <iostream>
#include <algorithm>
#include <numeric>

namespace hook_analyzer {

// Original CUDA function pointers
static cudaError_t (*real_cudaMalloc)(void**, size_t) = nullptr;
static cudaError_t (*real_cudaFree)(void*) = nullptr;
static cudaError_t (*real_cudaMallocManaged)(void**, size_t, unsigned int) = nullptr;
static cudaError_t (*real_cudaMemcpy)(void*, const void*, size_t, cudaMemcpyKind) = nullptr;

CudaHookManager& CudaHookManager::getInstance() {
    static CudaHookManager instance;
    return instance;
}

bool CudaHookManager::initialize() {
    std::lock_guard<std::mutex> lock(mutex_);

    if (initialized_) {
        return true;
    }

    // Load original CUDA functions using RTLD_NEXT
    real_cudaMalloc = (cudaError_t (*)(void**, size_t))dlsym(RTLD_NEXT, "cudaMalloc");
    real_cudaFree = (cudaError_t (*)(void*))dlsym(RTLD_NEXT, "cudaFree");
    real_cudaMallocManaged = (cudaError_t (*)(void**, size_t, unsigned int))dlsym(RTLD_NEXT, "cudaMallocManaged");
    real_cudaMemcpy = (cudaError_t (*)(void*, const void*, size_t, cudaMemcpyKind))dlsym(RTLD_NEXT, "cudaMemcpy");

    if (!real_cudaMalloc || !real_cudaFree) {
        std::cerr << "Failed to load original CUDA functions" << std::endl;
        return false;
    }

    initialized_ = true;
    std::cout << "[HookAnalyzer] CUDA hooks initialized successfully" << std::endl;
    return true;
}

void CudaHookManager::shutdown() {
    std::lock_guard<std::mutex> lock(mutex_);
    initialized_ = false;
    allocations_.clear();
    kernel_history_.clear();
    active_kernels_.clear();
}

void CudaHookManager::trackAllocation(void* ptr, size_t size, const std::string& tag) {
    if (!tracking_enabled_) return;

    std::lock_guard<std::mutex> lock(mutex_);

    int device_id = 0;
    cudaGetDevice(&device_id);

    MemoryAllocation alloc;
    alloc.ptr = ptr;
    alloc.size = size;
    alloc.timestamp = std::chrono::high_resolution_clock::now();
    alloc.tag = tag;
    alloc.device_id = device_id;

    allocations_[ptr] = alloc;
    current_allocated_ += size;
    total_allocations_++;

    if (current_allocated_ > peak_allocated_) {
        peak_allocated_ = current_allocated_;
    }
}

void CudaHookManager::trackDeallocation(void* ptr) {
    if (!tracking_enabled_) return;

    std::lock_guard<std::mutex> lock(mutex_);

    auto it = allocations_.find(ptr);
    if (it != allocations_.end()) {
        current_allocated_ -= it->second.size;
        allocations_.erase(it);
        total_deallocations_++;
    }
}

size_t CudaHookManager::getTotalAllocated() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return current_allocated_;
}

size_t CudaHookManager::getPeakMemoryUsage() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return peak_allocated_;
}

float CudaHookManager::getFragmentationRatio() const {
    std::lock_guard<std::mutex> lock(mutex_);

    if (allocations_.empty()) {
        return 0.0f;
    }

    // Simple fragmentation metric: ratio of allocation count to total size
    size_t total_size = current_allocated_;
    size_t num_allocations = allocations_.size();

    if (total_size == 0) return 0.0f;

    float avg_alloc_size = static_cast<float>(total_size) / num_allocations;
    float fragmentation = std::min(1.0f, num_allocations / (avg_alloc_size + 1.0f));

    return fragmentation;
}

void CudaHookManager::trackKernelLaunch(const std::string& name, dim3 grid, dim3 block,
                                        size_t shared_mem, cudaStream_t stream) {
    if (!tracking_enabled_) return;

    std::lock_guard<std::mutex> lock(mutex_);

    KernelLaunch launch;
    launch.name = name;
    launch.grid_dim = grid;
    launch.block_dim = block;
    launch.shared_mem = shared_mem;
    launch.stream = stream;
    launch.start_time = std::chrono::high_resolution_clock::now();

    active_kernels_[name] = launch;
}

void CudaHookManager::trackKernelEnd(const std::string& name) {
    if (!tracking_enabled_) return;

    std::lock_guard<std::mutex> lock(mutex_);

    auto it = active_kernels_.find(name);
    if (it != active_kernels_.end()) {
        it->second.end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
            it->second.end_time - it->second.start_time);
        it->second.duration_ms = duration.count() / 1000.0f;

        kernel_history_[name].push_back(it->second);
        active_kernels_.erase(it);
    }
}

CudaHookManager::MemoryStats CudaHookManager::getMemoryStats() const {
    std::lock_guard<std::mutex> lock(mutex_);

    MemoryStats stats;
    stats.current_allocated = current_allocated_;
    stats.peak_allocated = peak_allocated_;
    stats.total_allocations = total_allocations_;
    stats.total_deallocations = total_deallocations_;
    stats.fragmentation_ratio = getFragmentationRatio();
    stats.num_active_allocations = allocations_.size();

    return stats;
}

std::vector<CudaHookManager::KernelStats> CudaHookManager::getKernelStats() const {
    std::lock_guard<std::mutex> lock(mutex_);

    std::vector<KernelStats> stats;

    for (const auto& [name, launches] : kernel_history_) {
        if (launches.empty()) continue;

        KernelStats ks;
        ks.name = name;
        ks.call_count = launches.size();

        float total_time = 0.0f;
        float min_time = std::numeric_limits<float>::max();
        float max_time = 0.0f;

        for (const auto& launch : launches) {
            total_time += launch.duration_ms;
            min_time = std::min(min_time, launch.duration_ms);
            max_time = std::max(max_time, launch.duration_ms);
        }

        ks.total_time_ms = total_time;
        ks.avg_time_ms = total_time / launches.size();
        ks.min_time_ms = min_time;
        ks.max_time_ms = max_time;

        stats.push_back(ks);
    }

    return stats;
}

void CudaHookManager::resetStats() {
    std::lock_guard<std::mutex> lock(mutex_);

    allocations_.clear();
    kernel_history_.clear();
    active_kernels_.clear();
    current_allocated_ = 0;
    peak_allocated_ = 0;
    total_allocations_ = 0;
    total_deallocations_ = 0;
}

// RAII Scoped Tracker
ScopedCudaTracker::ScopedCudaTracker(const std::string& scope_name)
    : scope_name_(scope_name)
    , start_time_(std::chrono::high_resolution_clock::now()) {
}

ScopedCudaTracker::~ScopedCudaTracker() {
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
        end_time - start_time_);

    // You can log or store this information
    // std::cout << "[" << scope_name_ << "] took " << duration.count() / 1000.0 << " ms" << std::endl;
}

} // namespace hook_analyzer

// Hooked CUDA functions implementation
extern "C" {

cudaError_t cudaMalloc(void** devPtr, size_t size) {
    auto& manager = hook_analyzer::CudaHookManager::getInstance();

    if (!hook_analyzer::real_cudaMalloc) {
        manager.initialize();
    }

    cudaError_t result = hook_analyzer::real_cudaMalloc(devPtr, size);

    if (result == cudaSuccess && manager.isTrackingEnabled()) {
        manager.trackAllocation(*devPtr, size);
    }

    return result;
}

cudaError_t cudaFree(void* devPtr) {
    auto& manager = hook_analyzer::CudaHookManager::getInstance();

    if (manager.isTrackingEnabled()) {
        manager.trackDeallocation(devPtr);
    }

    return hook_analyzer::real_cudaFree(devPtr);
}

cudaError_t cudaMallocManaged(void** devPtr, size_t size, unsigned int flags) {
    auto& manager = hook_analyzer::CudaHookManager::getInstance();

    if (!hook_analyzer::real_cudaMallocManaged) {
        manager.initialize();
    }

    cudaError_t result = hook_analyzer::real_cudaMallocManaged(devPtr, size, flags);

    if (result == cudaSuccess && manager.isTrackingEnabled()) {
        manager.trackAllocation(*devPtr, size, "managed");
    }

    return result;
}

cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind) {
    return hook_analyzer::real_cudaMemcpy(dst, src, count, kind);
}

} // extern "C"
