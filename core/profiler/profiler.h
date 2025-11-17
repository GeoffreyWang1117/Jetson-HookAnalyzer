#pragma once

#include <cuda_runtime.h>
#include <string>
#include <vector>
#include <memory>
#include <chrono>
#include <thread>

#ifdef ENABLE_CUPTI
#include <cupti.h>
#endif

namespace hook_analyzer {

// Profiling event
struct ProfileEvent {
    std::string name;
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time;
    std::chrono::time_point<std::chrono::high_resolution_clock> end_time;
    float duration_ms;
    std::string category;  // "kernel", "memory", "compute", etc.
};

// GPU metrics
struct GPUMetrics {
    int device_id;
    float sm_utilization;          // SM (Streaming Multiprocessor) utilization %
    float memory_utilization;      // Memory utilization %
    float memory_bandwidth_gbps;   // Memory bandwidth in GB/s
    size_t free_memory;
    size_t total_memory;
    float power_usage_watts;
    float temperature_celsius;
};

// Kernel profiling info
struct KernelProfile {
    std::string name;
    uint64_t grid_size;
    uint64_t block_size;
    uint64_t shared_memory_bytes;
    uint64_t registers_per_thread;
    float duration_ms;
    float occupancy;               // Theoretical occupancy
    bool is_memory_bound;
    bool is_compute_bound;
};

// Profiler configuration
struct ProfilerConfig {
    bool enable_kernel_profiling = true;
    bool enable_memory_profiling = true;
    bool enable_metrics_collection = true;
    int metrics_interval_ms = 100;  // How often to collect GPU metrics
    bool export_chrome_trace = false;
    std::string trace_output_path = "trace.json";
};

// Performance Profiler
class Profiler {
public:
    explicit Profiler(const ProfilerConfig& config = ProfilerConfig());
    ~Profiler();

    // Initialize/shutdown
    bool initialize();
    void shutdown();

    // Event tracking
    void startEvent(const std::string& name, const std::string& category = "");
    void endEvent(const std::string& name);

    // CUDA event-based profiling
    cudaEvent_t createTimingEvent();
    void recordEvent(cudaEvent_t event, cudaStream_t stream = 0);
    float getElapsedTime(cudaEvent_t start, cudaEvent_t end);
    void destroyEvent(cudaEvent_t event);

    // GPU metrics
    GPUMetrics getGPUMetrics(int device_id = 0);

    // Get profiling results
    std::vector<ProfileEvent> getEvents() const;
    std::vector<KernelProfile> getKernelProfiles() const;

    // Export results
    bool exportChromeTrace(const std::string& filename);
    bool exportJSON(const std::string& filename);

    // Reset profiler
    void reset();

    // Enable/disable profiling
    void setEnabled(bool enabled) { enabled_ = enabled; }
    bool isEnabled() const { return enabled_; }

private:
    void metricsCollectionThread();

    ProfilerConfig config_;
    bool initialized_ = false;
    bool enabled_ = true;
    bool running_ = false;

    std::vector<ProfileEvent> events_;
    std::vector<KernelProfile> kernel_profiles_;

    // Metrics collection thread
    std::unique_ptr<std::thread> metrics_thread_;

#ifdef ENABLE_CUPTI
    CUpti_SubscriberHandle cupti_subscriber_;
#endif
};

// RAII helper for scoped profiling
class ScopedProfile {
public:
    ScopedProfile(Profiler& profiler, const std::string& name,
                  const std::string& category = "")
        : profiler_(profiler), name_(name) {
        profiler_.startEvent(name_, category);
    }

    ~ScopedProfile() {
        profiler_.endEvent(name_);
    }

private:
    Profiler& profiler_;
    std::string name_;
};

// RAII helper for CUDA event timing
class ScopedCudaTimer {
public:
    ScopedCudaTimer(Profiler& profiler, cudaStream_t stream = 0)
        : profiler_(profiler), stream_(stream) {
        start_event_ = profiler_.createTimingEvent();
        end_event_ = profiler_.createTimingEvent();
        profiler_.recordEvent(start_event_, stream_);
    }

    ~ScopedCudaTimer() {
        profiler_.recordEvent(end_event_, stream_);
        cudaStreamSynchronize(stream_);
        elapsed_ms_ = profiler_.getElapsedTime(start_event_, end_event_);
        profiler_.destroyEvent(start_event_);
        profiler_.destroyEvent(end_event_);
    }

    float getElapsedMs() const { return elapsed_ms_; }

private:
    Profiler& profiler_;
    cudaStream_t stream_;
    cudaEvent_t start_event_;
    cudaEvent_t end_event_;
    float elapsed_ms_ = 0.0f;
};

// Macros for convenient profiling
#define PROFILE_SCOPE(profiler, name) \
    hook_analyzer::ScopedProfile _profile(profiler, name)

#define PROFILE_CUDA_SCOPE(profiler, stream) \
    hook_analyzer::ScopedCudaTimer _cuda_timer(profiler, stream)

} // namespace hook_analyzer
