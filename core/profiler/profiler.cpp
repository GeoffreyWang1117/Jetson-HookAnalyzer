#include "profiler.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <iomanip>

namespace hook_analyzer {

Profiler::Profiler(const ProfilerConfig& config)
    : config_(config) {
}

Profiler::~Profiler() {
    shutdown();
}

bool Profiler::initialize() {
    if (initialized_) {
        return true;
    }

    // Start metrics collection thread if enabled
    if (config_.enable_metrics_collection) {
        running_ = true;
        metrics_thread_.reset(new std::thread(
            &Profiler::metricsCollectionThread, this));
    }

#ifdef ENABLE_CUPTI
    // Initialize CUPTI if available
    CUptiResult result = cuptiSubscribe(&cupti_subscriber_,
        (CUpti_CallbackFunc)nullptr, nullptr);
    if (result != CUPTI_SUCCESS) {
        std::cerr << "[Profiler] Failed to initialize CUPTI" << std::endl;
        return false;
    }
#endif

    initialized_ = true;
    std::cout << "[Profiler] Initialized successfully" << std::endl;
    return true;
}

void Profiler::shutdown() {
    if (!initialized_) {
        return;
    }

    running_ = false;

    if (metrics_thread_ && metrics_thread_->joinable()) {
        metrics_thread_->join();
    }

#ifdef ENABLE_CUPTI
    cuptiUnsubscribe(cupti_subscriber_);
#endif

    initialized_ = false;
}

void Profiler::startEvent(const std::string& name, const std::string& category) {
    if (!enabled_) return;

    ProfileEvent event;
    event.name = name;
    event.category = category;
    event.start_time = std::chrono::high_resolution_clock::now();

    events_.push_back(event);
}

void Profiler::endEvent(const std::string& name) {
    if (!enabled_) return;

    auto end_time = std::chrono::high_resolution_clock::now();

    // Find the most recent event with this name
    for (auto it = events_.rbegin(); it != events_.rend(); ++it) {
        if (it->name == name && it->duration_ms == 0.0f) {
            it->end_time = end_time;
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
                it->end_time - it->start_time);
            it->duration_ms = duration.count() / 1000.0f;
            break;
        }
    }
}

cudaEvent_t Profiler::createTimingEvent() {
    cudaEvent_t event;
    cudaEventCreate(&event);
    return event;
}

void Profiler::recordEvent(cudaEvent_t event, cudaStream_t stream) {
    cudaEventRecord(event, stream);
}

float Profiler::getElapsedTime(cudaEvent_t start, cudaEvent_t end) {
    float elapsed_ms = 0.0f;
    cudaEventElapsedTime(&elapsed_ms, start, end);
    return elapsed_ms;
}

void Profiler::destroyEvent(cudaEvent_t event) {
    cudaEventDestroy(event);
}

GPUMetrics Profiler::getGPUMetrics(int device_id) {
    GPUMetrics metrics;
    metrics.device_id = device_id;

    cudaSetDevice(device_id);

    // Get memory info
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    metrics.free_memory = free_mem;
    metrics.total_memory = total_mem;
    metrics.memory_utilization = 100.0f * (1.0f - (float)free_mem / total_mem);

    // Other metrics would require NVML (NVIDIA Management Library)
    // For now, set placeholder values
    metrics.sm_utilization = 0.0f;
    metrics.memory_bandwidth_gbps = 0.0f;
    metrics.power_usage_watts = 0.0f;
    metrics.temperature_celsius = 0.0f;

    return metrics;
}

std::vector<ProfileEvent> Profiler::getEvents() const {
    return events_;
}

std::vector<KernelProfile> Profiler::getKernelProfiles() const {
    return kernel_profiles_;
}

bool Profiler::exportChromeTrace(const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "[Profiler] Failed to open " << filename << std::endl;
        return false;
    }

    // Chrome Trace Event Format
    file << "[\n";

    bool first = true;
    for (const auto& event : events_) {
        if (!first) {
            file << ",\n";
        }
        first = false;

        auto start_us = std::chrono::duration_cast<std::chrono::microseconds>(
            event.start_time.time_since_epoch()).count();

        file << "  {\n";
        file << "    \"name\": \"" << event.name << "\",\n";
        file << "    \"cat\": \"" << event.category << "\",\n";
        file << "    \"ph\": \"X\",\n";  // Complete event
        file << "    \"ts\": " << start_us << ",\n";
        file << "    \"dur\": " << (event.duration_ms * 1000.0f) << ",\n";
        file << "    \"pid\": 0,\n";
        file << "    \"tid\": 0\n";
        file << "  }";
    }

    file << "\n]\n";
    file.close();

    std::cout << "[Profiler] Exported Chrome trace to " << filename << std::endl;
    return true;
}

bool Profiler::exportJSON(const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "[Profiler] Failed to open " << filename << std::endl;
        return false;
    }

    file << "{\n";
    file << "  \"events\": [\n";

    bool first = true;
    for (const auto& event : events_) {
        if (!first) {
            file << ",\n";
        }
        first = false;

        file << "    {\n";
        file << "      \"name\": \"" << event.name << "\",\n";
        file << "      \"category\": \"" << event.category << "\",\n";
        file << "      \"duration_ms\": " << event.duration_ms << "\n";
        file << "    }";
    }

    file << "\n  ]\n";
    file << "}\n";
    file.close();

    std::cout << "[Profiler] Exported JSON to " << filename << std::endl;
    return true;
}

void Profiler::reset() {
    events_.clear();
    kernel_profiles_.clear();
}

void Profiler::metricsCollectionThread() {
    while (running_) {
        // Collect metrics periodically
        // This is a placeholder - real implementation would use NVML

        std::this_thread::sleep_for(
            std::chrono::milliseconds(config_.metrics_interval_ms));
    }
}

} // namespace hook_analyzer
