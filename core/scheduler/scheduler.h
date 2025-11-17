#pragma once

#include <memory>
#include <queue>
#include <functional>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <vector>
#include <string>
#include <chrono>
#include <atomic>
#include <cuda_runtime.h>

namespace hook_analyzer {

// Inference task definition
struct InferenceTask {
    using TaskCallback = std::function<void()>;

    std::string model_id;
    void* input_data;
    void* output_data;
    size_t input_size;
    size_t output_size;
    int priority;  // Higher number = higher priority
    TaskCallback callback;
    cudaStream_t stream;

    // Metadata
    mutable std::chrono::time_point<std::chrono::high_resolution_clock> submit_time;
    mutable std::chrono::time_point<std::chrono::high_resolution_clock> start_time;
    mutable std::chrono::time_point<std::chrono::high_resolution_clock> end_time;

    // Task ID
    uint64_t task_id;

    // Priority comparison for priority queue
    bool operator<(const InferenceTask& other) const {
        return priority < other.priority;  // Max-heap (higher priority first)
    }
};

// Scheduler configuration
struct SchedulerConfig {
    int num_worker_threads = 2;
    int max_queue_size = 100;
    bool enable_dynamic_batching = true;
    int max_batch_size = 8;
    int batch_timeout_ms = 10;
    bool enable_priority_scheduling = true;
    bool enable_stream_optimization = true;
    int num_cuda_streams = 4;
};

// Scheduler statistics
struct SchedulerStats {
    uint64_t total_tasks_submitted;
    uint64_t total_tasks_completed;
    uint64_t total_tasks_failed;
    uint64_t tasks_in_queue;
    float avg_queue_wait_time_ms;
    float avg_execution_time_ms;
    float throughput_tasks_per_sec;
};

// Intelligent Inference Scheduler
class InferenceScheduler {
public:
    explicit InferenceScheduler(const SchedulerConfig& config = SchedulerConfig());
    ~InferenceScheduler();

    // Start/Stop scheduler
    bool start();
    void stop();
    bool isRunning() const { return running_; }

    // Submit task
    uint64_t submitTask(const InferenceTask& task);

    // Wait for specific task
    bool waitForTask(uint64_t task_id, int timeout_ms = -1);

    // Get statistics
    SchedulerStats getStats() const;

    // Configuration
    void updateConfig(const SchedulerConfig& config);
    SchedulerConfig getConfig() const;

    // Resource management
    cudaStream_t acquireStream();
    void releaseStream(cudaStream_t stream);

private:
    void workerThread();
    void processTask(const InferenceTask& task);
    std::vector<InferenceTask> tryBatchTasks(const InferenceTask& first_task);

    SchedulerConfig config_;
    std::atomic<bool> running_{false};

    // Task queue
    std::priority_queue<InferenceTask> task_queue_;
    mutable std::mutex queue_mutex_;
    std::condition_variable queue_cv_;

    // Worker threads
    std::vector<std::thread> workers_;

    // CUDA streams pool
    std::vector<cudaStream_t> stream_pool_;
    std::vector<bool> stream_available_;
    std::mutex stream_mutex_;

    // Statistics
    mutable std::mutex stats_mutex_;
    std::atomic<uint64_t> task_id_counter_{0};
    std::atomic<uint64_t> total_tasks_submitted_{0};
    std::atomic<uint64_t> total_tasks_completed_{0};
    std::atomic<uint64_t> total_tasks_failed_{0};

    std::vector<float> queue_wait_times_;
    std::vector<float> execution_times_;
};

// RAII helper for automatic stream management
class ScopedStream {
public:
    explicit ScopedStream(InferenceScheduler& scheduler)
        : scheduler_(scheduler)
        , stream_(scheduler.acquireStream()) {
    }

    ~ScopedStream() {
        scheduler_.releaseStream(stream_);
    }

    cudaStream_t get() const { return stream_; }

private:
    InferenceScheduler& scheduler_;
    cudaStream_t stream_;
};

} // namespace hook_analyzer
