#include "scheduler.h"
#include <iostream>
#include <numeric>
#include <algorithm>

namespace hook_analyzer {

InferenceScheduler::InferenceScheduler(const SchedulerConfig& config)
    : config_(config) {
}

InferenceScheduler::~InferenceScheduler() {
    stop();
}

bool InferenceScheduler::start() {
    if (running_) {
        std::cerr << "[Scheduler] Already running" << std::endl;
        return false;
    }

    // Initialize CUDA streams
    stream_pool_.resize(config_.num_cuda_streams);
    stream_available_.resize(config_.num_cuda_streams, true);

    for (int i = 0; i < config_.num_cuda_streams; ++i) {
        cudaError_t err = cudaStreamCreate(&stream_pool_[i]);
        if (err != cudaSuccess) {
            std::cerr << "[Scheduler] Failed to create CUDA stream: "
                      << cudaGetErrorString(err) << std::endl;
            return false;
        }
    }

    // Start worker threads
    running_ = true;
    workers_.reserve(config_.num_worker_threads);

    for (int i = 0; i < config_.num_worker_threads; ++i) {
        workers_.emplace_back(&InferenceScheduler::workerThread, this);
    }

    std::cout << "[Scheduler] Started with " << config_.num_worker_threads
              << " workers and " << config_.num_cuda_streams << " CUDA streams" << std::endl;

    return true;
}

void InferenceScheduler::stop() {
    if (!running_) {
        return;
    }

    running_ = false;
    queue_cv_.notify_all();

    // Wait for all workers to finish
    for (auto& worker : workers_) {
        if (worker.joinable()) {
            worker.join();
        }
    }
    workers_.clear();

    // Destroy CUDA streams
    for (auto stream : stream_pool_) {
        cudaStreamDestroy(stream);
    }
    stream_pool_.clear();
    stream_available_.clear();

    std::cout << "[Scheduler] Stopped" << std::endl;
}

uint64_t InferenceScheduler::submitTask(const InferenceTask& task) {
    if (!running_) {
        std::cerr << "[Scheduler] Cannot submit task: scheduler not running" << std::endl;
        return 0;
    }

    std::unique_lock<std::mutex> lock(queue_mutex_);

    // Check queue size
    if (task_queue_.size() >= static_cast<size_t>(config_.max_queue_size)) {
        std::cerr << "[Scheduler] Queue is full, rejecting task" << std::endl;
        return 0;
    }

    InferenceTask new_task = task;
    new_task.task_id = ++task_id_counter_;
    new_task.submit_time = std::chrono::high_resolution_clock::now();

    task_queue_.push(new_task);
    total_tasks_submitted_++;

    lock.unlock();
    queue_cv_.notify_one();

    return new_task.task_id;
}

bool InferenceScheduler::waitForTask(uint64_t task_id, int timeout_ms) {
    // Simple implementation: just wait for queue to drain
    // In a real implementation, you'd track individual tasks
    auto start = std::chrono::high_resolution_clock::now();

    while (running_) {
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            if (task_queue_.empty()) {
                return true;
            }
        }

        if (timeout_ms > 0) {
            auto now = std::chrono::high_resolution_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                now - start).count();
            if (elapsed >= timeout_ms) {
                return false;
            }
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    return false;
}

SchedulerStats InferenceScheduler::getStats() const {
    std::lock_guard<std::mutex> stats_lock(stats_mutex_);
    std::lock_guard<std::mutex> queue_lock(queue_mutex_);

    SchedulerStats stats;
    stats.total_tasks_submitted = total_tasks_submitted_;
    stats.total_tasks_completed = total_tasks_completed_;
    stats.total_tasks_failed = total_tasks_failed_;
    stats.tasks_in_queue = task_queue_.size();

    // Calculate averages
    if (!queue_wait_times_.empty()) {
        float sum = std::accumulate(queue_wait_times_.begin(),
                                   queue_wait_times_.end(), 0.0f);
        stats.avg_queue_wait_time_ms = sum / queue_wait_times_.size();
    } else {
        stats.avg_queue_wait_time_ms = 0.0f;
    }

    if (!execution_times_.empty()) {
        float sum = std::accumulate(execution_times_.begin(),
                                   execution_times_.end(), 0.0f);
        stats.avg_execution_time_ms = sum / execution_times_.size();
    } else {
        stats.avg_execution_time_ms = 0.0f;
    }

    // Calculate throughput (tasks per second)
    if (total_tasks_completed_ > 0 && !execution_times_.empty()) {
        float total_time_sec = std::accumulate(execution_times_.begin(),
                                              execution_times_.end(), 0.0f) / 1000.0f;
        stats.throughput_tasks_per_sec = total_tasks_completed_ / (total_time_sec + 0.001f);
    } else {
        stats.throughput_tasks_per_sec = 0.0f;
    }

    return stats;
}

void InferenceScheduler::updateConfig(const SchedulerConfig& config) {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    config_ = config;
}

SchedulerConfig InferenceScheduler::getConfig() const {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    return config_;
}

cudaStream_t InferenceScheduler::acquireStream() {
    std::lock_guard<std::mutex> lock(stream_mutex_);

    for (size_t i = 0; i < stream_available_.size(); ++i) {
        if (stream_available_[i]) {
            stream_available_[i] = false;
            return stream_pool_[i];
        }
    }

    // No stream available, return default stream
    return 0;
}

void InferenceScheduler::releaseStream(cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(stream_mutex_);

    for (size_t i = 0; i < stream_pool_.size(); ++i) {
        if (stream_pool_[i] == stream) {
            stream_available_[i] = true;
            break;
        }
    }
}

void InferenceScheduler::workerThread() {
    while (running_) {
        InferenceTask task;
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            queue_cv_.wait(lock, [this] {
                return !task_queue_.empty() || !running_;
            });

            if (!running_ && task_queue_.empty()) {
                break;
            }

            if (!task_queue_.empty()) {
                task = task_queue_.top();
                task_queue_.pop();
            } else {
                continue;
            }
        }

        processTask(task);
    }
}

void InferenceScheduler::processTask(const InferenceTask& task) {
    task.start_time = std::chrono::high_resolution_clock::now();

    // Calculate queue wait time
    auto wait_time = std::chrono::duration_cast<std::chrono::microseconds>(
        task.start_time - task.submit_time);
    float wait_ms = wait_time.count() / 1000.0f;

    try {
        // Execute the task callback
        if (task.callback) {
            task.callback();
        }

        task.end_time = std::chrono::high_resolution_clock::now();
        auto exec_time = std::chrono::duration_cast<std::chrono::microseconds>(
            task.end_time - task.start_time);
        float exec_ms = exec_time.count() / 1000.0f;

        // Update statistics
        {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            queue_wait_times_.push_back(wait_ms);
            execution_times_.push_back(exec_ms);

            // Keep only recent history (last 1000 tasks)
            if (queue_wait_times_.size() > 1000) {
                queue_wait_times_.erase(queue_wait_times_.begin());
            }
            if (execution_times_.size() > 1000) {
                execution_times_.erase(execution_times_.begin());
            }
        }

        total_tasks_completed_++;

    } catch (const std::exception& e) {
        std::cerr << "[Scheduler] Task " << task.task_id << " failed: "
                  << e.what() << std::endl;
        total_tasks_failed_++;
    }
}

std::vector<InferenceTask> InferenceScheduler::tryBatchTasks(const InferenceTask& first_task) {
    // Simple batching implementation
    // In a real system, you'd check model compatibility, input sizes, etc.
    std::vector<InferenceTask> batch;
    batch.push_back(first_task);

    if (!config_.enable_dynamic_batching) {
        return batch;
    }

    // Try to collect more tasks from the same model
    std::unique_lock<std::mutex> lock(queue_mutex_);

    while (batch.size() < static_cast<size_t>(config_.max_batch_size) &&
           !task_queue_.empty()) {
        const auto& next_task = task_queue_.top();

        if (next_task.model_id == first_task.model_id) {
            batch.push_back(next_task);
            task_queue_.pop();
        } else {
            break;
        }
    }

    return batch;
}

} // namespace hook_analyzer
