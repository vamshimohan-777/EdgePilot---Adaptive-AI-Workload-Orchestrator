#include "edgepilot/worker_pool.h"

// =============================================================================
// EdgePilot — P2: Execution Kernel
// worker_pool.cpp — Implementation of the WorkerPool.
// =============================================================================

namespace edgepilot {

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

WorkerPool::WorkerPool(std::size_t num_threads) {
    for (std::size_t i = 0; i < num_threads; ++i) {
        workers_.emplace_back([this]() {
            while (true) {
                std::function<void()> task;
                {
                    std::unique_lock<std::mutex> lock(this->queue_mutex_);
                    this->cv_.wait(lock, [this]() {
                        return this->stop_ || !this->tasks_.empty();
                    });

                    if (this->stop_ && this->tasks_.empty()) {
                        return;
                    }

                    task = std::move(this->tasks_.front());
                    this->tasks_.pop();
                }

                // Run the task
                task();
            }
        });
    }
}

// ---------------------------------------------------------------------------
// Destructor
// ---------------------------------------------------------------------------

WorkerPool::~WorkerPool() {
    Stop();
}

// ---------------------------------------------------------------------------
// Stop
// ---------------------------------------------------------------------------

void WorkerPool::Stop() {
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        if (stop_) {
            return;
        }
        stop_ = true;
    }
    cv_.notify_all();

    for (std::thread& worker : workers_) {
        if (worker.joinable()) {
            worker.join();
        }
    }
    workers_.clear();
}

// ---------------------------------------------------------------------------
// ThreadCount
// ---------------------------------------------------------------------------

std::size_t WorkerPool::ThreadCount() const {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    return workers_.size();
}

} // namespace edgepilot
