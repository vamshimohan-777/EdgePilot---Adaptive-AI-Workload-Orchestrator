#pragma once

#include <mutex>
#include <condition_variable>
#include <thread>
#include <vector>
#include <queue>
#include <functional>
#include <future>

// =============================================================================
// EdgePilot — P2: Execution Kernel
// worker_pool.h — A general-purpose thread pool for executing scheduled tasks.
//
// The Dispatcher enqueues tasks (e.g. executing model inference) to this pool
// which runs them concurrently across a configured number of worker threads.
// =============================================================================

namespace edgepilot {

class WorkerPool {
public:
    /// Constructs a worker pool with the specified number of threads.
    explicit WorkerPool(std::size_t num_threads);

    ~WorkerPool();

    // Non-copyable, non-movable
    WorkerPool(const WorkerPool&)            = delete;
    WorkerPool& operator=(const WorkerPool&) = delete;

    /// Enqueues a task for execution in the thread pool.
    /// Returns a std::future representing the eventual result of the task.
    template<class F, class... Args>
    auto Enqueue(F&& f, Args&&... args) 
        -> std::future<typename std::invoke_result<F, Args...>::type>;

    /// Stops the worker pool, joining all worker threads.
    void Stop();

    /// Returns the number of worker threads in the pool.
    std::size_t ThreadCount() const;

private:
    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;
    
    mutable std::mutex queue_mutex_;
    std::condition_variable cv_;
    bool stop_ = false;
};

// ---------------------------------------------------------------------------
// Template Implementation: Enqueue
// ---------------------------------------------------------------------------

template<class F, class... Args>
auto WorkerPool::Enqueue(F&& f, Args&&... args) 
    -> std::future<typename std::invoke_result<F, Args...>::type>
{
    using return_type = typename std::invoke_result<F, Args...>::type;

    auto task = std::make_shared<std::packaged_task<return_type()>>(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...)
    );
    
    std::future<return_type> res = task->get_future();
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        if (stop_) {
            throw std::runtime_error("Enqueue on stopped WorkerPool");
        }
        tasks_.emplace([task]() { (*task)(); });
    }
    cv_.notify_one();
    return res;
}

} // namespace edgepilot
