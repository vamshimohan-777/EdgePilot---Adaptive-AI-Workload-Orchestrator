#pragma once

#include "edgepilot/types.h"

#include <mutex>
#include <condition_variable>
#include <queue>
#include <vector>
#include <string>

// =============================================================================
// EdgePilot — P2: Execution Kernel
// job_queue.h — Thread-safe priority queue for AI inference jobs.
//
// Sorting order:
//   1. Priority (RealTime > High > Normal > Low)
//   2. Deadline (earlier deadlines take precedence)
//   3. Submission timestamp (FIFO for matching priority & deadline)
// =============================================================================

namespace edgepilot {

class JobQueue {
public:
    JobQueue()  = default;
    ~JobQueue() = default;

    // Non-copyable, non-movable
    JobQueue(const JobQueue&)            = delete;
    JobQueue& operator=(const JobQueue&) = delete;

    /// Pushes a job onto the priority queue.
    /// Thread-safe. Notifies any waiting worker threads.
    void Push(Job job);

    /// Pops the highest priority job from the queue.
    /// Thread-safe. Blocks until a job becomes available, or until Stop() is called.
    /// Returns std::nullopt if the queue is stopped and empty.
    std::optional<Job> Pop();

    /// Safely shuts down the queue. Wakes up all blocked worker threads.
    void Stop();

    /// Returns the number of jobs currently in the queue.
    /// Thread-safe.
    std::size_t Size() const;

    /// Returns true if the queue is empty.
    /// Thread-safe.
    bool IsEmpty() const;

    /// Clears all jobs from the queue.
    /// Thread-safe.
    void Clear();

private:
    struct JobComparator {
        bool operator()(const Job& lhs, const Job& rhs) const;
    };

    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::priority_queue<Job, std::vector<Job>, JobComparator> queue_;
    bool stopped_ = false;
};

} // namespace edgepilot
