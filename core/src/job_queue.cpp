#include "edgepilot/job_queue.h"

// =============================================================================
// EdgePilot — P2: Execution Kernel
// job_queue.cpp — Implementation of the JobQueue.
// =============================================================================

namespace edgepilot {

// ---------------------------------------------------------------------------
// JobComparator
// ---------------------------------------------------------------------------

bool JobQueue::JobComparator::operator()(const Job& lhs, const Job& rhs) const {
    // 1. Compare Priorities (higher value = higher priority)
    // RealTime (3) > High (2) > Normal (1) > Low (0)
    int l_pri = static_cast<int>(lhs.priority);
    int r_pri = static_cast<int>(rhs.priority);
    if (l_pri != r_pri) {
        return l_pri < r_pri; // lower priority goes to the back
    }

    // 2. Compare Deadlines (earlier deadline = processed first)
    // If one job has a deadline and the other doesn't, the deadline job goes first.
    if (lhs.deadline_at_us != rhs.deadline_at_us) {
        if (lhs.deadline_at_us == 0) return true;  // lhs has no deadline, rhs does -> lhs is "less" (goes back)
        if (rhs.deadline_at_us == 0) return false; // lhs has deadline, rhs doesn't -> lhs is "greater" (goes front)
        return lhs.deadline_at_us > rhs.deadline_at_us; // smaller deadline is greater priority
    }

    // 3. Compare Submission Timestamps (FIFO fallback: earlier = processed first)
    return lhs.submitted_at_us > rhs.submitted_at_us; // smaller timestamp is greater priority
}

// ---------------------------------------------------------------------------
// Push
// ---------------------------------------------------------------------------

void JobQueue::Push(Job job) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (stopped_) {
            return;
        }
        queue_.push(std::move(job));
    }
    // Wake up one worker thread
    cv_.notify_one();
}

// ---------------------------------------------------------------------------
// Pop
// ---------------------------------------------------------------------------

std::optional<Job> JobQueue::Pop() {
    std::unique_lock<std::mutex> lock(mutex_);
    
    // Block until queue has items or is stopped.
    cv_.wait(lock, [this]() {
        return !queue_.empty() || stopped_;
    });

    if (queue_.empty() && stopped_) {
        return std::nullopt;
    }

    Job job = std::move(const_cast<Job&>(queue_.top()));
    queue_.pop();
    return job;
}

// ---------------------------------------------------------------------------
// Stop
// ---------------------------------------------------------------------------

void JobQueue::Stop() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stopped_ = true;
    }
    // Wake up all blocked threads so they can exit gracefully.
    cv_.notify_all();
}

// ---------------------------------------------------------------------------
// Size
// ---------------------------------------------------------------------------

std::size_t JobQueue::Size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.size();
}

// ---------------------------------------------------------------------------
// IsEmpty
// ---------------------------------------------------------------------------

bool JobQueue::IsEmpty() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.empty();
}

// ---------------------------------------------------------------------------
// Clear
// ---------------------------------------------------------------------------

void JobQueue::Clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    while (!queue_.empty()) {
        queue_.pop();
    }
}

} // namespace edgepilot
