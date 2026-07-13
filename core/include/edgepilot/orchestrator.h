#pragma once

#include "edgepilot/types.h"
#include "edgepilot/model_registry.h"
#include "edgepilot/runtime_registry.h"
#include "edgepilot/lifecycle_manager.h"
#include "edgepilot/job_queue.h"
#include "edgepilot/worker_pool.h"
#include "edgepilot/scheduler_interface.h"
#include "edgepilot/pipeline_executor.h"
#include "edgepilot/telemetry_collector.h"

#include <mutex>
#include <thread>
#include <unordered_map>
#include <string>
#include <optional>

// =============================================================================
// EdgePilot — P2: Execution Kernel
// orchestrator.h — The main control daemon for the EdgePilot framework.
//
// Responsibilities:
//   - Starts/stops the Dispatcher, WorkerPool, and TelemetryCollector
//   - Manages job submission and tracks completed job results
//   - Runs a background IPC loopback listener to communicate with the Python REST API
// =============================================================================

namespace edgepilot {

class Orchestrator {
public:
    Orchestrator(ModelRegistry& model_registry,
                 RuntimeRegistry& runtime_registry,
                 ModelLifecycleManager& lifecycle_manager,
                 IScheduler& scheduler,
                 std::size_t num_worker_threads);

    ~Orchestrator();

    // Non-copyable
    Orchestrator(const Orchestrator&)            = delete;
    Orchestrator& operator=(const Orchestrator&) = delete;

    /// Starts the orchestrator background threads:
    ///   1. Dispatcher loop
    ///   2. Telemetry collector
    ///   3. IPC socket listener (default port: 12345)
    void Start(const std::string& telemetry_csv_path, int ipc_port = 12345);

    /// Shuts down all threads, joins pools, and flushes telemetry.
    void Stop();

    /// Submits a job to the scheduling queue.
    /// Thread-safe.
    void SubmitJob(Job job);

    /// Fetches the result of a completed or failed job.
    /// Returns std::nullopt if the job is still queued or executing.
    /// Thread-safe.
    std::optional<JobResult> GetJobResult(const std::string& job_id);

    /// Helper to fetch current system state.
    SystemState GetCurrentSystemState() const;

private:
    void DispatchLoop();
    void IpcListenerLoop(int port);
    void ExecuteJob(Job job, SchedulingDecision decision);

    ModelRegistry&         model_registry_;
    RuntimeRegistry&       runtime_registry_;
    ModelLifecycleManager& lifecycle_manager_;
    IScheduler&            scheduler_;
    PipelineExecutor       pipeline_executor_;
    TelemetryCollector     telemetry_collector_;

    JobQueue               job_queue_;
    WorkerPool             worker_pool_;
    std::size_t            num_workers_;

    bool                   running_ = false;
    std::thread            dispatch_thread_;
    std::thread            ipc_thread_;

    mutable std::mutex     results_mutex_;
    std::unordered_map<std::string, JobResult> job_results_;
    std::unordered_map<std::string, JobStatus> job_statuses_;
    std::unordered_map<std::string, SchedulingDecision> job_decisions_;

    int                    ipc_socket_ = -1;
};

} // namespace edgepilot
