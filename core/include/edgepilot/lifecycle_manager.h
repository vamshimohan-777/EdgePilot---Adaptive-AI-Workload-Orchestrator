#pragma once

#include "edgepilot/model_registry.h"
#include "edgepilot/runtime_registry.h"
#include "edgepilot/runtime_interface.h"

#include <condition_variable>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>
#include <thread>
#include <chrono>
#include <atomic>

// =============================================================================
// EdgePilot — P1: Runtime & Model Infrastructure
// lifecycle_manager.h — Coordinates model loading, unloading, caching, and
//                        lifecycle state tracking.
//
// Design:
//   The ModelLifecycleManager is the single point of control for loading and
//   unloading model instances. It coordinates between the ModelRegistry
//   (metadata lookup) and RuntimeRegistry (runtime resolution) to produce
//   IActiveModel handles for the Orchestrator.
//
// Cache behaviour:
//   - Loaded models are cached by model_id.
//   - Load() on an already-loaded model returns the cached handle (idempotent).
//   - Reload() forces an unload + fresh load.
//
// Concurrency:
//   - All public methods are thread-safe via cache_mutex_.
//   - Unload() waits for in-flight RunInference calls to complete before
//     proceeding, tracked via per-model reference counters and a condition
//     variable. The Orchestrator must call AcquireInference/ReleaseInference
//     around RunInference to enable this safety mechanism.
//
// Dependency injection:
//   Constructor-injected with ModelRegistry& and RuntimeRegistry&.
//
// Does NOT do:
//   - Scheduling decisions (when to load/unload — that's the AI Scheduler)
//   - Eviction policy (that's the AI Scheduler)
//   - Telemetry emission (P1 populates latency_us; Telemetry reads it)
// =============================================================================

namespace edgepilot {

class ModelLifecycleManager {
public:
    ModelLifecycleManager(ModelRegistry& model_registry,
                          RuntimeRegistry& runtime_registry);

    ~ModelLifecycleManager();

    // Non-copyable, non-movable.
    ModelLifecycleManager(const ModelLifecycleManager&)            = delete;
    ModelLifecycleManager& operator=(const ModelLifecycleManager&) = delete;
    ModelLifecycleManager(ModelLifecycleManager&&)                 = delete;
    ModelLifecycleManager& operator=(ModelLifecycleManager&&)      = delete;

    /// Loads a model and returns an active handle.
    ///
    /// @param model_id       The model to load (must exist in ModelRegistry).
    /// @param runtime_name   The runtime to use. If empty, falls back to
    ///                       ModelMetadata::preferred_runtime, then to the
    ///                       first entry in compatible_runtimes.
    /// @param config         Device and quantization configuration.
    ///
    /// If the model is already loaded (cache hit), returns the existing handle.
    /// Returns std::nullopt on failure (model not found, runtime not found,
    /// or LoadModel failed).
    /// Thread-safe.
    std::optional<std::shared_ptr<IActiveModel>>
        Load(const std::string& model_id,
             const std::string& runtime_name,
             const DeviceConfig& config);

    /// Unloads a model and removes it from the cache.
    ///
    /// Waits for any in-flight inference calls (tracked via AcquireInference/
    /// ReleaseInference) to complete before calling IActiveModel::Unload().
    ///
    /// Returns false if the model is not loaded or unload fails.
    /// Thread-safe.
    bool Unload(const std::string& model_id);

    /// Reloads a model: equivalent to Unload() + Load().
    /// Thread-safe.
    std::optional<std::shared_ptr<IActiveModel>>
        Reload(const std::string& model_id,
               const std::string& runtime_name,
               const DeviceConfig& config);

    /// Returns true if the model is currently in Loaded state.
    /// Thread-safe.
    bool IsLoaded(const std::string& model_id) const;

    /// Returns the lifecycle status of a model.
    /// Returns ModelStatus::NotLoaded if the model is not in the cache.
    /// Thread-safe.
    ModelStatus QueryStatus(const std::string& model_id) const;

    /// Returns the cached active model handle, or std::nullopt if not loaded.
    /// Thread-safe.
    std::optional<std::shared_ptr<IActiveModel>>
        GetHandle(const std::string& model_id) const;

    /// Returns the model_ids of all currently loaded models.
    /// Thread-safe.
    std::vector<std::string> GetAllLoaded() const;

    /// Must be called before RunInference on a model to prevent unload races.
    /// Increments the per-model inference counter.
    /// Thread-safe.
    void AcquireInference(const std::string& model_id);

    /// Must be called after RunInference on a model.
    /// Decrements the per-model inference counter and notifies waiting Unload().
    /// Thread-safe.
    void ReleaseInference(const std::string& model_id);

private:
    /// Resolves the runtime name from inputs and metadata fallback.
    std::string ResolveRuntimeName(const std::string& runtime_name,
                                   const ModelMetadata& metadata) const;

    void StartMonitor();
    void StopMonitor();
    void MonitorLoop();

    ModelRegistry&   model_registry_;
    RuntimeRegistry& runtime_registry_;

    mutable std::mutex cache_mutex_;
    std::condition_variable inference_cv_;

    /// Cached loaded model instances, keyed by model_id.
    std::unordered_map<std::string, std::shared_ptr<IActiveModel>> loaded_models_;

    /// Per-model status tracking (for models in transitional states).
    std::unordered_map<std::string, ModelStatus> status_map_;

    /// Per-model active inference call count (for safe unload).
    std::unordered_map<std::string, int> inference_count_;

    /// Background monitor thread for idle model eviction (sleep)
    std::thread monitor_thread_;
    std::atomic<bool> monitor_running_{false};
    std::condition_variable monitor_cv_;
    std::unordered_map<std::string, std::chrono::steady_clock::time_point> last_used_time_;
};

} // namespace edgepilot
