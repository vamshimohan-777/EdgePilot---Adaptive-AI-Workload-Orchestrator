#include "edgepilot/lifecycle_manager.h"
#include <iostream>

// =============================================================================
// EdgePilot — P1: Runtime & Model Infrastructure
// lifecycle_manager.cpp — Implementation of the ModelLifecycleManager.
// =============================================================================

namespace edgepilot {

ModelLifecycleManager::ModelLifecycleManager(
    ModelRegistry& model_registry,
    RuntimeRegistry& runtime_registry)
    : model_registry_(model_registry)
    , runtime_registry_(runtime_registry)
{
    StartMonitor();
}

ModelLifecycleManager::~ModelLifecycleManager() {
    StopMonitor();
}

// ---------------------------------------------------------------------------
// ResolveRuntimeName (private helper)
// ---------------------------------------------------------------------------

std::string ModelLifecycleManager::ResolveRuntimeName(
    const std::string& runtime_name,
    const ModelMetadata& metadata) const
{
    // 1. Caller-specified runtime name takes priority.
    if (!runtime_name.empty()) {
        return runtime_name;
    }

    // 2. Fall back to model's preferred runtime.
    if (!metadata.preferred_runtime.empty()) {
        return metadata.preferred_runtime;
    }

    // 3. Fall back to first compatible runtime listed in metadata.
    if (!metadata.compatible_runtimes.empty()) {
        return metadata.compatible_runtimes.front();
    }

    // 4. No runtime available.
    return "";
}

// ---------------------------------------------------------------------------
// Load
// ---------------------------------------------------------------------------

std::optional<std::shared_ptr<IActiveModel>>
ModelLifecycleManager::Load(const std::string& model_id,
                            const std::string& runtime_name,
                            const DeviceConfig& config)
{
    std::unique_lock<std::mutex> lock(cache_mutex_);

    // Cache hit: model already loaded — return existing handle.
    auto cached_it = loaded_models_.find(model_id);
    if (cached_it != loaded_models_.end()) {
        return cached_it->second;
    }

    // Look up model metadata.
    auto meta_opt = model_registry_.Get(model_id);
    if (!meta_opt.has_value()) {
        // Model not found in registry.
        return std::nullopt;
    }
    const auto& metadata = meta_opt.value();

    // Resolve which runtime to use.
    std::string resolved_runtime = ResolveRuntimeName(runtime_name, metadata);
    if (resolved_runtime.empty()) {
        // No runtime could be resolved.
        return std::nullopt;
    }

    // Look up the runtime.
    auto runtime_opt = runtime_registry_.Get(resolved_runtime);
    if (!runtime_opt.has_value()) {
        // Runtime not registered.
        return std::nullopt;
    }
    auto& runtime = runtime_opt.value();

    // Mark status as Loading.
    status_map_[model_id] = ModelStatus::Loading;

    // Release the lock during the potentially slow LoadModel() call.
    lock.unlock();

    auto model_opt = runtime->LoadModel(metadata, config);

    lock.lock();

    if (!model_opt.has_value()) {
        // LoadModel failed.
        status_map_[model_id] = ModelStatus::Error;
        return std::nullopt;
    }

    // Store in cache and update status.
    loaded_models_[model_id] = model_opt.value();
    status_map_[model_id]    = ModelStatus::Loaded;
    inference_count_[model_id] = 0;
    last_used_time_[model_id] = std::chrono::steady_clock::now();

    return model_opt.value();
}

// ---------------------------------------------------------------------------
// Unload
// ---------------------------------------------------------------------------

bool ModelLifecycleManager::Unload(const std::string& model_id) {
    std::unique_lock<std::mutex> lock(cache_mutex_);

    auto it = loaded_models_.find(model_id);
    if (it == loaded_models_.end()) {
        // Not loaded — nothing to do.
        return false;
    }

    status_map_[model_id] = ModelStatus::Unloading;

    // Wait for any in-flight inference calls to complete.
    inference_cv_.wait(lock, [this, &model_id]() {
        auto count_it = inference_count_.find(model_id);
        return count_it == inference_count_.end() || count_it->second == 0;
    });

    // Call Unload() on the model instance.
    auto model = it->second;
    bool ok = model->Unload();

    if (!ok) {
        status_map_[model_id] = ModelStatus::Error;
        return false;
    }

    // Remove from cache.
    loaded_models_.erase(it);
    status_map_[model_id] = ModelStatus::NotLoaded;
    inference_count_.erase(model_id);
    last_used_time_.erase(model_id);

    return true;
}

// ---------------------------------------------------------------------------
// Reload
// ---------------------------------------------------------------------------

std::optional<std::shared_ptr<IActiveModel>>
ModelLifecycleManager::Reload(const std::string& model_id,
                              const std::string& runtime_name,
                              const DeviceConfig& config)
{
    // Unload first (ignore failure — model may not have been loaded).
    Unload(model_id);

    // Load fresh.
    return Load(model_id, runtime_name, config);
}

// ---------------------------------------------------------------------------
// IsLoaded
// ---------------------------------------------------------------------------

bool ModelLifecycleManager::IsLoaded(const std::string& model_id) const {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    return loaded_models_.count(model_id) > 0;
}

// ---------------------------------------------------------------------------
// QueryStatus
// ---------------------------------------------------------------------------

ModelStatus
ModelLifecycleManager::QueryStatus(const std::string& model_id) const {
    std::lock_guard<std::mutex> lock(cache_mutex_);

    auto it = status_map_.find(model_id);
    if (it == status_map_.end()) {
        return ModelStatus::NotLoaded;
    }
    return it->second;
}

// ---------------------------------------------------------------------------
// GetHandle
// ---------------------------------------------------------------------------

std::optional<std::shared_ptr<IActiveModel>>
ModelLifecycleManager::GetHandle(const std::string& model_id) const {
    std::lock_guard<std::mutex> lock(cache_mutex_);

    auto it = loaded_models_.find(model_id);
    if (it == loaded_models_.end()) {
        return std::nullopt;
    }
    return it->second;
}

// ---------------------------------------------------------------------------
// GetAllLoaded
// ---------------------------------------------------------------------------

std::vector<std::string> ModelLifecycleManager::GetAllLoaded() const {
    std::lock_guard<std::mutex> lock(cache_mutex_);

    std::vector<std::string> result;
    result.reserve(loaded_models_.size());

    for (const auto& [id, model] : loaded_models_) {
        result.push_back(id);
    }
    return result;
}

// ---------------------------------------------------------------------------
// AcquireInference / ReleaseInference
// ---------------------------------------------------------------------------

void ModelLifecycleManager::AcquireInference(const std::string& model_id) {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    inference_count_[model_id]++;
    last_used_time_[model_id] = std::chrono::steady_clock::now();
}

void ModelLifecycleManager::ReleaseInference(const std::string& model_id) {
    {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        auto it = inference_count_.find(model_id);
        if (it != inference_count_.end() && it->second > 0) {
            it->second--;
        }
        last_used_time_[model_id] = std::chrono::steady_clock::now();
    }
    // Notify waiting Unload() calls.
    inference_cv_.notify_all();
}

void ModelLifecycleManager::StartMonitor() {
    monitor_running_ = true;
    monitor_thread_ = std::thread(&ModelLifecycleManager::MonitorLoop, this);
}

void ModelLifecycleManager::StopMonitor() {
    if (monitor_running_) {
        monitor_running_ = false;
        monitor_cv_.notify_all();
        if (monitor_thread_.joinable()) {
            monitor_thread_.join();
        }
    }
}

void ModelLifecycleManager::MonitorLoop() {
    while (monitor_running_) {
        std::unique_lock<std::mutex> lock(cache_mutex_);
        // Wait on monitor_cv_ or timeout of 1 second
        monitor_cv_.wait_for(lock, std::chrono::seconds(1), [this]() {
            return !monitor_running_;
        });

        if (!monitor_running_) {
            break;
        }

        auto now = std::chrono::steady_clock::now();
        std::vector<std::string> to_evict;

        for (const auto& [model_id, active_model] : loaded_models_) {
            // Only evict if inference_count is 0
            auto count_it = inference_count_.find(model_id);
            if (count_it != inference_count_.end() && count_it->second > 0) {
                continue;
            }

            auto time_it = last_used_time_.find(model_id);
            if (time_it != last_used_time_.end()) {
                auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - time_it->second).count();
                if (elapsed >= 5) {
                    to_evict.push_back(model_id);
                }
            }
        }

        lock.unlock();

        for (const auto& model_id : to_evict) {
            // Re-lock to verify model is still idle and hasn't had new activity
            std::unique_lock<std::mutex> inner_lock(cache_mutex_);
            auto count_it = inference_count_.find(model_id);
            if (count_it != inference_count_.end() && count_it->second > 0) {
                continue;
            }
            auto time_it = last_used_time_.find(model_id);
            if (time_it != last_used_time_.end()) {
                auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - time_it->second).count();
                if (elapsed < 5) {
                    continue;
                }
            }
            inner_lock.unlock();

            std::cout << "[EdgePilot Daemon] Model '" << model_id << "' has been idle for 5 seconds. Transitioning to sleep (unloaded)." << std::endl;
            Unload(model_id);
        }
    }
}

} // namespace edgepilot
