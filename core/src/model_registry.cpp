#include "edgepilot/model_registry.h"

#include <algorithm>

// =============================================================================
// EdgePilot — P1: Runtime & Model Infrastructure
// model_registry.cpp — Implementation of the ModelRegistry.
// =============================================================================

namespace edgepilot {

// ---------------------------------------------------------------------------
// Register
// ---------------------------------------------------------------------------

void ModelRegistry::Register(ModelMetadata metadata) {
    std::lock_guard<std::mutex> lock(mutex_);
    // Upsert: any existing entry with the same model_id is replaced.
    models_[metadata.model_id] = std::move(metadata);
}

// ---------------------------------------------------------------------------
// Unregister
// ---------------------------------------------------------------------------

void ModelRegistry::Unregister(const std::string& model_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    models_.erase(model_id);
}

// ---------------------------------------------------------------------------
// Get
// ---------------------------------------------------------------------------

std::optional<ModelMetadata>
ModelRegistry::Get(const std::string& model_id) const {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = models_.find(model_id);
    if (it == models_.end()) {
        return std::nullopt;
    }
    return it->second;
}

// ---------------------------------------------------------------------------
// List
// ---------------------------------------------------------------------------

std::vector<ModelMetadata> ModelRegistry::List() const {
    std::lock_guard<std::mutex> lock(mutex_);

    std::vector<ModelMetadata> result;
    result.reserve(models_.size());

    for (const auto& [id, meta] : models_) {
        result.push_back(meta);
    }
    return result;
}

// ---------------------------------------------------------------------------
// ListByTask
// ---------------------------------------------------------------------------

std::vector<ModelMetadata> ModelRegistry::ListByTask(TaskType task) const {
    std::lock_guard<std::mutex> lock(mutex_);

    std::vector<ModelMetadata> result;
    for (const auto& [id, meta] : models_) {
        if (meta.task_type == task) {
            result.push_back(meta);
        }
    }
    return result;
}

// ---------------------------------------------------------------------------
// ListByRuntime
// ---------------------------------------------------------------------------

std::vector<ModelMetadata>
ModelRegistry::ListByRuntime(const std::string& runtime_name) const {
    std::lock_guard<std::mutex> lock(mutex_);

    std::vector<ModelMetadata> result;
    for (const auto& [id, meta] : models_) {
        const auto& runtimes = meta.compatible_runtimes;
        if (std::find(runtimes.begin(), runtimes.end(), runtime_name) !=
            runtimes.end()) {
            result.push_back(meta);
        }
    }
    return result;
}

// ---------------------------------------------------------------------------
// Has
// ---------------------------------------------------------------------------

bool ModelRegistry::Has(const std::string& model_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return models_.count(model_id) > 0;
}

// ---------------------------------------------------------------------------
// Count
// ---------------------------------------------------------------------------

std::size_t ModelRegistry::Count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return models_.size();
}

// ---------------------------------------------------------------------------
// Clear
// ---------------------------------------------------------------------------

void ModelRegistry::Clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    models_.clear();
}

} // namespace edgepilot
