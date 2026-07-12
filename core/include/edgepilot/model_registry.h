#pragma once

#include "edgepilot/types.h"

#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

// =============================================================================
// EdgePilot — P1: Runtime & Model Infrastructure
// model_registry.h — Central metadata store for all registered AI models.
//
// Design pattern: Registry
//   Models are registered by model_id. The Orchestrator, LifecycleManager,
//   and QuantizationMetadataManager query this registry for metadata without
//   any coupling to loaded model instances or runtimes.
//
// Thread-safety:
//   All public methods are thread-safe. An internal std::mutex serialises
//   all reads and writes to the internal registry map.
//
// Duplicate registration policy:
//   Calling Register() with a ModelMetadata whose model_id already exists
//   replaces the existing entry (upsert semantics).
//
// Unregister policy:
//   Unregister() on a model_id that does not exist is a no-op.
//
// This component stores METADATA ONLY — it does not load models,
// schedule inference, or interact with runtimes.
// =============================================================================

namespace edgepilot {

class ModelRegistry {
public:
    ModelRegistry()  = default;
    ~ModelRegistry() = default;

    // Non-copyable, non-movable — single authoritative registry instance.
    ModelRegistry(const ModelRegistry&)            = delete;
    ModelRegistry& operator=(const ModelRegistry&) = delete;
    ModelRegistry(ModelRegistry&&)                 = delete;
    ModelRegistry& operator=(ModelRegistry&&)      = delete;

    /// Registers model metadata under its model_id.
    /// If a model with the same model_id is already registered, it is replaced.
    /// Thread-safe.
    void Register(ModelMetadata metadata);

    /// Removes the model metadata registered under the given model_id.
    /// No-op if no model with that id exists.
    /// Thread-safe.
    void Unregister(const std::string& model_id);

    /// Returns the metadata for the given model_id, or std::nullopt if not found.
    /// Thread-safe.
    std::optional<ModelMetadata> Get(const std::string& model_id) const;

    /// Returns a snapshot of all registered model metadata entries.
    /// Thread-safe.
    std::vector<ModelMetadata> List() const;

    /// Returns all models matching the given task type.
    /// Thread-safe.
    std::vector<ModelMetadata> ListByTask(TaskType task) const;

    /// Returns all models that list the given runtime_name in their
    /// compatible_runtimes vector. Case-sensitive exact match.
    /// Thread-safe.
    std::vector<ModelMetadata> ListByRuntime(const std::string& runtime_name) const;

    /// Returns true if a model with the given id is registered.
    /// Thread-safe.
    bool Has(const std::string& model_id) const;

    /// Returns the number of registered models.
    /// Thread-safe.
    std::size_t Count() const;

    /// Removes all registered models. Used during shutdown or testing.
    /// Thread-safe.
    void Clear();

private:
    mutable std::mutex mutex_;
    std::unordered_map<std::string, ModelMetadata> models_;
};

} // namespace edgepilot
