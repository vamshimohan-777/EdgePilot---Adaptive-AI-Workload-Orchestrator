#pragma once

#include "edgepilot/runtime_interface.h"

#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

// =============================================================================
// EdgePilot — P1: Runtime & Model Infrastructure
// runtime_registry.h — Central registry for all registered inference runtimes.
//
// Design pattern: Registry
//   Runtimes are registered by name at startup. The Orchestrator and Lifecycle
//   Manager look up runtimes by name without any coupling to concrete types.
//
// Thread-safety:
//   All public methods are thread-safe. An internal std::mutex serialises all
//   reads and writes to the internal registry map.
//
// Duplicate registration policy:
//   Calling Register() with a runtime whose name already exists replaces the
//   existing entry (upsert semantics). The replaced runtime's Shutdown() is
//   NOT called automatically — the caller is responsible for shutting down
//   the old runtime before re-registering.
//
// Unregister policy:
//   Unregister() on a name that does not exist is a no-op (no exception thrown).
// =============================================================================

namespace edgepilot {

class RuntimeRegistry {
public:
    RuntimeRegistry()  = default;
    ~RuntimeRegistry() = default;

    // Non-copyable, non-movable — single authoritative registry instance.
    RuntimeRegistry(const RuntimeRegistry&)            = delete;
    RuntimeRegistry& operator=(const RuntimeRegistry&) = delete;
    RuntimeRegistry(RuntimeRegistry&&)                 = delete;
    RuntimeRegistry& operator=(RuntimeRegistry&&)      = delete;

    /// Registers a runtime under its GetName() key.
    /// If a runtime with the same name is already registered, it is replaced.
    /// Thread-safe.
    void Register(std::shared_ptr<IInferenceRuntime> runtime);

    /// Removes the runtime registered under the given name.
    /// No-op if no runtime with that name exists.
    /// Thread-safe.
    void Unregister(const std::string& name);

    /// Returns the runtime registered under the given name, or std::nullopt
    /// if no runtime with that name has been registered.
    /// Thread-safe.
    std::optional<std::shared_ptr<IInferenceRuntime>>
        Get(const std::string& name) const;

    /// Returns a snapshot of all currently registered runtimes.
    /// The snapshot is a copy — modifications to the registry after this call
    /// are not reflected in the returned vector.
    /// Thread-safe.
    std::vector<std::shared_ptr<IInferenceRuntime>> GetAll() const;

    /// Returns true if a runtime with the given name is registered.
    /// Thread-safe.
    bool Has(const std::string& name) const;

    /// Returns the number of registered runtimes.
    /// Thread-safe.
    std::size_t Count() const;

    /// Removes all registered runtimes. Used during shutdown or testing.
    /// Thread-safe.
    void Clear();

private:
    mutable std::mutex mutex_;
    std::unordered_map<std::string, std::shared_ptr<IInferenceRuntime>> runtimes_;
};

} // namespace edgepilot
