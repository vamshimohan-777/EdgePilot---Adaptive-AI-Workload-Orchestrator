#include "edgepilot/runtime_registry.h"

#include <stdexcept>

// =============================================================================
// EdgePilot — P1: Runtime & Model Infrastructure
// runtime_registry.cpp — Implementation of the RuntimeRegistry.
// =============================================================================

namespace edgepilot {

// ---------------------------------------------------------------------------
// Register
// ---------------------------------------------------------------------------

void RuntimeRegistry::Register(std::shared_ptr<IInferenceRuntime> runtime) {
    if (!runtime) {
        // Silently reject null registrations — callers that pass nullptr have
        // a bug, but we do not throw here to keep the registry robust.
        return;
    }

    const std::string name = runtime->GetName();

    std::lock_guard<std::mutex> lock(mutex_);
    // Upsert: any existing entry is silently replaced.
    runtimes_[name] = std::move(runtime);
}

// ---------------------------------------------------------------------------
// Unregister
// ---------------------------------------------------------------------------

void RuntimeRegistry::Unregister(const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex_);
    // erase() is a no-op if the key does not exist — matches our contract.
    runtimes_.erase(name);
}

// ---------------------------------------------------------------------------
// Get
// ---------------------------------------------------------------------------

std::optional<std::shared_ptr<IInferenceRuntime>>
RuntimeRegistry::Get(const std::string& name) const {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = runtimes_.find(name);
    if (it == runtimes_.end()) {
        return std::nullopt;
    }
    return it->second;
}

// ---------------------------------------------------------------------------
// GetAll
// ---------------------------------------------------------------------------

std::vector<std::shared_ptr<IInferenceRuntime>>
RuntimeRegistry::GetAll() const {
    std::lock_guard<std::mutex> lock(mutex_);

    std::vector<std::shared_ptr<IInferenceRuntime>> result;
    result.reserve(runtimes_.size());

    for (const auto& [name, runtime] : runtimes_) {
        result.push_back(runtime);
    }
    return result;
}

// ---------------------------------------------------------------------------
// Has
// ---------------------------------------------------------------------------

bool RuntimeRegistry::Has(const std::string& name) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return runtimes_.count(name) > 0;
}

// ---------------------------------------------------------------------------
// Count
// ---------------------------------------------------------------------------

std::size_t RuntimeRegistry::Count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return runtimes_.size();
}

// ---------------------------------------------------------------------------
// Clear
// ---------------------------------------------------------------------------

void RuntimeRegistry::Clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    runtimes_.clear();
}

} // namespace edgepilot
