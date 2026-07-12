#pragma once

#include "edgepilot/model_registry.h"

#include <optional>
#include <string>
#include <vector>

// =============================================================================
// EdgePilot — P1: Runtime & Model Infrastructure
// quantization_manager.h — Façade over ModelRegistry for quantization queries.
//
// Design pattern: Façade
//   Provides a simplified, focused API for querying quantization variants
//   without exposing the full ModelRegistry interface. Does not own data —
//   all queries are delegated to the underlying ModelRegistry.
//
// Thread-safety:
//   Thread-safe — delegates to ModelRegistry which is internally locked.
//
// Dependency injection:
//   Constructed with a const reference to ModelRegistry.
//   This enables mock injection in unit tests.
// =============================================================================

namespace edgepilot {

class QuantizationMetadataManager {
public:
    /// Constructs the manager with a reference to the ModelRegistry.
    /// The registry must outlive this manager.
    explicit QuantizationMetadataManager(const ModelRegistry& registry);

    ~QuantizationMetadataManager() = default;

    // Non-copyable, movable.
    QuantizationMetadataManager(const QuantizationMetadataManager&)            = delete;
    QuantizationMetadataManager& operator=(const QuantizationMetadataManager&) = delete;
    QuantizationMetadataManager(QuantizationMetadataManager&&)                 = default;
    QuantizationMetadataManager& operator=(QuantizationMetadataManager&&)      = default;

    /// Returns all quantization variants available for the given model.
    /// Returns an empty vector if the model is not found or has no variants.
    std::vector<QuantizationVariant> GetVariants(const std::string& model_id) const;

    /// Returns a specific variant by model_id and variant_id.
    /// Returns std::nullopt if the model or variant is not found.
    std::optional<QuantizationVariant>
        GetVariant(const std::string& model_id, const std::string& variant_id) const;

    /// Returns the first (default) quantization variant for the given model.
    /// Returns std::nullopt if the model has no variants.
    std::optional<QuantizationVariant> GetDefault(const std::string& model_id) const;

    /// Returns true if the given model has any quantization variants registered.
    bool HasVariants(const std::string& model_id) const;

private:
    const ModelRegistry& registry_;
};

} // namespace edgepilot
