#include "edgepilot/quantization_manager.h"

#include <algorithm>

// =============================================================================
// EdgePilot — P1: Runtime & Model Infrastructure
// quantization_manager.cpp — Façade implementation over ModelRegistry.
// =============================================================================

namespace edgepilot {

QuantizationMetadataManager::QuantizationMetadataManager(
    const ModelRegistry& registry)
    : registry_(registry)
{
}

// ---------------------------------------------------------------------------
// GetVariants
// ---------------------------------------------------------------------------

std::vector<QuantizationVariant>
QuantizationMetadataManager::GetVariants(const std::string& model_id) const {
    auto meta = registry_.Get(model_id);
    if (!meta.has_value()) {
        return {};
    }
    return meta->quantization_variants;
}

// ---------------------------------------------------------------------------
// GetVariant
// ---------------------------------------------------------------------------

std::optional<QuantizationVariant>
QuantizationMetadataManager::GetVariant(const std::string& model_id,
                                        const std::string& variant_id) const {
    auto meta = registry_.Get(model_id);
    if (!meta.has_value()) {
        return std::nullopt;
    }

    const auto& variants = meta->quantization_variants;
    auto it = std::find_if(variants.begin(), variants.end(),
        [&variant_id](const QuantizationVariant& v) {
            return v.variant_id == variant_id;
        });

    if (it == variants.end()) {
        return std::nullopt;
    }
    return *it;
}

// ---------------------------------------------------------------------------
// GetDefault
// ---------------------------------------------------------------------------

std::optional<QuantizationVariant>
QuantizationMetadataManager::GetDefault(const std::string& model_id) const {
    auto meta = registry_.Get(model_id);
    if (!meta.has_value() || meta->quantization_variants.empty()) {
        return std::nullopt;
    }
    // Default = first variant in the list.
    return meta->quantization_variants.front();
}

// ---------------------------------------------------------------------------
// HasVariants
// ---------------------------------------------------------------------------

bool QuantizationMetadataManager::HasVariants(const std::string& model_id) const {
    auto meta = registry_.Get(model_id);
    if (!meta.has_value()) {
        return false;
    }
    return !meta->quantization_variants.empty();
}

} // namespace edgepilot
