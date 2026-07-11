#include "edgepilot/llama_cpp_adapter.h"

#include <algorithm>
#include <chrono>

// =============================================================================
// EdgePilot — P1: Runtime & Model Infrastructure
// llama_cpp_adapter.cpp — Stub implementation of the llama.cpp / GGUF adapter.
//
// Every method returns valid, predictable values so that the full lifecycle
// (Initialize → Register → LoadModel → RunInference → Unload → Shutdown)
// can be exercised and tested without linking the real llama.cpp library.
//
// TODO: Replace stub bodies with real llama.h calls:
//   - Initialize(): call llama_backend_init()
//   - LoadModel():  call llama_load_model_from_file(), llama_new_context_with_model()
//   - RunInference(): run the llama.cpp sampling loop (llama_decode, llama_sampler)
//   - Unload(): call llama_free(), llama_free_model()
//   - Shutdown(): call llama_backend_free()
// =============================================================================

namespace edgepilot {

// ===========================================================================
// LlamaCppActiveModel — Stub
// ===========================================================================

LlamaCppActiveModel::LlamaCppActiveModel(const std::string& model_id)
    : model_id_(model_id)
    , status_(ModelStatus::Loaded)
{
}

LlamaCppActiveModel::~LlamaCppActiveModel() {
    // Safety-net: release resources if caller forgot to call Unload().
    if (status_ == ModelStatus::Loaded) {
        Unload();
    }
}

std::string LlamaCppActiveModel::GetModelId() const {
    return model_id_;
}

ModelStatus LlamaCppActiveModel::GetStatus() const {
    return status_;
}

InferenceResult LlamaCppActiveModel::RunInference(const InferenceRequest& request) {
    InferenceResult result;
    result.request_id = request.request_id;

    // Guard: cannot run inference on an unloaded model.
    if (status_ != ModelStatus::Loaded) {
        result.success       = false;
        result.error_message = "Model '" + model_id_ + "' is not in Loaded state";
        return result;
    }

    auto start = std::chrono::steady_clock::now();

    // --- Stub inference logic ---
    // In the real implementation this runs the llama.cpp sampling loop.
    // For now, produce a predictable text response for testing.

    // If text_inputs has a "prompt" key, echo it with a stub completion.
    auto it = request.text_inputs.find("prompt");
    if (it != request.text_inputs.end()) {
        result.text_outputs["generated_text"] =
            "[llama-stub] Response to: " + it->second;
    } else {
        result.text_outputs["generated_text"] =
            "[llama-stub] No prompt provided";
    }

    // Echo any tensor inputs back as outputs (for round-trip testing).
    result.tensor_outputs = request.tensor_inputs;

    result.success = true;

    auto end = std::chrono::steady_clock::now();
    result.latency_us = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());

    return result;
}

bool LlamaCppActiveModel::Unload() {
    if (status_ == ModelStatus::NotLoaded) {
        // Already unloaded — idempotent no-op.
        return true;
    }

    status_ = ModelStatus::Unloading;

    // --- Stub unload logic ---
    // In the real implementation this calls llama_free() and llama_free_model().

    status_ = ModelStatus::NotLoaded;
    return true;
}


// ===========================================================================
// LlamaCppAdapter — Stub
// ===========================================================================

LlamaCppAdapter::~LlamaCppAdapter() {
    if (initialized_) {
        Shutdown();
    }
}

std::string LlamaCppAdapter::GetName() const {
    return "llama_cpp";
}

RuntimeCapabilities LlamaCppAdapter::GetCapabilities() const {
    RuntimeCapabilities caps;
    caps.runtime_name            = "llama_cpp";
    caps.runtime_version         = "b3000-stub";
    caps.supported_model_formats = {"gguf"};
    caps.supported_precisions    = {Precision::FP16, Precision::INT8, Precision::INT4};
    caps.supported_devices       = {HardwareDevice::CPU, HardwareDevice::GPU};
    caps.supports_streaming      = true;
    caps.extra_info["max_context_length"] = "4096";
    return caps;
}

bool LlamaCppAdapter::IsModelCompatible(const ModelMetadata& metadata) const {
    // Check if "llama_cpp" is listed in the model's compatible runtimes.
    const auto& runtimes = metadata.compatible_runtimes;
    return std::find(runtimes.begin(), runtimes.end(), "llama_cpp") != runtimes.end();
}

std::string LlamaCppAdapter::Initialize() {
    if (initialized_) {
        return "";  // Already initialized — idempotent.
    }

    // --- Stub initialization ---
    // In the real implementation this calls llama_backend_init().

    initialized_ = true;
    return "";  // Empty string = success.
}

void LlamaCppAdapter::Shutdown() {
    if (!initialized_) {
        return;
    }

    // --- Stub shutdown ---
    // In the real implementation this calls llama_backend_free().

    initialized_ = false;
}

std::optional<std::shared_ptr<IActiveModel>>
LlamaCppAdapter::LoadModel(const ModelMetadata& metadata,
                           const DeviceConfig& /*config*/) {
    if (!initialized_) {
        return std::nullopt;  // Runtime not initialized.
    }

    if (!IsModelCompatible(metadata)) {
        return std::nullopt;  // Model format not supported by this runtime.
    }

    // --- Stub model loading ---
    // In the real implementation this calls llama_load_model_from_file()
    // and llama_new_context_with_model().

    auto model = std::make_shared<LlamaCppActiveModel>(metadata.model_id);
    return model;
}

} // namespace edgepilot
