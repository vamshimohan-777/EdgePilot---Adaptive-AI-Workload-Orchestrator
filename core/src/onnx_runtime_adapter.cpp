#include "edgepilot/onnx_runtime_adapter.h"

#include <algorithm>
#include <chrono>

// =============================================================================
// EdgePilot — P1: Runtime & Model Infrastructure
// onnx_runtime_adapter.cpp — Stub implementation of the ONNX Runtime adapter.
//
// Every method returns valid, predictable values so that the full lifecycle
// (Initialize → Register → LoadModel → RunInference → Unload → Shutdown)
// can be exercised and tested without linking the real ONNX Runtime library.
//
// TODO: Replace stub bodies with real onnxruntime.h calls:
//   - Initialize(): create Ort::Env, configure logging
//   - LoadModel():  create Ort::Session from model file path
//   - RunInference(): run Ort::Session::Run with input/output tensors
//   - Unload(): release Ort::Session
//   - Shutdown(): release Ort::Env
// =============================================================================

namespace edgepilot {

// ===========================================================================
// OnnxActiveModel — Stub
// ===========================================================================

OnnxActiveModel::OnnxActiveModel(const std::string& model_id)
    : model_id_(model_id)
    , status_(ModelStatus::Loaded)
{
}

OnnxActiveModel::~OnnxActiveModel() {
    // Safety-net: release resources if caller forgot to call Unload().
    if (status_ == ModelStatus::Loaded) {
        Unload();
    }
}

std::string OnnxActiveModel::GetModelId() const {
    return model_id_;
}

ModelStatus OnnxActiveModel::GetStatus() const {
    return status_;
}

InferenceResult OnnxActiveModel::RunInference(const InferenceRequest& request) {
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
    // In the real implementation this calls Ort::Session::Run().
    // For now, echo back a stub tensor output and any text inputs.

    // Echo tensor inputs as outputs (useful for round-trip testing).
    result.tensor_outputs = request.tensor_inputs;

    // Echo text inputs as text outputs with a "[stub]" prefix.
    for (const auto& [key, value] : request.text_inputs) {
        result.text_outputs[key] = "[onnx-stub] " + value;
    }

    result.success = true;

    auto end = std::chrono::steady_clock::now();
    result.latency_us = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());

    return result;
}

bool OnnxActiveModel::Unload() {
    if (status_ == ModelStatus::NotLoaded) {
        // Already unloaded — idempotent no-op.
        return true;
    }

    status_ = ModelStatus::Unloading;

    // --- Stub unload logic ---
    // In the real implementation this releases the Ort::Session.

    status_ = ModelStatus::NotLoaded;
    return true;
}


// ===========================================================================
// OnnxRuntimeAdapter — Stub
// ===========================================================================

OnnxRuntimeAdapter::~OnnxRuntimeAdapter() {
    if (initialized_) {
        Shutdown();
    }
}

std::string OnnxRuntimeAdapter::GetName() const {
    return "onnx";
}

RuntimeCapabilities OnnxRuntimeAdapter::GetCapabilities() const {
    RuntimeCapabilities caps;
    caps.runtime_name           = "onnx";
    caps.runtime_version        = "1.18.0-stub";
    caps.supported_model_formats = {"onnx"};
    caps.supported_precisions    = {Precision::FP32, Precision::FP16, Precision::INT8};
    caps.supported_devices       = {HardwareDevice::CPU, HardwareDevice::GPU};
    caps.supports_streaming      = false;
    caps.extra_info["max_batch_size"] = "64";
    return caps;
}

bool OnnxRuntimeAdapter::IsModelCompatible(const ModelMetadata& metadata) const {
    // Check if "onnx" is listed in the model's compatible runtimes.
    const auto& runtimes = metadata.compatible_runtimes;
    return std::find(runtimes.begin(), runtimes.end(), "onnx") != runtimes.end();
}

std::string OnnxRuntimeAdapter::Initialize() {
    if (initialized_) {
        return "";  // Already initialized — idempotent.
    }

    // --- Stub initialization ---
    // In the real implementation this creates Ort::Env and session options.

    initialized_ = true;
    return "";  // Empty string = success.
}

void OnnxRuntimeAdapter::Shutdown() {
    if (!initialized_) {
        return;
    }

    // --- Stub shutdown ---
    // In the real implementation this releases Ort::Env.

    initialized_ = false;
}

std::optional<std::shared_ptr<IActiveModel>>
OnnxRuntimeAdapter::LoadModel(const ModelMetadata& metadata,
                              const DeviceConfig& /*config*/) {
    if (!initialized_) {
        return std::nullopt;  // Runtime not initialized.
    }

    if (!IsModelCompatible(metadata)) {
        return std::nullopt;  // Model format not supported by this runtime.
    }

    // --- Stub model loading ---
    // In the real implementation this creates an Ort::Session from the model
    // file specified in metadata.default_file_path (or the quantization
    // variant file if config.quantization_variant_id is set).

    auto model = std::make_shared<OnnxActiveModel>(metadata.model_id);
    return model;
}

} // namespace edgepilot
