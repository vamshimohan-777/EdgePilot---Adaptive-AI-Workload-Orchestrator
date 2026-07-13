#pragma once

#include "edgepilot/runtime_interface.h"

#include <memory>
#include <optional>
#include <string>
#include <mutex>

// =============================================================================
// EdgePilot — P1: Runtime & Model Infrastructure
// onnx_runtime_adapter.h — Adapter that wraps ONNX Runtime behind the
//                           IInferenceRuntime interface.
//
// Design pattern: Adapter + Factory Method
//   Adapts the ONNX Runtime C++ API to the EdgePilot IInferenceRuntime
//   interface. LoadModel() acts as a Factory Method, producing
//   OnnxActiveModel instances that implement IActiveModel.
//
// Current status: STUB implementation.
//   All methods return valid, predictable values for lifecycle and
//   integration testing. No actual ONNX Runtime library is linked.
//   Replace stub bodies with real onnxruntime.h calls when integrating.
// =============================================================================

namespace edgepilot {

// ---------------------------------------------------------------------------
// OnnxActiveModel — Stub loaded model handle
// ---------------------------------------------------------------------------

/// Stub implementation of IActiveModel for the ONNX Runtime adapter.
/// Simulates a loaded ONNX model with predictable inference results.
class OnnxActiveModel : public IActiveModel {
public:
    explicit OnnxActiveModel(const std::string& model_id);
    ~OnnxActiveModel() override;

    std::string     GetModelId() const override;
    ModelStatus     GetStatus()  const override;
    InferenceResult RunInference(const InferenceRequest& request) override;
    bool            Unload() override;

private:
    friend class OnnxRuntimeAdapter;
    std::string  model_id_;
    ModelStatus  status_;
    void*        session_ = nullptr;
    std::mutex   inference_mutex_;
};


// ---------------------------------------------------------------------------
// OnnxRuntimeAdapter — IInferenceRuntime implementation for ONNX
// ---------------------------------------------------------------------------

/// Stub adapter for ONNX Runtime.
///
/// Runtime name: "onnx"
/// Supported formats: "onnx"
/// Supported precisions: FP32, FP16, INT8
/// Supported devices: CPU, GPU
///
/// When integrated with the real ONNX Runtime library, Initialize() will
/// create an Ort::Env and session options, and LoadModel() will create
/// Ort::Session instances per model file.
class OnnxRuntimeAdapter : public IInferenceRuntime {
public:
    OnnxRuntimeAdapter()  = default;
    ~OnnxRuntimeAdapter() override;

    std::string          GetName()         const override;
    RuntimeCapabilities  GetCapabilities() const override;
    bool IsModelCompatible(const ModelMetadata& metadata) const override;

    std::string Initialize() override;
    void        Shutdown()   override;

    std::optional<std::shared_ptr<IActiveModel>>
        LoadModel(const ModelMetadata& metadata, const DeviceConfig& config) override;

private:
    bool  initialized_ = false;
    void* env_ = nullptr;
};

} // namespace edgepilot
