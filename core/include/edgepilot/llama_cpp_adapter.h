#pragma once

#include "edgepilot/runtime_interface.h"

#include <memory>
#include <optional>
#include <string>
#include <mutex>

// =============================================================================
// EdgePilot — P1: Runtime & Model Infrastructure
// llama_cpp_adapter.h — Adapter that wraps llama.cpp (GGUF models) behind
//                        the IInferenceRuntime interface.
//
// Design pattern: Adapter + Factory Method
//   Adapts the llama.cpp C API to the EdgePilot IInferenceRuntime interface.
//   LoadModel() acts as a Factory Method, producing LlamaCppActiveModel
//   instances that implement IActiveModel.
//
// Current status: STUB implementation.
//   All methods return valid, predictable values for lifecycle and
//   integration testing. No actual llama.cpp library is linked.
//   Replace stub bodies with real llama.h calls when integrating.
// =============================================================================

namespace edgepilot {

// ---------------------------------------------------------------------------
// LlamaCppActiveModel — Stub loaded model handle
// ---------------------------------------------------------------------------

/// Stub implementation of IActiveModel for the llama.cpp adapter.
/// Simulates a loaded GGUF model with predictable text generation results.
class LlamaCppActiveModel : public IActiveModel {
public:
    explicit LlamaCppActiveModel(const std::string& model_id);
    ~LlamaCppActiveModel() override;

    std::string     GetModelId() const override;
    ModelStatus     GetStatus()  const override;
    InferenceResult RunInference(const InferenceRequest& request) override;
    bool            Unload() override;

private:
    friend class LlamaCppAdapter;
    std::string  model_id_;
    ModelStatus  status_;
    void*        model_ = nullptr;
    void*        ctx_ = nullptr;
    std::mutex   inference_mutex_;
};


// ---------------------------------------------------------------------------
// LlamaCppAdapter — IInferenceRuntime implementation for GGUF / llama.cpp
// ---------------------------------------------------------------------------

/// Stub adapter for llama.cpp / GGUF models.
///
/// Runtime name: "llama_cpp"
/// Supported formats: "gguf"
/// Supported precisions: FP16, INT8, INT4
/// Supported devices: CPU, GPU
///
/// When integrated with the real llama.cpp library, Initialize() will
/// call llama_backend_init(), LoadModel() will call llama_load_model_from_file()
/// and llama_new_context_with_model(), and RunInference() will run the
/// sampling loop.
class LlamaCppAdapter : public IInferenceRuntime {
public:
    LlamaCppAdapter()  = default;
    ~LlamaCppAdapter() override;

    std::string          GetName()         const override;
    RuntimeCapabilities  GetCapabilities() const override;
    bool IsModelCompatible(const ModelMetadata& metadata) const override;

    std::string Initialize() override;
    void        Shutdown()   override;

    std::optional<std::shared_ptr<IActiveModel>>
        LoadModel(const ModelMetadata& metadata, const DeviceConfig& config) override;

private:
    bool initialized_ = false;
};

} // namespace edgepilot
