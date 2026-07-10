#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>
#include <optional>

// =============================================================================
// EdgePilot — P1: Runtime & Model Infrastructure
// types.h — Shared data structures used across all P1 components
//
// Design: All types are plain data structs with no logic.
//         No dependency on any inference runtime library.
// =============================================================================

namespace edgepilot {

// ---------------------------------------------------------------------------
// Enumerations
// ---------------------------------------------------------------------------

/// Numerical precision format for model weights and activations.
enum class Precision {
    FP32,   ///< 32-bit floating point (full precision)
    FP16,   ///< 16-bit floating point (half precision)
    INT8,   ///< 8-bit integer quantization
    INT4,   ///< 4-bit integer quantization
    UINT8,  ///< Unsigned 8-bit integer
    AUTO    ///< Let the runtime decide based on hardware
};

/// Execution hardware target.
enum class HardwareDevice {
    CPU,    ///< Central Processing Unit
    GPU,    ///< Graphics Processing Unit (CUDA / Metal / OpenCL)
    NPU,    ///< Neural Processing Unit (Qualcomm HTP, Apple ANE, etc.)
    AUTO    ///< Let the runtime select the best available device
};

/// High-level category of the AI task a model performs.
enum class TaskType {
    TextGeneration,
    TextEmbedding,
    ImageClassification,
    ObjectDetection,
    SpeechRecognition,
    Unknown
};

/// Lifecycle state of a model instance tracked by the ModelLifecycleManager.
enum class ModelStatus {
    NotLoaded,   ///< Model is registered but not loaded into memory
    Loading,     ///< Load is in progress
    Loaded,      ///< Model is loaded and ready to serve inference
    Unloading,   ///< Unload is in progress
    Error        ///< Load or unload failed; see logs for details
};


// ---------------------------------------------------------------------------
// Tensor types
// ---------------------------------------------------------------------------

/// Specification of a single tensor input or output (schema only, no data).
struct TensorSpec {
    std::string           name;
    std::vector<int64_t>  shape;    ///< -1 indicates a dynamic dimension
    std::string           dtype;    ///< "float32", "int64", "uint8", etc.
    bool                  optional; ///< true if the tensor may be omitted at runtime
};

/// A tensor carrying actual data — used in InferenceRequest and InferenceResult.
/// Data is stored as a raw byte buffer. The caller casts to the correct type
/// based on the `dtype` field.
struct Tensor {
    std::string           name;
    std::vector<int64_t>  shape;
    std::string           dtype;     ///< "float32", "int64", "uint8", etc.
    std::vector<uint8_t>  data;      ///< Raw byte buffer
};


// ---------------------------------------------------------------------------
// Quantization
// ---------------------------------------------------------------------------

/// Describes one quantized variant of a model (e.g. Q4_K_M for a GGUF model).
struct QuantizationVariant {
    std::string  variant_id;    ///< Identifier, e.g. "q4_k_m", "fp16", "int8"
    Precision    precision;
    uint64_t     size_bytes;    ///< On-disk size of this variant
    std::string  file_path;     ///< Absolute or relative path to model file
};


// ---------------------------------------------------------------------------
// Model Metadata
// ---------------------------------------------------------------------------

/// Complete descriptor for a registered model.
/// This struct is metadata only — it does not hold loaded weights or state.
struct ModelMetadata {
    std::string                      model_id;
    std::string                      display_name;
    std::string                      version;
    TaskType                         task_type;

    /// Runtime names this model is compatible with (e.g. "onnx", "llama_cpp").
    /// Must match the name returned by IInferenceRuntime::GetName().
    std::vector<std::string>         compatible_runtimes;

    /// Preferred runtime to use when the caller does not specify one.
    /// Empty string means no preference; the first compatible runtime is used.
    std::string                      preferred_runtime;

    std::vector<Precision>           supported_precisions;
    uint64_t                         file_size_bytes;
    std::string                      default_file_path;
    std::vector<QuantizationVariant> quantization_variants;
    std::vector<TensorSpec>          input_specs;
    std::vector<TensorSpec>          output_specs;

    /// Extensible key-value store for runtime-specific or future metadata.
    std::unordered_map<std::string, std::string> extra_metadata;
};


// ---------------------------------------------------------------------------
// Runtime Configuration
// ---------------------------------------------------------------------------

/// Configuration passed at model load time to control execution behaviour.
struct DeviceConfig {
    HardwareDevice device                 = HardwareDevice::CPU;
    int            num_threads            = 1;

    /// The quantization_variant_id to load. Empty string = use default precision.
    std::string    quantization_variant_id;

    /// Runtime-specific key-value options (e.g. "gpu_device_id": "0").
    std::unordered_map<std::string, std::string> extra_options;
};

/// Capabilities advertised by a registered inference runtime.
struct RuntimeCapabilities {
    std::string                  runtime_name;
    std::string                  runtime_version;

    /// Model format identifiers the runtime can load (e.g. "onnx", "gguf").
    std::vector<std::string>     supported_model_formats;

    std::vector<Precision>       supported_precisions;
    std::vector<HardwareDevice>  supported_devices;
    bool                         supports_streaming = false;

    /// Extensible key-value store for runtime-specific capability fields
    /// (e.g. "max_context_length": "4096", "max_batch_size": "32").
    std::unordered_map<std::string, std::string> extra_info;
};


// ---------------------------------------------------------------------------
// Inference I/O
// ---------------------------------------------------------------------------

/// Input to a single inference call.
struct InferenceRequest {
    std::string              model_id;
    std::string              request_id;

    /// Structured tensor inputs for non-LLM models.
    std::vector<Tensor>      tensor_inputs;

    /// Text-based inputs for LLM adapters (e.g. "prompt": "Hello").
    std::unordered_map<std::string, std::string> text_inputs;

    /// Runtime hints or adapter-specific options for this request.
    std::unordered_map<std::string, std::string> options;
};

/// Output of a single inference call.
struct InferenceResult {
    std::string              request_id;
    bool                     success;
    std::string              error_message;   ///< Non-empty when success == false

    /// Structured tensor outputs for non-LLM models.
    std::vector<Tensor>      tensor_outputs;

    /// Text outputs for LLM adapters (e.g. "generated_text": "...").
    std::unordered_map<std::string, std::string> text_outputs;

    /// Wall-clock latency measured by the adapter from call entry to return.
    /// Populated by the adapter; read by the Telemetry module from the result.
    uint64_t                 latency_us = 0;
};

} // namespace edgepilot
