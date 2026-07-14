#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>
#include <optional>

// =============================================================================
// EdgePilot — P1/P2/P3/P4: Runtime & Model Infrastructure + Execution Kernel
// types.h — Shared data structures used across all EdgePilot components
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

/// Execution status of an inference Job.
enum class JobStatus {
    Submitted,   ///< Job received and placed in the queue
    Scheduled,   ///< Job matched with a runtime and resource policy
    Executing,   ///< Job is running on a worker thread
    Completed,   ///< Inference finished successfully
    Failed       ///< Inference failed with an error
};

/// Priority of an inference workload.
enum class JobPriority {
    Low,
    Normal,
    High,
    RealTime
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
/// Data is stored as a raw byte buffer.
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
struct ModelMetadata {
    std::string                      model_id;
    std::string                      display_name;
    std::string                      version;
    TaskType                         task_type;

    /// Runtime names this model is compatible with (e.g. "onnx", "llama_cpp").
    std::vector<std::string>         compatible_runtimes;

    /// Preferred runtime to use when the caller does not specify one.
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

    /// Runtime-specific key-value options.
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

    /// Extensible key-value store for runtime-specific capability fields.
    std::unordered_map<std::string, std::string> extra_info;
};


// ---------------------------------------------------------------------------
// Inference I/O & Workloads
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
    bool                     success = false;
    std::string              error_message;

    /// Structured tensor outputs for non-LLM models.
    std::vector<Tensor>      tensor_outputs;

    /// Text outputs for LLM adapters (e.g. "generated_text": "...").
    std::unordered_map<std::string, std::string> text_outputs;

    /// Wall-clock latency measured by the adapter from call entry to return.
    uint64_t                 latency_us = 0;
};

/// An execution Job containing the request payload, priority, and timestamps.
struct Job {
    std::string             job_id;
    std::string             model_id;
    JobPriority             priority        = JobPriority::Normal;
    InferenceRequest        request;
    uint64_t                submitted_at_us = 0;
    uint64_t                deadline_at_us  = 0; ///< 0 means no deadline
};

/// Result of an executed job returned to the caller.
struct JobResult {
    std::string             job_id;
    JobStatus               status          = JobStatus::Failed;
    InferenceResult         result;
    std::string             error_message;
};


// ---------------------------------------------------------------------------
// Scheduler Inputs and Outputs
// ---------------------------------------------------------------------------

/// Represents real-time system resource usage metrics.
struct SystemState {
    float                    cpu_utilization    = 0.0f; ///< 0.0 to 100.0
    float                    gpu_utilization    = 0.0f; ///< 0.0 to 100.0
    uint64_t                 ram_free_bytes     = 0;
    float                    battery_level      = 100.0f; ///< 0.0 to 100.0
    float                    device_temperature = 0.0f; ///< Celsius
    uint32_t                 queue_length       = 0;
    std::vector<std::string> loaded_models;
};

/// Scheduling decisions generated by scheduling agents.
struct SchedulingDecision {
    std::string              runtime_name;
    std::string              quantization_variant_id;
    JobPriority              adjusted_priority;
    uint32_t                 execution_delay_ms = 0;
    std::vector<std::string> preload_models;
    std::vector<std::string> evict_models;
};

} // namespace edgepilot
