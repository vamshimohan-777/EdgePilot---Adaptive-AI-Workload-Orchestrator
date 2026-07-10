#pragma once

#include "edgepilot/types.h"

#include <memory>
#include <optional>
#include <string>

// =============================================================================
// EdgePilot — P1: Runtime & Model Infrastructure
// runtime_interface.h — Abstract interfaces for all inference runtimes and
//                        loaded model instances.
//
// Design patterns:
//   - Strategy   : IInferenceRuntime lets the caller swap backends transparently
//   - Factory Method : LoadModel() on each runtime creates its own IActiveModel
//   - Adapter    : Concrete adapters wrap backend-specific APIs behind these
//                  interfaces, hiding all runtime details from the Orchestrator
//
// Thread-safety contracts:
//   - IInferenceRuntime methods (Initialize, Shutdown, LoadModel, GetCapabilities)
//     are NOT required to be thread-safe by default. The Orchestrator must
//     serialize calls per runtime instance.
//   - IActiveModel::RunInference is NOT thread-safe per instance. The Execution
//     Kernel is responsible for serializing requests to a single model instance.
//   - IActiveModel::GetStatus and GetModelId are safe to call concurrently.
// =============================================================================

namespace edgepilot {

// ---------------------------------------------------------------------------
// IActiveModel
// ---------------------------------------------------------------------------

/// Represents a single loaded model instance inside a runtime.
///
/// Lifetime: Managed by shared_ptr. The ModelLifecycleManager caches the
/// shared_ptr; the Orchestrator also receives a copy. The model stays alive
/// as long as at least one shared_ptr refers to it.
///
/// RAII note: The concrete class destructor calls Unload() as a safety net
/// to prevent resource leaks. The primary teardown path is the explicit
/// Unload() method, which gives the caller controlled shutdown with an error
/// return. Do not rely on the destructor for error handling.
class IActiveModel {
public:
    virtual ~IActiveModel() = default;

    /// Returns the model_id this instance was loaded for.
    virtual std::string GetModelId() const = 0;

    /// Returns the current lifecycle state of this model instance.
    /// Safe to call from any thread.
    virtual ModelStatus GetStatus() const = 0;

    /// Runs inference synchronously and returns the result.
    ///
    /// Thread safety: NOT safe to call concurrently on the same instance.
    /// The Execution Kernel must serialize all requests to a single instance.
    ///
    /// Errors: Failures are reported via InferenceResult::success == false
    /// and InferenceResult::error_message. This method does not throw.
    ///
    /// Latency: The adapter populates InferenceResult::latency_us with the
    /// wall-clock time from entry to return. The Telemetry module reads this
    /// value from the result — P1 does not emit telemetry events directly.
    virtual InferenceResult RunInference(const InferenceRequest& request) = 0;

    /// Explicitly releases all runtime resources held by this model instance.
    ///
    /// Returns true on success, false if unload encountered an error
    /// (e.g. internal runtime state is inconsistent).
    ///
    /// Calling Unload() on an already-unloaded instance is a no-op (returns true).
    /// The destructor also calls Unload() as a fallback safety net.
    virtual bool Unload() = 0;
};


// ---------------------------------------------------------------------------
// IInferenceRuntime
// ---------------------------------------------------------------------------

/// Abstract interface that every inference runtime adapter must implement.
///
/// A runtime represents one backend execution engine (e.g. ONNX Runtime,
/// llama.cpp). It is responsible for:
///   - Advertising its capabilities to the RuntimeRegistry
///   - Determining whether it can execute a given model
///   - Loading a model file into memory and returning an IActiveModel handle
///
/// The choice of which runtime to use for a given model is made by the
/// Runtime Selector (not P1). P1 receives the chosen runtime name from the
/// Orchestrator and resolves it via RuntimeRegistry.
///
/// Lifecycle:
///   1. Instantiate concrete adapter
///   2. Call Initialize() — check return for error
///   3. Register with RuntimeRegistry
///   4. Call LoadModel() for each model to serve
///   5. Call Shutdown() on system teardown
class IInferenceRuntime {
public:
    virtual ~IInferenceRuntime() = default;

    /// Returns a unique, stable identifier for this runtime.
    /// This name is used as the key in RuntimeRegistry and must exactly match
    /// the names listed in ModelMetadata::compatible_runtimes.
    ///
    /// Examples: "onnx", "llama_cpp", "executorch", "litertf"
    virtual std::string GetName() const = 0;

    /// Returns the static capability descriptor for this runtime.
    /// Called by the Orchestrator / Runtime Selector during capability discovery.
    /// Should return consistent values across calls (no side effects).
    virtual RuntimeCapabilities GetCapabilities() const = 0;

    /// Returns true if this runtime is able to load and execute the given model.
    /// Implementations should check model format, task type, and precision
    /// against their supported capabilities.
    virtual bool IsModelCompatible(const ModelMetadata& metadata) const = 0;

    /// Performs one-time initialisation of the runtime backend (e.g. loading
    /// shared libraries, checking hardware availability, warming up thread pools).
    ///
    /// Returns an empty string on success.
    /// Returns a human-readable error description on failure.
    /// A runtime that fails to initialise must NOT be registered.
    virtual std::string Initialize() = 0;

    /// Releases all resources held by this runtime (thread pools, device handles,
    /// cached sessions). Called once during system shutdown.
    virtual void Shutdown() = 0;

    /// Loads a model from disk into this runtime and returns an active handle.
    ///
    /// Returns std::nullopt on failure. Callers must check for nullopt before
    /// using the result. Failure details are logged internally by the adapter.
    ///
    /// The returned IActiveModel is kept alive by shared_ptr reference counting.
    /// The ModelLifecycleManager caches one shared_ptr; the caller receives another.
    ///
    /// config.quantization_variant_id: if non-empty, the adapter loads the
    /// variant file specified in ModelMetadata::quantization_variants.
    /// If empty or not found, the adapter falls back to the default file path.
    virtual std::optional<std::shared_ptr<IActiveModel>>
        LoadModel(const ModelMetadata& metadata, const DeviceConfig& config) = 0;
};

} // namespace edgepilot
