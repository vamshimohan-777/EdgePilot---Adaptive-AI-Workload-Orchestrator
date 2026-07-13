# EdgePilot Core: Runtime & Model Infrastructure (P1)

This module provides the core abstraction, model metadata registry, and lifecycle management layer for on-device AI runtimes. It sits directly between the execution orchestrator and backend inference engines, shielding upper layers from hardware and engine-specific nuances.

---

## Architecture Overview

Upper Orchestrator Layers (Execution Kernel, Scheduler)
            │
            ▼
┌─────────────────────────────────────────────────────────────┐
│             P1: Runtime & Model Infrastructure              │
│                                                             │
│  ┌───────────────────────────────────────────────────────┐  │
│  │                 ModelLifecycleManager                 │  │
│  └──────┬──────────────────────┬───────────────────┬─────┘  │
│         │ queries              │ resolves          │ loads  │
│  ┌──────▼──────┐        ┌──────▼────────┐   ┌──────▼─────┐  │
│  │ModelRegistry│        │RuntimeRegistry│   │  Adapters  │  │
│  └──────┬──────┘        └───────────────┘   │ (ONNX/GGUF)│  │
│         │ façade                            └────────────┘  │
│  ┌──────▼──────────────────┐                                │
│  │QuantizationMetadataMngr │                                │
│  └─────────────────────────┘                                │
└─────────────────────────────────────────────────────────────┘

---

## Component Details

### 1. Model Registry (`ModelRegistry`)
- **Responsibility**: Houses and indexes `ModelMetadata` objects (inputs/outputs, version, runtime compatibility, available precisions, and quantization variants).
- **Storage**: In-memory database designed for thread-safe read/write access.
- **Filtering**: Supports querying models by `TaskType` or compatible `IInferenceRuntime` names.

### 2. Quantization Manager (`QuantizationMetadataManager`)
- **Responsibility**: A lightweight façade over `ModelRegistry` offering clean APIs to discover, retrieve, and select model quantization variants (such as `q4_k_m`, `fp16`, etc.) prior to scheduling load commands.

### 3. Runtime Registry (`RuntimeRegistry`)
- **Responsibility**: Acts as a central factory and locator for registered adapters (`IInferenceRuntime`).
- **Registration**: Thread-safe runtime plugin registration using upsert semantics (replacing duplicate registrations).

### 4. Runtime Abstraction Layer (RAL)
- **`IInferenceRuntime`**: Common strategy interface for runtimes. Responsible for initializing itself, shutting down, checking compatibility, and spawning active model handles via `LoadModel()`.
- **`IActiveModel`**: Representation of a loaded model in memory. Exposes `RunInference()` and an explicit `Unload()` release handler.

### 5. Model Lifecycle Manager (`ModelLifecycleManager`)
- **Responsibility**: Orchestrates loading, unloading, reloading, and status caching.
- **Cache Policy**: Loading a model is idempotent (cache-hit returns the existing active handle). A reload unloads and re-initializes.
- **Inference Synchronization**: To prevent a race condition where a model is unloaded while running inference on another thread, `Unload()` blocks until the active inference reference count reaches zero.

---

## Key Contracts and Guarantees

### Concurrency and Thread Safety
- **Registries and Lifecycle Manager**: Fully thread-safe. Thread safety is achieved via internal `std::mutex` locks.
- **`IActiveModel::RunInference()`**: **NOT thread-safe per instance.** The upper-level Execution Kernel is responsible for serializing inference requests to a single model instance.
- **Safe Unloading**: Upper layers must wrap inference execution calls as follows:
  ```cpp
  lifecycle_manager.AcquireInference(model_id);
  auto result = active_model->RunInference(request);
  lifecycle_manager.ReleaseInference(model_id);
  ```

### Error Handling
- Nullable operations return `std::optional<T>` rather than raw pointers to force compile-time check verification of failed loads or missing items.
- Run-time execution failures inside inference adapters are returned as structured messages in `InferenceResult::error_message` rather than throwing C++ exceptions.

### Telemetry Latency Boundary
- Telemetry measurements of execution run-times (`latency_us`) are captured internally by individual runtime adapters during the `RunInference` loop and written directly into the `InferenceResult`. Upper-level telemetry modules read this value upon result return (P1 does not write to disks or emit network telemetry events directly).

---

## Building and Running Tests

### Compilation
The core library requires a compiler supporting the **C++17** standard (e.g. GCC 7+, MSVC 2017+).

To configure and generate build scripts using CMake:
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

### Run Tests
To run unit and integration test suites:
```bash
# Day 2 Tests: Runtime Registry & Adapters
./build/tests/test_runtime_layer

# Day 3 Tests: Model Registry, Quantization, Lifecycle Manager
./build/tests/test_model_layer
```
Or use CTest:
```bash
cd build && ctest --output-on-failure
```
