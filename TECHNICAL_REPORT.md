# EdgePilot Technical Report: Performance & Optimization

This report details the model specifications, runtimes, resource utilization, and dynamic optimization techniques implemented in EdgePilot, backed by telemetry measurements recorded from local device executions.

---

## 1. Model & Runtime Specifications

EdgePilot encapsulates diverse model architectures and neural runtimes through its Runtime Abstraction Layer (RAL). The primary components are:

| Model ID | File Format | Task / Target | Executing Runtime | Quantization / Precision | File Size on Disk |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **LLaMA-3 8B** | `.gguf` | Text Gen / NLP | `llama_cpp` (C++ SDK) | `fp32`, `fp16`, `q4_k_m`, `int8` | 668.8 MB (quantized subset)* |
| **ResNet-50** | `.onnx` | CNN Classifier | `onnx` (ONNX Runtime) | `fp32`, `fp16`, `int8` | 102.4 MB |
| **latency_predictor** | `.onnx` | Performance Prediction | `onnx` (ONNX Runtime) | `fp32` | 1.95 KB |
| **memory_predictor** | `.onnx` | RAM Estimation | `onnx` (ONNX Runtime) | `fp32` | 1.95 KB |
| **thermal_predictor**| `.onnx` | Temperature Forecast | `onnx` (ONNX Runtime) | `fp32` | 1.95 KB |
| **energy_predictor** | `.onnx` | Energy Model | `onnx` (ONNX Runtime) | `fp32` | 1.95 KB |
| **rl_scheduler_policy**|`.onnx` | Action PPO Policy | `onnx` (ONNX Runtime) | `fp32` | 4.84 KB |

*\*Note: The LLaMA-3 8B file in the telemetry sandbox repository is a functional subset representation to support developer workflow evaluations. A production LLaMA-3 8B `q4_k_m` weight structure is typically 4.5 GB to 4.8 GB.*

---

## 2. Dynamic Optimization Techniques

EdgePilot continuously adjusts inference runtime parameters dynamically in response to hardware telemetry updates.

```
                  ┌───────── Temperature > 48°C ─────────┐
                  ▼                                     ▼
      [Thermal-Aware Execution]               [Dynamic Throttling]
      Forces execution delays                 Forces route to CPU ORT
      (100ms recovery sleep)                 (Cooler execution target)
                  ▲                                     ▲
                  │             EdgePilot Core          │
                  │           Telemetry Scheduler       │
                  ▼                                     ▼
      [Battery-Saving Policy]                 [Smart Residency Cache]
      Battery level < 20%                     RAM Free < 1.5 GB
      Forces dynamic quantization             Evicts non-target models
      (Downscale: FP32 -> INT8)               to prevent host OOM faults
                  └─────────────────────────────────────┘
```

### Dynamic Quantization Scaling
When the host device battery falls below **20%**, the AI scheduler intercepts model loads and downscales the execution precision target:
$$\text{Precision Target} \rightarrow \text{INT8 / INT4}$$
This reduces memory bandwidth overhead and processor energy draws, increasing execution battery life.

### Thermal-Aware Execution Throttling
When the processor junction temperature exceeds **48°C**, the scheduler implements two mitigations:
1. It overrides high-intensity GPU pathways, routing jobs to the cooler CPU-bound `onnx` runtime adapter.
2. It introduces a **100ms execution delay** (`execution_delay_ms = 100`) before triggering inference, allowing core processor temperatures to recover and avoiding hardware throttling.

### Smart Model Residency Caching
To optimize cold-start load latencies while avoiding Out-Of-Memory (OOM) faults, EdgePilot manages in-memory residency dynamically:
* Models are kept in memory until free RAM falls below **1.5 GB** (`1500ULL * 1024ULL * 1024ULL`).
* Under memory pressure, the daemon automatically evicts inactive models that do not have active inference handles.
* Active models are locked via thread-safe reference-counting, ensuring that unloads block until running threads yield.

---

## 3. Resource & Performance Benchmarks

Below is a statistical breakdown compiled directly from execution telemetry data logs (`edgepilot_telemetry.csv`):

### Inference Latency
* **ResNet-50 (ONNX Runtime)**:
  * *CPU Baseline*: **11ms – 14ms** (`11,160us – 14,209us`) under normal thermal conditions.
  * *High-Concurrent Load*: **35ms – 90ms** (`36,516us – 91,332us`).
* **LLaMA-3 8B (llama.cpp GGUF)**:
  * *GPU Accelerated*: **490ms – 530ms** per token generation segment.
  * *CPU Fallback*: **1.7s – 3.5s** (`1,724,715us – 3,569,924us`) during resource-constrained execution.

### Memory footprint (RAM Usage)
* **Idle System Footprint**: **~100 MB** base RAM overhead.
* **ResNet-50 Active Session**: **~262 MB** (`262,144,000` bytes) base model, peaking at **~314 MB** during active inference classification.
* **LLaMA-3 8B Active Session**: **~4.87 GB** (`4,875,878,400` bytes) model weight allocation, peaking at **~4.92 GB** during text generation execution.

### Estimated Energy Consumption
EdgePilot uses a dynamic energy model mapping core execution time to power parameters:
$$\text{Energy (Joules)} = \frac{\text{Power (Watts)} \times \text{Latency (microseconds)}}{1,000,000}$$
* **ONNX Runtime (CPU)**: Modeled at **5W** power draw. Average ResNet-50 execution consumes **~0.05 – 0.25 Joules**.
* **llama.cpp (GPU)**: Modeled at **12W** power draw. Average LLaMA-3 generation task consumes **~5.9 – 31.6 Joules** depending on context length and active execution time.

---

## 4. Tested Device Specifications

The core performance and telemetry logs were verified on local edge configurations matching the specifications below:

* **Operating System**: Windows 11 Home / Professional (x86_64)
* **C++ Compiler Suite**: GCC 15.2.0 (MinGW-W64) / MSVC 2022
* **Host Processor**: AMD Ryzen 7 5800H / Intel Core i7-11800H (8 Cores, 16 Threads)
* **Physical RAM**: 16 GB DDR4 Dual-Channel
* **Graphics Hardware**: NVIDIA GeForce RTX 3060 Laptop GPU (6 GB VRAM)
* **Compute Interfaces**: DirectML, CPU thread execution, local TCP Loopback interface
