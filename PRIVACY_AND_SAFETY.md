# EdgePilot Privacy and Safety Policy

This document details the data governance, security configurations, local storage architectures, safety thresholds, and risks associated with running EdgePilot.

---

## 1. Data Governance & Memory Lifecycles

EdgePilot implements a strict separation between **Inference Data** (user prompts, intermediate weights, output tokens) and **Execution Telemetry** (latencies, RAM bytes, CPU/GPU percentages).

### Inference Data Lifecycle
* **Volatile Allocation Only**: Any user prompt, text token, or image tensor submitted to the REST gateway is parsed into C++ structures (`InferenceRequest`) and stored in volatile heap memory (RAM).
* **Immediate Deallocation**: Once the execution adapter (`llama_cpp` or `onnx`) completes the inference pass, the outputs are compiled into `InferenceResult`, transmitted over the local TCP socket to the gateway, and the memory structures are immediately deallocated.
* **No Content Logging**: EdgePilot **never** writes user prompts, images, or output texts to log files, databases, or local disks.

### Telemetry Logging Lifecycle
The execution telemetry logged to [edgepilot_telemetry.csv](file:///c:/Users/kasiv/Desktop/EdgePilot---Adaptive-AI-Workload-Orchestrator/edgepilot_telemetry.csv) and [edgepilot_telemetry.db](file:///c:/Users/kasiv/Desktop/EdgePilot---Adaptive-AI-Workload-Orchestrator/edgepilot_telemetry.db) consists solely of non-identifying operational metrics:
* Timestamp (in microseconds)
* Model Identifier (e.g. `resnet50`, `llama3-8b`)
* Target Precision and Runtime name (e.g. `onnx`, `llama_cpp`)
* CPU and GPU utilization percentages
* Free memory (RAM) and battery level (%)
* Junction temperature (°C) and job queue length
* Latency (us), peak memory usage (bytes), and energy consumption (joules)

---

## 2. Network Security & Socket Bind Permissions

EdgePilot is not a network service. It runs on a local loopback architecture.

* **Loopback Binding**: The C++ Orchestrator Daemon (`orchestrator.cpp`) binds its TCP socket exclusively to:
  $$\text{IP Address} = \text{127.0.0.1} \quad (\text{localhost})$$
  This prevents the daemon from listening to network requests originating outside the host machine (e.g., from other devices on the LAN or the internet).
* **REST Access Control**: The FastAPI REST Gateway also binds to local interfaces, shielding the raw TCP socket endpoint (port `12345`) from external access.

---

## 3. Local Storage Policy

All persistent configurations, databases, and logs are kept in standard formats on the local storage partition:
1. **CSV Log File**: `edgepilot_telemetry.csv` provides flat-file append logging for telemetry.
2. **SQLite Database**: `edgepilot_telemetry.db` holds queryable indices of the telemetry log for fast baseline retrieval.
3. **Model Folder**: `./models/` holds local predictor ONNX weights and model configurations.

No root or administrative privileges are required to run any EdgePilot component. The system runs within standard user-space directories, respecting system file access controls.

---

## 4. Hardware Safety Boundaries & Execution Limits

To protect edge devices from hardware degradation, battery depletion, or thermal throttling, EdgePilot implements system-level execution safeguards:

```
               [Real-Time Hardware Telemetry Monitoring]
                                  │
         ┌────────────────────────┼────────────────────────┐
         ▼                        ▼                        ▼
  [Temp > 48.0°C]          [Battery < 20%]         [Free RAM < 1.5GB]
         │                        │                        │
         ▼                        ▼                        ▼
 [Thermal Safeguard]      [Power Safeguard]       [OOM Safeguard]
 Override GPU Routing;    Select INT8/INT4        Trigger Cache Eviction
 Introduce 100ms delay.   Quantization.           of Inactive Models.
```

### Thermal Safeguard
* **Trigger**: Junction temperature exceeds **48.0°C**.
* **Action**: The C++ scheduler overrides high-power GPU execution paths, routing jobs to the cooler CPU-bound `onnx` adapter. It also introduces a **100ms thread sleep** before starting inference, allowing heat dissipation.

### Power Safeguard
* **Trigger**: Host battery level drops below **20.0%**.
* **Action**: The scheduler restricts model loads to INT8 or INT4 quantization variants. This reduces DRAM weight-streaming bandwidth and power requirements.

### Memory Safeguard
* **Trigger**: Free physical memory drops below **1.5 GB**.
* **Action**: The Model Lifecycle Manager evicts inactive resident models from the residency cache.

---

## 5. Potential Risks & Limitations

### Simulated Backend Adapters
In simulation mode (when actual runtime libraries are not linked), the adapter layers execute mock profiling stubs. In production, these adapters must be compiled against real target libraries (`onnxruntime.dll` and `llama.dll`). Developers must verify that dynamic loading configurations do not point to untrusted system paths.

### Native UI Dialog Blockers
The model registration utility opens a native OS file browser dialog via Python `tkinter`. Since this dialog is spawned on the host desktop environment, it runs synchronously. If deployed in headless environments, this script will block indefinitely. Model registrations in headless containers must bypass this dialog and supply paths directly to the `/models/register` endpoint.
