# EdgePilot Evaluation: Scheduler Benchmarks & Failure Modes

This document evaluates the scheduling effectiveness of EdgePilot, detailing our benchmarking methodologies, baseline comparisons, and known failure cases.

---

## 1. Benchmarking Methodology

EdgePilot uses a dual-phase profiling framework to characterize model workloads and continuously optimize scheduling.

```
                   [New Model Registered]
                             │
                             ▼
                 [ModelCharacterizer (P3)]
              Runs 5 synthetic profiling steps
           Loads model, benchmarks latency/memory
                             │
                             ▼
               [Baseline Profile Created]
         Written to edgepilot_telemetry.csv & DB
                             │
                             ▼
             [Adaptive Telemetry Feedback Loop]
         Real-time execution stats logged per job
```

### Model Characterization (P3)
To resolve the cold-start problem (where a newly registered model lacks historical telemetry, causing the AI scheduler to make poor routing choices), EdgePilot invokes the `ModelCharacterizer` when a model is registered:
* It simulates **5 execution cycles** across all compatible runtimes.
* It measures initialization overhead, active inference time (`latency_us`), RAM consumption (`peak_memory_bytes`), and energy footprints (`energy_estimate_joules`).
* The average of these 5 runs is saved as a baseline query profile in the SQLite telemetry database (`edgepilot_telemetry.db`), enabling the AI scheduler to make informed decisions immediately.

### Continuous Telemetry Logging
During standard operations, the C++ execution kernel captures latency metrics for every job. These stats are appended to `edgepilot_telemetry.csv` and synced back to SQLite, creating a dataset for continuous online retraining.

---

## 2. Scheduler Evaluation & Baseline Comparisons

EdgePilot's scheduler was evaluated by comparing the **Heuristic Scheduler (Rule-Based)** against the **AI Scheduler (Hybrid Predictor + PPO RL)** across various system stress scenarios.

### Scenario A: Thermal Stress (Junction Temp > 48°C)
* **Heuristic Scheduler**: Keeps sending text generation jobs to the heavy `llama_cpp` GPU runtime. The processor overheats, triggering hardware-level thermal throttling, causing average inference latency to spike by **300%**.
* **AI Scheduler**: Senses temperature spikes, overrides default routing, and sends jobs to the cooler CPU-bound `onnx` adapter. It also introduces a **100ms execution delay**, allowing CPU junction temperatures to stabilize.

### Scenario B: Battery Depletion (Battery < 20%)
* **Heuristic Scheduler**: Executes ResNet-50 and LLaMA-3 at full FP32 precision. Power draw remains at maximum (12W on GPU), depleting the edge battery rapidly.
* **AI Scheduler**: Detects low battery, selects the INT8 or INT4 quantization variants (`q4_k_m`), and switches the model execution precision. Peak power draw drops from **12W to 5W**, extending remaining battery life.

### Scenario C: Memory Pressure (RAM Free < 1.5 GB)
* **Heuristic Scheduler**: Attempts to load LLaMA-3 8B alongside ResNet-50 simultaneously. The combined memory footprint exceeds host RAM boundaries, triggering an Out-of-Memory (OOM) fault and crashing the daemon.
* **AI Scheduler**: Evaluates free RAM beforehand, identifies memory pressure, and triggers evictions for inactive resident models before loading the target model, maintaining system stability.

---

## 3. Known Failure Cases & Mitigations

Despite its design, EdgePilot faces specific operating constraints. Below is a analysis of these failure modes and their mitigations:

| Failure Mode | Root Cause | System Impact | Mitigation |
| :--- | :--- | :--- | :--- |
| **Inference Stalling during Unload** | A heavy model (LLaMA-3) is being unloaded while a worker thread is running active inference on it. | Segmentation fault / Daemon crash. | **Reference-Count Lock**: `ModelLifecycleManager::Unload()` blocks on a `std::condition_variable` until the model's `inference_count` drops to 0, ensuring safe releases. |
| **Out-Of-Memory (OOM) under Concurrent Batches** | Multiple clients submit high-context text jobs concurrently, consuming all available system memory before evictions can complete. | Host OS terminates the daemon process. | **Worker Pool Concurrency Limit**: The daemon's worker thread pool limits parallel executions to a default of 2 workers (`num_worker_threads = 2`), capping parallel memory allocation. |
| **Telemetry Metric Noise** | OS background tasks (Windows updates, browser processes) spike CPU/GPU usage, causing telemetry readings to deviate from the model's actual performance. | The training agent updates the policy based on skewed data. | **Running Averages**: The SQLite database calculates average metrics (`avg_latency_us`) over multiple runs rather than relying on a single data point, smoothing out performance outliers. |
| **Policy Cold-Start Overshooting** | A newly retrained RL policy outputs actions that cause rapid scheduler transitions (e.g. continuous load and unload loops). | High CPU thrashing and disk read overhead. | **Action-Space Clipping**: The C++ scheduling adapters filter RL outputs, ignoring duplicate reload/eviction requests if the target model state matches current residency. |
