# EdgePilot Local AI Verification

EdgePilot is designed as a local-first, zero-cloud execution framework. This document verifies the local execution boundaries, data privacy guarantees, and networking configurations of the system.

---

## 1. On-Device Execution Scope

All core services, database engines, ML runtimes, and training loops execute locally within the user's OS user space. No external APIs or cloud computations are referenced.

```
                  ┌──────────────────────────────────────────────┐
                  │            Local Device Sandbox              │
                  │                                              │
                  │  ┌──────────────────┐  ┌──────────────────┐  │
                  │  │   C++ Daemon     │  │ FastAPI REST API │  │
                  │  │  (Orchestrator)  │  │    (Gateway)     │  │
                  │  └────────┬─────────┘  └────────┬─────────┘  │
                  │           │                     │            │
                  │           ▼                     ▼            │
                  │  ┌──────────────────┐  ┌──────────────────┐  │
                  │  │  Inference RAL   │  │ PyTorch Training │  │
                  │  │  (ONNX / GGUF)   │  │   (RL Policy)    │  │
                  │  └──────────────────┘  └──────────────────┘  │
                  │                                              │
                  └──────────────────────────────────────────────┘
                                          ▲
                                          │
                                 [ ZERO CONNECTIONS ]
                                          │
                                          ▼
                                   [ Public Internet ]
```

Specifically:
1. **Inference Engines**: Adapters for ONNX Runtime (C++) and llama.cpp run locally. Model weights (`.onnx` and `.gguf` formats) are loaded from the local storage disk directly into system RAM / VRAM.
2. **Telemetry Database**: All logging utilizes a local CSV file (`edgepilot_telemetry.csv`) and a lightweight SQLite database file (`edgepilot_telemetry.db`) stored in the workspace directory.
3. **Orchestrator Daemon**: The core executor (`edgepilot_daemon` or `test_main_orchestrator`) is compiled to native machine instructions and executes on local CPU worker threads.
4. **FastAPI Web Server**: Mounts the HTML/CSS/JS dashboard and routes developer requests locally.
5. **AI Retraining Loop**: The PyTorch agent performs PPO gradient descent optimization steps on the local device, drawing from the local CSV logs and overwriting the local ONNX policy model (`rl_scheduler_policy.onnx`).

---

## 2. Internet Connectivity Requirements

* **Required Connections**: **Zero (0)**.
* **Offline Operation**: EdgePilot operates fully offline. The C++ compiler toolchains (GCC/MSVC), CMake builds, python environment setups, model loaders, REST dashboard, and training modules require no internet connection after installation.
* **Network Isolation**: The system does not attempt DNS resolution of external domains, does not perform HTTP requests to external servers, and lacks any third-party telemetry, crash reporting, or remote monitoring hooks.

---

## 3. Data Leakage and Privacy Boundaries

EdgePilot maintains absolute data privacy. User data is guaranteed never to leave the local machine.

### Loopback IP Restrictions
The C++ orchestrator daemon binds its TCP socket exclusively to:
$$\text{Host} = \text{127.0.0.1} \quad (\text{localhost})$$
This restricts communication to loopback traffic within the network namespace of the host machine. The daemon does not listen on public network interfaces (e.g. `0.0.0.0`), preventing external devices on the local area network (LAN) or WAN from accessing the socket port `12345`.

### User Input Isolation
Any prompts, code samples, image payloads, or classification requests submitted to the SDKs:
* Are processed in volatile memory (RAM) during inference.
* Are not cached to disk or log files (the telemetry database only logs structural metrics: execution latency, memory footprint, and energy consumption—never the text content of the prompts or generated results).
* Are immediately discarded from memory once the REST API forwards the response.

### Model Storage Boundaries
Custom model files uploaded via the dashboard's Model Profiler remain in their specified local directories. The file selection utility uses a native Windows dialog hook (`tkinter.filedialog.askopenfilename`) executed locally, meaning model file paths are resolved locally without sending metadata to external servers.
