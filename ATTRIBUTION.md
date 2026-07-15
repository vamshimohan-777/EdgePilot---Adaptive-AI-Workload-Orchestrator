# EdgePilot Attribution and References

EdgePilot relies on open-source libraries, pretrained model weights, execution engines, and web frameworks. This document lists the attributions, credits, and licensing details for the technologies that make this orchestrator possible.

---

## 1. Pretrained Model Weights

### LLaMA-3 (Large Language Model Meta AI)
* **Creator**: Meta AI
* **Version**: LLaMA-3 8B
* **File Format**: `.gguf` (quantized format optimized for llama.cpp weight streaming)
* **License**: Meta LLaMA 3 Community License Agreement
* **Usage**: Text generation, natural language processing sandbox, and high-memory execution profiling targets.

### ResNet-50 (Deep Residual Learning for Image Recognition)
* **Creator**: Microsoft Research / ONNX Model Zoo
* **File Format**: `.onnx`
* **License**: MIT License
* **Usage**: Image classification inference workloads, baseline ONNX runtime adapter executions, and computer vision classification sandbox.

---

## 2. Core Execution Runtimes

### ONNX Runtime
* **Creator**: Microsoft Corporation
* **Interface**: C++ API (`onnxruntime_cxx_api.h`)
* **License**: MIT License
* **Usage**: Evaluates the telemetry predictors (latency, memory, energy, thermals) and the PPO scheduler policy network. It also serves as the runtime platform for computer vision classification models.

### llama.cpp
* **Creator**: Georgi Gerganov and the llama.cpp open-source community
* **Interface**: C++ RAL Adapter Wrapper
* **License**: MIT License
* **Usage**: Powering local transformer weight streaming, CPU/GPU hybrid offloading, and dynamic precision weight quantization.

---

## 3. Machine Learning & Training Frameworks

### PyTorch (Torch)
* **Creator**: Meta AI & the Linux Foundation open-source community
* **Interface**: Python API (`torch`, `torch.nn`, `torch.optim`)
* **License**: Modified BSD License
* **Usage**: Powers the reinforcement learning policy (`train_rl_scheduler.py`) Actor-Critic (PPO) training network, online continuous retraining updates (`relearn_online`), and the regression networks for system metrics.

---

## 4. Web Gateway & Application Architecture

### FastAPI
* **Creator**: Sebastián Ramírez (tiangolo)
* **Interface**: REST API Framework
* **License**: MIT License
* **Usage**: Serving REST gateway endpoints (`jobs/submit`, `jobs/status`, `metrics`), hosting native model explorer interfaces, and mounting the Web Dashboard UI.

### SQLite3
* **Creator**: D. Richard Hipp
* **Interface**: Python standard library (`sqlite3`)
* **License**: Public Domain
* **Usage**: Local relational database engine powering telemetry log queries and baseline query lookups.

---

## 5. Support Libraries & Build Systems

* **CMake**: Cross-platform open-source build system generator. Used for managing the C++ core compile sequences.
* **Pandas & NumPy**: Data analysis and mathematical array libraries. Used in Python gateways for telemetry parsing, database synchronization, and state feature scaling.
* **Winsock2 (ws2_32)**: Windows Sockets API. Used in the C++ daemon for low-level TCP IPC communication and socket listener loops on Windows.
