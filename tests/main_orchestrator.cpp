#include "edgepilot/types.h"
#include "edgepilot/model_registry.h"
#include "edgepilot/runtime_registry.h"
#include "edgepilot/lifecycle_manager.h"
#include "edgepilot/ai_scheduler.h"
#include "edgepilot/orchestrator.h"
#include "edgepilot/onnx_runtime_adapter.h"
#include "edgepilot/llama_cpp_adapter.h"

#include <iostream>
#include <thread>
#include <chrono>

using namespace edgepilot;

int main() {
    std::cout << "=== EdgePilot Orchestrator Daemon Startup ===" << std::endl;

    // 1. Instantiate Registries & Managers
    ModelRegistry model_registry;
    RuntimeRegistry runtime_registry;
    ModelLifecycleManager lifecycle_manager(model_registry, runtime_registry);

    // 2. Register Adapters
    auto onnx = std::make_shared<OnnxRuntimeAdapter>();
    auto llama = std::make_shared<LlamaCppAdapter>();
    
    if (!onnx->Initialize().empty() || !llama->Initialize().empty()) {
        std::cerr << "Failed to initialize adapters!" << std::endl;
        return 1;
    }
    runtime_registry.Register(onnx);
    runtime_registry.Register(llama);

    // 3. Register Models
    ModelMetadata m1;
    m1.model_id = "resnet50";
    m1.display_name = "ResNet 50 Image Classifier";
    m1.compatible_runtimes = {"onnx"};
    m1.preferred_runtime = "onnx";
    m1.file_size_bytes = 100 * 1024 * 1024;
    m1.default_file_path = "models/resnet50.onnx";
    model_registry.Register(m1);

    ModelMetadata m2;
    m2.model_id = "llama3-8b";
    m2.display_name = "LLaMA 3 8B Text Gen";
    m2.compatible_runtimes = {"llama_cpp"};
    m2.preferred_runtime = "llama_cpp";
    m2.file_size_bytes = 4500ULL * 1024ULL * 1024ULL;
    m2.default_file_path = "models/llama3-8b.gguf";
    model_registry.Register(m2);

    // 4. Instantiate AIScheduler
    AIScheduler scheduler(model_registry, lifecycle_manager);
    scheduler.ConfigureModelPaths("models");

    // 5. Start Orchestrator (2 worker threads, listening on port 12345)
    Orchestrator orchestrator(model_registry, runtime_registry, lifecycle_manager, scheduler, 2);
    orchestrator.Start("edgepilot_telemetry.csv", 12345);

    std::cout << "[Orchestrator] Daemon is running in persistent server mode. Close terminal or kill process to terminate." << std::endl;
    
    // Run indefinitely as a server
    while (true) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    orchestrator.Stop();
    std::cout << "=== EdgePilot Orchestrator Daemon Shutdown Clean ===" << std::endl;
    return 0;
}
