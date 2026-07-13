#include "edgepilot/ai_scheduler.h"

#ifdef __GNUC__
#ifndef _stdcall
#define _stdcall __stdcall
#endif
#endif

#include <onnxruntime_cxx_api.h>

#include <iostream>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <vector>

// =============================================================================
// EdgePilot — P2/P3: Execution Kernel / AI Intelligence
// ai_scheduler.cpp — Real implementation of Heuristic and AI Schedulers.
//
// Loads real RL policy ONNX files and evaluates them dynamically via ORT C++
// to recommend adaptive actions (runtimes, quantization, evictions).
// =============================================================================

namespace edgepilot {

// ===========================================================================
// HeuristicScheduler Implementation
// ===========================================================================

HeuristicScheduler::HeuristicScheduler(const ModelRegistry& model_registry)
    : model_registry_(model_registry)
{
}

std::string HeuristicScheduler::GetName() const {
    return "HeuristicScheduler (Rule-Based)";
}

SchedulingDecision HeuristicScheduler::Schedule(const Job& job, const SystemState& /*state*/) {
    SchedulingDecision decision;
    decision.adjusted_priority = job.priority;
    decision.execution_delay_ms = 0;

    auto meta_opt = model_registry_.Get(job.model_id);
    if (!meta_opt.has_value()) {
        return decision;
    }
    const auto& metadata = meta_opt.value();

    if (!metadata.preferred_runtime.empty()) {
        decision.runtime_name = metadata.preferred_runtime;
    } else if (!metadata.compatible_runtimes.empty()) {
        decision.runtime_name = metadata.compatible_runtimes.front();
    }

    if (!metadata.quantization_variants.empty()) {
        decision.quantization_variant_id = metadata.quantization_variants.front().variant_id;
    }

    return decision;
}


// ===========================================================================
// AIScheduler Implementation
// ===========================================================================

AIScheduler::AIScheduler(const ModelRegistry& model_registry,
                         ModelLifecycleManager& lifecycle_manager)
    : model_registry_(model_registry)
    , lifecycle_manager_(lifecycle_manager)
    , policy_session_(nullptr)
    , ort_env_(nullptr)
{
}

AIScheduler::~AIScheduler() {
    if (policy_session_) {
        delete static_cast<Ort::Session*>(policy_session_);
        policy_session_ = nullptr;
    }
    if (ort_env_) {
        delete static_cast<Ort::Env*>(ort_env_);
        ort_env_ = nullptr;
    }
}

std::string AIScheduler::GetName() const {
    return "AIScheduler (Hybrid Predictor + RL)";
}

void AIScheduler::ConfigureModelPaths(const std::string& model_dir) {
    model_dir_ = model_dir;
    std::string policy_path = model_dir_ + "/rl_scheduler_policy.onnx";

    std::ifstream check(policy_path);
    if (!check.good()) {
        use_onnx_models_ = false;
        return;
    }
    check.close();

    try {
        if (policy_session_) {
            delete static_cast<Ort::Session*>(policy_session_);
            policy_session_ = nullptr;
        }
        if (!ort_env_) {
            ort_env_ = new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "EdgePilotAISchedulerEnv");
        }

        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);

#ifdef _WIN32
        std::wstring wpath(policy_path.begin(), policy_path.end());
        policy_session_ = new Ort::Session(*static_cast<Ort::Env*>(ort_env_), wpath.c_str(), session_options);
#else
        policy_session_ = new Ort::Session(*static_cast<Ort::Env*>(ort_env_), policy_path.c_str(), session_options);
#endif
        use_onnx_models_ = true;
        std::cout << "[AIScheduler] Loaded real RL policy ONNX model successfully!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "[AIScheduler] Failed to load policy ONNX: " << e.what() << std::endl;
        use_onnx_models_ = false;
    }
}

SchedulingDecision AIScheduler::Schedule(const Job& job, const SystemState& state) {
    if (use_onnx_models_ && policy_session_) {
        return RunPredictionModel(job, state);
    } else {
        return RunSimulationModel(job, state);
    }
}

SchedulingDecision AIScheduler::RunPredictionModel(const Job& job, const SystemState& state) {
    SchedulingDecision decision;
    decision.adjusted_priority = job.priority;
    decision.execution_delay_ms = 0;

    auto meta_opt = model_registry_.Get(job.model_id);
    if (!meta_opt.has_value()) {
        return decision;
    }
    const auto& metadata = meta_opt.value();

    // Default runtime configuration fallback
    std::string default_runtime = metadata.preferred_runtime;
    if (default_runtime.empty() && !metadata.compatible_runtimes.empty()) {
        default_runtime = metadata.compatible_runtimes.front();
    }
    decision.runtime_name = default_runtime;

    if (!metadata.quantization_variants.empty()) {
        decision.quantization_variant_id = metadata.quantization_variants.front().variant_id;
    }

    try {
        Ort::Session* session = static_cast<Ort::Session*>(policy_session_);

        // Total physical RAM approximation (16GB) to scale used RAM
        uint64_t total_ram = 16ULL * 1024ULL * 1024ULL * 1024ULL;
        uint64_t used_ram = total_ram > state.ram_free_bytes ? total_ram - state.ram_free_bytes : 1000000000ULL;

        // Construct RL Gym State Vector (Size 10)
        std::vector<float> input_tensor_values = {
            state.cpu_utilization / 100.0f,
            state.gpu_utilization / 100.0f,
            static_cast<float>(used_ram) / 1e10f,
            state.battery_level / 100.0f,
            state.device_temperature / 100.0f,
            static_cast<float>(state.queue_length) / 10.0f,
            0.01f, // latency baseline
            0.02f, // RAM baseline
            0.05f, // energy baseline
            state.device_temperature / 100.0f
        };

        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        std::vector<int64_t> input_dims = {1, 10};
        
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, input_tensor_values.data(), 10, input_dims.data(), 2
        );

        const char* input_names[] = {"system_state"};
        const char* output_names[] = {"action_probs"};

        // Run forward pass of PPO policy
        auto output_tensors = session->Run(
            Ort::RunOptions{nullptr},
            input_names,
            &input_tensor,
            1,
            output_names,
            1
        );

        int action = 0;
        if (!output_tensors.empty() && output_tensors[0].IsTensor()) {
            float* probs = output_tensors[0].GetTensorMutableData<float>();
            float max_prob = -1.0f;
            for (int i = 0; i < 6; ++i) {
                if (probs[i] > max_prob) {
                    max_prob = probs[i];
                    action = i;
                }
            }
        }

        // Apply PPO Action Mapping
        std::string recommended_runtime = "";
        if (action == 0) {
            recommended_runtime = "onnx";
        } else if (action == 1) {
            recommended_runtime = "llama_cpp";
        }

        if (!recommended_runtime.empty()) {
            if (std::find(metadata.compatible_runtimes.begin(), metadata.compatible_runtimes.end(), recommended_runtime) != metadata.compatible_runtimes.end()) {
                decision.runtime_name = recommended_runtime;
            } else {
                decision.runtime_name = default_runtime;
            }
        }

        if (action == 2) {
            // Force quantization compression
            for (const auto& var : metadata.quantization_variants) {
                if (var.variant_id == "int8" || var.variant_id == "int4") {
                    decision.quantization_variant_id = var.variant_id;
                    break;
                }
            }
        } else if (action == 3) {
            // Apply delay to prevent thermal throttling
            decision.execution_delay_ms = 100;
            decision.adjusted_priority = JobPriority::Low;
        } else if (action == 4) {
            decision.preload_models.push_back(job.model_id);
        } else if (action == 5) {
            // Evict other resident models
            for (const auto& loaded : state.loaded_models) {
                if (loaded != job.model_id) {
                    decision.evict_models.push_back(loaded);
                }
            }
        }

    } catch (const std::exception& e) {
        std::cerr << "[AIScheduler] ONNX inference failed, using rule-based fallback: " << e.what() << std::endl;
        return RunSimulationModel(job, state);
    }

    return decision;
}

SchedulingDecision AIScheduler::RunSimulationModel(const Job& job, const SystemState& state) {
    SchedulingDecision decision;
    decision.adjusted_priority = job.priority;
    decision.execution_delay_ms = 0;

    auto meta_opt = model_registry_.Get(job.model_id);
    if (!meta_opt.has_value()) {
        return decision;
    }
    const auto& metadata = meta_opt.value();

    bool is_llm = (metadata.task_type == TaskType::TextGeneration);
    std::string selected_quant = "";
    std::string selected_runtime = "";

    // Low free memory (< 1.5 GB) evictions
    if (state.ram_free_bytes < 1500ULL * 1024ULL * 1024ULL) {
        for (const auto& loaded : state.loaded_models) {
            if (loaded != job.model_id) {
                decision.evict_models.push_back(loaded);
            }
        }
    }

    // Low battery (< 20%) quantization fallback
    if (state.battery_level < 20.0f && !metadata.quantization_variants.empty()) {
        auto min_variant = std::min_element(metadata.quantization_variants.begin(),
                                            metadata.quantization_variants.end(),
                                            [](const QuantizationVariant& a, const QuantizationVariant& b) {
                                                return a.size_bytes < b.size_bytes;
                                            });
        selected_quant = min_variant->variant_id;
    } else {
        if (!metadata.quantization_variants.empty()) {
            selected_quant = metadata.quantization_variants.front().variant_id;
        }
    }

    // High temperature (> 48°C) thermal throttling
    if (state.device_temperature > 48.0f) {
        selected_runtime = "onnx";
        decision.execution_delay_ms = 100;
        decision.adjusted_priority = JobPriority::Low;
    } else {
        if (is_llm && std::find(metadata.compatible_runtimes.begin(),
                                metadata.compatible_runtimes.end(),
                                "llama_cpp") != metadata.compatible_runtimes.end()) {
            selected_runtime = "llama_cpp";
        } else {
            selected_runtime = !metadata.preferred_runtime.empty() ?
                               metadata.preferred_runtime :
                               (!metadata.compatible_runtimes.empty() ? metadata.compatible_runtimes.front() : "");
        }
    }

    if (std::find(state.loaded_models.begin(), state.loaded_models.end(), job.model_id) == state.loaded_models.end()) {
        decision.preload_models.push_back(job.model_id);
    }

    decision.runtime_name = selected_runtime;
    decision.quantization_variant_id = selected_quant;

    return decision;
}

} // namespace edgepilot
