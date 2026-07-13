#include "edgepilot/pipeline_executor.h"

#include <iostream>
#include <chrono>

// =============================================================================
// EdgePilot — P2: Execution Kernel
// pipeline_executor.cpp — Implementation of the PipelineExecutor.
// =============================================================================

namespace edgepilot {

PipelineExecutor::PipelineExecutor(ModelLifecycleManager& lifecycle_manager)
    : lifecycle_manager_(lifecycle_manager)
{
}

// ---------------------------------------------------------------------------
// Execute
// ---------------------------------------------------------------------------

InferenceResult PipelineExecutor::Execute(const Pipeline& pipeline,
                                        const InferenceRequest& initial_request)
{
    InferenceResult final_result;
    final_result.request_id = initial_request.request_id;
    final_result.success    = false;

    if (pipeline.stages.empty()) {
        final_result.error_message = "Pipeline has no execution stages";
        return final_result;
    }

    auto start_time = std::chrono::steady_clock::now();
    uint64_t accumulated_latency = 0;

    // We copy the request, and for subsequent stages, we modify the inputs.
    InferenceRequest current_request = initial_request;

    for (std::size_t i = 0; i < pipeline.stages.size(); ++i) {
        const auto& stage = pipeline.stages[i];

        // 1. Load the model via LifecycleManager
        auto model_opt = lifecycle_manager_.Load(stage.model_id, stage.runtime_name, stage.device_config);
        if (!model_opt.has_value()) {
            final_result.error_message = "Pipeline failed at stage " + std::to_string(i) +
                                         ": Failed to load model '" + stage.model_id + "'";
            return final_result;
        }
        auto model = model_opt.value();

        // 2. Prepare the request (set target model ID)
        current_request.model_id = stage.model_id;

        // 3. Execute inference safely with Acquire/Release locks
        lifecycle_manager_.AcquireInference(stage.model_id);
        InferenceResult stage_result = model->RunInference(current_request);
        lifecycle_manager_.ReleaseInference(stage.model_id);

        if (!stage_result.success) {
            final_result.error_message = "Pipeline failed at stage " + std::to_string(i) +
                                         " (" + stage.model_id + "): " + stage_result.error_message;
            return final_result;
        }

        accumulated_latency += stage_result.latency_us;

        // 4. Pipe output of stage N as input to stage N+1
        if (i < pipeline.stages.size() - 1) {
            current_request.tensor_inputs = stage_result.tensor_outputs;
            
            // Standard Text Generation piping:
            // If the model generated text, set that text as the "prompt" for the next stage.
            auto it = stage_result.text_outputs.find("generated_text");
            if (it != stage_result.text_outputs.end()) {
                current_request.text_inputs["prompt"] = it->second;
            } else {
                current_request.text_inputs = stage_result.text_outputs;
            }
        } else {
            // Final stage: construct final result
            final_result.tensor_outputs = stage_result.tensor_outputs;
            final_result.text_outputs   = stage_result.text_outputs;
        }
    }

    auto end_time = std::chrono::steady_clock::now();
    uint64_t total_elapsed = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count());

    final_result.success    = true;
    final_result.latency_us = total_elapsed; // total clock wall-time of the pipeline

    return final_result;
}

} // namespace edgepilot
