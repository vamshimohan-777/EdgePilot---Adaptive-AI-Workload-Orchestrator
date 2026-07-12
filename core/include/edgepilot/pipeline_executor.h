#pragma once

#include "edgepilot/types.h"
#include "edgepilot/lifecycle_manager.h"

#include <vector>
#include <string>
#include <memory>
#include <optional>

// =============================================================================
// EdgePilot — P2: Execution Kernel
// pipeline_executor.h — Handles sequential execution of multiple models where
//                       outputs of one stage feed into inputs of the next.
// =============================================================================

namespace edgepilot {

struct PipelineStage {
    std::string model_id;
    std::string runtime_name;      ///< Empty = use fallback
    DeviceConfig device_config;
};

struct Pipeline {
    std::string pipeline_id;
    std::vector<PipelineStage> stages;
};

class PipelineExecutor {
public:
    explicit PipelineExecutor(ModelLifecycleManager& lifecycle_manager);
    ~PipelineExecutor() = default;

    // Non-copyable
    PipelineExecutor(const PipelineExecutor&)            = delete;
    PipelineExecutor& operator=(const PipelineExecutor&) = delete;

    /// Executes a sequential pipeline of models.
    /// Pipes text or tensor results from stage N to stage N+1.
    /// Thread-safe.
    InferenceResult Execute(const Pipeline& pipeline, const InferenceRequest& initial_request);

private:
    ModelLifecycleManager& lifecycle_manager_;
};

} // namespace edgepilot
