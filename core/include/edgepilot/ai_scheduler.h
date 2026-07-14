#pragma once

#include "edgepilot/scheduler_interface.h"
#include "edgepilot/model_registry.h"
#include "edgepilot/lifecycle_manager.h"

#include <memory>
#include <string>

// =============================================================================
// EdgePilot — P2/P3: Execution Kernel / AI Intelligence
// ai_scheduler.h — Declares Heuristic and AI-based scheduling engines.
// =============================================================================

namespace edgepilot {

// ---------------------------------------------------------------------------
// HeuristicScheduler (Rule-Based Fallback)
// ---------------------------------------------------------------------------

class HeuristicScheduler : public IScheduler {
public:
    explicit HeuristicScheduler(const ModelRegistry& model_registry);
    ~HeuristicScheduler() override = default;

    std::string GetName() const override;
    SchedulingDecision Schedule(const Job& job, const SystemState& state) override;

private:
    const ModelRegistry& model_registry_;
};


// ---------------------------------------------------------------------------
// AIScheduler (ONNX-Loaded Hybrid Prediction / RL Agent)
// ---------------------------------------------------------------------------

class AIScheduler : public IScheduler {
public:
    /// Constructs the AI Scheduler.
    ///
    /// If prediction ONNX models exist on disk, it uses ModelLifecycleManager
    /// to load them and evaluate state vectors. Otherwise, it falls back
    /// to a high-fidelity simulation model.
    AIScheduler(const ModelRegistry& model_registry,
                ModelLifecycleManager& lifecycle_manager);

    ~AIScheduler() override;

    std::string GetName() const override;
    SchedulingDecision Schedule(const Job& job, const SystemState& state) override;

    /// Configures the folder containing exported ONNX files (predictors & policy).
    void ConfigureModelPaths(const std::string& model_dir);

private:
    SchedulingDecision RunPredictionModel(const Job& job, const SystemState& state);
    SchedulingDecision RunSimulationModel(const Job& job, const SystemState& state);

    const ModelRegistry&   model_registry_;
    ModelLifecycleManager& lifecycle_manager_;
    std::string            model_dir_;
    bool                   use_onnx_models_ = false;
    void*                  policy_session_ = nullptr;
    void*                  ort_env_ = nullptr;
};

} // namespace edgepilot
