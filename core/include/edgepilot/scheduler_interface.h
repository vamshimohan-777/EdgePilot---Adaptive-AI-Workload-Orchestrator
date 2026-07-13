#pragma once

#include "edgepilot/types.h"

#include <string>

// =============================================================================
// EdgePilot — P2: Execution Kernel
// scheduler_interface.h — Defines the interface that all scheduling engines
//                          must implement (e.g. Heuristic, Reinforcement Learning).
// =============================================================================

namespace edgepilot {

class IScheduler {
public:
    virtual ~IScheduler() = default;

    /// Returns a unique, human-readable name of the scheduling policy/agent.
    virtual std::string GetName() const = 0;

    /// Evaluates system telemetry and job properties to return an execution decision.
    ///
    /// @param job    The inference job requesting execution.
    /// @param state  The current real-time telemetry metrics of the device.
    virtual SchedulingDecision Schedule(const Job& job, const SystemState& state) = 0;
};

} // namespace edgepilot
