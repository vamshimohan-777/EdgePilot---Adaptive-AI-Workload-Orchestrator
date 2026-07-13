#pragma once

#include "edgepilot/types.h"

#include <string>
#include <mutex>
#include <fstream>

// =============================================================================
// EdgePilot — P2: Execution Kernel / Telemetry
// telemetry_collector.h — Gathers system stats and logs execution metrics.
//
// Telemetry records are written to a structured CSV file (`edgepilot_telemetry.csv`)
// which serves as the local dataset for the offline Python ML trainer.
// =============================================================================

namespace edgepilot {

struct TelemetryRecord {
    std::string model_id;
    std::string runtime_name;
    std::string quantization_variant_id;
    float cpu_utilization;
    float gpu_utilization;
    uint64_t ram_usage_bytes;
    float battery_level;
    float device_temperature;
    uint32_t queue_length;
    uint64_t latency_us;
    uint64_t peak_memory_bytes;
    float energy_estimate_joules;
    uint64_t timestamp_us;
};

class TelemetryCollector {
public:
    TelemetryCollector()  = default;
    ~TelemetryCollector();

    // Non-copyable
    TelemetryCollector(const TelemetryCollector&)            = delete;
    TelemetryCollector& operator=(const TelemetryCollector&) = delete;

    /// Initializes the collector and opens the CSV log file.
    /// Writes the header row if the file is new.
    void Initialize(const std::string& log_path);

    /// Logs a single execution record to the CSV file.
    /// Thread-safe.
    void LogExecution(const TelemetryRecord& record);

    /// Samples the real-time system metrics (CPU, memory, battery, temperature).
    /// Thread-safe.
    SystemState GetCurrentState(uint32_t current_queue_length,
                                const std::vector<std::string>& loaded_models);

private:
    std::string log_path_;
    std::ofstream log_file_;
    mutable std::mutex file_mutex_;

    // Platform-specific sampler helpers
    float SampleCPU();
    float SampleGPU();
    uint64_t GetFreeRAM();
    float GetBatteryLevel();
    float GetTemperature();
};

} // namespace edgepilot
