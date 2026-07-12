import time
import os
import numpy as np
import pandas as pd
try:
    from learning.telemetry_db import TelemetryDatabase
except ImportError:
    from telemetry_db import TelemetryDatabase

# =============================================================================
# EdgePilot — P3: AI Intelligence
# characterizer.py — Model Characterization Engine. Runs synthetic benchmark
#                    workloads on newly registered models to capture initial
#                    performance profiles (solving the cold-start problem).
# =============================================================================

class ModelCharacterizer:
    def __init__(self, csv_path="edgepilot_telemetry.csv"):
        self.db = TelemetryDatabase(csv_path=csv_path)

    def run_characterization(self, model_id, compatible_runtimes, model_path=None):
        """Simulates running synthetic benchmarks on the model to profile it."""
        print(f"[ModelCharacterizer] Starting characterization for model '{model_id}'...")
        time.sleep(1) # Simulate profiling workload run

        file_size = 0
        if model_path and os.path.exists(model_path):
            file_size = os.path.getsize(model_path)
            print(f"[ModelCharacterizer] Found physical model file at '{model_path}' ({file_size / (1024*1024):.2f} MB)")

        records = []
        timestamp = int(time.time() * 1e6)

        # Profile the model across its compatible runtimes
        for runtime in compatible_runtimes:
            print(f"[ModelCharacterizer] Profiling '{model_id}' on runtime '{runtime}'...")
            
            # Simulate 5 benchmark runs to build a baseline profile
            for run in range(5):
                # Heuristic profiling: LLMs have higher latency/memory, ONNX is CPU/GPU based
                if runtime == "llama_cpp":
                    latency_us = np.random.uniform(45000, 95000)
                    peak_memory = file_size + 150 * 1024 * 1024 if file_size > 0 else np.random.uniform(3.5e9, 4.2e9)
                    energy = np.random.uniform(0.5, 1.2)
                else:
                    latency_us = np.random.uniform(5000, 15000)
                    peak_memory = file_size + 50 * 1024 * 1024 if file_size > 0 else np.random.uniform(2e8, 3e8)
                    energy = np.random.uniform(0.05, 0.15)

                records.append({
                    "timestamp_us": timestamp + run * 1000,
                    "model_id": model_id,
                    "runtime_name": runtime,
                    "quantization_variant_id": "fp32",
                    "cpu_utilization": np.random.uniform(20, 60),
                    "gpu_utilization": np.random.uniform(10, 50),
                    "ram_usage_bytes": peak_memory - 50 * 1024 * 1024,
                    "battery_level": 90.0,
                    "device_temperature": 42.0,
                    "queue_length": 0,
                    "latency_us": int(latency_us),
                    "peak_memory_bytes": int(peak_memory),
                    "energy_estimate_joules": energy
                })

        df_new = pd.DataFrame(records)
        header = not os.path.exists(self.db.csv_path)
        df_new.to_csv(self.db.csv_path, mode='a', index=False, header=header)
        
        # Sync to SQLite
        self.db.sync_csv_to_db()

        # Query and show stats
        stats = self.db.get_model_baseline(model_id)
        if stats:
            print(f"[ModelCharacterizer] Cold-start profile created successfully for '{model_id}':")
            print(f"  - Avg Latency: {stats['avg_latency_us']:.2f} us")
            print(f"  - Avg Peak Memory: {stats['avg_peak_memory_bytes'] / 1e6:.2f} MB")
            print(f"  - Avg Energy: {stats['avg_energy_joules']:.4f} Joules")
        else:
            print("[ModelCharacterizer] Failed to read profile from database.")

if __name__ == "__main__":
    characterizer = ModelCharacterizer()
    characterizer.run_characterization("resnet50", ["onnx"])
    characterizer.run_characterization("llama3-8b", ["llama_cpp"])
