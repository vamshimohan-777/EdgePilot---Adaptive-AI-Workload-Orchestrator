import os
import numpy as np
import pandas as pd
from telemetry_db import TelemetryDatabase

# We use standard library try/except block to allow script to run
# even if PyTorch is not pre-installed in the user's local terminal.
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# =============================================================================
# EdgePilot — P3: AI Intelligence
# train_predictors.py — Trains neural networks to predict model performance
#                       (Latency, Memory, Thermals, Energy) and exports to ONNX.
# =============================================================================

class MLPRegressor(nn.Module if TORCH_AVAILABLE else object):
    def __init__(self, input_dim=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        return self.net(x)

def train_and_export_onnx(csv_path="edgepilot_telemetry.csv", output_dir="models"):
    if not TORCH_AVAILABLE:
        print("[PredictorTrainer] PyTorch is not installed. Skipping model training.")
        return

    os.makedirs(output_dir, exist_ok=True)
    db = TelemetryDatabase(csv_path=csv_path)
    df = db.load_training_dataset()

    if df.empty or len(df) < 5:
        print("[PredictorTrainer] Not enough telemetry records in CSV. Creating synthetic data for training.")
        # Create synthetic data to run the exporter
        records = []
        for _ in range(100):
            records.append({
                "cpu_utilization": np.random.uniform(10, 80),
                "gpu_utilization": np.random.uniform(5, 90),
                "ram_usage_bytes": np.random.uniform(1e8, 2e9),
                "battery_level": np.random.uniform(20, 100),
                "device_temperature": np.random.uniform(35, 60),
                "queue_length": np.random.randint(0, 10),
                "runtime_name": "onnx" if np.random.rand() > 0.5 else "llama_cpp",
                "quantization_variant_id": "fp32" if np.random.rand() > 0.5 else "int8",
                "latency_us": np.random.uniform(1000, 50000),
                "peak_memory_bytes": np.random.uniform(2e8, 3e9),
                "energy_estimate_joules": np.random.uniform(0.01, 2.5)
            })
        df = pd.DataFrame(records)

    # Feature engineering
    # Features: [cpu, gpu, ram, battery, temp, queue_len, runtime_idx, quant_idx]
    X = []
    for _, row in df.iterrows():
        runtime_idx = 1.0 if row["runtime_name"] == "llama_cpp" else 0.0
        quant_idx = 1.0 if row["quantization_variant_id"] == "int8" else 0.0
        X.append([
            row["cpu_utilization"],
            row["gpu_utilization"],
            row["ram_usage_bytes"] / 1e9, # Scale RAM to GB
            row["battery_level"] / 100.0,  # Scale battery
            row["device_temperature"],
            float(row["queue_length"]),
            runtime_idx,
            quant_idx
        ])
    
    X = np.array(X, dtype=np.float32)
    
    targets = {
        "latency_predictor.onnx": df["latency_us"].values.astype(np.float32),
        "memory_predictor.onnx": df["peak_memory_bytes"].values.astype(np.float32) / 1e9, # target in GB
        "thermal_predictor.onnx": df["device_temperature"].values.astype(np.float32),
        "energy_predictor.onnx": df["energy_estimate_joules"].values.astype(np.float32)
    }

    for model_name, y in targets.items():
        print(f"[PredictorTrainer] Training {model_name}...")
        
        # Instantiate model
        model = MLPRegressor(input_dim=8)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)

        # PyTorch tensors
        inputs_t = torch.tensor(X)
        targets_t = torch.tensor(y).unsqueeze(1)

        # Simple epoch training loop
        model.train()
        for epoch in range(100):
            optimizer.zero_grad()
            outputs = model(inputs_t)
            loss = criterion(outputs, targets_t)
            loss.backward()
            optimizer.step()

        # Export to ONNX
        model.eval()
        dummy_input = torch.randn(1, 8, dtype=torch.float32)
        onnx_path = os.path.join(output_dir, model_name)
        
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['system_state'],
            output_names=['prediction'],
            dynamic_axes={'system_state': {0: 'batch_size'}, 'prediction': {0: 'batch_size'}}
        )
        print(f"[PredictorTrainer] Exported prediction model successfully to {onnx_path}")

if __name__ == "__main__":
    train_and_export_onnx()
