import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from sdk.python.edgepilot import EdgePilotClient

def main():
    print("==================================================")
    # 1. Instantiate SDK Client
    client = EdgePilotClient("http://127.0.0.1:8000")

    # 2. Submit ONNX job (ResNet-50)
    print("[Verification] Submitting ResNet-50 classification task...")
    try:
        res = client.run_inference("resnet50", "Classify this image")
        print(f"[Verification] ResNet-50 Output: {res}")
    except Exception as e:
        print(f"[Verification] ResNet-50 Failed: {e}")

    # 3. Submit GGUF job (Llama 3)
    print("\n[Verification] Submitting Llama3-8b prompt task...")
    try:
        res = client.run_inference("llama3-8b", "Explain EdgePilot.")
        print(f"[Verification] Llama3-8b Output: {res}")
    except Exception as e:
        print(f"[Verification] Llama3-8b Failed: {e}")

    # 4. Fetch metrics
    print("\n[Verification] Querying real-time telemetry metrics...")
    try:
        metrics = client.get_system_metrics()
        print(f"[Verification] Metrics: {metrics}")
    except Exception as e:
        print(f"[Verification] Metrics Query Failed: {e}")

    # 5. Fetch loaded models
    print("\n[Verification] Querying loaded models in residency cache...")
    try:
        loaded = client.get_loaded_models()
        print(f"[Verification] Loaded models: {loaded}")
    except Exception as e:
        print(f"[Verification] Loaded Models Query Failed: {e}")

    print("==================================================")

if __name__ == "__main__":
    main()
