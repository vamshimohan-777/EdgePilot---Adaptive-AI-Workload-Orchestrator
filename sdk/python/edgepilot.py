import requests
import time

# =============================================================================
# EdgePilot — P4: Platform Layer
# edgepilot.py — Client SDK binding for Python applications.
#
# Provides a simple developer API to interact with the EdgePilot REST gateway.
# =============================================================================

class EdgePilotClient:
    def __init__(self, api_url="http://127.0.0.1:8000"):
        self.api_url = api_url.rstrip("/")

    def submit_job(self, model_id: str, prompt: str, priority: int = 1) -> str:
        """Submits an inference job to the orchestrator queue and returns the job ID."""
        url = f"{self.api_url}/jobs/submit"
        payload = {"model_id": model_id, "priority": priority, "prompt": prompt}
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()["job_id"]

    def query_status(self, job_id: str) -> dict:
        """Queries the current status of a job."""
        url = f"{self.api_url}/jobs/status/{job_id}"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()

    def run_inference(self, model_id: str, prompt: str, priority: int = 1, timeout_secs: float = 10.0) -> str:
        """Synchronously submits a job and blocks waiting for the completed output."""
        job_id = self.submit_job(model_id, prompt, priority)
        
        start_time = time.time()
        while time.time() - start_time < timeout_secs:
            status_info = self.query_status(job_id)
            status = status_info["status"]
            
            if status == "COMPLETED":
                return status_info.get("result", "Success")
            elif status == "FAILED":
                raise RuntimeError(f"EdgePilot inference job '{job_id}' failed.")
            
            time.sleep(0.2)
            
        raise TimeoutError(f"Inference job '{job_id}' timed out after {timeout_secs} seconds.")

    def get_loaded_models(self) -> list:
        """Lists all active models loaded in memory cache."""
        url = f"{self.api_url}/models/loaded"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()

    def get_system_metrics(self) -> dict:
        """Fetches real-time hardware telemetry statistics."""
        url = f"{self.api_url}/metrics"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()

# --- Example Usage Verification ---
if __name__ == "__main__":
    client = EdgePilotClient()
    try:
        print("[SDK] Querying active model registry residency...")
        loaded = client.get_loaded_models()
        print(f"[SDK] Loaded models: {loaded}")
    except Exception as e:
        print(f"[SDK] REST Server Offline (normal if not running): {e}")
