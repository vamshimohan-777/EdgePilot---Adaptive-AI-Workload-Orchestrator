"""
EdgePilot API Server
====================
FastAPI server exposing Job, Model, and Metrics endpoints.
Run with:  uvicorn api.app:app --reload --port 8000
"""

import asyncio
import json
import uuid
import time

import psutil
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import (
    JobRequest, JobStatus, JobState, JobListResponse,
    ModelInfo, ModelListResponse,
    MetricSnapshot, SchedulerDecision,
    SuccessResponse, Priority, RuntimeType
)


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="EdgePilot API",
    description="Intelligent on-device AI workload orchestrator — schedules AI inference jobs across CPU/GPU/NPU runtimes.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# In-memory stores  (replaced by P1/P2 internals on Day 4)
# ---------------------------------------------------------------------------

_jobs: dict[str, dict] = {}

_mock_models = [
    ModelInfo(model_id="m1", name="llama-3.2-1b", format="gguf",    size_mb=4200, is_loaded=True),
    ModelInfo(model_id="m2", name="whisper-tiny",  format="onnx",   size_mb=290,  is_loaded=False),
    ModelInfo(model_id="m3", name="mobilenet-v3",  format="tflite", size_mb=14,   is_loaded=True),
]

_scheduler_decisions: list[dict] = []


# ---------------------------------------------------------------------------
# Root
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse, tags=["Root"])
def root():
    return """
    <!DOCTYPE html>
    <html>
    <head><title>EdgePilot</title></head>
    <body style="font-family:sans-serif;padding:40px;background:#0d1117;color:#e6edf3">
        <h1>⚡ EdgePilot API</h1>
        <p>Intelligent on-device AI workload orchestrator</p>
        <a href="/docs"  style="color:#58a6ff">📄 Swagger Docs</a> &nbsp;|&nbsp;
        <a href="/redoc" style="color:#58a6ff">📘 ReDoc</a> &nbsp;|&nbsp;
        <a href="/health" style="color:#58a6ff">❤️ Health</a>
    </body>
    </html>
    """


@app.get("/health", tags=["Root"])
def health():
    return {"status": "ok", "service": "EdgePilot", "version": "1.0.0"}


# ---------------------------------------------------------------------------
# Jobs  —  /jobs/*
# ---------------------------------------------------------------------------

@app.post("/jobs/submit", response_model=JobStatus, tags=["Jobs"])
def submit_job(request: JobRequest):
    """Submit a new AI inference job to the EdgePilot scheduler."""
    job_id = str(uuid.uuid4())[:8]

    # TODO Day 4: replace with P2's JobQueue.submit()
    job = {
        "job_id":     job_id,
        "state":      JobState.PENDING,
        "model_name": request.model_name,
        "priority":   request.priority,
        "runtime":    request.runtime.value if request.runtime != RuntimeType.AUTO else None,
        "latency_ms": None,
        "result":     None,
        "error":      None,
    }
    _jobs[job_id] = job

    # Mock: immediately mark as completed with fake result
    # Remove these 3 lines on Day 4
    _jobs[job_id]["state"]      = JobState.COMPLETED
    _jobs[job_id]["latency_ms"] = round(120 + len(request.input_data) * 0.5, 2)
    _jobs[job_id]["result"]     = f"[mock] Ran '{request.model_name}' on: {request.input_data[:50]}"
    _jobs[job_id]["runtime"]    = "onnx"

    return _jobs[job_id]


@app.get("/jobs/{job_id}", response_model=JobStatus, tags=["Jobs"])
def get_job(job_id: str):
    """Get the current status of a job by its ID."""
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    return _jobs[job_id]


@app.get("/jobs", response_model=JobListResponse, tags=["Jobs"])
def list_jobs():
    """List all jobs (pending, running, completed, failed)."""
    jobs = list(_jobs.values())
    return {"jobs": jobs, "total": len(jobs)}


@app.delete("/jobs/{job_id}", response_model=SuccessResponse, tags=["Jobs"])
def cancel_job(job_id: str):
    """Cancel a pending or running job."""
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    if _jobs[job_id]["state"] in [JobState.COMPLETED, JobState.FAILED, JobState.CANCELLED]:
        return {"success": False, "message": f"Job '{job_id}' is already {_jobs[job_id]['state']}"}
    _jobs[job_id]["state"] = JobState.CANCELLED
    return {"success": True, "message": f"Job '{job_id}' cancelled"}


# ---------------------------------------------------------------------------
# Models  —  /models/*
# ---------------------------------------------------------------------------

@app.get("/models/list", response_model=ModelListResponse, tags=["Models"])
def list_models():
    """List all registered AI models and their load status."""
    # TODO Day 4: replace with P1's ModelRegistry.list()
    return {"models": _mock_models, "total": len(_mock_models)}


@app.post("/models/{model_id}/load", response_model=SuccessResponse, tags=["Models"])
def load_model(model_id: str):
    """Load a model into memory (warm it up)."""
    # TODO Day 4: replace with P1's ModelLoader.load(model_id)
    for m in _mock_models:
        if m.model_id == model_id:
            m.is_loaded = True
            return {"success": True, "message": f"Model '{m.name}' loaded"}
    raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")


@app.post("/models/{model_id}/unload", response_model=SuccessResponse, tags=["Models"])
def unload_model(model_id: str):
    """Unload a model from memory to free up RAM."""
    # TODO Day 4: replace with P1's ModelLoader.unload(model_id)
    for m in _mock_models:
        if m.model_id == model_id:
            m.is_loaded = False
            return {"success": True, "message": f"Model '{m.name}' unloaded"}
    raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")


# ---------------------------------------------------------------------------
# Metrics  —  /metrics/*
# ---------------------------------------------------------------------------

@app.get("/metrics/snapshot", response_model=MetricSnapshot, tags=["Metrics"])
def metrics_snapshot():
    """Get a single real-time snapshot of device resource usage."""
    mem = psutil.virtual_memory()
    active  = sum(1 for j in _jobs.values() if j["state"] == JobState.RUNNING)
    queued  = sum(1 for j in _jobs.values() if j["state"] == JobState.PENDING)
    return MetricSnapshot(
        cpu_percent   = psutil.cpu_percent(interval=0.2),
        ram_percent   = mem.percent,
        ram_used_mb   = round(mem.used / (1024 * 1024), 1),
        gpu_percent   = None,   # TODO: add pynvml for GPU
        temperature_c = None,   # TODO: psutil.sensors_temperatures() on Linux
        active_jobs   = active,
        queued_jobs   = queued,
    )


@app.get("/metrics/decisions", tags=["Metrics"])
def get_scheduler_decisions():
    """
    Get the list of recent scheduler decisions made by the AI engine (P3).
    Shows which runtime was chosen and why.
    """
    # TODO Day 4: replace with P3's PolicyEngine.get_decisions()
    mock = [
        {"job_id": "abc123", "model_name": "llama-3.2-1b", "chosen_runtime": "onnx",
         "thread_count": 4, "quantization": "int8",  "reason": "GPU temp 87°C — using CPU/ONNX"},
        {"job_id": "def456", "model_name": "mobilenet-v3",  "chosen_runtime": "tflite",
         "thread_count": 2, "quantization": "fp16",  "reason": "Low RAM — using lightweight runtime"},
        {"job_id": "ghi789", "model_name": "whisper-tiny",  "chosen_runtime": "onnx",
         "thread_count": 4, "quantization": "fp32",  "reason": "High priority job — full precision"},
    ]
    return {"decisions": mock, "total": len(mock)}


# ---------------------------------------------------------------------------
# WebSocket  —  /ws/telemetry  (live streaming)
# ---------------------------------------------------------------------------

@app.websocket("/ws/telemetry")
async def telemetry(websocket: WebSocket):
    """
    WebSocket endpoint that streams real-time device metrics every second.
    Connect from the dashboard with:  new WebSocket('ws://localhost:8000/ws/telemetry')
    """
    await websocket.accept()
    try:
        while True:
            mem = psutil.virtual_memory()
            active = sum(1 for j in _jobs.values() if j["state"] == JobState.RUNNING)
            queued = sum(1 for j in _jobs.values() if j["state"] == JobState.PENDING)
            payload = {
                "cpu":       psutil.cpu_percent(),
                "ram":       mem.percent,
                "ram_mb":    round(mem.used / (1024 * 1024), 1),
                "gpu":       None,
                "temp":      None,
                "active":    active,
                "queued":    queued,
                "timestamp": time.time(),
            }
            await websocket.send_text(json.dumps(payload))
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        pass
