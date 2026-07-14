from pydantic import BaseModel
from typing import Optional, List
from enum import Enum


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class JobState(str, Enum):
    PENDING   = "pending"
    RUNNING   = "running"
    COMPLETED = "completed"
    FAILED    = "failed"
    CANCELLED = "cancelled"


class Priority(str, Enum):
    LOW    = "low"
    MEDIUM = "medium"
    HIGH   = "high"


class RuntimeType(str, Enum):
    ONNX     = "onnx"
    TFLITE   = "tflite"
    GGUF     = "gguf"
    EXECUTORCH = "executorch"
    AUTO     = "auto"


# ---------------------------------------------------------------------------
# Job Schemas
# ---------------------------------------------------------------------------

class JobRequest(BaseModel):
    model_name:  str
    input_data:  str
    priority:    Priority    = Priority.MEDIUM
    runtime:     RuntimeType = RuntimeType.AUTO
    deadline_ms: Optional[int] = None      # optional max latency in ms


class JobStatus(BaseModel):
    job_id:      str
    state:       JobState
    model_name:  str
    priority:    Priority
    runtime:     Optional[str]   = None    # chosen runtime (filled by P3 scheduler)
    latency_ms:  Optional[float] = None    # filled when job completes
    result:      Optional[str]   = None    # output of the model
    error:       Optional[str]   = None    # error message if failed


class JobListResponse(BaseModel):
    jobs:  List[JobStatus]
    total: int


# ---------------------------------------------------------------------------
# Model Schemas
# ---------------------------------------------------------------------------

class ModelInfo(BaseModel):
    model_id:  str
    name:      str
    format:    str    # "onnx", "tflite", "gguf"
    size_mb:   float
    is_loaded: bool


class ModelListResponse(BaseModel):
    models: List[ModelInfo]
    total:  int


# ---------------------------------------------------------------------------
# Metrics Schemas
# ---------------------------------------------------------------------------

class MetricSnapshot(BaseModel):
    cpu_percent:   float
    ram_percent:   float
    ram_used_mb:   float
    gpu_percent:   Optional[float] = None   # None if no GPU available
    temperature_c: Optional[float] = None   # None if sensor unavailable
    active_jobs:   int
    queued_jobs:   int


class SchedulerDecision(BaseModel):
    job_id:      str
    model_name:  str
    chosen_runtime:  str
    thread_count:    int
    quantization:    str     # "int8", "fp16", "fp32"
    reason:          str     # human-readable explanation


# ---------------------------------------------------------------------------
# Generic Responses
# ---------------------------------------------------------------------------

class SuccessResponse(BaseModel):
    success: bool
    message: str