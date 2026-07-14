// =============================================================================
// EdgePilot — P4: Platform Layer
// app.js — Client dashboard script executing real-time profiling,
//          explainable scheduler parsing, and batch loops.
// =============================================================================

const API_BASE = window.location.origin;

// State tracking
let jobIds = [];
let batchJobs = [];
let currentDemoStep = 1;

// DOM elements
const cpuVal = document.getElementById('cpu-val');
const cpuBar = document.getElementById('cpu-bar');
const gpuVal = document.getElementById('gpu-val');
const gpuBar = document.getElementById('gpu-bar');
const ramVal = document.getElementById('ram-val');
const tempVal = document.getElementById('temp-val');
const tempStatus = document.getElementById('temp-status');
const battVal = document.getElementById('batt-val');
const battStatus = document.getElementById('batt-status');
const queueVal = document.getElementById('queue-val');
const statusBadge = document.getElementById('status-badge');

const loadedModelsList = document.getElementById('loaded-models-list');
const historyRows = document.getElementById('job-history-rows');

// Sandbox DOM elements
const sandboxConsole = document.getElementById('sandbox-console');
const sandboxBatchGrid = document.getElementById('sandbox-batch-grid');

// ---------------------------------------------------------------------------
// Tab Switcher
// ---------------------------------------------------------------------------

function switchTab(tabId) {
    document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
    document.getElementById(`tab-${tabId}`).classList.add('active');

    document.querySelectorAll('.tab-button').forEach(btn => btn.classList.remove('active'));
    
    const clickedBtn = Array.from(document.querySelectorAll('.tab-button')).find(
        btn => btn.innerText.includes(tabId.toUpperCase()) || btn.innerText.includes("SANDBOX") && tabId === "sandbox" || btn.innerText.includes("WALKTHROUGH") && tabId === "demostory"
    );
    if (clickedBtn) clickedBtn.classList.add('active');
}

// ---------------------------------------------------------------------------
// Telemetry Metrics Polling
// ---------------------------------------------------------------------------

async function pollMetrics() {
    try {
        const response = await fetch(`${API_BASE}/metrics`);
        if (!response.ok) throw new Error("API Offline");
        const data = await response.json();

        // Update Gauges
        cpuVal.innerText = `${data.cpu.toFixed(1)}%`;
        cpuBar.style.width = `${data.cpu}%`;

        gpuVal.innerText = `${data.gpu.toFixed(1)}%`;
        gpuBar.style.width = `${data.gpu}%`;

        const ramGB = data.ram / (1024 * 1024 * 1024);
        ramVal.innerText = `${ramGB.toFixed(2)} GB`;

        tempVal.innerText = `${data.temperature.toFixed(1)}°C`;
        if (data.temperature > 48.0) {
            tempStatus.innerText = "Hot (Throttling)";
            tempStatus.style.color = "#ef4444";
        } else {
            tempStatus.innerText = "Normal";
            tempStatus.style.color = "#9f9cb5";
        }

        battVal.innerText = `${data.battery.toFixed(0)}%`;
        battStatus.innerText = data.battery < 20 ? "Low Battery" : "Discharging";
        battStatus.style.color = data.battery < 20 ? "#ef4444" : "#9f9cb5";

        queueVal.innerText = data.queue_length;

        // Reset badge
        statusBadge.className = "connection-status";
        statusBadge.innerHTML = `<span class="status-dot pulse"></span><span class="status-text">DAEMON CONNECTED</span>`;

    } catch (e) {
        statusBadge.className = "connection-status error";
        statusBadge.innerHTML = `<span class="status-dot error-dot"></span><span class="status-text" style="color: #ef4444;">DAEMON DISCONNECTED</span>`;
    }
}

// ---------------------------------------------------------------------------
// Residency Cache Polling
// ---------------------------------------------------------------------------

async function pollLoadedModels() {
    try {
        const response = await fetch(`${API_BASE}/models/loaded`);
        if (!response.ok) return;
        const models = await response.json();

        if (models.length === 0) {
            loadedModelsList.innerHTML = `<li class="empty-state">No models loaded in memory.</li>`;
            return;
        }

        loadedModelsList.innerHTML = models.map(m => `
            <li class="residency-item">
                <span class="model-tag">${m}</span>
                <span class="active-badge">IN MEMORY</span>
            </li>
        `).join('');

    } catch (e) {
        // ignore offline
    }
}

// ---------------------------------------------------------------------------
// Sandbox Terminal Logging
// ---------------------------------------------------------------------------

function logToConsole(text, type = "system") {
    const line = document.createElement('div');
    line.className = `console-line ${type}-line`;
    
    const time = new Date().toLocaleTimeString();
    line.innerHTML = `[${time}] ${text}`;
    
    sandboxConsole.appendChild(line);
    sandboxConsole.scrollTop = sandboxConsole.scrollHeight;
}

// Helper to wait/poll for a specific job to finish
async function waitForJobCompletion(jobId) {
    const maxRetries = 50;
    for (let i = 0; i < maxRetries; i++) {
        try {
            const response = await fetch(`${API_BASE}/jobs/status/${jobId}`);
            if (response.ok) {
                const data = await response.json();
                if (data.status === "COMPLETED") {
                    return { success: true, result: data.result };
                } else if (data.status === "FAILED") {
                    return { success: false, error: "Orchestrator returned failed status" };
                }
            }
        } catch (e) {
            // ignore socket glitch
        }
        await new Promise(resolve => setTimeout(resolve, 200));
    }
    return { success: false, error: "Timeout waiting for job completion" };
}

// ---------------------------------------------------------------------------
// Scenario 1: Run Sequential Model Pipeline
// ---------------------------------------------------------------------------

async function runSequentialPipeline() {
    sandboxBatchGrid.style.display = "none";
    sandboxConsole.style.height = "100%";

    const initialInput = document.getElementById('pipe-prompt').value.trim() || "Classification target";

    logToConsole("==================================================", "system");
    logToConsole(`[PIPELINE] Initializing sequential pipeline execution...`, "info");
    logToConsole(`[PIPELINE] Input: "${initialInput}"`, "info");

    logToConsole(`[STAGE 1] Loading 'resnet50' via ModelLifecycleManager...`, "system");
    
    try {
        const stage1Response = await fetch(`${API_BASE}/jobs/submit`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model_id: "resnet50", priority: 2, prompt: initialInput })
        });
        
        if (!stage1Response.ok) throw new Error("Stage 1 submit failed");
        const stage1Job = await stage1Response.json();
        
        // Add to jobIds list for the history table
        jobIds.push({ id: stage1Job.job_id, model_id: "resnet50", priority: 2, status: "QUEUED" });
        
        logToConsole(`[STAGE 1] Workload dispatched. Job ID: ${stage1Job.job_id}. Priority: HIGH.`, "info");
        logToConsole(`[STAGE 1] Worker assigned. Running inference...`, "system");
        
        const stage1Result = await waitForJobCompletion(stage1Job.job_id);
        if (!stage1Result.success) throw new Error(stage1Result.error);
        
        logToConsole(`[STAGE 1] Completed successfully. ResNet-50 classified pipeline object context.`, "success");

        const pipedPrompt = `Explain the following classified target: ${initialInput}`;
        logToConsole(`[PIPELINE] Piping classification context to next stage...`, "system");
        logToConsole(`[STAGE 2] Submitting to 'llama3-8b' on GGUF runtime...`, "system");
        logToConsole(`[STAGE 2] Piped Prompt: "${pipedPrompt}"`, "info");

        const stage2Response = await fetch(`${API_BASE}/jobs/submit`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model_id: "llama3-8b", priority: 1, prompt: pipedPrompt })
        });
        
        if (!stage2Response.ok) throw new Error("Stage 2 submit failed");
        const stage2Job = await stage2Response.json();
        
        // Add to jobIds list for the history table
        jobIds.push({ id: stage2Job.job_id, model_id: "llama3-8b", priority: 1, status: "QUEUED" });
        
        logToConsole(`[STAGE 2] Workload dispatched. Job ID: ${stage2Job.job_id}. Priority: NORMAL.`, "info");
        logToConsole(`[STAGE 2] Worker assigned. Running text decoding...`, "system");
        
        const stage2Result = await waitForJobCompletion(stage2Job.job_id);
        if (!stage2Result.success) throw new Error(stage2Result.error);
        
        logToConsole(`[STAGE 2] Completed successfully. text generated.`, "success");
        logToConsole(`[RESULT] pipeline_output: "${stage2Result.result}"`, "success");
        logToConsole("==================================================", "system");

        pollJobStatuses();

    } catch (err) {
        logToConsole(`[ERROR] Pipeline failed: ${err.message}`, "error");
        logToConsole("==================================================", "system");
    }
}

// ---------------------------------------------------------------------------
// Scenario 2: Run Concurrent Batch Load Test
// ---------------------------------------------------------------------------

async function runConcurrentBatch() {
    sandboxBatchGrid.style.display = "grid";
    sandboxConsole.style.height = "250px";
    
    logToConsole("==================================================", "system");
    logToConsole(`[BATCH-RUN] Generating 10 concurrent workloads...`, "info");

    const batchModels = ["resnet50", "llama3-8b"];
    const priorities = [0, 1, 2, 3];
    batchJobs = [];

    sandboxBatchGrid.innerHTML = "";

    const submitPromises = [];
    for (let i = 0; i < 10; i++) {
        const model_id = batchModels[i % 2];
        const priority = priorities[i % 4];
        const prompt = `Batch benchmark workload index ${i}`;

        const promise = fetch(`${API_BASE}/jobs/submit`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model_id, priority, prompt })
        }).then(res => res.json()).then(data => {
            const jobObj = {
                id: data.job_id,
                model_id,
                priority,
                status: "QUEUED"
            };
            batchJobs.push(jobObj);
            jobIds.push(jobObj);
            
            const card = document.createElement('div');
            card.className = "batch-job-card queued";
            card.id = `batch-card-${data.job_id}`;
            
            const priorityNames = ["LOW", "NORMAL", "HIGH", "REAL-TIME"];
            card.innerHTML = `
                <div class="batch-job-id">${data.job_id}</div>
                <div class="batch-job-model">${model_id}</div>
                <div style="font-size: 10px; color: var(--text-secondary);">PRIORITY: ${priorityNames[priority]}</div>
                <div class="batch-job-status status-queued" id="batch-status-${data.job_id}">QUEUED</div>
            `;
            sandboxBatchGrid.appendChild(card);
            return jobObj;
        });
        
        submitPromises.push(promise);
    }

    const submittedJobs = await Promise.all(submitPromises);
    logToConsole(`[BATCH-RUN] 10 jobs submitted. C++ Orchestrator sorting dispatch queue...`, "info");

    submittedJobs.forEach(job => {
        pollBatchJob(job);
    });
}

async function pollBatchJob(job) {
    const maxRetries = 60;
    const priorityNames = ["LOW", "NORMAL", "HIGH", "REAL-TIME"];
    
    const interval = setInterval(async () => {
        try {
            const response = await fetch(`${API_BASE}/jobs/status/${job.id}`);
            if (!response.ok) return;
            const data = await response.json();

            const card = document.getElementById(`batch-card-${job.id}`);
            const statusLabel = document.getElementById(`batch-status-${job.id}`);
            
            if (!card) {
                clearInterval(interval);
                return;
            }

            if (data.status !== job.status) {
                job.status = data.status;
                logToConsole(`[SCHEDULER] Job ${job.id} (${job.model_id}) transitioned to ${data.status} [Priority: ${priorityNames[job.priority]}]`, "system");
                
                card.className = `batch-job-card ${data.status.toLowerCase()}`;
                statusLabel.className = `batch-job-status status-${data.status.toLowerCase()}`;
                statusLabel.innerText = data.status;
            }

            if (data.status === "COMPLETED") {
                clearInterval(interval);
                card.classList.add('completed');
                logToConsole(`[SUCCESS] Job ${job.id} completed. Telemetry metrics logged to SQLite.`, "success");
            } 
            else if (data.status === "FAILED") {
                clearInterval(interval);
                card.classList.add('failed');
                logToConsole(`[ERROR] Job ${job.id} failed in orchestrator kernel.`, "error");
            }

        } catch (e) {
            // ignore network glitch
        }
    }, 250);
}

// ---------------------------------------------------------------------------
// Job Status Polling (Sync with main timeline logs)
// ---------------------------------------------------------------------------

async function pollJobStatuses() {
    for (let job of jobIds) {
        if (job.status === "COMPLETED" || job.status === "FAILED") {
            continue;
        }
        try {
            const response = await fetch(`${API_BASE}/jobs/status/${job.id}`);
            if (response.ok) {
                const data = await response.json();
                job.status = data.status;
                if (data.status === "COMPLETED") {
                    job.result = data.result;
                }
            }
        } catch (e) {}
    }
    updateHistoryTable();
}

function updateHistoryTable() {
    if (jobIds.length === 0) {
        historyRows.innerHTML = `<tr><td colspan="5" class="empty-table">No workloads dispatched in this session.</td></tr>`;
        return;
    }

    const priorityNames = ["Low", "Normal", "High", "Real-Time"];

    historyRows.innerHTML = jobIds.map(job => {
        let badgeClass = "badge-status";
        if (job.status === "COMPLETED") badgeClass += " status-completed";
        else if (job.status === "EXECUTING" || job.status === "QUEUED") badgeClass += " status-executing";
        else badgeClass += " status-failed";

        return `
            <tr>
                <td style="font-family: monospace; font-weight: 600;">${job.id}</td>
                <td style="font-family: monospace; color: var(--accent-violet);">${job.model_id}</td>
                <td><span class="${badgeClass}">${job.status}</span></td>
                <td>${priorityNames[job.priority]}</td>
                <td style="color: var(--text-secondary); max-width: 300px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">
                    ${job.status === "COMPLETED" ? (job.result || "Success") : (job.status === "FAILED" ? "Error executing job" : "Processing...")}
                </td>
            </tr>
        `;
    }).reverse().join('');
}

async function registerAndProfileModel() {
    sandboxBatchGrid.style.display = "none";
    sandboxConsole.style.height = "100%";

    const modelIdInput = document.getElementById('reg-model-id');
    const modelId = modelIdInput.value.trim().toLowerCase();
    const modelPathInput = document.getElementById('reg-model-path');
    const modelPath = modelPathInput.value.trim();
    
    if (!modelId) {
        logToConsole("[ERROR] Model Identifier cannot be empty.", "error");
        return;
    }

    if (!modelPath) {
        logToConsole("[ERROR] Local Model File Path cannot be empty.", "error");
        return;
    }

    const runtimes = [];
    if (document.getElementById('reg-rt-onnx').checked) runtimes.push("onnx");
    if (document.getElementById('reg-rt-llama').checked) runtimes.push("llama_cpp");

    if (runtimes.length === 0) {
        logToConsole("[ERROR] Select at least one compatible runtime.", "error");
        return;
    }

    logToConsole("==================================================", "system");
    logToConsole(`[CHARACTERIZER] Initiating live hardware characterization for '${modelId}'...`, "info");
    logToConsole(`[CHARACTERIZER] Target Runtimes: ${runtimes.join(', ')}`, "info");
    logToConsole(`[CHARACTERIZER] Running hardware loops. Monitoring RAM, Thermals, and Energy...`, "system");

    try {
        const response = await fetch(`${API_BASE}/models/register`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model_id: modelId, model_path: modelPath, compatible_runtimes: runtimes })
        });

        if (!response.ok) {
            const data = await response.json();
            throw new Error(data.detail || "Characterization request failed");
        }

        const stats = await response.json();
        logToConsole(`[CHARACTERIZER] Characterization complete! Model '${modelId}' registered.`, "success");
        logToConsole(`[METRICS] - Avg Latency: ${stats.avg_latency_ms} ms`, "success");
        logToConsole(`[METRICS] - Peak RAM:    ${stats.avg_peak_memory_mb} MB`, "success");
        logToConsole(`[METRICS] - Est. Energy: ${stats.avg_energy_joules} Joules`, "success");
        logToConsole("==================================================", "system");

        modelIdInput.value = "";
        modelPathInput.value = "";
        pollLoadedModels();

    } catch (err) {
        logToConsole(`[ERROR] Characterization failed: ${err.message}`, "error");
        logToConsole("==================================================", "system");
    }
}

// Poll everything
setInterval(pollMetrics, 1500);
setInterval(pollLoadedModels, 2000);
setInterval(pollJobStatuses, 1000);

// Init
pollMetrics();
pollLoadedModels();
updateHistoryTable();

async function browseModelFile() {
    const btn = document.getElementById('btn-browse-model');
    const pathInput = document.getElementById('reg-model-path');

    // Show loading state on button
    const originalHTML = btn.innerHTML;
    btn.disabled = true;
    btn.innerHTML = `
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"
             style="animation: spin 1s linear infinite;">
            <path d="M21 12a9 9 0 1 1-6.219-8.56"/>
        </svg>
        Opening…`;
    btn.style.opacity = '0.7';
    btn.style.cursor = 'not-allowed';

    // Ensure spin animation exists
    if (!document.getElementById('ep-spin-style')) {
        const style = document.createElement('style');
        style.id = 'ep-spin-style';
        style.textContent = '@keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }';
        document.head.appendChild(style);
    }

    try {
        const response = await fetch(`${API_BASE}/models/select-file`);
        if (!response.ok) {
            const data = await response.json();
            throw new Error(data.detail || "Failed to open file dialog");
        }
        const result = await response.json();
        if (result.path) {
            // Populate path, update input styling to look active
            pathInput.value = result.path;
            pathInput.style.fontStyle = 'normal';
            pathInput.style.color = '#e2e8f0';
            pathInput.style.borderColor = 'rgba(139,92,246,0.6)';

            logToConsole(`[SYSTEM] Selected model file: ${result.path}`, "success");

            // Auto-populate Model Identifier from file basename if empty
            const modelIdInput = document.getElementById('reg-model-id');
            if (!modelIdInput.value.trim()) {
                const parts = result.path.split(/[/\\]/);
                const filename = parts[parts.length - 1];
                const baseName = filename.substring(0, filename.lastIndexOf('.')) || filename;
                modelIdInput.value = baseName.toLowerCase().replace(/[^a-z0-9_-]/g, '_');
                logToConsole(`[SYSTEM] Auto-populated model identifier: ${modelIdInput.value}`, "info");
            }

            // Auto-select runtime checkbox based on extension
            const ext = result.path.split('.').pop().toLowerCase();
            if (ext === 'gguf') {
                document.getElementById('reg-rt-llama').checked = true;
                document.getElementById('reg-rt-onnx').checked = false;
            } else if (ext === 'onnx') {
                document.getElementById('reg-rt-onnx').checked = true;
                document.getElementById('reg-rt-llama').checked = false;
            }
        } else {
            logToConsole("[SYSTEM] File selection cancelled.", "warning");
        }
    } catch (err) {
        logToConsole(`[ERROR] File selection failed: ${err.message}`, "error");
    } finally {
        // Restore button
        btn.disabled = false;
        btn.innerHTML = originalHTML;
        btn.style.opacity = '1';
        btn.style.cursor = 'pointer';
    }
}
