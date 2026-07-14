#include "edgepilot/orchestrator.h"

#include <iostream>
#include <sstream>
#include <chrono>
#include <cstring>

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
using socket_t = SOCKET;
#define CLOSE_SOCKET(s) closesocket(s)
#define IS_VALIDSOCKET(s) ((s) != INVALID_SOCKET)
#else
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
using socket_t = int;
#define CLOSE_SOCKET(s) ::close(s)
#define IS_VALIDSOCKET(s) ((s) >= 0)
#define INVALID_SOCKET (-1)
#define SOCKET_ERROR (-1)
#endif

// =============================================================================
// EdgePilot — P2: Execution Kernel
// orchestrator.cpp — Implementation of the Orchestrator.
// =============================================================================

namespace edgepilot {

Orchestrator::Orchestrator(ModelRegistry& model_registry,
                           RuntimeRegistry& runtime_registry,
                           ModelLifecycleManager& lifecycle_manager,
                           IScheduler& scheduler,
                           std::size_t num_worker_threads)
    : model_registry_(model_registry)
    , runtime_registry_(runtime_registry)
    , lifecycle_manager_(lifecycle_manager)
    , scheduler_(scheduler)
    , pipeline_executor_(lifecycle_manager)
    , worker_pool_(num_worker_threads)
    , num_workers_(num_worker_threads)
{
}

Orchestrator::~Orchestrator() {
    if (running_) {
        Stop();
    }
}

// ---------------------------------------------------------------------------
// Start
// ---------------------------------------------------------------------------

void Orchestrator::Start(const std::string& telemetry_csv_path, int ipc_port) {
    if (running_) {
        return;
    }

#ifdef _WIN32
    // Initialize Winsock
    WSADATA wsaData;
    int wsa_err = WSAStartup(MAKEWORD(2, 2), &wsaData);
    if (wsa_err != 0) {
        std::cerr << "[Orchestrator] Winsock startup failed: " << wsa_err << std::endl;
        return;
    }
#endif

    running_ = true;

    // Initialize telemetry
    telemetry_collector_.Initialize(telemetry_csv_path);

    // Start background loops
    dispatch_thread_ = std::thread(&Orchestrator::DispatchLoop, this);
    ipc_thread_      = std::thread(&Orchestrator::IpcListenerLoop, this, ipc_port);

    std::cout << "[Orchestrator] Service started with " << num_workers_ 
              << " workers. Listening on loopback port " << ipc_port << std::endl;
}

// ---------------------------------------------------------------------------
// Stop
// ---------------------------------------------------------------------------

void Orchestrator::Stop() {
    if (!running_) {
        return;
    }
    running_ = false;

    // Shutdown IPC socket to break the blocking accept() call.
    if (ipc_socket_ != -1) {
        CLOSE_SOCKET(ipc_socket_);
        ipc_socket_ = -1;
    }

    // Stop and flush queue
    job_queue_.Stop();
    worker_pool_.Stop();

    if (dispatch_thread_.joinable()) {
        dispatch_thread_.join();
    }
    if (ipc_thread_.joinable()) {
        ipc_thread_.join();
    }

#ifdef _WIN32
    WSACleanup();
#endif

    std::cout << "[Orchestrator] Service stopped." << std::endl;
}

// ---------------------------------------------------------------------------
// SubmitJob
// ---------------------------------------------------------------------------

void Orchestrator::SubmitJob(Job job) {
    {
        std::lock_guard<std::mutex> lock(results_mutex_);
        job_statuses_[job.job_id] = JobStatus::Submitted;
    }
    job_queue_.Push(std::move(job));
}

// ---------------------------------------------------------------------------
// GetJobResult
// ---------------------------------------------------------------------------

std::optional<JobResult> Orchestrator::GetJobResult(const std::string& job_id) {
    std::lock_guard<std::mutex> lock(results_mutex_);
    auto it = job_results_.find(job_id);
    if (it == job_results_.end()) {
        return std::nullopt; // Still queued or running
    }
    return it->second;
}

// ---------------------------------------------------------------------------
// GetCurrentSystemState
// ---------------------------------------------------------------------------

SystemState Orchestrator::GetCurrentSystemState() const {
    uint32_t queue_len = static_cast<uint32_t>(job_queue_.Size());
    std::vector<std::string> loaded = lifecycle_manager_.GetAllLoaded();
    return const_cast<TelemetryCollector&>(telemetry_collector_).GetCurrentState(queue_len, loaded);
}

// ---------------------------------------------------------------------------
// DispatchLoop (P2 - Worker Dispatcher)
// ---------------------------------------------------------------------------

void Orchestrator::DispatchLoop() {
    while (running_) {
        auto job_opt = job_queue_.Pop();
        if (!job_opt.has_value()) {
            // Queue stopped or empty
            continue;
        }

        Job job = std::move(job_opt.value());
        {
            std::lock_guard<std::mutex> lock(results_mutex_);
            job_statuses_[job.job_id] = JobStatus::Scheduled;
        }

        // Gather real-time system metrics
        SystemState state = GetCurrentSystemState();

        // Query scheduling policy decision (AI or Heuristic)
        SchedulingDecision decision = scheduler_.Schedule(job, state);

        {
            std::lock_guard<std::mutex> lock(results_mutex_);
            job_decisions_[job.job_id] = decision;
        }

        // Enqueue execution to our worker thread pool
        worker_pool_.Enqueue([this, j = std::move(job), d = decision]() mutable {
            this->ExecuteJob(std::move(j), std::move(d));
        });
    }
}

// ---------------------------------------------------------------------------
// ExecuteJob (Job Execution and Telemetry Capture)
// ---------------------------------------------------------------------------

void Orchestrator::ExecuteJob(Job job, SchedulingDecision decision) {
    {
        std::lock_guard<std::mutex> lock(results_mutex_);
        job_statuses_[job.job_id] = JobStatus::Executing;
    }

    // Apply preloads and evictions recommended by the scheduler
    for (const auto& evict_id : decision.evict_models) {
        lifecycle_manager_.Unload(evict_id);
    }
    for (const auto& preload_id : decision.preload_models) {
        DeviceConfig preload_config;
        preload_config.quantization_variant_id = decision.quantization_variant_id;
        preload_config.device = HardwareDevice::AUTO;
        lifecycle_manager_.Load(preload_id, decision.runtime_name, preload_config);
    }

    // Insert any delay scheduled (thermals/throttling)
    if (decision.execution_delay_ms > 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(decision.execution_delay_ms));
    }

    // Resolve execution configuration
    DeviceConfig config;
    config.quantization_variant_id = decision.quantization_variant_id;
    config.device = HardwareDevice::AUTO;
    
    // Sample state right before execution
    SystemState pre_state = GetCurrentSystemState();

    auto start_time = std::chrono::steady_clock::now();

    // Load model & run inference safely using Acquire/Release counts
    auto model_opt = lifecycle_manager_.Load(job.model_id, decision.runtime_name, config);
    
    JobResult result;
    result.job_id = job.job_id;

    if (!model_opt.has_value()) {
        result.status = JobStatus::Failed;
        result.error_message = "Failed to load model target '" + job.model_id + "'";
    } else {
        auto model = model_opt.value();
        
        lifecycle_manager_.AcquireInference(job.model_id);
        result.result = model->RunInference(job.request);
        lifecycle_manager_.ReleaseInference(job.model_id);

        if (result.result.success) {
            result.status = JobStatus::Completed;
        } else {
            result.status = JobStatus::Failed;
            result.error_message = result.result.error_message;
        }
    }

    auto end_time = std::chrono::steady_clock::now();
    uint64_t actual_latency_us = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();

    // Cache the result
    {
        std::lock_guard<std::mutex> lock(results_mutex_);
        job_results_[job.job_id]   = result;
        job_statuses_[job.job_id]  = result.status;
    }

    // Log telemetry metrics (P3)
    TelemetryRecord rec;
    rec.timestamp_us             = std::chrono::duration_cast<std::chrono::microseconds>(
                                       std::chrono::system_clock::now().time_since_epoch()).count();
    rec.model_id                 = job.model_id;
    rec.runtime_name             = decision.runtime_name;
    rec.quantization_variant_id  = decision.quantization_variant_id;
    rec.cpu_utilization          = pre_state.cpu_utilization;
    rec.gpu_utilization          = pre_state.gpu_utilization;
    // Estimate peak memory and energy
    auto meta_opt = model_registry_.Get(job.model_id);
    uint64_t model_sz = meta_opt.has_value() ? meta_opt->file_size_bytes : 0;
    rec.ram_usage_bytes          = model_sz + 150ULL * 1024ULL * 1024ULL; // base model + overhead
    rec.battery_level            = pre_state.battery_level;
    rec.device_temperature       = pre_state.device_temperature;
    rec.queue_length             = pre_state.queue_length;
    rec.latency_us               = actual_latency_us;
    rec.peak_memory_bytes        = rec.ram_usage_bytes + (result.status == JobStatus::Completed ? 50ULL * 1024ULL * 1024ULL : 0);
    // Simple energy model: GPU = 12W, CPU = 5W
    float watts = (decision.runtime_name == "llama_cpp") ? 12.0f : 5.0f;
    rec.energy_estimate_joules   = (watts * static_cast<float>(actual_latency_us)) / 1000000.0f;

    telemetry_collector_.LogExecution(rec);
}

// ---------------------------------------------------------------------------
// IpcListenerLoop (P4 - REST API integration bridge)
// ---------------------------------------------------------------------------

void Orchestrator::IpcListenerLoop(int port) {
    socket_t server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (!IS_VALIDSOCKET(server_fd)) {
        std::cerr << "[Orchestrator] Socket creation failed" << std::endl;
        return;
    }

    // Allow port reuse
    int opt = 1;
#ifdef _WIN32
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, reinterpret_cast<const char*>(&opt), sizeof(opt));
#else
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
#endif

    sockaddr_in address;
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY; // loopback & local interfaces
    address.sin_port = htons(port);

    if (bind(server_fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) == SOCKET_ERROR) {
        std::cerr << "[Orchestrator] Socket bind failed on port " << port << std::endl;
        CLOSE_SOCKET(server_fd);
        return;
    }

    if (listen(server_fd, 10) == SOCKET_ERROR) {
        std::cerr << "[Orchestrator] Socket listen failed" << std::endl;
        CLOSE_SOCKET(server_fd);
        return;
    }

    ipc_socket_ = server_fd;

    while (running_) {
        sockaddr_in client_addr;
        int addr_len = sizeof(client_addr);
#ifdef _WIN32
        socket_t client_socket = accept(server_fd, reinterpret_cast<sockaddr*>(&client_addr), &addr_len);
#else
        socket_t client_socket = accept(server_fd, reinterpret_cast<sockaddr*>(&client_addr), reinterpret_cast<socklen_t*>(&addr_len));
#endif

        if (!IS_VALIDSOCKET(client_socket)) {
            continue; // likely stopped
        }

        // Process request
        char buffer[1024] = {0};
        int bytes_read = recv(client_socket, buffer, sizeof(buffer) - 1, 0);
        if (bytes_read > 0) {
            std::string raw_req(buffer, bytes_read);
            std::stringstream ss(raw_req);
            std::string cmd;
            ss >> cmd;

            std::stringstream response;

            if (cmd == "SUBMIT") {
                // Command: SUBMIT <job_id> <model_id> <priority_int> <prompt>
                std::string job_id, model_id;
                int priority_val;
                ss >> job_id >> model_id >> priority_val;
                
                std::string prompt;
                std::getline(ss, prompt);
                // Strip leading space
                if (!prompt.empty() && prompt[0] == ' ') {
                    prompt = prompt.substr(1);
                }

                Job job;
                job.job_id   = job_id;
                job.model_id = model_id;
                job.priority = static_cast<JobPriority>(priority_val);
                job.request.model_id   = model_id;
                job.request.request_id = job_id;
                job.request.text_inputs["prompt"] = prompt;
                job.submitted_at_us = std::chrono::duration_cast<std::chrono::microseconds>(
                                         std::chrono::system_clock::now().time_since_epoch()).count();

                SubmitJob(std::move(job));
                response << "OK " << job_id << "\n";
            }
            else if (cmd == "STATUS") {
                std::string job_id;
                ss >> job_id;

                std::lock_guard<std::mutex> lock(results_mutex_);
                auto it = job_statuses_.find(job_id);
                if (it == job_statuses_.end()) {
                    response << "UNKNOWN\n";
                } else {
                    JobStatus status = it->second;
                    std::string dec_str = "runtime:unknown | quant:unknown | delay:0";
                    auto dec_it = job_decisions_.find(job_id);
                    if (dec_it != job_decisions_.end()) {
                        dec_str = "runtime:" + dec_it->second.runtime_name + 
                                  " | quant:" + dec_it->second.quantization_variant_id + 
                                  " | delay:" + std::to_string(dec_it->second.execution_delay_ms);
                    }

                    if (status == JobStatus::Submitted) {
                        response << "QUEUED | " << dec_str << "\n";
                    } else if (status == JobStatus::Scheduled || status == JobStatus::Executing) {
                        response << "EXECUTING | " << dec_str << "\n";
                    } else if (status == JobStatus::Completed) {
                        auto res_it = job_results_.find(job_id);
                        std::string out = "";
                        if (res_it != job_results_.end()) {
                            auto text_it = res_it->second.result.text_outputs.find("generated_text");
                            if (text_it != res_it->second.result.text_outputs.end()) {
                                out = text_it->second;
                            }
                        }
                        response << "COMPLETED " << out << " | " << dec_str << "\n";
                    } else {
                        response << "FAILED | " << dec_str << "\n";
                    }
                }
            }
            else if (cmd == "LIST") {
                auto loaded = lifecycle_manager_.GetAllLoaded();
                response << "LOADED ";
                for (std::size_t i = 0; i < loaded.size(); ++i) {
                    response << loaded[i] << (i < loaded.size() - 1 ? "," : "");
                }
                response << "\n";
            }
            else if (cmd == "METRICS") {
                SystemState state = GetCurrentSystemState();
                response << "METRICS cpu:" << state.cpu_utilization
                         << " gpu:" << state.gpu_utilization
                         << " ram:" << state.ram_free_bytes
                         << " batt:" << state.battery_level
                         << " temp:" << state.device_temperature
                         << " queue:" << state.queue_length << "\n";
            }
            else if (cmd == "REGISTER") {
                std::string model_id, runtime_name, file_path;
                uint64_t file_size;
                ss >> model_id >> runtime_name >> file_path >> file_size;

                ModelMetadata metadata;
                metadata.model_id = model_id;
                metadata.display_name = model_id;
                metadata.compatible_runtimes = {runtime_name};
                metadata.preferred_runtime = runtime_name;
                metadata.file_size_bytes = file_size;
                metadata.default_file_path = file_path;

                const_cast<ModelRegistry&>(model_registry_).Register(metadata);
                response << "OK\n";
            }
            else {
                response << "ERROR Unknown Command\n";
            }

            std::string res_str = response.str();
            send(client_socket, res_str.c_str(), static_cast<int>(res_str.length()), 0);
        }
        CLOSE_SOCKET(client_socket);
    }
    CLOSE_SOCKET(server_fd);
}

} // namespace edgepilot
