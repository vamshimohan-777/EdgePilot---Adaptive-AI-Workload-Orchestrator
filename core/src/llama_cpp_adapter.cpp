#include "edgepilot/llama_cpp_adapter.h"
#include <llama.h>

#include <algorithm>
#include <chrono>
#include <thread>
#include <iostream>
#include <vector>

// =============================================================================
// EdgePilot — P1: Runtime & Model Infrastructure
// llama_cpp_adapter.cpp — Production implementation of llama.cpp GGUF adapter.
//
// Loads GGUF models, tokenizes prompts, decodes tokens via llama_decode,
// and auto-regressively samples text outputs.
// =============================================================================

namespace edgepilot {

// Helper function to insert a token into a llama_batch
static void batch_add(llama_batch& batch, llama_token id, llama_pos pos, llama_seq_id seq_id, bool logits) {
    batch.token[batch.n_tokens] = id;
    batch.pos[batch.n_tokens] = pos;
    batch.n_seq_id[batch.n_tokens] = 1;
    batch.seq_id[batch.n_tokens][0] = seq_id;
    batch.logits[batch.n_tokens] = logits;
    batch.n_tokens++;
}

// ===========================================================================
// LlamaCppActiveModel — Real llama_model & llama_context wrapper
// ===========================================================================

LlamaCppActiveModel::LlamaCppActiveModel(const std::string& model_id)
    : model_id_(model_id)
    , status_(ModelStatus::Loaded)
    , model_(nullptr)
    , ctx_(nullptr)
{
}

LlamaCppActiveModel::~LlamaCppActiveModel() {
    if (status_ == ModelStatus::Loaded) {
        Unload();
    }
}

std::string LlamaCppActiveModel::GetModelId() const {
    return model_id_;
}

ModelStatus LlamaCppActiveModel::GetStatus() const {
    return status_;
}

InferenceResult LlamaCppActiveModel::RunInference(const InferenceRequest& request) {
    std::lock_guard<std::mutex> lock(inference_mutex_);
    InferenceResult result;
    result.request_id = request.request_id;

    if (status_ != ModelStatus::Loaded || !model_ || !ctx_) {
        result.success       = false;
        result.error_message = "Llama model/context is not loaded";
        return result;
    }

    auto start = std::chrono::steady_clock::now();

    // Find the prompt
    std::string prompt = "Hello";
    for (const auto& [key, value] : request.text_inputs) {
        prompt = value;
        break; // Use the first input text as prompt
    }

    try {
        llama_model* model = static_cast<llama_model*>(model_);
        llama_context* ctx = static_cast<llama_context*>(ctx_);

        std::cout << "[LlamaAdapter] Tokenizing prompt: \"" << prompt << "\"..." << std::endl;
        
        // Tokenize prompt
        std::vector<llama_token> tokens(prompt.size() + 4);
        int n_tokens = llama_tokenize(model, prompt.c_str(), prompt.size(), tokens.data(), tokens.size(), true, false);
        if (n_tokens < 0) {
            tokens.resize(-n_tokens);
            n_tokens = llama_tokenize(model, prompt.c_str(), prompt.size(), tokens.data(), tokens.size(), true, false);
        }
        
        if (n_tokens < 0) {
            throw std::runtime_error("Prompt tokenization failed");
        }
        tokens.resize(n_tokens);
        std::cout << "[LlamaAdapter] Prompt tokenized to " << n_tokens << " tokens." << std::endl;

        // Prepare batch
        llama_batch batch = llama_batch_init(512, 0, 1);
        std::cout << "[LlamaAdapter] Batch initialized. Adding tokens..." << std::endl;
        for (int i = 0; i < n_tokens; ++i) {
            batch_add(batch, tokens[i], i, 0, i == n_tokens - 1);
        }

        std::cout << "[LlamaAdapter] Decoding initial batch..." << std::endl;
        if (llama_decode(ctx, batch) != 0) {
            llama_batch_free(batch);
            throw std::runtime_error("Initial llama_decode failed");
        }
        std::cout << "[LlamaAdapter] Decode completed successfully." << std::endl;

        // Greedy decoding/sampling loop
        std::string response = "";
        llama_token new_token = 0;
        int max_tokens = 32; // Limit generation length for speed/resource efficiency
        int n_past = n_tokens;

        // Sample first token
        std::cout << "[LlamaAdapter] Sampling first token..." << std::endl;
        auto logits = llama_get_logits_ith(ctx, batch.n_tokens - 1);
        if (!logits) {
            llama_batch_free(batch);
            throw std::runtime_error("llama_get_logits_ith returned nullptr");
        }
        
        auto n_vocab = llama_n_vocab(model);
        float max_logit = -99999.0f;
        llama_token best_token = 0;
        for (llama_token id = 0; id < n_vocab; ++id) {
            if (logits[id] > max_logit) {
                max_logit = logits[id];
                best_token = id;
            }
        }
        new_token = best_token;

        // Add token piece to response
        char buf[256];
        int n_char = llama_token_to_piece(model, new_token, buf, sizeof(buf), 0, false);
        if (n_char > 0) {
            response.append(buf, n_char);
        }

        // Loop for successive tokens
        std::cout << "[LlamaAdapter] Entering generation loop..." << std::endl;
        for (int i = 0; i < max_tokens; ++i) {
            if (new_token == llama_token_eos(model)) {
                break;
            }

            batch.n_tokens = 0;
            batch_add(batch, new_token, n_past, 0, true);
            n_past++;

            if (llama_decode(ctx, batch) != 0) {
                break;
            }

            logits = llama_get_logits_ith(ctx, batch.n_tokens - 1);
            if (!logits) break;
            
            max_logit = -99999.0f;
            best_token = 0;
            for (llama_token id = 0; id < n_vocab; ++id) {
                if (logits[id] > max_logit) {
                    max_logit = logits[id];
                    best_token = id;
                }
            }
            new_token = best_token;

            n_char = llama_token_to_piece(model, new_token, buf, sizeof(buf), 0, false);
            if (n_char > 0) {
                response.append(buf, n_char);
            }
        }

        std::cout << "[LlamaAdapter] Response generation complete." << std::endl;
        llama_batch_free(batch);
        result.text_outputs["generated_text"] = response;
        result.success = true;

    } catch (const std::exception& e) {
        result.success = false;
        result.error_message = std::string("llama.cpp execution error: ") + e.what();
        std::cerr << "[LlamaAdapter] Error during RunInference: " << e.what() << std::endl;
    }

    auto end = std::chrono::steady_clock::now();
    result.latency_us = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());

    return result;
}

bool LlamaCppActiveModel::Unload() {
    if (status_ == ModelStatus::NotLoaded) {
        return true;
    }

    status_ = ModelStatus::Unloading;

    if (ctx_) {
        llama_free(static_cast<llama_context*>(ctx_));
        ctx_ = nullptr;
    }

    if (model_) {
        llama_free_model(static_cast<llama_model*>(model_));
        model_ = nullptr;
    }

    status_ = ModelStatus::NotLoaded;
    return true;
}


// ===========================================================================
// LlamaCppAdapter — Environment & Lifecycle
// ===========================================================================

LlamaCppAdapter::~LlamaCppAdapter() {
    if (initialized_) {
        Shutdown();
    }
}

std::string LlamaCppAdapter::GetName() const {
    return "llama_cpp";
}

RuntimeCapabilities LlamaCppAdapter::GetCapabilities() const {
    RuntimeCapabilities caps;
    caps.runtime_name           = "llama_cpp";
    caps.runtime_version        = "b3400";
    caps.supported_model_formats = {"gguf"};
    caps.supported_precisions    = {Precision::FP16, Precision::INT8, Precision::INT4};
    caps.supported_devices       = {HardwareDevice::CPU};
    caps.supports_streaming      = false;
    return caps;
}

bool LlamaCppAdapter::IsModelCompatible(const ModelMetadata& metadata) const {
    const auto& runtimes = metadata.compatible_runtimes;
    return std::find(runtimes.begin(), runtimes.end(), "llama_cpp") != runtimes.end();
}

std::string LlamaCppAdapter::Initialize() {
    if (initialized_) {
        return "";
    }

    try {
        llama_backend_init();
        initialized_ = true;
        return "";
    } catch (const std::exception& e) {
        return std::string("Failed to initialize llama_backend: ") + e.what();
    }
}

void LlamaCppAdapter::Shutdown() {
    if (!initialized_) {
        return;
    }

    llama_backend_free();
    initialized_ = false;
}

std::optional<std::shared_ptr<IActiveModel>>
LlamaCppAdapter::LoadModel(const ModelMetadata& metadata,
                              const DeviceConfig& /*config*/) {
    if (!initialized_) {
        return std::nullopt;
    }

    if (!IsModelCompatible(metadata)) {
        return std::nullopt;
    }

    try {
        llama_model_params model_params = llama_model_default_params();
        model_params.n_gpu_layers = 0;

        std::string path = metadata.default_file_path;
        llama_model* model = llama_load_model_from_file(path.c_str(), model_params);
        if (!model) {
            std::cerr << "[LlamaAdapter] Failed to load model file: " << path << std::endl;
            return std::nullopt;
        }

        llama_context_params ctx_params = llama_context_default_params();
        ctx_params.n_ctx = 512;
        ctx_params.n_threads = 2;

        llama_context* ctx = llama_new_context_with_model(model, ctx_params);
        if (!ctx) {
            llama_free_model(model);
            std::cerr << "[LlamaAdapter] Failed to create context for: " << path << std::endl;
            return std::nullopt;
        }

        auto active_model = std::make_shared<LlamaCppActiveModel>(metadata.model_id);
        active_model->model_ = model;
        active_model->ctx_ = ctx;
        return active_model;

    } catch (const std::exception& e) {
        std::cerr << "[LlamaAdapter] Exception during model load: " << e.what() << std::endl;
        return std::nullopt;
    }
}

} // namespace edgepilot
