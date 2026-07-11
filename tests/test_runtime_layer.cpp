#include "edgepilot/types.h"
#include "edgepilot/runtime_interface.h"
#include "edgepilot/runtime_registry.h"
#include "edgepilot/onnx_runtime_adapter.h"
#include "edgepilot/llama_cpp_adapter.h"

#include <cassert>
#include <iostream>
#include <memory>
#include <string>

// =============================================================================
// EdgePilot — P1: Runtime & Model Infrastructure
// test_runtime_layer.cpp — Unit tests for Day 2 deliverables:
//   - RuntimeRegistry (register, get, unregister, clear, duplicate handling)
//   - OnnxRuntimeAdapter (capabilities, compatibility, lifecycle)
//   - LlamaCppAdapter   (capabilities, compatibility, lifecycle)
//   - Full adapter lifecycle: Init → Load → Inference → Unload → Shutdown
//
// Test framework: None — uses assert() and manual pass/fail counting.
// Build: Link against edgepilot_core static library.
// =============================================================================

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) \
    static void test_##name(); \
    static struct Register_##name { \
        Register_##name() { \
            std::cout << "  Running: " #name "... "; \
            try { \
                test_##name(); \
                std::cout << "PASS" << std::endl; \
                tests_passed++; \
            } catch (const std::exception& e) { \
                std::cout << "FAIL (" << e.what() << ")" << std::endl; \
                tests_failed++; \
            } catch (...) { \
                std::cout << "FAIL (unknown exception)" << std::endl; \
                tests_failed++; \
            } \
        } \
    } register_##name; \
    static void test_##name()

#define ASSERT_TRUE(expr)  do { if (!(expr)) throw std::runtime_error("ASSERT_TRUE failed: " #expr); } while(0)
#define ASSERT_FALSE(expr) do { if (expr)    throw std::runtime_error("ASSERT_FALSE failed: " #expr); } while(0)
#define ASSERT_EQ(a, b)    do { if ((a) != (b)) throw std::runtime_error("ASSERT_EQ failed: " #a " != " #b); } while(0)

using namespace edgepilot;

// ===========================================================================
// RuntimeRegistry Tests
// ===========================================================================

TEST(registry_initially_empty) {
    RuntimeRegistry registry;
    ASSERT_EQ(registry.Count(), 0u);
    ASSERT_FALSE(registry.Has("onnx"));
    auto all = registry.GetAll();
    ASSERT_TRUE(all.empty());
}

TEST(registry_register_and_get) {
    RuntimeRegistry registry;
    auto onnx = std::make_shared<OnnxRuntimeAdapter>();
    onnx->Initialize();
    registry.Register(onnx);

    ASSERT_TRUE(registry.Has("onnx"));
    ASSERT_EQ(registry.Count(), 1u);

    auto retrieved = registry.Get("onnx");
    ASSERT_TRUE(retrieved.has_value());
    ASSERT_EQ(retrieved.value()->GetName(), "onnx");
}

TEST(registry_get_missing_returns_nullopt) {
    RuntimeRegistry registry;
    auto result = registry.Get("nonexistent");
    ASSERT_FALSE(result.has_value());
}

TEST(registry_register_multiple) {
    RuntimeRegistry registry;
    auto onnx  = std::make_shared<OnnxRuntimeAdapter>();
    auto llama = std::make_shared<LlamaCppAdapter>();
    onnx->Initialize();
    llama->Initialize();

    registry.Register(onnx);
    registry.Register(llama);

    ASSERT_EQ(registry.Count(), 2u);
    ASSERT_TRUE(registry.Has("onnx"));
    ASSERT_TRUE(registry.Has("llama_cpp"));

    auto all = registry.GetAll();
    ASSERT_EQ(all.size(), 2u);
}

TEST(registry_upsert_replaces) {
    RuntimeRegistry registry;
    auto onnx1 = std::make_shared<OnnxRuntimeAdapter>();
    auto onnx2 = std::make_shared<OnnxRuntimeAdapter>();
    onnx1->Initialize();
    onnx2->Initialize();

    registry.Register(onnx1);
    registry.Register(onnx2);  // Should replace, not add.

    ASSERT_EQ(registry.Count(), 1u);
    auto retrieved = registry.Get("onnx");
    ASSERT_TRUE(retrieved.has_value());
    // The second registration should have replaced the first.
    ASSERT_TRUE(retrieved.value().get() == onnx2.get());
}

TEST(registry_unregister) {
    RuntimeRegistry registry;
    auto onnx = std::make_shared<OnnxRuntimeAdapter>();
    onnx->Initialize();
    registry.Register(onnx);
    ASSERT_TRUE(registry.Has("onnx"));

    registry.Unregister("onnx");
    ASSERT_FALSE(registry.Has("onnx"));
    ASSERT_EQ(registry.Count(), 0u);
}

TEST(registry_unregister_nonexistent_is_noop) {
    RuntimeRegistry registry;
    // Should not throw or crash.
    registry.Unregister("does_not_exist");
    ASSERT_EQ(registry.Count(), 0u);
}

TEST(registry_clear) {
    RuntimeRegistry registry;
    auto onnx  = std::make_shared<OnnxRuntimeAdapter>();
    auto llama = std::make_shared<LlamaCppAdapter>();
    onnx->Initialize();
    llama->Initialize();
    registry.Register(onnx);
    registry.Register(llama);
    ASSERT_EQ(registry.Count(), 2u);

    registry.Clear();
    ASSERT_EQ(registry.Count(), 0u);
    ASSERT_FALSE(registry.Has("onnx"));
    ASSERT_FALSE(registry.Has("llama_cpp"));
}

TEST(registry_null_registration_ignored) {
    RuntimeRegistry registry;
    registry.Register(nullptr);
    ASSERT_EQ(registry.Count(), 0u);
}

// ===========================================================================
// OnnxRuntimeAdapter Tests
// ===========================================================================

TEST(onnx_adapter_name) {
    OnnxRuntimeAdapter adapter;
    ASSERT_EQ(adapter.GetName(), "onnx");
}

TEST(onnx_adapter_capabilities) {
    OnnxRuntimeAdapter adapter;
    auto caps = adapter.GetCapabilities();
    ASSERT_EQ(caps.runtime_name, "onnx");
    ASSERT_FALSE(caps.supported_model_formats.empty());
    ASSERT_EQ(caps.supported_model_formats[0], "onnx");
    ASSERT_FALSE(caps.supported_precisions.empty());
    ASSERT_FALSE(caps.supported_devices.empty());
    ASSERT_FALSE(caps.supports_streaming);
    // Extensibility: extra_info should have max_batch_size
    ASSERT_TRUE(caps.extra_info.count("max_batch_size") > 0);
}

TEST(onnx_adapter_initialize_and_shutdown) {
    OnnxRuntimeAdapter adapter;
    std::string err = adapter.Initialize();
    ASSERT_TRUE(err.empty());  // Success

    // Idempotent initialization
    err = adapter.Initialize();
    ASSERT_TRUE(err.empty());

    adapter.Shutdown();
}

TEST(onnx_adapter_compatibility_check) {
    OnnxRuntimeAdapter adapter;

    ModelMetadata compatible;
    compatible.model_id = "test-onnx-model";
    compatible.compatible_runtimes = {"onnx"};

    ModelMetadata incompatible;
    incompatible.model_id = "test-gguf-model";
    incompatible.compatible_runtimes = {"llama_cpp"};

    ASSERT_TRUE(adapter.IsModelCompatible(compatible));
    ASSERT_FALSE(adapter.IsModelCompatible(incompatible));
}

TEST(onnx_adapter_load_without_init_fails) {
    OnnxRuntimeAdapter adapter;
    ModelMetadata meta;
    meta.model_id = "test";
    meta.compatible_runtimes = {"onnx"};
    DeviceConfig config;

    auto result = adapter.LoadModel(meta, config);
    ASSERT_FALSE(result.has_value());  // Not initialized
}

TEST(onnx_adapter_load_incompatible_fails) {
    OnnxRuntimeAdapter adapter;
    adapter.Initialize();

    ModelMetadata meta;
    meta.model_id = "test";
    meta.compatible_runtimes = {"llama_cpp"};  // Not compatible with onnx
    DeviceConfig config;

    auto result = adapter.LoadModel(meta, config);
    ASSERT_FALSE(result.has_value());
}

TEST(onnx_adapter_full_lifecycle) {
    OnnxRuntimeAdapter adapter;
    adapter.Initialize();

    ModelMetadata meta;
    meta.model_id = "resnet50";
    meta.compatible_runtimes = {"onnx"};
    DeviceConfig config;

    // Load
    auto loaded = adapter.LoadModel(meta, config);
    ASSERT_TRUE(loaded.has_value());
    auto model = loaded.value();
    ASSERT_EQ(model->GetModelId(), "resnet50");
    ASSERT_EQ(model->GetStatus(), ModelStatus::Loaded);

    // Inference
    InferenceRequest req;
    req.request_id = "req-001";
    req.model_id   = "resnet50";
    req.text_inputs["prompt"] = "hello";

    InferenceResult result = model->RunInference(req);
    ASSERT_TRUE(result.success);
    ASSERT_EQ(result.request_id, "req-001");
    ASSERT_TRUE(result.text_outputs.count("prompt") > 0);

    // Unload
    bool ok = model->Unload();
    ASSERT_TRUE(ok);
    ASSERT_EQ(model->GetStatus(), ModelStatus::NotLoaded);

    // Inference after unload should fail
    InferenceResult fail_result = model->RunInference(req);
    ASSERT_FALSE(fail_result.success);

    // Idempotent unload
    ok = model->Unload();
    ASSERT_TRUE(ok);

    adapter.Shutdown();
}

// ===========================================================================
// LlamaCppAdapter Tests
// ===========================================================================

TEST(llama_adapter_name) {
    LlamaCppAdapter adapter;
    ASSERT_EQ(adapter.GetName(), "llama_cpp");
}

TEST(llama_adapter_capabilities) {
    LlamaCppAdapter adapter;
    auto caps = adapter.GetCapabilities();
    ASSERT_EQ(caps.runtime_name, "llama_cpp");
    ASSERT_EQ(caps.supported_model_formats[0], "gguf");
    ASSERT_TRUE(caps.supports_streaming);
    ASSERT_TRUE(caps.extra_info.count("max_context_length") > 0);
}

TEST(llama_adapter_compatibility_check) {
    LlamaCppAdapter adapter;

    ModelMetadata compatible;
    compatible.model_id = "llama3";
    compatible.compatible_runtimes = {"llama_cpp"};

    ModelMetadata incompatible;
    incompatible.model_id = "resnet50";
    incompatible.compatible_runtimes = {"onnx"};

    ASSERT_TRUE(adapter.IsModelCompatible(compatible));
    ASSERT_FALSE(adapter.IsModelCompatible(incompatible));
}

TEST(llama_adapter_load_without_init_fails) {
    LlamaCppAdapter adapter;
    ModelMetadata meta;
    meta.model_id = "test";
    meta.compatible_runtimes = {"llama_cpp"};
    DeviceConfig config;

    auto result = adapter.LoadModel(meta, config);
    ASSERT_FALSE(result.has_value());
}

TEST(llama_adapter_full_lifecycle) {
    LlamaCppAdapter adapter;
    adapter.Initialize();

    ModelMetadata meta;
    meta.model_id = "llama3-8b";
    meta.compatible_runtimes = {"llama_cpp"};
    DeviceConfig config;

    // Load
    auto loaded = adapter.LoadModel(meta, config);
    ASSERT_TRUE(loaded.has_value());
    auto model = loaded.value();
    ASSERT_EQ(model->GetModelId(), "llama3-8b");
    ASSERT_EQ(model->GetStatus(), ModelStatus::Loaded);

    // Inference with prompt
    InferenceRequest req;
    req.request_id = "req-llm-001";
    req.model_id   = "llama3-8b";
    req.text_inputs["prompt"] = "What is EdgePilot?";

    InferenceResult result = model->RunInference(req);
    ASSERT_TRUE(result.success);
    ASSERT_EQ(result.request_id, "req-llm-001");
    ASSERT_TRUE(result.text_outputs.count("generated_text") > 0);
    // Verify stub prefix
    auto generated = result.text_outputs.at("generated_text");
    ASSERT_TRUE(generated.find("[llama-stub]") != std::string::npos);

    // Unload
    bool ok = model->Unload();
    ASSERT_TRUE(ok);
    ASSERT_EQ(model->GetStatus(), ModelStatus::NotLoaded);

    // Inference after unload should fail
    InferenceResult fail_result = model->RunInference(req);
    ASSERT_FALSE(fail_result.success);

    adapter.Shutdown();
}

// ===========================================================================
// Cross-Adapter: Registry + Adapters Integration
// ===========================================================================

TEST(integration_registry_with_both_adapters) {
    RuntimeRegistry registry;

    auto onnx  = std::make_shared<OnnxRuntimeAdapter>();
    auto llama = std::make_shared<LlamaCppAdapter>();

    ASSERT_TRUE(onnx->Initialize().empty());
    ASSERT_TRUE(llama->Initialize().empty());

    registry.Register(onnx);
    registry.Register(llama);

    ASSERT_EQ(registry.Count(), 2u);

    // Retrieve ONNX and check caps
    auto rt_onnx = registry.Get("onnx");
    ASSERT_TRUE(rt_onnx.has_value());
    auto onnx_caps = rt_onnx.value()->GetCapabilities();
    ASSERT_EQ(onnx_caps.supported_model_formats[0], "onnx");

    // Retrieve llama_cpp and check caps
    auto rt_llama = registry.Get("llama_cpp");
    ASSERT_TRUE(rt_llama.has_value());
    auto llama_caps = rt_llama.value()->GetCapabilities();
    ASSERT_EQ(llama_caps.supported_model_formats[0], "gguf");

    // Load a model through the registry-resolved runtime
    ModelMetadata meta;
    meta.model_id = "test-gguf";
    meta.compatible_runtimes = {"llama_cpp"};
    DeviceConfig config;

    auto model_opt = rt_llama.value()->LoadModel(meta, config);
    ASSERT_TRUE(model_opt.has_value());
    ASSERT_EQ(model_opt.value()->GetModelId(), "test-gguf");
    ASSERT_EQ(model_opt.value()->GetStatus(), ModelStatus::Loaded);

    model_opt.value()->Unload();

    onnx->Shutdown();
    llama->Shutdown();
    registry.Clear();
}

// ===========================================================================
// Main — run all tests via static initializers
// ===========================================================================

int main() {
    std::cout << "\n========================================" << std::endl;
    std::cout << " EdgePilot P1 — Runtime Layer Test Suite" << std::endl;
    std::cout << "========================================\n" << std::endl;

    // Tests already ran via static initializers above.

    std::cout << "\n----------------------------------------" << std::endl;
    std::cout << " Results: " << tests_passed << " passed, "
              << tests_failed << " failed, "
              << (tests_passed + tests_failed) << " total" << std::endl;
    std::cout << "----------------------------------------\n" << std::endl;

    return tests_failed > 0 ? 1 : 0;
}
