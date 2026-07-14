#include "edgepilot/types.h"
#include "edgepilot/runtime_interface.h"
#include "edgepilot/runtime_registry.h"
#include "edgepilot/model_registry.h"
#include "edgepilot/quantization_manager.h"
#include "edgepilot/lifecycle_manager.h"
#include "edgepilot/onnx_runtime_adapter.h"
#include "edgepilot/llama_cpp_adapter.h"

#include <iostream>
#include <string>
#include <stdexcept>

// =============================================================================
// EdgePilot — P1: Runtime & Model Infrastructure
// test_model_layer.cpp — Day 3 tests covering:
//   - ModelRegistry (register, get, filter, upsert, clear)
//   - QuantizationMetadataManager (variants, default, lookup)
//   - ModelLifecycleManager (load, cache-hit, unload, reload, status, errors)
//   - Full integration: registry → lifecycle → inference → unload
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

// ---------------------------------------------------------------------------
// Helper: create sample ModelMetadata
// ---------------------------------------------------------------------------

static ModelMetadata make_onnx_model(const std::string& id = "resnet50") {
    ModelMetadata m;
    m.model_id            = id;
    m.display_name        = "ResNet-50";
    m.version             = "1.0";
    m.task_type           = TaskType::ImageClassification;
    m.compatible_runtimes = {"onnx"};
    m.preferred_runtime   = "onnx";
    m.supported_precisions = {Precision::FP32, Precision::FP16};
    m.file_size_bytes     = 102400000;
    m.default_file_path   = "/models/resnet50.onnx";

    QuantizationVariant fp32;
    fp32.variant_id = "fp32";
    fp32.precision  = Precision::FP32;
    fp32.size_bytes = 102400000;
    fp32.file_path  = "/models/resnet50_fp32.onnx";

    QuantizationVariant int8;
    int8.variant_id = "int8";
    int8.precision  = Precision::INT8;
    int8.size_bytes = 25600000;
    int8.file_path  = "/models/resnet50_int8.onnx";

    m.quantization_variants = {fp32, int8};
    return m;
}

static ModelMetadata make_llama_model(const std::string& id = "llama3-8b") {
    ModelMetadata m;
    m.model_id            = id;
    m.display_name        = "LLaMA 3 8B";
    m.version             = "3.0";
    m.task_type           = TaskType::TextGeneration;
    m.compatible_runtimes = {"llama_cpp"};
    m.preferred_runtime   = "llama_cpp";
    m.supported_precisions = {Precision::FP16, Precision::INT4};
    m.file_size_bytes     = 4500000000ULL;
    m.default_file_path   = "/models/llama3-8b.gguf";

    QuantizationVariant q4;
    q4.variant_id = "q4_k_m";
    q4.precision  = Precision::INT4;
    q4.size_bytes = 4700000000ULL;
    q4.file_path  = "/models/llama3-8b-q4_k_m.gguf";

    QuantizationVariant fp16;
    fp16.variant_id = "fp16";
    fp16.precision  = Precision::FP16;
    fp16.size_bytes = 16000000000ULL;
    fp16.file_path  = "/models/llama3-8b-fp16.gguf";

    m.quantization_variants = {q4, fp16};
    return m;
}


// ===========================================================================
// ModelRegistry Tests
// ===========================================================================

TEST(model_registry_initially_empty) {
    ModelRegistry registry;
    ASSERT_EQ(registry.Count(), 0u);
    ASSERT_FALSE(registry.Has("resnet50"));
}

TEST(model_registry_register_and_get) {
    ModelRegistry registry;
    registry.Register(make_onnx_model());

    ASSERT_TRUE(registry.Has("resnet50"));
    ASSERT_EQ(registry.Count(), 1u);

    auto meta = registry.Get("resnet50");
    ASSERT_TRUE(meta.has_value());
    ASSERT_EQ(meta->model_id, "resnet50");
    ASSERT_EQ(meta->display_name, "ResNet-50");
    ASSERT_EQ(meta->task_type, TaskType::ImageClassification);
}

TEST(model_registry_get_missing_returns_nullopt) {
    ModelRegistry registry;
    auto result = registry.Get("nonexistent");
    ASSERT_FALSE(result.has_value());
}

TEST(model_registry_upsert_replaces) {
    ModelRegistry registry;
    auto model = make_onnx_model();
    registry.Register(model);

    model.display_name = "ResNet-50 v2";
    registry.Register(model);

    ASSERT_EQ(registry.Count(), 1u);
    auto meta = registry.Get("resnet50");
    ASSERT_EQ(meta->display_name, "ResNet-50 v2");
}

TEST(model_registry_register_multiple) {
    ModelRegistry registry;
    registry.Register(make_onnx_model());
    registry.Register(make_llama_model());

    ASSERT_EQ(registry.Count(), 2u);
    auto all = registry.List();
    ASSERT_EQ(all.size(), 2u);
}

TEST(model_registry_list_by_task) {
    ModelRegistry registry;
    registry.Register(make_onnx_model());
    registry.Register(make_llama_model());

    auto img_models = registry.ListByTask(TaskType::ImageClassification);
    ASSERT_EQ(img_models.size(), 1u);
    ASSERT_EQ(img_models[0].model_id, "resnet50");

    auto txt_models = registry.ListByTask(TaskType::TextGeneration);
    ASSERT_EQ(txt_models.size(), 1u);
    ASSERT_EQ(txt_models[0].model_id, "llama3-8b");

    auto empty = registry.ListByTask(TaskType::SpeechRecognition);
    ASSERT_TRUE(empty.empty());
}

TEST(model_registry_list_by_runtime) {
    ModelRegistry registry;
    registry.Register(make_onnx_model());
    registry.Register(make_llama_model());

    auto onnx_models = registry.ListByRuntime("onnx");
    ASSERT_EQ(onnx_models.size(), 1u);
    ASSERT_EQ(onnx_models[0].model_id, "resnet50");

    auto llama_models = registry.ListByRuntime("llama_cpp");
    ASSERT_EQ(llama_models.size(), 1u);

    auto none = registry.ListByRuntime("executorch");
    ASSERT_TRUE(none.empty());
}

TEST(model_registry_unregister) {
    ModelRegistry registry;
    registry.Register(make_onnx_model());
    registry.Unregister("resnet50");
    ASSERT_FALSE(registry.Has("resnet50"));
    ASSERT_EQ(registry.Count(), 0u);
}

TEST(model_registry_clear) {
    ModelRegistry registry;
    registry.Register(make_onnx_model());
    registry.Register(make_llama_model());
    registry.Clear();
    ASSERT_EQ(registry.Count(), 0u);
}


// ===========================================================================
// QuantizationMetadataManager Tests
// ===========================================================================

TEST(quant_manager_get_variants) {
    ModelRegistry registry;
    registry.Register(make_onnx_model());
    QuantizationMetadataManager qm(registry);

    auto variants = qm.GetVariants("resnet50");
    ASSERT_EQ(variants.size(), 2u);
    ASSERT_EQ(variants[0].variant_id, "fp32");
    ASSERT_EQ(variants[1].variant_id, "int8");
}

TEST(quant_manager_get_variants_missing_model) {
    ModelRegistry registry;
    QuantizationMetadataManager qm(registry);

    auto variants = qm.GetVariants("nonexistent");
    ASSERT_TRUE(variants.empty());
}

TEST(quant_manager_get_specific_variant) {
    ModelRegistry registry;
    registry.Register(make_llama_model());
    QuantizationMetadataManager qm(registry);

    auto variant = qm.GetVariant("llama3-8b", "q4_k_m");
    ASSERT_TRUE(variant.has_value());
    ASSERT_EQ(variant->variant_id, "q4_k_m");
    ASSERT_EQ(variant->precision, Precision::INT4);

    auto missing = qm.GetVariant("llama3-8b", "nonexistent_variant");
    ASSERT_FALSE(missing.has_value());
}

TEST(quant_manager_get_default) {
    ModelRegistry registry;
    registry.Register(make_llama_model());
    QuantizationMetadataManager qm(registry);

    auto def = qm.GetDefault("llama3-8b");
    ASSERT_TRUE(def.has_value());
    ASSERT_EQ(def->variant_id, "q4_k_m");  // First in list
}

TEST(quant_manager_get_default_no_variants) {
    ModelRegistry registry;
    ModelMetadata m;
    m.model_id = "bare-model";
    // No quantization_variants
    registry.Register(m);
    QuantizationMetadataManager qm(registry);

    auto def = qm.GetDefault("bare-model");
    ASSERT_FALSE(def.has_value());
}

TEST(quant_manager_has_variants) {
    ModelRegistry registry;
    registry.Register(make_onnx_model());
    ModelMetadata bare;
    bare.model_id = "bare";
    registry.Register(bare);
    QuantizationMetadataManager qm(registry);

    ASSERT_TRUE(qm.HasVariants("resnet50"));
    ASSERT_FALSE(qm.HasVariants("bare"));
    ASSERT_FALSE(qm.HasVariants("nonexistent"));
}


// ===========================================================================
// ModelLifecycleManager Tests
// ===========================================================================

// Helper: set up full infrastructure
struct TestFixture {
    RuntimeRegistry runtime_registry;
    ModelRegistry model_registry;
    std::shared_ptr<OnnxRuntimeAdapter> onnx;
    std::shared_ptr<LlamaCppAdapter> llama;

    TestFixture() {
        onnx = std::make_shared<OnnxRuntimeAdapter>();
        llama = std::make_shared<LlamaCppAdapter>();
        onnx->Initialize();
        llama->Initialize();
        runtime_registry.Register(onnx);
        runtime_registry.Register(llama);
        model_registry.Register(make_onnx_model());
        model_registry.Register(make_llama_model());
    }

    ~TestFixture() {
        onnx->Shutdown();
        llama->Shutdown();
    }
};

TEST(lifecycle_load_model) {
    TestFixture fix;
    ModelLifecycleManager mgr(fix.model_registry, fix.runtime_registry);

    DeviceConfig config;
    auto result = mgr.Load("resnet50", "onnx", config);
    ASSERT_TRUE(result.has_value());
    ASSERT_EQ(result.value()->GetModelId(), "resnet50");
    ASSERT_EQ(result.value()->GetStatus(), ModelStatus::Loaded);
    ASSERT_TRUE(mgr.IsLoaded("resnet50"));
    ASSERT_EQ(mgr.QueryStatus("resnet50"), ModelStatus::Loaded);
}

TEST(lifecycle_cache_hit) {
    TestFixture fix;
    ModelLifecycleManager mgr(fix.model_registry, fix.runtime_registry);

    DeviceConfig config;
    auto first  = mgr.Load("resnet50", "onnx", config);
    auto second = mgr.Load("resnet50", "onnx", config);

    ASSERT_TRUE(first.has_value());
    ASSERT_TRUE(second.has_value());
    // Same pointer — cache hit, not a reload
    ASSERT_TRUE(first.value().get() == second.value().get());
}

TEST(lifecycle_load_model_not_found) {
    TestFixture fix;
    ModelLifecycleManager mgr(fix.model_registry, fix.runtime_registry);

    DeviceConfig config;
    auto result = mgr.Load("nonexistent", "onnx", config);
    ASSERT_FALSE(result.has_value());
}

TEST(lifecycle_load_runtime_not_found) {
    TestFixture fix;
    ModelLifecycleManager mgr(fix.model_registry, fix.runtime_registry);

    DeviceConfig config;
    auto result = mgr.Load("resnet50", "executorch", config);
    ASSERT_FALSE(result.has_value());
}

TEST(lifecycle_load_with_preferred_runtime_fallback) {
    TestFixture fix;
    ModelLifecycleManager mgr(fix.model_registry, fix.runtime_registry);

    DeviceConfig config;
    // Empty runtime_name → falls back to preferred_runtime = "onnx"
    auto result = mgr.Load("resnet50", "", config);
    ASSERT_TRUE(result.has_value());
    ASSERT_EQ(result.value()->GetModelId(), "resnet50");
}

TEST(lifecycle_unload_model) {
    TestFixture fix;
    ModelLifecycleManager mgr(fix.model_registry, fix.runtime_registry);

    DeviceConfig config;
    mgr.Load("resnet50", "onnx", config);
    ASSERT_TRUE(mgr.IsLoaded("resnet50"));

    bool ok = mgr.Unload("resnet50");
    ASSERT_TRUE(ok);
    ASSERT_FALSE(mgr.IsLoaded("resnet50"));
    ASSERT_EQ(mgr.QueryStatus("resnet50"), ModelStatus::NotLoaded);
}

TEST(lifecycle_unload_not_loaded_returns_false) {
    TestFixture fix;
    ModelLifecycleManager mgr(fix.model_registry, fix.runtime_registry);

    bool ok = mgr.Unload("resnet50");
    ASSERT_FALSE(ok);
}

TEST(lifecycle_reload_model) {
    TestFixture fix;
    ModelLifecycleManager mgr(fix.model_registry, fix.runtime_registry);

    DeviceConfig config;
    auto first = mgr.Load("resnet50", "onnx", config);
    auto reloaded = mgr.Reload("resnet50", "onnx", config);

    ASSERT_TRUE(reloaded.has_value());
    // Reload creates a new instance — different pointer
    ASSERT_TRUE(first.value().get() != reloaded.value().get());
    ASSERT_TRUE(mgr.IsLoaded("resnet50"));
}

TEST(lifecycle_get_handle) {
    TestFixture fix;
    ModelLifecycleManager mgr(fix.model_registry, fix.runtime_registry);

    auto handle = mgr.GetHandle("resnet50");
    ASSERT_FALSE(handle.has_value());  // Not loaded yet

    DeviceConfig config;
    mgr.Load("resnet50", "onnx", config);
    handle = mgr.GetHandle("resnet50");
    ASSERT_TRUE(handle.has_value());
    ASSERT_EQ(handle.value()->GetModelId(), "resnet50");
}

TEST(lifecycle_get_all_loaded) {
    TestFixture fix;
    ModelLifecycleManager mgr(fix.model_registry, fix.runtime_registry);

    ASSERT_TRUE(mgr.GetAllLoaded().empty());

    DeviceConfig config;
    mgr.Load("resnet50", "onnx", config);
    mgr.Load("llama3-8b", "llama_cpp", config);

    auto loaded = mgr.GetAllLoaded();
    ASSERT_EQ(loaded.size(), 2u);
}

TEST(lifecycle_query_status_not_loaded) {
    TestFixture fix;
    ModelLifecycleManager mgr(fix.model_registry, fix.runtime_registry);

    ASSERT_EQ(mgr.QueryStatus("resnet50"), ModelStatus::NotLoaded);
}


// ===========================================================================
// Full Integration: Registry → Lifecycle → Inference → Unload
// ===========================================================================

TEST(integration_full_onnx_pipeline) {
    TestFixture fix;
    ModelLifecycleManager mgr(fix.model_registry, fix.runtime_registry);
    QuantizationMetadataManager qm(fix.model_registry);

    // 1. Query quantization variants
    auto variants = qm.GetVariants("resnet50");
    ASSERT_EQ(variants.size(), 2u);

    // 2. Load model
    DeviceConfig config;
    config.quantization_variant_id = variants[0].variant_id;  // fp32
    auto model = mgr.Load("resnet50", "", config);
    ASSERT_TRUE(model.has_value());

    // 3. Run inference
    mgr.AcquireInference("resnet50");
    InferenceRequest req;
    req.request_id = "integ-001";
    req.model_id   = "resnet50";
    InferenceResult result = model.value()->RunInference(req);
    mgr.ReleaseInference("resnet50");

    ASSERT_TRUE(result.success);
    ASSERT_EQ(result.request_id, "integ-001");

    // 4. Unload
    bool ok = mgr.Unload("resnet50");
    ASSERT_TRUE(ok);
    ASSERT_FALSE(mgr.IsLoaded("resnet50"));
}

TEST(integration_full_llama_pipeline) {
    TestFixture fix;
    ModelLifecycleManager mgr(fix.model_registry, fix.runtime_registry);
    QuantizationMetadataManager qm(fix.model_registry);

    // 1. Query quantization
    auto def = qm.GetDefault("llama3-8b");
    ASSERT_TRUE(def.has_value());
    ASSERT_EQ(def->variant_id, "q4_k_m");

    // 2. Load model (auto-resolve runtime)
    DeviceConfig config;
    config.quantization_variant_id = def->variant_id;
    auto model = mgr.Load("llama3-8b", "", config);
    ASSERT_TRUE(model.has_value());

    // 3. Run inference with prompt
    mgr.AcquireInference("llama3-8b");
    InferenceRequest req;
    req.request_id = "integ-llm-001";
    req.model_id   = "llama3-8b";
    req.text_inputs["prompt"] = "Explain EdgePilot in one sentence.";
    InferenceResult result = model.value()->RunInference(req);
    mgr.ReleaseInference("llama3-8b");

    ASSERT_TRUE(result.success);
    ASSERT_TRUE(result.text_outputs.count("generated_text") > 0);
    auto text = result.text_outputs.at("generated_text");
    ASSERT_TRUE(text.find("[llama-stub]") != std::string::npos);

    // 4. Unload
    mgr.Unload("llama3-8b");
    ASSERT_FALSE(mgr.IsLoaded("llama3-8b"));
}

TEST(integration_multi_model_concurrent_load) {
    TestFixture fix;
    ModelLifecycleManager mgr(fix.model_registry, fix.runtime_registry);

    DeviceConfig config;
    auto onnx_model  = mgr.Load("resnet50", "onnx", config);
    auto llama_model = mgr.Load("llama3-8b", "llama_cpp", config);

    ASSERT_TRUE(onnx_model.has_value());
    ASSERT_TRUE(llama_model.has_value());
    ASSERT_EQ(mgr.GetAllLoaded().size(), 2u);

    // Run inference on both
    InferenceRequest req1;
    req1.request_id = "multi-onnx";
    req1.model_id   = "resnet50";
    auto r1 = onnx_model.value()->RunInference(req1);
    ASSERT_TRUE(r1.success);

    InferenceRequest req2;
    req2.request_id = "multi-llama";
    req2.model_id   = "llama3-8b";
    req2.text_inputs["prompt"] = "Hello";
    auto r2 = llama_model.value()->RunInference(req2);
    ASSERT_TRUE(r2.success);

    // Unload both
    mgr.Unload("resnet50");
    mgr.Unload("llama3-8b");
    ASSERT_TRUE(mgr.GetAllLoaded().empty());
}


// ===========================================================================
// Main
// ===========================================================================

int main() {
    std::cout << "\n========================================" << std::endl;
    std::cout << " EdgePilot P1 — Model Layer Test Suite  " << std::endl;
    std::cout << "========================================\n" << std::endl;

    // Tests already ran via static initializers.

    std::cout << "\n----------------------------------------" << std::endl;
    std::cout << " Results: " << tests_passed << " passed, "
              << tests_failed << " failed, "
              << (tests_passed + tests_failed) << " total" << std::endl;
    std::cout << "----------------------------------------\n" << std::endl;

    return tests_failed > 0 ? 1 : 0;
}
