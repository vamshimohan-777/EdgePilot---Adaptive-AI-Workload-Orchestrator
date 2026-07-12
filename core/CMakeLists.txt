# ---------------------------------------------------------------------------
# edgepilot_core — Static library for EdgePilot core orchestrator and runtimes
# ---------------------------------------------------------------------------

# Public headers — any target that links edgepilot_core can use these
set(EDGEPILOT_CORE_HEADERS
    include/edgepilot/types.h
    include/edgepilot/runtime_interface.h
    include/edgepilot/runtime_registry.h
    include/edgepilot/onnx_runtime_adapter.h
    include/edgepilot/llama_cpp_adapter.h
    include/edgepilot/model_registry.h
    include/edgepilot/quantization_manager.h
    include/edgepilot/lifecycle_manager.h
    include/edgepilot/scheduler_interface.h
    include/edgepilot/job_queue.h
    include/edgepilot/worker_pool.h
    include/edgepilot/telemetry_collector.h
    include/edgepilot/pipeline_executor.h
    include/edgepilot/ai_scheduler.h
    include/edgepilot/orchestrator.h
)

# Implementation sources — compiled into the static library
set(EDGEPILOT_CORE_SOURCES
    src/runtime_registry.cpp
    src/onnx_runtime_adapter.cpp
    src/llama_cpp_adapter.cpp
    src/model_registry.cpp
    src/quantization_manager.cpp
    src/lifecycle_manager.cpp
    src/job_queue.cpp
    src/worker_pool.cpp
    src/telemetry_collector.cpp
    src/pipeline_executor.cpp
    src/ai_scheduler.cpp
    src/orchestrator.cpp
)

add_library(edgepilot_core STATIC
    ${EDGEPILOT_CORE_HEADERS}
    ${EDGEPILOT_CORE_SOURCES}
)

# Make headers available to consumers via target_link_libraries
target_include_directories(edgepilot_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/onnxruntime/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/llama_cpp/include
)

# Link Winsock on Windows for our IPC socket server
if(WIN32)
    target_link_libraries(edgepilot_core PUBLIC ws2_32)
    target_link_libraries(edgepilot_core PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/onnxruntime/lib/onnxruntime.lib
        ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/llama_cpp/llama.dll
    )
endif()

# Enforce C++17 on consumers as well
target_compile_features(edgepilot_core PUBLIC cxx_std_17)


# ---------------------------------------------------------------------------
# IDE source grouping (optional, helps Visual Studio / CLion)
# ---------------------------------------------------------------------------
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/include" PREFIX "Headers" FILES
    ${EDGEPILOT_CORE_HEADERS}
)
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/src" PREFIX "Sources" FILES
    ${EDGEPILOT_CORE_SOURCES}
)
