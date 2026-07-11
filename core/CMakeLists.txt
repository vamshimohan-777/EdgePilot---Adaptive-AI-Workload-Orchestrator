# ---------------------------------------------------------------------------
# edgepilot_core — Static library for the P1 Runtime & Model Infrastructure
# ---------------------------------------------------------------------------

# Public headers — any target that links edgepilot_core can use these
set(EDGEPILOT_CORE_HEADERS
    include/edgepilot/types.h
    include/edgepilot/runtime_interface.h
    include/edgepilot/runtime_registry.h
    include/edgepilot/onnx_runtime_adapter.h
)

# Implementation sources — compiled into the static library
set(EDGEPILOT_CORE_SOURCES
    src/runtime_registry.cpp
    src/onnx_runtime_adapter.cpp
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
)

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
