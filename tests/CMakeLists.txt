# ---------------------------------------------------------------------------
# EdgePilot — Test Suite
# ---------------------------------------------------------------------------

# Runtime Layer Tests (Day 2)
add_executable(test_runtime_layer
    test_runtime_layer.cpp
)
target_link_libraries(test_runtime_layer PRIVATE edgepilot_core)
add_test(NAME RuntimeLayerTests COMMAND test_runtime_layer)

# Model Layer Tests (Day 3)
add_executable(test_model_layer
    test_model_layer.cpp
)
target_link_libraries(test_model_layer PRIVATE edgepilot_core)
add_test(NAME ModelLayerTests COMMAND test_model_layer)

# Orchestrator Main Daemon Executable (P2/P3/P4)
add_executable(edgepilot_daemon
    main_orchestrator.cpp
)
target_link_libraries(edgepilot_daemon PRIVATE edgepilot_core)

# Performance Benchmark Executable (P4)
add_executable(edgepilot_benchmark
    benchmark_framework.cpp
)
target_link_libraries(edgepilot_benchmark PRIVATE edgepilot_core)



