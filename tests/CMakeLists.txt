# ---------------------------------------------------------------------------
# EdgePilot — Test Suite
# ---------------------------------------------------------------------------

# Runtime Layer Tests (Day 2)
add_executable(test_runtime_layer
    test_runtime_layer.cpp
)

target_link_libraries(test_runtime_layer PRIVATE edgepilot_core)

add_test(NAME RuntimeLayerTests COMMAND test_runtime_layer)
