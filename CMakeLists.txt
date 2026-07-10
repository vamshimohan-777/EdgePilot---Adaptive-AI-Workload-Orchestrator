cmake_minimum_required(VERSION 3.16)

project(EdgePilot
    VERSION 0.1.0
    DESCRIPTION "An OS-Inspired Orchestration Framework for Adaptive On-Device AI"
    LANGUAGES CXX
)

# ---------------------------------------------------------------------------
# Global C++ Standard
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)   # Use -std=c++17, not -std=gnu++17

# ---------------------------------------------------------------------------
# Build type defaults
# ---------------------------------------------------------------------------
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Debug" CACHE STRING
        "Build type: Debug | Release | RelWithDebInfo | MinSizeRel" FORCE)
endif()

message(STATUS "EdgePilot build type : ${CMAKE_BUILD_TYPE}")
message(STATUS "EdgePilot C++ standard: C++${CMAKE_CXX_STANDARD}")

# ---------------------------------------------------------------------------
# Compiler warnings
# ---------------------------------------------------------------------------
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(
        -Wall
        -Wextra
        -Wpedantic
        -Wshadow
        -Wnon-virtual-dtor
        -Woverloaded-virtual
    )
elseif(MSVC)
    add_compile_options(/W4 /permissive-)
endif()

# ---------------------------------------------------------------------------
# Subprojects
# ---------------------------------------------------------------------------
add_subdirectory(core)

# Tests are optional — only built if testing is enabled
option(EDGEPILOT_BUILD_TESTS "Build unit and integration tests" ON)
if(EDGEPILOT_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
