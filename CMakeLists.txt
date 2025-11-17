cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(HookAnalyzer LANGUAGES CXX CUDA VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Build options
option(BUILD_TESTS "Build unit tests" ON)
option(BUILD_BENCHMARKS "Build benchmarks" ON)
option(BUILD_EXAMPLES "Build examples" ON)
option(ENABLE_TENSORRT "Enable TensorRT support" ON)
option(ENABLE_ONNX "Enable ONNX Runtime support" ON)
option(ENABLE_PROFILING "Enable CUPTI profiling" ON)

# Set build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pthread")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Wall")

# CUDA architecture (Jetson Orin Nano uses Ampere SM 8.7)
set(CMAKE_CUDA_ARCHITECTURES 87 75 72 61)

# Find packages
find_package(CUDA REQUIRED)
find_package(Threads REQUIRED)

message(STATUS "CUDA Version: ${CUDA_VERSION}")
message(STATUS "CUDA Toolkit Root: ${CUDA_TOOLKIT_ROOT_DIR}")

# CUPTI for profiling
if(ENABLE_PROFILING)
    find_library(CUPTI_LIBRARY
        NAMES cupti
        PATHS ${CUDA_TOOLKIT_ROOT_DIR}
        PATH_SUFFIXES lib64 lib/x64 lib
    )
    if(CUPTI_LIBRARY)
        message(STATUS "Found CUPTI: ${CUPTI_LIBRARY}")
    else()
        message(WARNING "CUPTI not found, profiling features will be limited")
    endif()
endif()

# TensorRT
if(ENABLE_TENSORRT)
    find_path(TENSORRT_INCLUDE_DIR NvInfer.h
        HINTS /usr/include/aarch64-linux-gnu /usr/include/x86_64-linux-gnu
        PATHS /usr/include /usr/local/include
    )
    find_library(TENSORRT_LIBRARY nvinfer
        HINTS /usr/lib/aarch64-linux-gnu /usr/lib/x86_64-linux-gnu
        PATHS /usr/lib /usr/local/lib
    )
    if(TENSORRT_INCLUDE_DIR AND TENSORRT_LIBRARY)
        message(STATUS "Found TensorRT: ${TENSORRT_LIBRARY}")
        set(TENSORRT_FOUND TRUE)
    else()
        message(WARNING "TensorRT not found, TensorRT adapter will be disabled")
        set(TENSORRT_FOUND FALSE)
    endif()
endif()

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/core
    ${CUDA_INCLUDE_DIRS}
)

if(CUPTI_LIBRARY)
    include_directories(${CUDA_TOOLKIT_ROOT_DIR}/extras/CUPTI/include)
endif()

if(TENSORRT_FOUND)
    include_directories(${TENSORRT_INCLUDE_DIR})
endif()

# Subdirectories
add_subdirectory(core/cuda_hook)
add_subdirectory(core/scheduler)
add_subdirectory(core/profiler)
add_subdirectory(kernels/optimized)

if(TENSORRT_FOUND)
    add_subdirectory(engines/tensorrt_adapter)
endif()

if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if(BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Main library (without cuda_hook to avoid symbol conflicts)
add_library(hook_analyzer SHARED
    core/scheduler/scheduler.cpp
    core/profiler/profiler.cpp
)

target_link_libraries(hook_analyzer
    ${CUDA_LIBRARIES}
    Threads::Threads
)

# Note: cuda_hook is a separate library to avoid CUDA runtime symbol conflicts
# Use it with LD_PRELOAD: LD_PRELOAD=./libcuda_hook.so ./your_app

if(CUPTI_LIBRARY)
    target_link_libraries(hook_analyzer ${CUPTI_LIBRARY})
endif()

# Installation
install(TARGETS hook_analyzer
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(DIRECTORY core/
    DESTINATION include/hook_analyzer
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)

# Summary
message(STATUS "")
message(STATUS "=== HookAnalyzer Configuration ===")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "TensorRT: ${TENSORRT_FOUND}")
message(STATUS "CUPTI Profiling: ${ENABLE_PROFILING}")
message(STATUS "Build Tests: ${BUILD_TESTS}")
message(STATUS "Build Benchmarks: ${BUILD_BENCHMARKS}")
message(STATUS "==================================")
message(STATUS "")
