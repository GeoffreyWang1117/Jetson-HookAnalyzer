# Benchmark executables
add_executable(benchmark_kernels
    benchmark_kernels.cpp
)

target_include_directories(benchmark_kernels PRIVATE
    ${CMAKE_SOURCE_DIR}/core
    ${CUDA_INCLUDE_DIRS}
)

target_link_libraries(benchmark_kernels
    optimized_kernels
    profiler
    ${CUDA_LIBRARIES}
    ${CUDA_CUBLAS_LIBRARIES}
)
