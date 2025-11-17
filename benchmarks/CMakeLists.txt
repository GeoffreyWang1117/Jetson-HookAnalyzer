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

# GEMM analysis tool
add_executable(gemm_analysis
    gemm_analysis.cpp
)

target_include_directories(gemm_analysis PRIVATE
    ${CMAKE_SOURCE_DIR}/core
    ${CUDA_INCLUDE_DIRS}
)

target_link_libraries(gemm_analysis
    optimized_kernels
    ${CUDA_LIBRARIES}
    ${CUDA_CUBLAS_LIBRARIES}
)

# GEMM comparison tool
add_executable(gemm_compare
    gemm_compare.cpp
)

target_include_directories(gemm_compare PRIVATE
    ${CMAKE_SOURCE_DIR}/core
    ${CUDA_INCLUDE_DIRS}
)

target_link_libraries(gemm_compare
    optimized_kernels
    ${CUDA_LIBRARIES}
    ${CUDA_CUBLAS_LIBRARIES}
)
