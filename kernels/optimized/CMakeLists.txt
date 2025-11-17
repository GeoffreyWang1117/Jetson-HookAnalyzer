add_library(optimized_kernels SHARED
    kernels.cu
    gemm_optimized_v2.cu
)

target_include_directories(optimized_kernels PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CUDA_INCLUDE_DIRS}
)

set_target_properties(optimized_kernels PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    POSITION_INDEPENDENT_CODE ON
)

target_link_libraries(optimized_kernels
    ${CUDA_LIBRARIES}
)
