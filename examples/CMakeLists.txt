# Minimal demo without cuda_hook (to avoid linking issues)
add_executable(simple_demo
    simple_demo_minimal.cpp
)

target_include_directories(simple_demo PRIVATE
    ${CMAKE_SOURCE_DIR}/core
    ${CUDA_INCLUDE_DIRS}
)

target_link_libraries(simple_demo
    scheduler
    profiler
    optimized_kernels
    ${CUDA_LIBRARIES}
    Threads::Threads
)

# Kernel-only test (most stable)
add_executable(kernel_test
    kernel_test.cpp
)

target_include_directories(kernel_test PRIVATE
    ${CMAKE_SOURCE_DIR}/core
    ${CUDA_INCLUDE_DIRS}
)

target_link_libraries(kernel_test
    optimized_kernels
    ${CUDA_LIBRARIES}
)
