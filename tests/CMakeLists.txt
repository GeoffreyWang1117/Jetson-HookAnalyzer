# Simple test placeholder
# Add actual tests here using Google Test or similar framework

add_executable(test_basic
    test_basic.cpp
)

target_include_directories(test_basic PRIVATE
    ${CMAKE_SOURCE_DIR}/core
    ${CUDA_INCLUDE_DIRS}
)

target_link_libraries(test_basic
    cuda_hook
    scheduler
    profiler
    ${CUDA_cudart_LIBRARY}  # Use shared CUDA runtime instead of static
    ${CMAKE_DL_LIBS}
)

add_test(NAME BasicTest COMMAND test_basic)
