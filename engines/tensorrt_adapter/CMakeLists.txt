# TensorRT adapter implementation
add_library(tensorrt_adapter SHARED
    tensorrt_engine.cpp
)

target_include_directories(tensorrt_adapter PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CUDA_INCLUDE_DIRS}
)

# Find TensorRT
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    HINTS /usr/include/aarch64-linux-gnu
          /usr/local/cuda/include
)

find_library(TENSORRT_LIBRARY nvinfer
    HINTS /usr/lib/aarch64-linux-gnu
)

if(TENSORRT_INCLUDE_DIR AND TENSORRT_LIBRARY)
    message(STATUS "TensorRT found:")
    message(STATUS "  Include: ${TENSORRT_INCLUDE_DIR}")
    message(STATUS "  Library: ${TENSORRT_LIBRARY}")

    target_include_directories(tensorrt_adapter PUBLIC ${TENSORRT_INCLUDE_DIR})
    target_link_libraries(tensorrt_adapter
        ${TENSORRT_LIBRARY}
        ${CUDA_LIBRARIES}
    )
else()
    message(WARNING "TensorRT not found, adapter will not be built")
endif()
