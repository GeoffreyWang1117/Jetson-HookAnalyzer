add_library(cuda_hook SHARED
    cuda_hook.cpp
)

target_include_directories(cuda_hook PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CUDA_INCLUDE_DIRS}
)

target_link_libraries(cuda_hook
    ${CMAKE_DL_LIBS}
    # Note: Don't link ${CUDA_LIBRARIES} to avoid symbol conflicts with hook functions
)

set_target_properties(cuda_hook PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)
