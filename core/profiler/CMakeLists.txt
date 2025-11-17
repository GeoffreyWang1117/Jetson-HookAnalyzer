add_library(profiler SHARED
    profiler.cpp
)

target_include_directories(profiler PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CUDA_INCLUDE_DIRS}
)

target_link_libraries(profiler
    ${CUDA_LIBRARIES}
    Threads::Threads
)

if(CUPTI_LIBRARY)
    target_link_libraries(profiler ${CUPTI_LIBRARY})
    target_compile_definitions(profiler PRIVATE ENABLE_CUPTI)
endif()

set_target_properties(profiler PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)
