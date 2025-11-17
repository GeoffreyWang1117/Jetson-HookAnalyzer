add_library(scheduler SHARED
    scheduler.cpp
)

target_include_directories(scheduler PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CUDA_INCLUDE_DIRS}
)

target_link_libraries(scheduler
    ${CUDA_LIBRARIES}
    Threads::Threads
)

set_target_properties(scheduler PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)
