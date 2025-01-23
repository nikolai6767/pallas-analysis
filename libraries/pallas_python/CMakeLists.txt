cmake_minimum_required(VERSION 3.14.0)

# POLICIES #####################
if (POLICY CMP0076)
    cmake_policy(SET CMP0076 NEW)
endif ()

find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})
add_library(pallas_python SHARED)

target_sources(pallas_python
        PRIVATE
        pallas_python.cpp
)

target_compile_options(pallas
        PUBLIC
        -D_GNU_SOURCE
        PRIVATE
        -Wall -Wextra ${WARNINGS}
)
message("Warning list: ${WARNINGS}")

target_link_libraries(pallas
        PUBLIC
        dl
        PRIVATE
        ${CMAKE_DL_LIBS}
        rt
        m
        ${PYTHON_LIBRARIES}
)
target_link_directories(pallas PRIVATE ${PYTHON_LIBRARIES})