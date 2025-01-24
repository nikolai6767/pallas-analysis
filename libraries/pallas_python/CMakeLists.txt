cmake_minimum_required(VERSION 3.14.0)

# POLICIES #####################
if (POLICY CMP0076)
    cmake_policy(SET CMP0076 NEW)
endif ()

find_package(Python3 COMPONENTS Interpreter Development NumPy REQUIRED)

add_library(pallas_python SHARED)

target_sources(pallas_python
        PRIVATE
        pallas_python.cpp
)

set_target_properties(
        pallas_python
        PROPERTIES
            PREFIX ""
            OUTPUT_NAME "pallas_python"
            LINKER_LANGUAGE CXX
)
target_compile_options(pallas_python
        PUBLIC
        -D_GNU_SOURCE
        PRIVATE
        -Wall -Wextra ${WARNINGS}
)
message("Warning list: ${WARNINGS}")

target_link_libraries(pallas_python
        PUBLIC
        dl
        PRIVATE
        pallas
        ${CMAKE_DL_LIBS}
        rt
        m
        ${Python3_LIBRARIES}
)

target_include_directories(pallas_python PUBLIC ${Python3_INCLUDE_DIRS})
target_link_directories(pallas_python PRIVATE ${Python3_LIBRARIES})