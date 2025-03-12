cmake_minimum_required(VERSION 3.14.0)
set (CMAKE_CXX_STANDARD 20)

find_package(Python3 COMPONENTS Interpreter Development)

if (Python3_FOUND)
    if (Python3_VERSION VERSION_LESS 3.10)
        message("Python version was too low: ${Python3_VERSION} <= 3.10")
        message("Skipping Python Library Generation")
        return(-1)
    endif()
endif()

execute_process(COMMAND pybind11-config --cmakedir OUTPUT_VARIABLE pybind11_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
find_package(pybind11 CONFIG REQUIRED)


set(CMAKE_SHARED_MODULE_PREFIX "")  # Remove default 'lib' prefix
set(CMAKE_SHARED_LIBRARY_SUFFIX ${Python3_SOABI}.so)  # Ensure correct suffix for Python

pybind11_add_module(pallas_python MODULE pallas_python.cpp)
set_property(TARGET pallas_python PROPERTY PUBLIC_HEADER pallas_python.h)

list(APPEND WARNINGS "-Wno-missing-field-initializers")

target_compile_options(pallas_python
        PUBLIC
        -D_GNU_SOURCE
        PRIVATE
        -Wall -Wextra ${WARNINGS}
)

target_include_directories(pallas_python PUBLIC ${Python3_INCLUDE_DIRS})
target_link_directories(pallas_python PRIVATE ${Python3_LIBRARY_DIRS})

target_link_libraries(pallas_python
        PRIVATE
        dl
        pallas
        ${Python3_LIBRARIES}
)

install(TARGETS pallas_python LIBRARY DESTINATION ${INSTALL_LIBDIR})