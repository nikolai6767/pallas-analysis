cmake_minimum_required(VERSION 3.14.0)
set (CMAKE_CXX_STANDARD 20)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development NumPy)


set(CMAKE_SHARED_MODULE_PREFIX "")  # Remove default 'lib' prefix
set(CMAKE_SHARED_LIBRARY_SUFFIX ${Python3_SOABI}.so)  # Ensure correct suffix for Python

add_library(pallas_python MODULE pallas_python.cpp)

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