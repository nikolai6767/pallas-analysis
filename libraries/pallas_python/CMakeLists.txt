cmake_minimum_required(VERSION 3.14.0)
set (CMAKE_CXX_STANDARD 20)

set(CMAKE_SHARED_MODULE_PREFIX "")  # Remove default 'lib' prefix
set(CMAKE_SHARED_LIBRARY_SUFFIX ${Python3_SOABI}.so)  # Ensure correct suffix for Python

add_library(pallas_python MODULE pallas_python.cpp grammar.cpp structure.cpp)
set_property(TARGET pallas_python PROPERTY PUBLIC_HEADER grammar.h structure.h pallas_python.h)

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
        Python3::NumPy
)

install(TARGETS pallas_python LIBRARY DESTINATION ${INSTALL_LIBDIR})