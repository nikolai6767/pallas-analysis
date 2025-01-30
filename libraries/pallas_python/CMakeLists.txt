cmake_minimum_required(VERSION 3.14.0)

# POLICIES #####################
if (POLICY CMP0076)
    cmake_policy(SET CMP0076 NEW)
endif ()

find_package(Python3 COMPONENTS Interpreter Development NumPy REQUIRED)

execute_process(
        COMMAND ${Python_EXECUTABLE} -c "import skbuild; import os; print(os.path.join(os.path.dirname(skbuild.__file__), 'resources', 'cmake'))"
        OUTPUT_VARIABLE SCIKIT_BUILD_CMAKE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
)
message(SCIKIT_BUILD_CMAKE_DIR=${SCIKIT_BUILD_CMAKE_DIR})
cmake_policy(SET CMP0148 OLD)
# Add the scikit-build cmake directory to CMAKE_MODULE_PATH
list(APPEND CMAKE_MODULE_PATH ${SCIKIT_BUILD_CMAKE_DIR})

find_package(PythonExtensions REQUIRED)
add_library(pallas_python MODULE pallas_python.cpp)

python_extension_module(pallas_python)

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

target_link_libraries(pallas_python
        ${Python3_LIBRARIES}
        pallas
)

target_include_directories(pallas_python PUBLIC ${Python3_INCLUDE_DIRS})
target_link_directories(pallas_python PRIVATE ${Python3_LIBRARIES})