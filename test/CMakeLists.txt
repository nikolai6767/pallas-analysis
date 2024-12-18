cmake_minimum_required(VERSION 3.14)
# add_subdirectory(mpi)

add_compile_options(-Wall -Wextra -Wpedantic -Wno-sign-compare ${WARNINGS})
include_directories(${CMAKE_SOURCE_DIR}/src/)
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
link_libraries(pallas Threads::Threads)


enable_testing()

add_executable(test_collection test_collection.cpp)
target_link_libraries(test_collection GTest::gtest_main)


include(GoogleTest)
gtest_discover_tests(test_collection)

####################################################
add_executable(write_benchmark write_benchmark.c)
add_executable(write_benchmark_CPP write_benchmark.cpp)

SET(TRACE_NAME ${CMAKE_CURRENT_BINARY_DIR}/write_benchmark_trace)
SET(N_THREADS 4)
SET(N_ITER 40)

add_test(NAME write_benchmark COMMAND write_benchmark -n ${N_ITER} -t ${N_THREADS})

add_test(NAME info_benchmark COMMAND pallas_info ${TRACE_NAME}/main.pallas)
add_test(NAME print_benchmark COMMAND pallas_print ${TRACE_NAME}/main.pallas)
add_test(NAME print_benchmark_structure COMMAND pallas_print -S ${TRACE_NAME}/main.pallas)

add_test (benchmark_checks bash
        "${CMAKE_CURRENT_SOURCE_DIR}/write_benchmark.sh"
        "${CMAKE_BINARY_DIR}" ${TRACE_NAME} -n ${N_ITER} -t ${N_THREADS})

set_tests_properties(info_benchmark print_benchmark print_benchmark_structure PROPERTIES
        REQUIRED_FILES ${TRACE_NAME})
set_tests_properties(benchmark_checks PROPERTIES
        REQUIRED "info_benchmark;print_benchmark;print_benchmark_structure")


add_test(NAME write_benchmark_CPP COMMAND write_benchmark -n ${N_ITER} -t ${N_THREADS})

add_test(NAME info_benchmark_CPP COMMAND pallas_info ${TRACE_NAME}/main.pallas)
add_test(NAME print_benchmark_CPP COMMAND pallas_print ${TRACE_NAME}/main.pallas)
add_test(NAME print_benchmark_structure_CPP COMMAND pallas_print -S ${TRACE_NAME}/main.pallas)

add_test (benchmark_checks_CPP bash
        "${CMAKE_CURRENT_SOURCE_DIR}/write_benchmark.sh"
        "${CMAKE_BINARY_DIR}" ${TRACE_NAME} -n ${N_ITER} -t ${N_THREADS})

set_tests_properties(info_benchmark_CPP print_benchmark_CPP print_benchmark_structure_CPP PROPERTIES
        REQUIRED_FILES ${TRACE_NAME})
set_tests_properties(benchmark_checks_CPP PROPERTIES
        REQUIRED "info_benchmark_CPP;print_benchmark_CPP;print_benchmark_structure_CPP")


add_test(NAME write_benchmark_logical COMMAND write_benchmark -n ${N_ITER} -t ${N_THREADS} -l)

add_test(NAME info_benchmark_logical COMMAND pallas_info ${TRACE_NAME}/main.pallas)
add_test(NAME print_benchmark_logical COMMAND pallas_print ${TRACE_NAME}/main.pallas)
add_test(NAME print_benchmark_structure_logical COMMAND pallas_print -S ${TRACE_NAME}/main.pallas)

add_test (benchmark_checks_logical bash
        "${CMAKE_CURRENT_SOURCE_DIR}/write_benchmark.sh"
        "${CMAKE_BINARY_DIR}" ${TRACE_NAME} -n ${N_ITER} -t ${N_THREADS} -l)


set_tests_properties(info_benchmark_logical print_benchmark_logical print_benchmark_structure_logical PROPERTIES
        REQUIRED_FILES ${TRACE_NAME})
set_tests_properties(benchmark_checks_logical PROPERTIES
        REQUIRED "info_benchmark_logical;print_benchmark_logical;print_benchmark_structure_logical")