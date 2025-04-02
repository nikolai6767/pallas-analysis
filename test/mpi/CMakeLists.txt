cmake_minimum_required(VERSION 3.9)

find_package(MPI REQUIRED)

add_executable(mpi_benchmark mpi_benchmark.c)
target_link_libraries(mpi_benchmark PUBLIC pallas MPI::MPI_C)
target_include_directories(mpi_benchmark PUBLIC MPI::MPI_C_INCLUDE_DIRS ${CMAKE_SOURCE_DIR}/src/pallas/include)
add_test(NAME mpi_benchmark COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} ./mpi_benchmark ${MPIEXEC_POSTFLAGS})