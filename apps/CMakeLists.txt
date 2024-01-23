cmake_minimum_required(VERSION 3.13)

add_compile_options(-Wall -Wextra -Wpedantic)
include_directories(${CMAKE_SOURCE_DIR}/src/)
link_libraries(pallas zstd)
configure_file(../config.json config.json COPYONLY)

add_executable(pallas_print pallas_print.cpp)
add_executable(pallas_info pallas_info.cpp)
add_executable(pallas_comm_matrix pallas_comm_matrix.cpp)

install(
  TARGETS pallas_print pallas_info
  LIBRARY DESTINATION ${INSTALL_LIBDIR}
  RUNTIME DESTINATION ${INSTALL_BINDIR}
  INCLUDES DESTINATION ${INSTALL_INCLUDEDIR}
)
