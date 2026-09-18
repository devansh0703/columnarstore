cmake_minimum_required(VERSION 3.20)

add_executable(basic_scan basic_scan.cpp)
target_link_libraries(basic_scan PRIVATE columnarstore)
