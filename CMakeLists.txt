cmake_minimum_required(VERSION 3.10)
project(opencl_bench)

set(CMAKE_CXX_STANDARD 17)      # Use C++17
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(OpenCL_INCLUDE_DIR "")
set(OpenCL_LIBRARY "")


include_directories(${OpenCL_INCLUDE_DIR})

add_executable(benchmark src/benchmark.cpp)

target_link_libraries(benchmark ${OpenCL_LIBRARY})