cmake_minimum_required(VERSION 3.10)
project(opencl_bench)

set(CMAKE_CXX_STANDARD 17)      # Use C++17
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Let CMake find OpenCL automatically
find_package(OpenCL REQUIRED)

# Find OpenMP
find_package(OpenMP REQUIRED)


include_directories(${OpenCL_INCLUDE_DIR})

add_executable(benchmark src/benchmark.cpp src/openMP_functions.cpp src/openCL_functions.cpp)

target_link_libraries(benchmark ${OpenCL_LIBRARY} OpenMP::OpenMP_CXX)